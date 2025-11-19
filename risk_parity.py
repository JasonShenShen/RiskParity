import argparse
import json
import math
import os
import random
import time
import urllib.request
import csv
import io

def _matvec(m, v):
    # 矩阵-向量乘法：计算 m × v
    n = len(v)
    r = [0.0] * n
    for i in range(n):
        s = 0.0
        row = m[i]
        for j in range(n):
            s += row[j] * v[j]  # 计算第i行与向量v的点积
        r[i] = s
    return r

def _dot(a, b):
    # 向量点积：计算 a · b
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s

def _normalize_nonnegative(w):
    # 确保权重非负且和为1
    for i in range(len(w)):
        if w[i] < 0.0:
            w[i] = 0.0  # 负权重置为0
    s = sum(w)
    if s == 0.0:
        u = 1.0 / len(w)
        return [u] * len(w)  # 如果和为0，返回等权重
    return [x / s for x in w]  # 归一化

def _covariance_matrix(returns):
    """
    计算资产收益率的协方差矩阵
    n: 观测数量
    k: 资产数量
    returns: 资产收益率矩阵（n行k列）
    """
    n = len(returns)  # 观测数量
    if n <= 1:
        return []  # 不足2个观测点无法计算协方差
    
    k = len(returns[0])  # 资产数量
    
    # 1. 计算均值
    means = [0.0] * k
    for r in returns:
        for i in range(k):
            means[i] += r[i]
    for i in range(k):
        means[i] /= n
    
    # 2. 计算协方差矩阵
    m = [[0.0 for _ in range(k)] for _ in range(k)]
    for r in returns:
        for i in range(k):
            di = r[i] - means[i]  # 偏差
            for j in range(k):
                dj = r[j] - means[j]
                m[i][j] += di * dj  # 协方差累加
    
    # 3. 除以(n-1)得到无偏估计
    c = 1.0 / (n - 1) if n > 1 else 0.0
    for i in range(k):
        for j in range(k):
            m[i][j] *= c
    return m

def risk_parity_weights(sigma, tol=1e-6, max_iter=5000, damping=0.5):
    """
    风险平价权重计算的核心函数
    sigma: 协方差矩阵
    tol: 收敛容忍度
    max_iter: 最大迭代次数
    damping: 阻尼系数，控制更新步长
    """
    n = len(sigma)
    
    # 1. 初始权重：反比于波动率
    vols = [math.sqrt(max(sigma[i][i], 0.0)) for i in range(n)]
    w = [1.0 / (v if v > 0.0 else 1.0) for v in vols]
    w = _normalize_nonnegative(w)
    
    # 2. 迭代优化
    for _ in range(max_iter):
        sw = _matvec(sigma, w)  # Σ × w
        varp = _dot(w, sw)      # w^T × Σ × w (组合方差)
        
        if varp <= 0.0:
            break
            
        # 3. 计算风险贡献
        rc = [w[i] * sw[i] for i in range(n)]  # 每个资产的风险贡献
        target = varp / n  # 目标风险贡献（等分）
        
        # 4. 检查收敛
        diff = 0.0
        for i in range(n):
            diff = max(diff, abs(rc[i] - target) / varp)
        if diff < tol:
            break  # 收敛
            
        # 5. 更新权重
        for i in range(n):
            scale = target / rc[i] if rc[i] > 0.0 else 1.0
            # 阻尼更新：w_new = (1-damping)*w_old + damping*w_old*scale
            w[i] = w[i] * (1.0 - damping) + w[i] * damping * scale
        w = _normalize_nonnegative(w)
    
    return _normalize_nonnegative(w)

def _cholesky(a):
    """
    Cholesky分解：将正定矩阵A分解为A = L × L^T
    用于生成具有特定相关性的随机数
    """
    n = len(a)
    l = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += l[i][k] * l[j][k]
            if i == j:  # 对角元素
                val = a[i][i] - s
                l[i][j] = math.sqrt(val if val > 0.0 else 0.0)
            else:  # 非对角元素
                denom = l[j][j]
                l[i][j] = (a[i][j] - s) / denom if denom != 0.0 else 0.0
    return l

def _synthetic_returns(n_days, vols_annual, corr):
    """
    生成具有指定波动率和相关性的合成收益率数据
    n_days: 天数
    vols_annual: 年化波动率列表
    corr: 相关性矩阵
    """
    n = len(vols_annual)
    vols_daily = [v / math.sqrt(252.0) for v in vols_annual]  # 转为日波动率
    l = _cholesky(corr)  # Cholesky分解相关性矩阵
    
    data = []
    for _ in range(n_days):
        # 生成独立标准正态随机数
        z = [random.gauss(0.0, 1.0) for _ in range(n)]
        # 通过Cholesky分解引入相关性
        y = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += l[i][j] * z[j]
            y[i] = s * vols_daily[i]  # 缩放到目标波动率
        data.append(y)
    return data

def _yahoo_symbol(sym):
    if sym.endswith('.SH'):
        return sym[:-3] + '.SS'
    return sym

def _download_yahoo_csv(symbol, start_ts, end_ts):
    """从Yahoo Finance下载历史价格数据"""
    u = (
        'https://query1.finance.yahoo.com/v7/finance/download/'
        + symbol
        + '?period1='
        + str(start_ts)
        + '&period2='
        + str(end_ts)
        + '&interval=1d&events=history&includeAdjustedClose=true'
    )
    r = urllib.request.urlopen(u, timeout=15)
    b = r.read()
    s = b.decode('utf-8')
    return s

def _parse_prices(csv_text):
    f = io.StringIO(csv_text)
    reader = csv.DictReader(f)
    out = []
    for row in reader:
        d = row.get('Date')
        a = row.get('Adj Close')
        if d is None or a is None:
            continue
        try:
            v = float(a)
        except Exception:
            continue
        out.append((d, v))
    return out

def _log_returns_from_prices(prices):
    """将价格序列转换为对数收益率"""
    r = []
    for i in range(1, len(prices)):
        p0 = prices[i - 1][1]
        p1 = prices[i][1]
        if p0 > 0.0 and p1 > 0.0:
            r.append((prices[i][0], math.log(p1 / p0)))  # ln(P_t/P_{t-1})
    return r

def _align_returns(symbols, lookback_days=252):
    """
    获取多个标的的对齐收益率数据
    只保留所有标的都有数据的交易日
    """
    end_ts = int(time.time())
    start_ts = end_ts - 5 * 365 * 24 * 3600  # 5年历史数据
    
    # 1. 下载各标的数据
    series = {}
    for s in symbols:
        ys = _yahoo_symbol(s)  # 转换标的代码
        try:
            csv_text = _download_yahoo_csv(ys, start_ts, end_ts)
            prices = _parse_prices(csv_text)
            returns = _log_returns_from_prices(prices)
            series[s] = returns
        except Exception:
            series[s] = []
    
    # 2. 找到所有标的共同的交易日
    dates = None
    for s in symbols:
        d = set([x[0] for x in series[s]])
        dates = d if dates is None else dates.intersection(d)
    
    # 3. 构建对齐的收益率矩阵
    if dates is None:
        return []
    
    ds = sorted(list(dates))
    ds = ds[-lookback_days:] if len(ds) > lookback_days else ds
    
    # 创建日期->收益率的映射
    idx = {}
    for s in symbols:
        m = {}
        for dt, v in series[s]:
            m[dt] = v
        idx[s] = m
    
    # 构建对齐的收益率数据
    rows = []
    for dt in ds:
        row = []
        ok = True
        for s in symbols:
            mv = idx[s].get(dt)
            if mv is None:
                ok = False
                break
            row.append(mv)
        if ok:
            rows.append(row)
    return rows

def run(source, user_symbols=None):
    """主执行函数"""
    if source == "yahoo" and user_symbols:
        # 使用真实市场数据
        returns = _align_returns(user_symbols, 252)
        if not returns:
            # 如果获取失败，使用默认合成数据
            returns = _synthetic_returns(
                252,
                [0.15, 0.12, 0.25, 0.20],  # 年化波动率
                [
                    [1.0, -0.2, 0.1, 0.1],   # 相关性矩阵
                    [-0.2, 1.0, -0.3, -0.2],
                    [0.1, -0.3, 1.0, 0.6],
                    [0.1, -0.2, 0.6, 1.0],
                ],
            )
        sigma = _covariance_matrix(returns)  # 计算协方差矩阵
        w = risk_parity_weights(sigma)       # 计算风险平价权重
        out = {user_symbols[i]: w[i] for i in range(len(user_symbols))}
        print(json.dumps({"weights": out}, ensure_ascii=False))
        return
    
    # 默认使用合成数据和预设标的
    tickers = ["GLD", "TLT", "QQQ", "CSI300"]
    returns = _synthetic_returns(
        252,
        [0.15, 0.12, 0.25, 0.20],  # 年化波动率
        [
            [1.0, -0.2, 0.1, 0.1],   # 相关性矩阵
            [-0.2, 1.0, -0.3, -0.2],
            [0.1, -0.3, 1.0, 0.6],
            [0.1, -0.2, 0.6, 1.0],
        ],
    )
    sigma = _covariance_matrix(returns)
    w = risk_parity_weights(sigma)
    out = {tickers[i]: w[i] for i in range(len(tickers))}
    print(json.dumps({"weights": out}, ensure_ascii=False))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="synthetic")
    p.add_argument("--tickers", nargs='*')
    args = p.parse_args()
    run(args.source, args.tickers)

if __name__ == "__main__":
    main()
