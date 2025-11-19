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
    n = len(v)
    r = [0.0] * n
    for i in range(n):
        s = 0.0
        row = m[i]
        for j in range(n):
            s += row[j] * v[j]
        r[i] = s
    return r

def _dot(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s

def _normalize_nonnegative(w):
    for i in range(len(w)):
        if w[i] < 0.0:
            w[i] = 0.0
    s = sum(w)
    if s == 0.0:
        u = 1.0 / len(w)
        return [u] * len(w)
    return [x / s for x in w]

def _covariance_matrix(returns):
    n = len(returns)
    if n == 0:
        return []
    k = len(returns[0])
    means = [0.0] * k
    for r in returns:
        for i in range(k):
            means[i] += r[i]
    for i in range(k):
        means[i] /= n
    m = [[0.0 for _ in range(k)] for _ in range(k)]
    for r in returns:
        for i in range(k):
            di = r[i] - means[i]
            for j in range(k):
                dj = r[j] - means[j]
                m[i][j] += di * dj
    c = 1.0 / (n - 1) if n > 1 else 0.0
    for i in range(k):
        for j in range(k):
            m[i][j] *= c
    return m

def risk_parity_weights(sigma, tol=1e-6, max_iter=5000, damping=0.5):
    n = len(sigma)
    vols = [math.sqrt(max(sigma[i][i], 0.0)) for i in range(n)]
    w = [1.0 / (v if v > 0.0 else 1.0) for v in vols]
    w = _normalize_nonnegative(w)
    for _ in range(max_iter):
        sw = _matvec(sigma, w)
        varp = _dot(w, sw)
        if varp <= 0.0:
            break
        rc = [w[i] * sw[i] for i in range(n)]
        target = varp / n
        diff = 0.0
        for i in range(n):
            diff = max(diff, abs(rc[i] - target) / varp)
        if diff < tol:
            break
        for i in range(n):
            scale = target / rc[i] if rc[i] > 0.0 else 1.0
            w[i] = w[i] * (1.0 - damping) + w[i] * damping * scale
        w = _normalize_nonnegative(w)
    return _normalize_nonnegative(w)

def _cholesky(a):
    n = len(a)
    l = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += l[i][k] * l[j][k]
            if i == j:
                val = a[i][i] - s
                l[i][j] = math.sqrt(val if val > 0.0 else 0.0)
            else:
                denom = l[j][j]
                l[i][j] = (a[i][j] - s) / denom if denom != 0.0 else 0.0
    return l

def _synthetic_returns(n_days, vols_annual, corr):
    n = len(vols_annual)
    vols_daily = [v / math.sqrt(252.0) for v in vols_annual]
    l = _cholesky(corr)
    data = []
    for _ in range(n_days):
        z = [random.gauss(0.0, 1.0) for _ in range(n)]
        y = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += l[i][j] * z[j]
            y[i] = s * vols_daily[i]
        data.append(y)
    return data

def _yahoo_symbol(sym):
    if sym.endswith('.SH'):
        return sym[:-3] + '.SS'
    return sym

def _download_yahoo_csv(symbol, start_ts, end_ts):
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
    r = []
    for i in range(1, len(prices)):
        p0 = prices[i - 1][1]
        p1 = prices[i][1]
        if p0 > 0.0 and p1 > 0.0:
            r.append((prices[i][0], math.log(p1 / p0)))
    return r

def _align_returns(symbols, lookback_days=252):
    end_ts = int(time.time())
    start_ts = end_ts - 5 * 365 * 24 * 3600
    series = {}
    for s in symbols:
        ys = _yahoo_symbol(s)
        try:
            t = _download_yahoo_csv(ys, start_ts, end_ts)
            px = _parse_prices(t)
            rr = _log_returns_from_prices(px)
            series[s] = rr
        except Exception:
            series[s] = []
    dates = None
    for s in symbols:
        d = set([x[0] for x in series[s]])
        dates = d if dates is None else dates.intersection(d)
    if dates is None:
        return []
    ds = sorted(list(dates))
    ds = ds[-lookback_days:] if len(ds) > lookback_days else ds
    idx = {}
    for s in symbols:
        m = {}
        for dt, v in series[s]:
            m[dt] = v
        idx[s] = m
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
    if source == "yahoo" and user_symbols:
        returns = _align_returns(user_symbols, 252)
        if not returns:
            returns = _synthetic_returns(
                252,
                [0.15, 0.12, 0.25, 0.20],
                [
                    [1.0, -0.2, 0.1, 0.1],
                    [-0.2, 1.0, -0.3, -0.2],
                    [0.1, -0.3, 1.0, 0.6],
                    [0.1, -0.2, 0.6, 1.0],
                ],
            )
        sigma = _covariance_matrix(returns)
        w = risk_parity_weights(sigma)
        out = {user_symbols[i]: w[i] for i in range(len(user_symbols))}
        print(json.dumps({"weights": out}, ensure_ascii=False))
        return
    tickers = ["GLD", "TLT", "QQQ", "CSI300"]
    returns = _synthetic_returns(
        252,
        [0.15, 0.12, 0.25, 0.20],
        [
            [1.0, -0.2, 0.1, 0.1],
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