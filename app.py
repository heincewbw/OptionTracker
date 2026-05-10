"""
Put Option Anomaly Scanner
Detects sell put opportunities where IV is significantly higher than HV
(options priced expensively vs realized volatility — good for option sellers).

Requires Python 3.10+
"""
import logging
import json
import uuid
import threading
from datetime import datetime, timedelta
from typing import Optional

import requests
import numpy as np
from flask import Flask, render_template, request, jsonify
import yfinance as yf

# ── Force HTTP timeouts on every yfinance network call ─────────────────────────
# yfinance >= 1.x uses curl_cffi (NOT standard requests) to bypass anti-bot.
# We patch BOTH libraries so no underlying HTTP call can hang indefinitely.
HTTP_TIMEOUT = 8

_orig_requests_request = requests.Session.request
def _requests_request_with_timeout(self, method, url, **kwargs):
    kwargs["timeout"] = HTTP_TIMEOUT  # force override, yfinance passes timeout=30 explicitly
    return _orig_requests_request(self, method, url, **kwargs)
requests.Session.request = _requests_request_with_timeout

try:
    from curl_cffi import requests as curl_requests
    _orig_curl_request = curl_requests.Session.request
    def _curl_request_with_timeout(self, method, url, **kwargs):
        kwargs["timeout"] = HTTP_TIMEOUT  # force override, yfinance passes timeout=30 explicitly
        return _orig_curl_request(self, method, url, **kwargs)
    curl_requests.Session.request = _curl_request_with_timeout
except ImportError:
    pass

# ── In-memory job store ─────────────────────────────────────────────────────────
# { job_id: { "status": "running"|"done"|"error",
#             "processed": int, "total": int,
#             "log": [...], "results": [...] } }
JOBS: dict = {}

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Default ticker universe (S&P 100 + high-beta names) ────────────────────────
DEFAULT_TICKERS = [
    # Mega-cap tech
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AVGO",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "SCHW", "C", "PNC",
    # Healthcare
    "UNH", "LLY", "JNJ", "MRK", "ABBV", "TMO", "ABT", "MDT", "AMGN",
    "GILD", "REGN", "VRTX", "SYK", "BSX", "ZTS", "CVS", "ELV", "CI", "HUM",
    # Consumer
    "WMT", "HD", "COST", "MCD", "PEP", "KO", "PG", "PM", "MO", "TJX",
    "SBUX", "LOW", "NKE", "BKNG",
    # Energy
    "XOM", "CVX",
    # Industrials
    "CAT", "DE", "BA", "GE", "RTX", "HON", "ETN", "ITW", "MMM", "EMR", "UPS",
    # Tech
    "CRM", "ADBE", "ORCL", "INTU", "CSCO", "IBM", "TXN", "AMD", "INTC",
    "PANW", "KLAC", "LRCX", "SNPS", "NOW", "ISRG", "ACN",
    # Growth / High-beta
    "NFLX", "UBER", "ABNB", "CRWD", "PLTR", "COIN", "ARM", "SNOW",
    # Other large-cap
    "V", "MA", "BRK-B", "LIN", "SPGI", "CME", "ICE", "MMC", "ADP",
    "NEE", "SO", "DUK", "PLD", "CB",
]


def calculate_hv(ticker_obj: yf.Ticker, window: int = 30) -> Optional[float]:
    """Annualized historical volatility over `window` trading days."""
    try:
        hist = ticker_obj.history(period="6mo", auto_adjust=True)
        if len(hist) < window + 5:
            return None
        log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        hv = float(log_ret.rolling(window).std().iloc[-1]) * np.sqrt(252)
        return None if np.isnan(hv) else hv
    except Exception:
        return None


def process_ticker(
    sym: str,
    min_mktcap: float,
    min_dte: int,
    max_dte: int,
    iv_hv_thr: float,
    iv_min: float,
    only_undervalued: bool = False,
) -> list:
    """Return list of anomalous put records for one ticker."""
    results = []
    try:
        tk = yf.Ticker(sym)

        # ── Market cap check ──────────────────────────────────────────────────
        mktcap = None
        try:
            mktcap = tk.fast_info.market_cap
        except Exception:
            pass
        if not mktcap:
            try:
                mktcap = (tk.info or {}).get("marketCap", 0)
            except Exception:
                return results

        if not mktcap or mktcap < min_mktcap:
            return results

        # ── Basic info ────────────────────────────────────────────────────────
        try:
            info = tk.info or {}
            company = info.get("longName") or info.get("shortName", sym)
            price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or 0
            )
        except Exception:
            company = sym
            try:
                price = tk.fast_info.last_price or 0
            except Exception:
                return results

        if not price or price <= 0:
            return results

        mktcap_b = round(mktcap / 1e9, 1)

        # ── Undervalued check ─────────────────────────────────────────────────
        analyst_target = None
        upside_pct     = None
        try:
            analyst_target = info.get("targetMeanPrice") or info.get("targetMedianPrice")
            if analyst_target and price > 0:
                upside_pct = round((analyst_target - price) / price * 100, 1)
        except Exception:
            pass

        if only_undervalued:
            pe  = info.get("trailingPE") or info.get("forwardPE")
            pb  = info.get("priceToBook")
            peg = info.get("pegRatio")
            has_upside  = upside_pct is not None and upside_pct > 5
            low_pe      = pe  is not None and pe  < 20
            low_pb      = pb  is not None and pb  < 1.5
            low_peg     = peg is not None and peg < 1.0
            if not (has_upside or low_pe or low_pb or low_peg):
                return results

        # ── Historical volatility ─────────────────────────────────────────────
        hv = calculate_hv(tk)

        # ── Option expiration dates within max_dte ────────────────────────────
        try:
            all_dates = tk.options  # tuple of 'YYYY-MM-DD'
        except Exception:
            return results

        if not all_dates:
            return results

        now = datetime.now()
        floor  = now + timedelta(days=min_dte)
        cutoff = now + timedelta(days=max_dte)

        valid = []
        for d in all_dates:
            try:
                dt = datetime.strptime(d, "%Y-%m-%d")
                if floor <= dt <= cutoff:
                    valid.append((d, dt))
            except ValueError:
                pass

        if not valid:
            return results

        # ── Iterate over valid expirations ────────────────────────────────────
        for exp_str, exp_dt in valid:
            try:
                chain = tk.option_chain(exp_str)
                puts = chain.puts.copy()
            except Exception:
                continue

            if puts.empty:
                continue

            dte = max(1, (exp_dt - now).days)

            # Only strikes from 70 % to 105 % of current price
            puts = puts[
                (puts["strike"] >= price * 0.70) & (puts["strike"] <= price * 1.05)
            ]

            for _, row in puts.iterrows():
                iv_raw = row.get("impliedVolatility")
                if iv_raw is None:
                    continue
                iv = float(iv_raw)
                if np.isnan(iv) or iv < iv_min:
                    continue

                iv_hv_ratio = (iv / hv) if (hv and hv > 0) else None

                # Anomaly: IV/HV exceeds threshold, OR IV is extreme (≥ 70 %)
                is_anomaly = (
                    iv_hv_ratio is not None and iv_hv_ratio >= iv_hv_thr
                ) or iv >= 0.70

                if not is_anomaly:
                    continue

                bid  = float(row.get("bid",       0) or 0)
                ask  = float(row.get("ask",       0) or 0)
                last = float(row.get("lastPrice", 0) or 0)
                mid  = (bid + ask) / 2 if (bid > 0 and ask > 0) else last

                strike          = float(row["strike"])
                premium_pct     = (mid / strike * 100) if strike > 0 else 0
                ann_premium_pct = (premium_pct * 365 / dte) if dte > 0 else 0
                moneyness_pct   = (strike - price) / price * 100

                results.append(
                    {
                        "ticker":          sym,
                        "company":         company,
                        "market_cap_b":    mktcap_b,
                        "current_price":   round(price,  2),
                        "analyst_target":  round(analyst_target, 2) if analyst_target else None,
                        "upside_pct":      upside_pct,
                        "strike":          strike,
                        "exp_date":        exp_str,
                        "dte":             dte,
                        "iv":              round(iv * 100, 1),
                        "hv":              round(hv * 100, 1) if hv else None,
                        "iv_hv_ratio":     round(iv_hv_ratio, 2) if iv_hv_ratio else None,
                        "bid":             round(bid,  2),
                        "ask":             round(ask,  2),
                        "mid":             round(mid,  2),
                        "volume":          int(row.get("volume",       0) or 0),
                        "open_interest":   int(row.get("openInterest", 0) or 0),
                        "premium_pct":     round(premium_pct,     2),
                        "ann_premium_pct": round(ann_premium_pct, 1),
                        "moneyness_pct":   round(moneyness_pct,   1),
                    }
                )

    except Exception:
        pass

    return results


# ── Flask routes ────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan/start", methods=["POST"])
def scan_start():
    data = request.get_json(force=True, silent=True) or {}

    min_mktcap       = float(data.get("min_market_cap",   100))  * 1e9
    min_dte          = int(  data.get("min_days_to_exp",   30))
    max_dte          = int(  data.get("days_to_exp",       60))
    iv_hv_thr        = float(data.get("iv_hv_threshold",  1.3))
    iv_min           = float(data.get("iv_min",            30))  / 100
    only_undervalued = str(  data.get("only_undervalued", "true")).lower() == "true"
    custom_raw       =       data.get("custom_tickers",    "")

    tickers = list(DEFAULT_TICKERS)
    if custom_raw:
        extras = [
            t.strip().upper()
            for t in custom_raw.replace(";", ",").split(",")
            if t.strip()
        ]
        tickers = list(dict.fromkeys(extras + tickers))

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status":     "running",
        "processed":  0,
        "total":      len(tickers),
        "log":        [],
        "results":    [],
        "lock":       threading.Lock(),
        "started_at": __import__("time").monotonic(),
    }

    def _scan_ticker(sym):
        """Run in its own daemon thread. Always puts to queue."""
        try:
            res = process_ticker(sym, min_mktcap, min_dte, max_dte,
                                 iv_hv_thr, iv_min, only_undervalued)
        except Exception:
            res = []
        return sym, res

    def worker():
        import time, queue as q_mod
        job  = JOBS[job_id]
        rq   = q_mod.Queue()

        # Launch all tickers as daemon threads
        for sym in tickers:
            def _run(s=sym):
                try:
                    res = process_ticker(s, min_mktcap, min_dte, max_dte,
                                        iv_hv_thr, iv_min, only_undervalued)
                except Exception:
                    res = []
                rq.put((s, res))
            threading.Thread(target=_run, daemon=True).start()

        collected  = set()
        deadline   = time.monotonic() + 60   # absolute 60s cap
        last_tick  = time.monotonic()
        STALL      = 8                        # 8s without any ticker = done

        while len(collected) < len(tickers):
            now = time.monotonic()
            if now - deadline > 0 or now - last_tick > STALL:
                break
            try:
                sym, res = rq.get(timeout=0.5)
            except q_mod.Empty:
                continue
            if sym in collected:
                continue
            collected.add(sym)
            last_tick = time.monotonic()
            with job["lock"]:
                job["processed"] += 1
                job["results"].extend(res)
                job["log"].append({
                    "sym":   sym,
                    "found": len(res),
                    "pct":   round(job["processed"] / job["total"] * 100),
                })

        # Mark skipped tickers
        with job["lock"]:
            for sym in tickers:
                if sym not in collected:
                    job["processed"] += 1
                    job["log"].append({
                        "sym":   sym,
                        "found": 0,
                        "pct":   round(job["processed"] / job["total"] * 100),
                    })
            job["results"].sort(
                key=lambda x: x.get("iv_hv_ratio") or (x.get("iv", 0) / 100),
                reverse=True)
            job["results"] = job["results"][:50]
            job["status"]  = "done"

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job_id, "total": len(tickers)})


@app.route("/api/scan/status/<job_id>")
def scan_status(job_id):
    import time
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    # Safety net: force done after 75s in case worker thread itself hung
    if job["status"] == "running":
        elapsed = time.monotonic() - job.get("started_at", time.monotonic())
        if elapsed > 75:
            with job["lock"]:
                if job["status"] == "running":   # re-check inside lock
                    job["results"].sort(
                        key=lambda x: x.get("iv_hv_ratio") or (x.get("iv", 0) / 100),
                        reverse=True)
                    job["results"] = job["results"][:50]
                    job["status"] = "done"

    with job["lock"]:
        status    = job["status"]
        processed = job["processed"]
        total     = job["total"]
        log       = list(job["log"][-30:])
        count     = len(job["results"])
        results   = list(job["results"]) if status == "done" else []

    return jsonify({
        "status":    status,
        "processed": processed,
        "total":     total,
        "pct":       round(processed / total * 100) if total else 0,
        "log":       log,
        "count":     count,
        "results":   results,
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, use_reloader=False, port=port, threaded=True)
