from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


APP_VERSION = "2026.09.11-tv-bridge"
DB_PATH = Path(os.getenv("OPEXE_DB_PATH", "/tmp/opexe_tv.sqlite3"))
PROJECTX_BASE = "https://api.topstepx.com/api"
TV_FRESH_MINUTES = int(os.getenv("TV_FRESH_MINUTES", "20"))

PRIMARY_WEIGHTS = {
    "BOS": 20.0,
    "FVG": 18.0,
    "SMA": 18.0,
    "LIQUIDITY_SWEEP": 18.0,
}
SECONDARY_WEIGHTS = {
    "CHOCH": 6.0,
    "DISPLACEMENT": 5.0,
    "VWAP": 4.0,
    "EMA": 3.0,
    "RSI": 3.0,
    "ADX": 3.0,
    "VOLUME": 1.0,
    "ORB": 1.0,
}
TIMEFRAMES = ["1m", "5m", "30m", "4h", "1d", "1w"]

YF_CONFIG = {
    "1m": ("7d", "1m", None),
    "5m": ("60d", "5m", None),
    "30m": ("60d", "30m", None),
    "4h": ("2y", "1h", "4h"),
    "1d": ("2y", "1d", None),
    "1w": ("10y", "1wk", None),
}

FUTURES_SPECS = {
    "MES": (0.25, 1.25, "ES=F"),
    "ES": (0.25, 12.50, "ES=F"),
    "MNQ": (0.25, 0.50, "NQ=F"),
    "NQ": (0.25, 5.00, "NQ=F"),
    "M2K": (0.10, 0.50, "RTY=F"),
    "RTY": (0.10, 5.00, "RTY=F"),
    "MYM": (1.00, 0.50, "YM=F"),
    "YM": (1.00, 5.00, "YM=F"),
    "MGC": (0.10, 1.00, "GC=F"),
    "GC": (0.10, 10.00, "GC=F"),
    "MCL": (0.01, 1.00, "CL=F"),
    "CL": (0.01, 10.00, "CL=F"),
}

app = FastAPI(title="OP.exe", version=APP_VERSION)
db_lock = threading.Lock()


# ----------------------------
# Database: latest TradingView snapshot
# ----------------------------

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with db_lock, db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tv_snapshots (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                received_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(symbol, timeframe)
            )
        """)
        conn.commit()


init_db()


def normalize_symbol(s: str) -> str:
    return (s or "").strip().upper().replace(" ", "")


def normalize_tf(tf: str) -> str:
    x = str(tf or "").strip().lower()
    aliases = {
        "1": "1m", "1m": "1m",
        "5": "5m", "5m": "5m",
        "30": "30m", "30m": "30m",
        "240": "4h", "4h": "4h",
        "d": "1d", "1d": "1d", "1day": "1d",
        "w": "1w", "1w": "1w", "1week": "1w",
    }
    return aliases.get(x, x)


def root_symbol(s: str) -> str:
    s = normalize_symbol(s)
    for root in sorted(FUTURES_SPECS, key=len, reverse=True):
        if s.startswith(root):
            return root
    return s


def save_tv_snapshot(payload: Dict[str, Any]):
    symbol = normalize_symbol(payload.get("symbol", ""))
    timeframe = normalize_tf(payload.get("timeframe", ""))
    if not symbol or timeframe not in TIMEFRAMES:
        raise ValueError("Payload needs valid symbol and timeframe.")
    now = datetime.now(timezone.utc).isoformat()
    payload = dict(payload)
    payload["symbol"] = symbol
    payload["timeframe"] = timeframe
    payload["received_at"] = now
    with db_lock, db() as conn:
        conn.execute(
            "INSERT INTO tv_snapshots(symbol,timeframe,received_at,payload) VALUES(?,?,?,?) "
            "ON CONFLICT(symbol,timeframe) DO UPDATE SET received_at=excluded.received_at,payload=excluded.payload",
            (symbol, timeframe, now, json.dumps(payload)),
        )
        conn.commit()


def get_tv_snapshot(symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
    symbol = normalize_symbol(symbol)
    timeframe = normalize_tf(timeframe)
    with db_lock, db() as conn:
        row = conn.execute(
            "SELECT received_at,payload FROM tv_snapshots WHERE symbol=? AND timeframe=?",
            (symbol, timeframe),
        ).fetchone()
    if not row:
        return None
    payload = json.loads(row["payload"])
    received = datetime.fromisoformat(row["received_at"])
    age = datetime.now(timezone.utc) - received
    payload["_fresh"] = age <= timedelta(minutes=TV_FRESH_MINUTES)
    payload["_age_seconds"] = age.total_seconds()
    return payload


# ----------------------------
# Market data
# ----------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = [c[0] if isinstance(c, tuple) else c for c in x.columns]
    x = x.rename(columns={c: str(c).title() for c in x.columns})
    need = ["Open", "High", "Low", "Close"]
    if not all(c in x.columns for c in need):
        return pd.DataFrame()
    if "Volume" not in x.columns:
        x["Volume"] = 0.0
    for c in need + ["Volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.dropna(subset=need).sort_index()


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return pd.DataFrame({
        "Open": df["Open"].resample(rule).first(),
        "High": df["High"].resample(rule).max(),
        "Low": df["Low"].resample(rule).min(),
        "Close": df["Close"].resample(rule).last(),
        "Volume": df["Volume"].resample(rule).sum(),
    }).dropna()


def yahoo_data(symbol: str, tf: str) -> Tuple[pd.DataFrame, str]:
    period, interval, rs = YF_CONFIG[tf]
    root = root_symbol(symbol)
    yfs = FUTURES_SPECS[root][2] if root in FUTURES_SPECS else normalize_symbol(symbol)
    df = yf.download(
        yfs, period=period, interval=interval, auto_adjust=False,
        progress=False, threads=False, prepost=True
    )
    df = clean(df)
    if not df.empty and rs:
        df = resample(df, rs)
    source = "Yahoo Finance"
    if root in FUTURES_SPECS:
        source += " continuous-futures proxy"
    return df, source


# ----------------------------
# Indicators
# ----------------------------

def tr(df):
    pc = df["Close"].shift(1)
    return pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"] - pc).abs(),
    ], axis=1).max(axis=1)


def atr(df, n=14):
    return tr(df).ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def rsi(df, n=14):
    d = df["Close"].diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al = l.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100/(1+rs)


def adx(df, n=14):
    up = df["High"].diff()
    dn = -df["Low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * pdm.ewm(alpha=1/n, adjust=False).mean() / a
    mdi = 100 * mdm.ewm(alpha=1/n, adjust=False).mean() / a
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean(), pdi, mdi


def vwap(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(np.nan, index=df.index)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].replace(0, np.nan)
    dates = pd.Series(df.index.date, index=df.index)
    return (typical*vol).groupby(dates).cumsum() / vol.groupby(dates).cumsum()


def indicators(df):
    x = df.copy()
    x["SMA20"] = x["Close"].rolling(20).mean()
    x["SMA50"] = x["Close"].rolling(50).mean()
    x["EMA9"] = x["Close"].ewm(span=9, adjust=False).mean()
    x["EMA21"] = x["Close"].ewm(span=21, adjust=False).mean()
    x["ATR"] = atr(x)
    x["RSI"] = rsi(x)
    x["ADX"], x["PDI"], x["MDI"] = adx(x)
    x["VOL20"] = x["Volume"].rolling(20).mean()
    x["VWAP"] = vwap(x)
    return x


def swings(df, left=2, right=2):
    hs, ls = [], []
    h, l = df["High"].values, df["Low"].values
    for i in range(left, len(df)-right):
        wh = h[i-left:i+right+1]
        wl = l[i-left:i+right+1]
        if h[i] == np.max(wh) and np.sum(wh == h[i]) == 1:
            hs.append((i, float(h[i])))
        if l[i] == np.min(wl) and np.sum(wl == l[i]) == 1:
            ls.append((i, float(l[i])))
    return hs, ls


def latest_before(items, idx):
    vals = [x for x in items if x[0] < idx]
    return vals[-1] if vals else None


def fvg_list(df, lookback=120):
    out = []
    for i in range(max(2, len(df)-lookback), len(df)):
        h2, l2 = float(df["High"].iloc[i-2]), float(df["Low"].iloc[i-2])
        h, l = float(df["High"].iloc[i]), float(df["Low"].iloc[i])
        if l > h2:
            later = df["Low"].iloc[i+1:]
            filled = bool((later <= h2).any()) if len(later) else False
            out.append({"dir": 1, "low": h2, "high": l, "i": i, "filled": filled})
        if h < l2:
            later = df["High"].iloc[i+1:]
            filled = bool((later >= l2).any()) if len(later) else False
            out.append({"dir": -1, "low": h, "high": l2, "i": i, "filled": filled})
    return out


# ----------------------------
# Evidence
# ----------------------------

def local_primary(df):
    hs, ls = swings(df)
    idx = len(df)-1
    sh, sl = latest_before(hs, idx), latest_before(ls, idx)
    c = float(df["Close"].iloc[-1])
    pc = float(df["Close"].iloc[-2])
    hi = float(df["High"].iloc[-1])
    lo = float(df["Low"].iloc[-1])

    bos = 0
    bos_detail = "No active BOS"
    if sh and c > sh[1]:
        bos = 1
        bos_detail = f"Above confirmed swing high {sh[1]:.4f}"
    elif sl and c < sl[1]:
        bos = -1
        bos_detail = f"Below confirmed swing low {sl[1]:.4f}"

    sweep = 0
    sweep_detail = "No latest-bar liquidity sweep"
    sweep_extreme = None
    if sl and lo < sl[1] and c > sl[1]:
        sweep = 1
        sweep_detail = f"Swept below {sl[1]:.4f} and reclaimed"
        sweep_extreme = lo
    elif sh and hi > sh[1] and c < sh[1]:
        sweep = -1
        sweep_detail = f"Swept above {sh[1]:.4f} and rejected"
        sweep_extreme = hi

    row = df.iloc[-1]
    sma = 0
    if c > row["SMA20"] > row["SMA50"]:
        sma = 1
    elif c < row["SMA20"] < row["SMA50"]:
        sma = -1
    sma_detail = f"Close={c:.4f}, SMA20={row['SMA20']:.4f}, SMA50={row['SMA50']:.4f}"

    gaps = [g for g in fvg_list(df) if not g["filled"]]
    gap = None
    fvg = 0
    fvg_detail = "No open recent FVG"
    if gaps:
        av = float(row["ATR"]) if pd.notna(row["ATR"]) else max(abs(c)*0.002, 1e-9)
        gap = min(gaps, key=lambda g: abs(c - (g["low"]+g["high"])/2)/max(av,1e-9) + .025*(idx-g["i"]))
        if gap["dir"] == 1 and c >= gap["low"]:
            fvg = 1
            fvg_detail = f"Bullish FVG {gap['low']:.4f}-{gap['high']:.4f}"
        elif gap["dir"] == -1 and c <= gap["high"]:
            fvg = -1
            fvg_detail = f"Bearish FVG {gap['low']:.4f}-{gap['high']:.4f}"

    return {
        "BOS": (bos, bos_detail),
        "FVG": (fvg, fvg_detail),
        "SMA": (sma, sma_detail),
        "LIQUIDITY_SWEEP": (sweep, sweep_detail),
        "_sh": sh,
        "_sl": sl,
        "_gap": gap,
        "_sweep_extreme": sweep_extreme,
    }


def tv_primary(snapshot: Optional[dict]) -> Optional[dict]:
    if not snapshot or not snapshot.get("_fresh"):
        return None

    def d(key):
        try:
            x = int(float(snapshot.get(key, 0)))
            return 1 if x > 0 else -1 if x < 0 else 0
        except Exception:
            return 0

    return {
        "BOS": (d("bos"), f"TradingView chart BOS={d('bos')}"),
        "FVG": (d("fvg"), f"TradingView chart FVG={d('fvg')}"),
        "SMA": (d("sma"), f"TradingView chart SMA={d('sma')}"),
        "LIQUIDITY_SWEEP": (d("liquidity_sweep"), f"TradingView chart sweep={d('liquidity_sweep')}"),
    }


def secondary(df):
    row = df.iloc[-1]
    out = {}

    # CHoCH
    hs, ls = swings(df)
    choch = 0
    if len(hs) >= 2 and len(ls) >= 2:
        last_h, prev_h = hs[-1], hs[-2]
        last_l, prev_l = ls[-1], ls[-2]
        prior_bull = last_h[1] > prev_h[1] and last_l[1] > prev_l[1]
        prior_bear = last_h[1] < prev_h[1] and last_l[1] < prev_l[1]
        if prior_bull and row["Close"] < last_l[1]:
            choch = -1
        elif prior_bear and row["Close"] > last_h[1]:
            choch = 1
    out["CHOCH"] = choch

    av = float(row["ATR"]) if pd.notna(row["ATR"]) else 0
    body = abs(float(row["Close"]-row["Open"]))
    rng = max(float(row["High"]-row["Low"]), 1e-9)
    out["DISPLACEMENT"] = (
        1 if av > 0 and body >= .8*av and body/rng >= .6 and row["Close"] > row["Open"]
        else -1 if av > 0 and body >= .8*av and body/rng >= .6 and row["Close"] < row["Open"]
        else 0
    )

    out["VWAP"] = 0 if pd.isna(row["VWAP"]) else (1 if row["Close"] > row["VWAP"] else -1 if row["Close"] < row["VWAP"] else 0)
    out["EMA"] = 1 if row["EMA9"] > row["EMA21"] and row["Close"] > row["EMA9"] else -1 if row["EMA9"] < row["EMA21"] and row["Close"] < row["EMA9"] else 0

    rv = row["RSI"]
    out["RSI"] = 0 if pd.isna(rv) else (1 if 55 <= rv <= 75 else -1 if 25 <= rv <= 45 else 0)

    ax = row["ADX"]
    out["ADX"] = 0 if pd.isna(ax) else (1 if ax >= 20 and row["PDI"] > row["MDI"] else -1 if ax >= 20 and row["MDI"] > row["PDI"] else 0)

    vm = row["VOL20"]
    if pd.isna(vm) or vm <= 0 or row["Volume"] < 1.25*vm:
        out["VOLUME"] = 0
    else:
        out["VOLUME"] = 1 if row["Close"] > row["Open"] else -1 if row["Close"] < row["Open"] else 0

    out["ORB"] = 0
    return out


def score(primary, sec):
    bull = bear = 0.0
    details = []

    for name, weight in PRIMARY_WEIGHTS.items():
        direction, detail = primary[name]
        if direction > 0:
            bull += weight
        elif direction < 0:
            bear += weight
        details.append({"name": name, "type": "PRIMARY", "direction": direction, "weight": weight, "detail": detail})

    for name, weight in SECONDARY_WEIGHTS.items():
        direction = int(sec.get(name, 0))
        if direction > 0:
            bull += weight
        elif direction < 0:
            bear += weight
        details.append({"name": name, "type": "SECONDARY", "direction": direction, "weight": weight, "detail": ""})

    active = bull + bear
    if active == 0:
        return "WAIT", "MIXED", 0.0, 0.0, details, bull, bear

    agreement = max(bull, bear)/active*100
    p_bull = sum(PRIMARY_WEIGHTS[k] for k in PRIMARY_WEIGHTS if primary[k][0] > 0)
    p_bear = sum(PRIMARY_WEIGHTS[k] for k in PRIMARY_WEIGHTS if primary[k][0] < 0)
    p_active = p_bull+p_bear
    p_agree = max(p_bull,p_bear)/p_active*100 if p_active else 0

    net = bull-bear
    signal = "LONG" if net >= 12 and agreement >= 57 else "SHORT" if net <= -12 and agreement >= 57 else "WAIT"

    if signal != "WAIT" and agreement >= 78 and p_agree >= 75 and abs(net) >= 34:
        tier = "STRONG"
    elif signal != "WAIT" and agreement >= 66 and abs(net) >= 22:
        tier = "MODERATE"
    elif signal != "WAIT":
        tier = "QUALIFYING"
    else:
        tier = "MIXED"

    return signal, tier, agreement, p_agree, details, bull, bear


# ----------------------------
# Entry / TP / SL
# ----------------------------

def plan(df, signal, primary, tvsnap=None):
    if signal not in ("LONG", "SHORT"):
        return None
    row = df.iloc[-1]
    entry = float(tvsnap.get("close")) if tvsnap and tvsnap.get("_fresh") and tvsnap.get("close") is not None else float(row["Close"])
    av = float(tvsnap.get("atr")) if tvsnap and tvsnap.get("_fresh") and tvsnap.get("atr") not in (None, 0, "") else float(row["ATR"])
    if not np.isfinite(av) or av <= 0:
        av = max(entry*0.002, 1e-6)

    hs, ls = swings(df)
    rh = [p for _,p in hs[-10:]]
    rl = [p for _,p in ls[-10:]]
    gaps = [g for g in fvg_list(df, 180) if not g["filled"]]

    tv_sh = None
    tv_sl = None
    tv_fvg_low = None
    tv_fvg_high = None
    if tvsnap and tvsnap.get("_fresh"):
        for key in ["swing_high", "swing_low", "fvg_low", "fvg_high"]:
            try:
                val = float(tvsnap.get(key))
                if math.isfinite(val):
                    if key == "swing_high": tv_sh = val
                    elif key == "swing_low": tv_sl = val
                    elif key == "fvg_low": tv_fvg_low = val
                    elif key == "fvg_high": tv_fvg_high = val
            except Exception:
                pass

    targets = []
    stops = []

    if signal == "LONG":
        for p in rh:
            if p > entry + .35*av: targets.append((p, "Confirmed swing high"))
        for g in gaps:
            if g["dir"] < 0 and g["low"] > entry: targets.append((g["low"], "Opposing bearish FVG"))
        if tv_sh and tv_sh > entry + .35*av: targets.append((tv_sh, "TradingView confirmed swing high"))
        if tv_fvg_low and tv_fvg_low > entry + .35*av: targets.append((tv_fvg_low, "TradingView opposing FVG"))
        for p in rl:
            if p < entry and entry-p >= .35*av: stops.append((p-.1*av, "Below confirmed swing low"))
        if tv_sl and tv_sl < entry and entry-tv_sl >= .35*av: stops.append((tv_sl-.1*av, "Below TradingView swing low"))
        target = min(targets, key=lambda x:x[0]) if targets else (entry+1.5*av, "1.5 ATR extension")
        stop = max(stops, key=lambda x:x[0]) if stops else (entry-1.0*av, "1 ATR invalidation")
    else:
        for p in rl:
            if p < entry - .35*av: targets.append((p, "Confirmed swing low"))
        for g in gaps:
            if g["dir"] > 0 and g["high"] < entry: targets.append((g["high"], "Opposing bullish FVG"))
        if tv_sl and tv_sl < entry - .35*av: targets.append((tv_sl, "TradingView confirmed swing low"))
        if tv_fvg_high and tv_fvg_high < entry - .35*av: targets.append((tv_fvg_high, "TradingView opposing FVG"))
        for p in rh:
            if p > entry and p-entry >= .35*av: stops.append((p+.1*av, "Above confirmed swing high"))
        if tv_sh and tv_sh > entry and tv_sh-entry >= .35*av: stops.append((tv_sh+.1*av, "Above TradingView swing high"))
        target = max(targets, key=lambda x:x[0]) if targets else (entry-1.5*av, "1.5 ATR extension")
        stop = min(stops, key=lambda x:x[0]) if stops else (entry+1.0*av, "1 ATR invalidation")

    reward = abs(target[0]-entry)
    risk = abs(entry-stop[0])
    rr = reward/risk if risk else None
    return {
        "entry": entry,
        "take_profit": float(target[0]),
        "stop_loss": float(stop[0]),
        "rr": rr,
        "target_basis": target[1],
        "stop_basis": stop[1],
        "atr_target_distance": reward/av if av else None,
    }


def pnl(symbol, a, b, contracts):
    root = root_symbol(symbol)
    if root not in FUTURES_SPECS:
        return None
    tick, value, _ = FUTURES_SPECS[root]
    return abs(b-a)/tick*value*contracts


def analyze(symbol, tf, contracts=1):
    df, source = yahoo_data(symbol, tf)
    if df.empty or len(df) < 60:
        return {"timeframe": tf, "signal": "WAIT", "tier": "NO DATA", "error": f"Only {len(df)} bars", "source": source}

    df = indicators(df)
    local = local_primary(df)
    snap = get_tv_snapshot(symbol, tf)
    tvp = tv_primary(snap)

    # TradingView primary pillars become the main source whenever a fresh chart
    # snapshot is available. Local feed remains fallback/cross-check.
    primary = {}
    primary_source = "market-feed calculation"
    if tvp:
        primary_source = "TradingView chart snapshot"
        for k in PRIMARY_WEIGHTS:
            tv_dir, tv_detail = tvp[k]
            loc_dir, loc_detail = local[k]
            # Use TV direction when non-neutral. If TV says neutral, preserve the
            # independently calculated local signal instead of throwing information away.
            primary[k] = (tv_dir, tv_detail + " • primary") if tv_dir != 0 else (loc_dir, loc_detail + " • local fallback")
    else:
        for k in PRIMARY_WEIGHTS:
            primary[k] = local[k]

    sec = secondary(df)
    signal, tier, agreement, p_agree, details, bull, bear = score(primary, sec)
    trade = plan(df, signal, primary, snap)

    out = {
        "timeframe": tf,
        "signal": signal,
        "tier": tier,
        "price": float(df["Close"].iloc[-1]),
        "agreement": agreement,
        "primary_agreement": p_agree,
        "bull_points": bull,
        "bear_points": bear,
        "source": source,
        "primary_source": primary_source,
        "tradingview_fresh": bool(snap and snap.get("_fresh")),
        "tradingview_age_seconds": snap.get("_age_seconds") if snap else None,
        "evidence": details,
        "trade": trade,
    }
    if trade:
        out["projected_profit"] = pnl(symbol, trade["entry"], trade["take_profit"], contracts)
        out["projected_risk"] = pnl(symbol, trade["entry"], trade["stop_loss"], contracts)
    return out


# ----------------------------
# API / TradingView webhook
# ----------------------------

class WebhookPayload(BaseModel):
    symbol: str
    timeframe: str
    close: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    atr: Optional[float] = None
    sma20: Optional[float] = None
    sma50: Optional[float] = None
    bos: int = 0
    fvg: int = 0
    liquidity_sweep: int = 0
    sma: int = 0
    swing_high: Optional[float] = None
    swing_low: Optional[float] = None
    fvg_low: Optional[float] = None
    fvg_high: Optional[float] = None
    secret: Optional[str] = None


@app.post("/tradingview/webhook")
async def tradingview_webhook(payload: WebhookPayload):
    required_secret = os.getenv("TRADINGVIEW_WEBHOOK_SECRET", "")
    if required_secret and payload.secret != required_secret:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")
    data = payload.model_dump()
    save_tv_snapshot(data)
    return {"ok": True, "symbol": normalize_symbol(payload.symbol), "timeframe": normalize_tf(payload.timeframe)}


@app.get("/api/analyze")
def api_analyze(symbol: str, timeframe: str, contracts: int = 1):
    tf = normalize_tf(timeframe)
    if tf not in TIMEFRAMES:
        raise HTTPException(status_code=400, detail="Unsupported timeframe")
    return analyze(symbol, tf, contracts)


@app.get("/api/analyze-all")
def api_analyze_all(symbol: str, contracts: int = 1):
    return {"symbol": normalize_symbol(symbol), "results": [analyze(symbol, tf, contracts) for tf in TIMEFRAMES]}


@app.get("/api/tradingview-status")
def tv_status(symbol: str):
    return {
        tf: get_tv_snapshot(symbol, tf)
        for tf in TIMEFRAMES
    }


# ----------------------------
# HTML dashboard
# ----------------------------

PAGE = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OP.exe</title>
<style>
body{font-family:Inter,system-ui,-apple-system,sans-serif;background:#0e1117;color:#f1f5f9;margin:0}
.wrap{max-width:1400px;margin:auto;padding:24px}
h1{margin:0 0 4px;font-size:34px}.muted{color:#94a3b8}
.controls{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}
input,button{font:inherit;border-radius:9px;border:1px solid #334155;padding:10px 12px;background:#111827;color:#fff}
button{cursor:pointer;background:#1d4ed8;border-color:#1d4ed8;font-weight:700}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.card{background:#111827;border:1px solid #263244;border-radius:12px;padding:14px}
.long{color:#4ade80}.short{color:#fb7185}.wait{color:#fbbf24}
table{width:100%;border-collapse:collapse;margin-top:18px;font-size:14px}
th,td{padding:10px;border-bottom:1px solid #263244;text-align:left}
th{color:#94a3b8}
.badge{font-size:12px;border:1px solid #334155;border-radius:999px;padding:3px 7px}
.primary{font-weight:700}
details{margin:12px 0;background:#111827;border:1px solid #263244;border-radius:10px;padding:12px}
code{background:#0b1220;padding:2px 5px;border-radius:4px}
.notice{border-left:4px solid #3b82f6;padding:12px 14px;background:#111827;margin:16px 0}
</style>
</head>
<body>
<div class="wrap">
<h1>OP.exe</h1>
<div class="muted">BOS + FVG + SMA + Liquidity Sweeps = primary engine. Entry, TP, SL, R:R and secondary confluences retained.</div>
<div class="notice">
TradingView cannot be read directly from another browser tab by a Render server. When the TradingView bridge is configured, fresh TradingView chart snapshots become the <b>primary source for the four main confluences</b>; the market feed remains a fallback/cross-check.
</div>
<div class="controls">
<input id="symbol" value="MNQ" placeholder="Ticker">
<input id="contracts" type="number" value="1" min="1" max="100">
<button onclick="run()">Analyze</button>
</div>
<div id="cards" class="grid"></div>
<div id="table"></div>
<div id="detail"></div>
</div>
<script>
function cls(sig){return sig==="LONG"?"long":sig==="SHORT"?"short":"wait"}
function n(v,d=2){return (v===null||v===undefined||Number.isNaN(Number(v)))?"—":Number(v).toFixed(d)}
async function run(){
 let symbol=document.getElementById("symbol").value.trim();
 let contracts=document.getElementById("contracts").value;
 let res=await fetch(`/api/analyze-all?symbol=${encodeURIComponent(symbol)}&contracts=${contracts}`);
 let data=await res.json();
 let cards="", rows="", det="";
 for(let r of data.results){
   cards+=`<div class="card"><div class="muted">${r.timeframe}</div><div class="${cls(r.signal)}" style="font-size:25px;font-weight:800">${r.signal}</div><div>${r.tier||""}</div><div class="muted">${r.tradingview_fresh?"TradingView synced":"Feed analysis"}</div></div>`;
   let t=r.trade||{};
   rows+=`<tr><td>${r.timeframe}</td><td class="${cls(r.signal)}"><b>${r.signal}</b></td><td>${r.tier||""}</td><td>${n(r.agreement,0)}%</td><td>${n(r.primary_agreement,0)}%</td><td>${n(t.entry,4)}</td><td>${n(t.take_profit,4)}</td><td>${n(t.stop_loss,4)}</td><td>${n(t.rr,2)}</td><td>${r.tradingview_fresh?"YES":"NO"}</td></tr>`;
   let ev=(r.evidence||[]).map(e=>`<tr><td>${e.type}</td><td>${e.name}</td><td>${e.direction>0?"BULLISH":e.direction<0?"BEARISH":"NEUTRAL"}</td><td>${e.weight}</td><td>${e.detail||""}</td></tr>`).join("");
   det+=`<details><summary><b>${r.timeframe} • ${r.signal} • ${r.tier||""}</b></summary>
   <p>Primary source: <b>${r.primary_source||""}</b></p>
   ${r.trade?`<p>Entry ${n(t.entry,4)} • TP ${n(t.take_profit,4)} • SL ${n(t.stop_loss,4)} • R:R ${n(t.rr,2)}:1</p><p>TP basis: ${t.target_basis}<br>SL basis: ${t.stop_basis}</p>`:""}
   <table><tr><th>Type</th><th>Confluence</th><th>Direction</th><th>Weight</th><th>Detail</th></tr>${ev}</table></details>`;
 }
 document.getElementById("cards").innerHTML=cards;
 document.getElementById("table").innerHTML=`<table><tr><th>TF</th><th>Signal</th><th>Strength</th><th>Agreement</th><th>Main 4</th><th>Entry</th><th>TP</th><th>SL</th><th>R:R</th><th>TV sync</th></tr>${rows}</table>`;
 document.getElementById("detail").innerHTML=det;
}
run();
setInterval(run,30000);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


@app.get("/health")
def health():
    return {"ok": True, "version": APP_VERSION}
