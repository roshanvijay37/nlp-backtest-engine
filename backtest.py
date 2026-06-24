#!/usr/bin/env python3
# Natural Language Backtesting Engine
# Type strategies in plain English and backtest instantly

import sys, re, json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import yfinance as yf
import pandas as pd
import numpy as np


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    side: str
    entry_price: float
    exit_price: float
    qty: int
    pnl: float
    pnl_percent: float
    exit_reason: str
    bars_held: int


def parse_strategy(text: str) -> dict:
    t = text.lower()
    
    # Symbol mapping - check longer names first
    sym_map = {
        "bank nifty": "^NSEBANK",
        "nifty bank": "^NSEBANK",
        "nifty 50": "^NSEI",
        "fin nifty": "NIFTY_FIN_SERVICE.NS",
        "sensex": "^BSESN",
        "reliance": "RELIANCE.NS",
        "tcs": "TCS.NS", "infy": "INFY.NS",
        "hdfcbank": "HDFCBANK.NS", "sbin": "SBIN.NS",
        "nifty": "^NSEI",
    }
    symbol = "^NSEI"
    for name, code in sym_map.items():
        if name in t:
            symbol = code
            break
    
    # Timeframe
    tf = "1d"
    if "1min" in t or "1 min" in t or "1 minute" in t: tf = "1m"
    elif "5min" in t or "5 min" in t or "5 minute" in t: tf = "5m"
    elif "15min" in t or "15 min" in t or "15 minute" in t: tf = "15m"
    elif "30min" in t or "30 min" in t or "30 minute" in t: tf = "30m"
    elif "1 hour" in t or "hourly" in t: tf = "1h"
    elif "daily" in t or "day" in t: tf = "1d"
    elif "weekly" in t: tf = "1wk"
    
    # Dates
    end = datetime.now()
    start = end - timedelta(days=90)
    if "last 1 year" in t or "1 year" in t: start = end - timedelta(days=365)
    elif "last 2 year" in t or "2 year" in t: start = end - timedelta(days=730)
    elif "last 6 month" in t or "6 month" in t: start = end - timedelta(days=180)
    elif "last 3 month" in t or "3 month" in t: start = end - timedelta(days=90)
    elif "last 1 month" in t or "1 month" in t: start = end - timedelta(days=30)
    
    # Capital
    capital = 1_000_000
    cap_match = re.search(r'(\d+)\s*(?:lac|lakh|lacs|lakhs)', t)
    if cap_match: 
        capital = int(cap_match.group(1)) * 100_000
    else:
        # Check for "10 lakh" without space
        cap_match2 = re.search(r'(\d+)lakh', t)
        if cap_match2:
            capital = int(cap_match2.group(1)) * 100_000
    
    # Risk per trade
    risk = 1.0
    risk_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*risk', t)
    if risk_match: risk = float(risk_match.group(1))
    
    # Stop loss
    sl = None
    sl_match = re.search(r'stop\s*loss\s+(\d+(?:\.\d+)?)\s*%', t)
    if sl_match: sl = float(sl_match.group(1)) / 100
    
    # Take profit / R:R
    tp = None
    tp_match = re.search(r'target\s+(\d+(?:\.\d+)?)\s*%', t)
    rr_match = re.search(r'risk[\s:]*reward\s*(?:ratio)?\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*[:\-]?\s*(\d+(?:\.\d+)?)', t)
    if tp_match:
        tp = float(tp_match.group(1)) / 100
    elif rr_match:
        tp = float(rr_match.group(2)) / float(rr_match.group(1))
    
    # RSI
    rsi_period = 14
    rsi_entry = None
    rsi_exit = None
    
    rsi_match = re.search(r'rsi\s*\(?\s*(\d+)?\s*\)?\s*(<|below|under)\s*(\d+)', t)
    if rsi_match:
        if rsi_match.group(1): rsi_period = int(rsi_match.group(1))
        rsi_entry = int(rsi_match.group(3))
        rsi_exit = 70 if rsi_entry < 50 else 30
    
    rsi_exit_match = re.search(r'(?:sell|exit).+rsi\s*(>|above)\s*(\d+)', t)
    if rsi_exit_match:
        rsi_exit = int(rsi_exit_match.group(2))
    
    # Golden/Death cross
    golden = "golden cross" in t
    death = "death cross" in t
    
    sma_fast, sma_slow = None, None
    sma_match = re.search(r'(\d+)\s*sma\s*(?:crosses|crosse[sd])\s*(?:above|below)\s*(\d+)\s*sma', t)
    if sma_match:
        golden = True
        sma_fast = int(sma_match.group(1))
        sma_slow = int(sma_match.group(2))
    
    # Bollinger
    bb = "bollinger" in t or "bb" in t
    bb_lower = "lower" in t or "bounce" in t
    
    return {
        "symbol": symbol,
        "timeframe": tf,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "capital": capital,
        "risk_per_trade": risk,
        "stop_loss": sl,
        "take_profit": tp,
        "rsi_period": rsi_period,
        "rsi_entry": rsi_entry,
        "rsi_exit": rsi_exit,
        "golden_cross": golden,
        "death_cross": death,
        "sma_fast": sma_fast or 50,
        "sma_slow": sma_slow or 200,
        "bollinger": bb,
        "bollinger_lower": bb_lower,
    }


def fetch_data(symbol: str, start: str, end: str, tf: str) -> pd.DataFrame:
    # Yahoo Finance intraday data limitation: max 60 days for sub-daily
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    days_diff = (end_dt - start_dt).days
    
    intraday_tfs = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
    if tf in intraday_tfs and days_diff > 60:
        print(f"⚠️  Yahoo Finance limits intraday data to last 60 days.")
        print(f"   Adjusting start date from {start} to {(end_dt - timedelta(days=60)).strftime('%Y-%m-%d')}")
        start_dt = end_dt - timedelta(days=60)
        start = start_dt.strftime("%Y-%m-%d")
    
    print(f"\\n📊 Fetching {symbol} [{tf}] from {start} to {end}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=tf)
    if df.empty:
        raise ValueError(f"No data for {symbol} - try a shorter date range or daily timeframe")
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df = df.reset_index()
    col0 = df.columns[0]
    df = df.rename(columns={col0: "datetime"})
    return df


def add_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    c = df["close"]
    
    # RSI
    period = config.get("rsi_period", 14)
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))
    
    # SMAs
    for p in [9, 20, 50, 100, 200]:
        df[f"sma_{p}"] = c.rolling(p).mean()
    
    # Bollinger
    df["bb_mid"] = df["sma_20"]
    std = c.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * std
    df["bb_lower"] = df["bb_mid"] - 2 * std
    
    return df


def run_backtest(config: dict) -> dict:
    df = fetch_data(config["symbol"], config["start_date"], config["end_date"], config["timeframe"])
    df = add_indicators(df, config)
    
    capital = config["capital"]
    risk_pct = config["risk_per_trade"] / 100
    trades: List[Trade] = []
    equity = [{"date": str(df.iloc[0]["datetime"]), "equity": capital}]
    peak = capital
    position = None
    
    warmup = max(config.get("sma_slow", 200), config.get("rsi_period", 14)) + 5
    
    for i in range(warmup, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        if capital > peak: peak = capital
        
        # Exit logic
        if position:
            bars = i - position["entry_idx"]
            exit_price = None
            reason = None
            
            if position["side"] == "LONG" and row["low"] <= position["sl"]:
                exit_price = max(row["open"], position["sl"])
                reason = "STOP_LOSS"
            elif position["side"] == "SHORT" and row["high"] >= position["sl"]:
                exit_price = min(row["open"], position["sl"])
                reason = "STOP_LOSS"
            elif position["side"] == "LONG" and row["high"] >= position["tp"]:
                exit_price = min(row["open"], position["tp"])
                reason = "TARGET"
            elif position["side"] == "SHORT" and row["low"] <= position["tp"]:
                exit_price = max(row["open"], position["tp"])
                reason = "TARGET"
            elif config.get("rsi_exit"):
                if position["side"] == "LONG" and row["rsi"] > config["rsi_exit"]:
                    exit_price = row["close"]
                    reason = "RSI_EXIT"
                elif position["side"] == "SHORT" and row["rsi"] < config["rsi_exit"]:
                    exit_price = row["close"]
                    reason = "RSI_EXIT"
            
            if exit_price:
                pnl = (exit_price - position["entry"]) * position["qty"] if position["side"] == "LONG" else (position["entry"] - exit_price) * position["qty"]
                capital += pnl
                trades.append(Trade(
                    entry_date=str(position["date"]), exit_date=str(row["datetime"]),
                    side=position["side"], entry_price=position["entry"], exit_price=exit_price,
                    qty=position["qty"], pnl=round(pnl, 2),
                    pnl_percent=round((pnl / (position["entry"] * position["qty"])) * 100, 2),
                    exit_reason=reason, bars_held=bars
                ))
                equity.append({"date": str(row["datetime"]), "equity": round(capital, 2)})
                position = None
            continue
        
        # Entry logic
        signal = None
        
        if config.get("rsi_entry") and row["rsi"] < config["rsi_entry"]:
            signal = "LONG"
        elif config.get("golden_cross"):
            f, s = config["sma_fast"], config["sma_slow"]
            if prev[f"sma_{f}"] <= prev[f"sma_{s}"] and row[f"sma_{f}"] > row[f"sma_{s}"]:
                signal = "LONG"
        elif config.get("death_cross"):
            f, s = config["sma_fast"], config["sma_slow"]
            if prev[f"sma_{f}"] >= prev[f"sma_{s}"] and row[f"sma_{f}"] < row[f"sma_{s}"]:
                signal = "SHORT"
        elif config.get("bollinger") and config.get("bollinger_lower"):
            if row["close"] <= row["bb_lower"]:
                signal = "LONG"
        
        if signal:
            entry = row["close"]
            sl_dist = entry * (config.get("stop_loss") or 0.01)
            if config.get("take_profit"):
                rr = config["take_profit"]
                if rr > 0 and rr < 10:
                    tp_dist = sl_dist * rr
                else:
                    tp_dist = entry * rr
            else:
                tp_dist = sl_dist * 2
            
            qty = int((capital * risk_pct) / sl_dist) if sl_dist > 0 else 0
            
            if qty > 0:
                position = {
                    "side": signal,
                    "entry": entry,
                    "qty": qty,
                    "sl": entry - sl_dist if signal == "LONG" else entry + sl_dist,
                    "tp": entry + tp_dist if signal == "LONG" else entry - tp_dist,
                    "date": row["datetime"],
                    "entry_idx": i,
                }
    
    # Close open position
    if position:
        last = df.iloc[-1]
        pnl = (last["close"] - position["entry"]) * position["qty"] if position["side"] == "LONG" else (position["entry"] - last["close"]) * position["qty"]
        capital += pnl
        trades.append(Trade(
            entry_date=str(position["date"]), exit_date=str(last["datetime"]),
            side=position["side"], entry_price=position["entry"], exit_price=last["close"],
            qty=position["qty"], pnl=round(pnl, 2),
            pnl_percent=round((pnl / (position["entry"] * position["qty"])) * 100, 2),
            exit_reason="END_OF_DATA", bars_held=len(df) - position["entry_idx"]
        ))
    
    # Summary
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades)
    win_sum = sum(t.pnl for t in wins)
    loss_sum = abs(sum(t.pnl for t in losses))
    
    summary = {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 2) if trades else 0,
        "total_return_pct": round((capital - config["capital"]) / config["capital"] * 100, 2),
        "total_pnl": round(total_pnl, 2),
        "profit_factor": round(win_sum / loss_sum, 2) if loss_sum > 0 else float("inf"),
        "avg_win": round(win_sum / len(wins), 2) if wins else 0,
        "avg_loss": round(loss_sum / len(losses), 2) if losses else 0,
        "final_capital": round(capital, 2),
    }
    
    return {
        "config": config,
        "summary": summary,
        "trades": [asdict(t) for t in trades],
        "equity": equity,
    }


def print_results(result: dict):
    s = result["summary"]
    c = result["config"]
    
    print("\\n" + "=" * 60)
    print(f"📈 BACKTEST RESULTS: {c['symbol']}")
    print("=" * 60)
    print(f"Period:    {c['start_date']} to {c['end_date']} ({c['timeframe']})")
    print(f"Capital:   ₹{c['capital']:,.0f} | Risk: {c['risk_per_trade']}% per trade")
    print("-" * 60)
    print(f"Total Trades:  {s['total_trades']}")
    print(f"Win Rate:      {s['win_rate']}%")
    print(f"Profit Factor: {s['profit_factor']}")
    print(f"Total Return:  {s['total_return_pct']}%")
    print(f"Total P&L:     ₹{s['total_pnl']:,.2f}")
    print(f"Final Capital: ₹{s['final_capital']:,.2f}")
    print(f"Avg Win:       ₹{s['avg_win']:,.2f}")
    print(f"Avg Loss:      ₹{s['avg_loss']:,.2f}")
    print("=" * 60)
    
    if result["trades"]:
        print("\\n📋 LAST 10 TRADES:")
        print(f"{'#':<4} {'Side':<6} {'Entry':>10} {'Exit':>10} {'P&L':>12} {'Exit':<12}")
        for t in result["trades"][-10:]:
            print(f"{'-':<4} {t['side']:<6} ₹{t['entry_price']:>8.2f} ₹{t['exit_price']:>8.2f} ₹{t['pnl']:>10.2f} {t['exit_reason']:<12}")


def main():
    print("=" * 60)
    print("🧠 Natural Language Backtesting Engine")
    print("=" * 60)
    print("\\nType your strategy in plain English.")
    print('Examples:')
    print('  "RSI 2 period below 10 on Bank Nifty 5min, target RSI 90, 1% risk, last 3 months"')
    print('  "Golden cross 50/200 SMA on Nifty daily, 2% stop, 1:3 RR, last 1 year 10 lakh"')
    print('  "Bollinger bounce on RELIANCE 15min, RSI confirm, last 6 months"')
    print("\\nType 'quit' to exit\\n")
    
    while True:
        try:
            text = input("Strategy > ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            
            config = parse_strategy(text)
            result = run_backtest(config)
            print_results(result)
            
            save = input("\\nSave results to JSON? (y/n): ").strip().lower()
            if save == "y":
                fname = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(fname, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"Saved to {fname}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\\n❌ Error: {e}")
    
    print("\\n👋 Goodbye!")


if __name__ == "__main__":
    main()