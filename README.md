# Natural Language Backtesting Engine

Type trading strategies in plain English and backtest them instantly.

## Quick Start

```bash
pip install -r requirements.txt
python backtest.py
```

## Examples

```
Strategy > RSI 2 below 10 on Bank Nifty 5min, sell when RSI above 90, 1% risk, last 3 months
Strategy > Golden cross 50/200 SMA on Nifty daily, 2% stop, 1:3 RR, last 1 year 10 lakh
Strategy > Bollinger bounce on RELIANCE 15min, RSI confirm, last 6 months 5 lakh
```

## Supported Patterns

- **RSI**: "RSI 14 below 30"
- **Golden Cross**: "50 SMA crosses above 200 SMA"
- **Bollinger**: "Price touches lower Bollinger Band"
- **Stop Loss**: "1% stop loss" or "2x ATR stop"
- **Target**: "Target 6%" or "1:3 risk-reward"
- **Timeframes**: 1m, 5m, 15m, 30m, 1h, daily, weekly