OP.exe — TradingView Bridge + BOS/FVG/SMA/Liquidity Engine

This version preserves OP.exe's signal table, entry, take profit, stop loss, reward, projected futures P&L logic, independent timeframes, and secondary confluences.

Main directional hierarchy

The four highest-weight confluences are:

Break of Structure (BOS)

Fair Value Gap (FVG)

SMA20 / SMA50 structure

Liquidity sweep

Secondary evidence remains:

CHoCH

displacement

VWAP

EMA

RSI

ADX

volume expansion

ORB slot

How TradingView synchronization actually works

A Render server cannot directly read the pixels, DOM, or currently open chart from another browser tab.

The reliable bridge is TradingView's own alert/webhook system:

Add tradingview_bridge.pine to the TradingView chart.

Create an alert on that script.

Choose Any alert() function call.

Use Once per bar close behavior from the script.

Set the webhook URL to:

https://YOUR-RENDER-APP.onrender.com/tradingview/webhook

Set a private random string in Render as:
TRADINGVIEW_WEBHOOK_SECRET

Put the exact same string into the Pine indicator's Webhook secret input.

Do not put usernames, passwords, brokerage API keys, or other credentials into the webhook body.

Important TradingView behavior

TradingView alerts run from the symbol/timeframe/settings captured when the alert is created. Changing the chart later does not automatically convert the existing alert to the new timeframe.

For exact multi-timeframe synchronization, create one alert for each timeframe you want fed to OP.exe:

1m

5m

30m

4h / 240

1d / D

1w / W

What happens when TradingView data is fresh

The TradingView snapshot becomes the primary source for:

BOS

FVG

SMA

liquidity sweep

The app's own market feed still calculates:

secondary confluences

structural levels

fallback/cross-check data

TP / SL / R

TradingView swing/FVG values are also considered when selecting structural TP and SL.

Render

Build:

pip install -r requirements.txt

Start:

uvicorn main:app --host 0.0.0.0 --port $PORT

This uses a single FastAPI web service so the dashboard and TradingView webhook share the same HTTPS Render URL.

Accuracy

The strength/agreement score measures rule agreement. It is not a probability of winning and is not a promise of profitability.
