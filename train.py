"""
train.py

This script is an interactive, self-guided training on key technical indicators.
It provides multiple examples, visualizations, and questions to help you understand each indicator in depth.

Run: python train.py
It will generate train.html, which you can open in your browser for a 5-10 minute guided learning session.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
import os
import shutil

np.random.seed(42)

# Helper to create different market scenarios
def make_scenario(trend=0, volatility=1, n=100, start=100):
    steps = np.random.randn(n) * volatility + trend
    prices = np.cumsum(steps) + start
    volumes = np.random.randint(1000, 5000, size=n)
    return prices, volumes

# Scenarios: uptrend, downtrend, sideways, volatile
# Add more diverse scenarios for deeper understanding
scenarios = [
    ("Uptrend", *make_scenario(trend=0.3, volatility=1.2)),
    ("Downtrend", *make_scenario(trend=-0.3, volatility=1.2)),
    ("Sideways", *make_scenario(trend=0, volatility=1.2)),
    ("Volatile", *make_scenario(trend=0, volatility=3)),
    ("Sharp Rally", *make_scenario(trend=1.0, volatility=2)),
    ("Sharp Drop", *make_scenario(trend=-1.0, volatility=2)),
    ("Calm Market", *make_scenario(trend=0, volatility=0.5)),
    ("Choppy Market", *make_scenario(trend=0, volatility=4)),
]

dates = pd.date_range(end=pd.Timestamp.today(), periods=100)

# Prepare HTML content
html_sections = []
img_files = []

def plot_and_save(fig, name):
    fname = f"train_{name}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    img_files.append(fname)
    return fname

# --- SMA & EMA ---
sma_ema_html = "<h2>SMA & EMA: Smoothing Price Data</h2>"
sma_ema_html += """
<p>
<b>SMA</b> (Simple Moving Average) and <b>EMA</b> (Exponential Moving Average) help you see the trend by smoothing out price noise.<br>
<b>EMA</b> reacts faster to price changes, which can help you spot reversals sooner, but may also give more false signals in choppy markets.<br>
<b>Tip:</b> In a strong trend, both SMA and EMA will slope in the trend direction. In sideways or choppy markets, they may cross the price often.
</p>
"""
for label, prices, volumes in scenarios:
    df = pd.DataFrame({"Date": dates, "Close": prices, "Volume": volumes})
    df["SMA10"] = df["Close"].rolling(window=10).mean()
    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["Date"], df["Close"], label="Close", color="black")
    ax.plot(df["Date"], df["SMA10"], label="SMA10", color="orange")
    ax.plot(df["Date"], df["EMA10"], label="EMA10", color="blue")
    ax.set_title(f"{label} Example: SMA vs EMA")
    ax.legend()
    fname = plot_and_save(fig, f"smaema_{label.lower()}")
    sma_ema_html += f"""
    <h3>{label} Market</h3>
    <img src="{fname}" alt="{label} SMA/EMA">
    <p>
    <b>What to notice:</b> In a {label.lower()}, see how the moving averages behave. 
    Do they follow the price closely or lag behind? Are there whipsaws (false signals)?
    </p>
    """

# --- Bollinger Bands ---
bb_html = "<h2>Bollinger Bands: Volatility Envelopes</h2>"
bb_html += """
<p>
Bollinger Bands help you visualize volatility. When the bands are wide, the market is volatile; when narrow, it's calm.<br>
<b>Tip:</b> Price touching or crossing the bands can signal overbought/oversold, but in strong trends, price can 'ride' the band.
</p>
"""
for label, prices, volumes in scenarios:
    df = pd.DataFrame({"Date": dates, "Close": prices, "Volume": volumes})
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["UpperBB"] = df["SMA20"] + 2 * df["Close"].rolling(window=20).std()
    df["LowerBB"] = df["SMA20"] - 2 * df["Close"].rolling(window=20).std()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["Date"], df["Close"], label="Close", color="black")
    ax.plot(df["Date"], df["SMA20"], label="SMA20", color="orange")
    ax.fill_between(df["Date"], df["UpperBB"], df["LowerBB"], color="gray", alpha=0.2, label="Bollinger Bands")
    ax.set_title(f"{label} Example: Bollinger Bands")
    ax.legend()
    fname = plot_and_save(fig, f"bb_{label.lower()}")
    bb_html += f"""
    <h3>{label} Market</h3>
    <img src="{fname}" alt="{label} Bollinger Bands">
    <p>
    <b>What to notice:</b> In a {label.lower()}, do the bands expand or contract? Does price bounce off the bands or break through?
    </p>
    """

# --- VWAP ---
vwap_html = "<h2>VWAP: Volume Weighted Average Price</h2>"
vwap_html += """
<p>
VWAP is used by professionals to judge the average price paid for a stock during the day. If price is above VWAP, buyers are in control; below, sellers are.<br>
<b>Tip:</b> VWAP is most useful for intraday trading, but can help you see if you're buying/selling at a fair price.
</p>
"""
for label, prices, volumes in scenarios:
    df = pd.DataFrame({"Date": dates, "Close": prices, "Volume": volumes})
    df["Open"] = df["Close"] + np.random.randn(100)
    df["High"] = df[["Open", "Close"]].max(axis=1) + np.abs(np.random.randn(100))
    df["Low"] = df[["Open", "Close"]].min(axis=1) - np.abs(np.random.randn(100))
    df["VWAP"] = VolumeWeightedAveragePrice(
        high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]
    ).volume_weighted_average_price()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["Date"], df["Close"], label="Close", color="black")
    ax.plot(df["Date"], df["VWAP"], label="VWAP", color="purple")
    ax.set_title(f"{label} Example: VWAP")
    ax.legend()
    fname = plot_and_save(fig, f"vwap_{label.lower()}")
    vwap_html += f"""
    <h3>{label} Market</h3>
    <img src="{fname}" alt="{label} VWAP">
    <p>
    <b>What to notice:</b> In a {label.lower()}, does price stay above or below VWAP? Does it cross VWAP often?
    </p>
    """

# --- RSI ---
rsi_html = "<h2>RSI: Relative Strength Index</h2>"
rsi_html += """
<p>
RSI measures momentum. Above 70 is overbought, below 30 is oversold.<br>
<b>Tip:</b> In strong trends, RSI can stay overbought/oversold for a long time. Look for divergences (price makes new high, RSI does not) as warning signs.
</p>
"""
for label, prices, volumes in scenarios:
    df = pd.DataFrame({"Date": dates, "Close": prices, "Volume": volumes})
    df["RSI"] = RSIIndicator(close=df["Close"]).rsi()
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(df["Date"], df["Close"], label="Close", color="black")
    ax2 = ax1.twinx()
    ax2.plot(df["Date"], df["RSI"], label="RSI", color="green")
    ax2.axhline(70, color="red", linestyle="--", alpha=0.5)
    ax2.axhline(30, color="red", linestyle="--", alpha=0.5)
    ax1.set_title(f"{label} Example: RSI")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fname = plot_and_save(fig, f"rsi_{label.lower()}")
    rsi_html += f"""
    <h3>{label} Market</h3>
    <img src="{fname}" alt="{label} RSI">
    <p>
    <b>What to notice:</b> In a {label.lower()}, does RSI stay in the overbought/oversold zone? Are there divergences?
    </p>
    """

# --- MACD ---
macd_html = "<h2>MACD: Trend and Momentum</h2>"
macd_html += """
<p>
MACD shows the relationship between two moving averages. Crossovers can signal trend changes.<br>
<b>Tip:</b> Look for MACD crossing above/below its signal line, and for divergence between MACD and price.
</p>
"""
for label, prices, volumes in scenarios:
    df = pd.DataFrame({"Date": dates, "Close": prices, "Volume": volumes})
    macd = MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["Date"], df["MACD"], label="MACD", color="teal")
    ax.plot(df["Date"], df["MACD_Signal"], label="MACD Signal", color="magenta")
    ax.set_title(f"{label} Example: MACD")
    ax.legend()
    fname = plot_and_save(fig, f"macd_{label.lower()}")
    macd_html += f"""
    <h3>{label} Market</h3>
    <img src="{fname}" alt="{label} MACD">
    <p>
    <b>What to notice:</b> In a {label.lower()}, do you see clear crossovers? Is MACD confirming the trend or showing divergence?
    </p>
    """

# --- ATR ---
atr_html = "<h2>ATR: Average True Range (Volatility)</h2>"
atr_html += """
<p>
ATR measures volatility. High ATR means big price swings; low ATR means calm markets.<br>
<b>Tip:</b> Use ATR to set stop-losses: in volatile markets, use wider stops.
</p>
"""
for label, prices, volumes in scenarios:
    df = pd.DataFrame({"Date": dates, "Close": prices, "Volume": volumes})
    df["Open"] = df["Close"] + np.random.randn(100)
    df["High"] = df[["Open", "Close"]].max(axis=1) + np.abs(np.random.randn(100))
    df["Low"] = df[["Open", "Close"]].min(axis=1) - np.abs(np.random.randn(100))
    df["ATR"] = AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"]
    ).average_true_range()
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(df["Date"], df["Close"], label="Close", color="black")
    ax2 = ax1.twinx()
    ax2.plot(df["Date"], df["ATR"], label="ATR", color="brown")
    ax1.set_title(f"{label} Example: ATR")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fname = plot_and_save(fig, f"atr_{label.lower()}")
    atr_html += f"""
    <h3>{label} Market</h3>
    <img src="{fname}" alt="{label} ATR">
    <p>
    <b>What to notice:</b> In a {label.lower()}, does ATR rise or fall? Does it warn you of increased risk?
    </p>
    """

# --- Interactive Quiz Section with Hover-to-Reveal Answers ---
quiz_html = """
<h2>Test Your Knowledge</h2>
<ol>
<li class="quiz-q">What does it mean if the RSI is above 70? Below 30?
    <span class="quiz-a">Above 70: Overbought (potential reversal down).<br>Below 30: Oversold (potential reversal up).</span>
</li>
<li class="quiz-q">How do Bollinger Bands behave during periods of high volatility?
    <span class="quiz-a">The bands widen as volatility increases and contract as volatility decreases.</span>
</li>
<li class="quiz-q">What is the main difference between SMA and EMA?
    <span class="quiz-a">EMA gives more weight to recent prices, so it reacts faster to price changes than SMA.</span>
</li>
<li class="quiz-q">How can you use MACD crossovers in trading?
    <span class="quiz-a">A bullish signal occurs when MACD crosses above its signal line; bearish when it crosses below.</span>
</li>
<li class="quiz-q">What does a rising ATR indicate?
    <span class="quiz-a">Increasing volatility in the market.</span>
</li>
<li class="quiz-q">How does VWAP help traders?
    <span class="quiz-a">VWAP shows the average price weighted by volume; traders use it to gauge if they bought/sold at a good price.</span>
</li>
</ol>
<p>Hover over each question to reveal the answer!</p>
"""

# --- Assemble HTML ---
html = f"""
<html>
<head>
    <title>Technical Indicator Training</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #181818;
            color: #eee;
            padding: 40px;
        }}
        h2 {{
            color: #ffb347;
        }}
        h3 {{
            color: #7ec8e3;
        }}
        ul, ol {{
            font-size: 1.1em;
        }}
        img {{
            display: block;
            margin: 30px auto;
            border: 2px solid #444;
            max-width: 100%;
        }}
        p {{
            max-width: 800px;
            margin: 0 auto 20px auto;
        }}
        .quiz-q {{
            position: relative;
            cursor: pointer;
            background: #222;
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 10px;
            transition: background 0.2s;
        }}
        .quiz-q:hover {{
            background: #333;
        }}
        .quiz-a {{
            display: none;
            position: absolute;
            left: 0;
            top: 100%;
            background: #333;
            color: #ffe066;
            padding: 10px;
            border-radius: 6px;
            margin-top: 4px;
            min-width: 250px;
            z-index: 10;
            box-shadow: 0 2px 8px #0008;
        }}
        .quiz-q:hover .quiz-a {{
            display: block;
        }}
    </style>
</head>
<body>
    <h1>Technical Indicator Training</h1>
    <p>This interactive session will walk you through several technical indicators with multiple market scenarios. Take your time to study each example and try the quiz at the end!</p>
    {sma_ema_html}
    {bb_html}
    {vwap_html}
    {rsi_html}
    {macd_html}
    {atr_html}
    {quiz_html}
    <p style="text-align:center; color:#aaa;">End of training. Review this file as often as needed!</p>
</body>
</html>
"""

# --- Create 'training' folder and move all output files there ---
output_dir = "training"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Write HTML file to the training folder
html_path = os.path.join(output_dir, "train.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ Training complete! Open {html_path} in your browser for a full guided session.")

# Move all PNG images to the training folder
for fname in img_files:
    if os.path.exists(fname):
        shutil.move(fname, os.path.join(output_dir, fname))

print(f"✅ All images moved to '{output_dir}' folder.")