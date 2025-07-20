# asts_dashboard_all_indicators_single_chart.py

import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from plotly.subplots import make_subplots
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from dateutil.relativedelta import relativedelta

def deduplicate_columns(cols):
    seen = {}
    new_cols = []
    for col in cols:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}.{seen[col]}")
    return new_cols

def fetch_data(start=None, end=None):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if start is None:
        start_date = datetime.today() - relativedelta(months=9)
        start = start_date.strftime("%Y-%m-%d")
    df = yf.download("ASTS", start=start, end=end, auto_adjust=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [col.split()[-1] if isinstance(col, str) else col for col in df.columns]
    df.columns = deduplicate_columns(df.columns)
    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    df.reset_index(inplace=True)

    # Standard Indicators
    df["SMA10"] = df["Close"].rolling(window=10).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["UpperBB"] = df["SMA10"] + 2 * df["Close"].rolling(window=10).std()
    df["LowerBB"] = df["SMA10"] - 2 * df["Close"].rolling(window=10).std()
    df["VWAP"] = VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()

    # Advanced Indicators
    df["RSI"] = RSIIndicator(close=df["Close"]).rsi()
    macd = MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()

    # Scaled versions for single-panel charting
    max_close = df["Close"].max()
    df["RSI_scaled"] = df["RSI"] / 100 * max_close
    df["MACD_scaled"] = (df["MACD"] - df["MACD"].min()) / (df["MACD"].max() - df["MACD"].min()) * max_close
    df["MACD_Signal_scaled"] = (df["MACD_Signal"] - df["MACD_Signal"].min()) / (df["MACD_Signal"].max() - df["MACD_Signal"].min()) * max_close
    df["ATR_scaled"] = df["ATR"] / df["ATR"].max() * max_close

    return df

def resample_weekly(df):
    df_weekly = df.copy()
    df_weekly.set_index("Date", inplace=True)
    df_weekly = df_weekly.resample("W").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna().reset_index()
    # Recalculate indicators for weekly data
    df_weekly["SMA10"] = df_weekly["Close"].rolling(window=10).mean()
    df_weekly["EMA20"] = df_weekly["Close"].ewm(span=20, adjust=False).mean()
    df_weekly["UpperBB"] = df_weekly["SMA10"] + 2 * df_weekly["Close"].rolling(window=10).std()
    df_weekly["LowerBB"] = df_weekly["SMA10"] - 2 * df_weekly["Close"].rolling(window=10).std()
    df_weekly["VWAP"] = df_weekly["Close"]  # VWAP not meaningful on weekly, so just plot Close
    df_weekly["RSI"] = RSIIndicator(close=df_weekly["Close"]).rsi()
    macd = MACD(close=df_weekly["Close"])
    df_weekly["MACD"] = macd.macd()
    df_weekly["MACD_Signal"] = macd.macd_signal()
    df_weekly["ATR"] = AverageTrueRange(df_weekly["High"], df_weekly["Low"], df_weekly["Close"]).average_true_range()
    max_close = df_weekly["Close"].max()
    df_weekly["RSI_scaled"] = df_weekly["RSI"] / 100 * max_close
    df_weekly["MACD_scaled"] = (df_weekly["MACD"] - df_weekly["MACD"].min()) / (df_weekly["MACD"].max() - df_weekly["MACD"].min()) * max_close
    df_weekly["MACD_Signal_scaled"] = (df_weekly["MACD_Signal"] - df_weekly["MACD_Signal"].min()) / (df_weekly["MACD_Signal"].max() - df_weekly["MACD_Signal"].min()) * max_close
    df_weekly["ATR_scaled"] = df_weekly["ATR"] / df_weekly["ATR"].max() * max_close

    return df_weekly

def generate_plotly_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)

    # 🕯️ Candlestick + Indicators
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"],
                                 increasing_line_color='limegreen', decreasing_line_color='orangered',
                                 name="Candlestick"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA10"], line=dict(color="orange", width=1), name="SMA10"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA20"], line=dict(color="deepskyblue", width=1), name="EMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["UpperBB"], line=dict(color="gray", dash="dot"), name="Upper BB"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["LowerBB"], line=dict(color="gray", dash="dot"), name="Lower BB"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["VWAP"], line=dict(color="white", width=1), name="VWAP"), row=1, col=1)

    # 🔁 Scaled Indicators
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI_scaled"], line=dict(color="violet", width=1, dash="dot"), name="RSI"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_scaled"], line=dict(color="lightgreen", width=1, dash="dash"), name="MACD"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal_scaled"], line=dict(color="orange", width=1, dash="dash"), name="MACD Signal"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["ATR_scaled"], line=dict(color="pink", width=1, dash="dot"), name="ATR"), row=1, col=1)

    # 📊 Volume
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], marker_color="royalblue",
                         marker_line_width=0, name="Volume", showlegend=False), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=960,
                      margin=dict(t=30, b=30, l=20, r=20),
                      xaxis=dict(rangeslider=dict(visible=True), type="date"),
                      legend=dict(x=0.01, y=0.99))
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def build_html_table(df):
    headers = ["Date", "Open", "High", "Low", "Close", "Volume"]
    rows = []
    for _, row in df.iterrows():
        row_data = [row["Date"].strftime("%Y-%m-%d"),
                    f"{row['Open']:.2f}", f"{row['High']:.2f}",
                    f"{row['Low']:.2f}", f"{row['Close']:.2f}",
                    f"{int(row['Volume']):,}"]
        rows.append("<tr><td>" + "</td><td>".join(row_data) + "</td></tr>")
    return "<table><tr><th>" + "</th><th>".join(headers) + "</th></tr>\n" + "\n".join(rows) + "</table>"

def save_html(chart_html, weekly_chart_html, table_html, output_path):
    html = f"""
    <html>
    <head>
        <title>ASTS Full Dashboard</title>
        <style>
            body {{
                font-family: monospace;
                background-color: #111;
                color: #eee;
                padding: 40px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 40px;
            }}
            th, td {{
                border: 1px solid #444;
                padding: 6px;
                text-align: center;
            }}
            th {{
                background-color: #222;
            }}
            tr:nth-child(even) {{
                background-color: #1a1a1a;
            }}
        </style>
    </head>
    <body>
        {chart_html}
        <h2 style='text-align:center;'>Weekly Chart</h2>
        {weekly_chart_html}
        {table_html}
    </body>
    </html>
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Dashboard saved to: {output_path}")

def main():
    df = fetch_data()
    chart_html = generate_plotly_chart(df)
    # --- Add weekly chart ---
    df_weekly = resample_weekly(df)
    weekly_chart_html = generate_plotly_chart(df_weekly)
    table_html = build_html_table(df)
    output_path = "asts.html"
    save_html(chart_html, weekly_chart_html, table_html, output_path)

if __name__ == "__main__":
    main()