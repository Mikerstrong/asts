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
import random

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

def create_mock_data(symbol, days=200):
    """Create mock stock data when real data is unavailable"""
    # Set different seeds for different stocks to get different data
    seed_map = {"ASTS": 42, "ASTX": 123, "XXRP": 456}
    random.seed(seed_map.get(symbol, 42))
    
    dates = pd.date_range(end=datetime.today(), periods=days, freq='D')
    
    # Different base prices and trends for different stocks
    base_prices = {"ASTS": 25.0, "ASTX": 15.0, "XXRP": 8.0}
    base_price = base_prices.get(symbol, 20.0)
    
    trends = {"ASTS": 0.001, "ASTX": -0.0005, "XXRP": 0.0008}
    trend = trends.get(symbol, 0)
    
    # Generate realistic price movements
    prices = [base_price]
    for i in range(1, days):
        change = random.uniform(-0.04, 0.04) + trend  # Daily volatility + trend
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # Ensure price doesn't go below $1
    
    # Create OHLC data
    opens = [p * random.uniform(0.99, 1.01) for p in prices]
    closes = prices
    highs = [max(o, c) * random.uniform(1.00, 1.04) for o, c in zip(opens, closes)]
    lows = [min(o, c) * random.uniform(0.96, 1.00) for o, c in zip(opens, closes)]
    volumes = [random.randint(1000000, 20000000) for _ in range(days)]
    
    # Shares outstanding varies by stock
    shares_map = {"ASTS": 150000000, "ASTX": 100000000, "XXRP": 80000000}
    shares_outstanding = shares_map.get(symbol, 100000000)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes,
        'Shares Outstanding': [shares_outstanding] * days
    })
    
    # Reset index to match the expected structure
    df = df.reset_index(drop=True)
    
    return df

def fetch_data(symbol="ASTS", start=None, end=None):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if start is None:
        start_date = datetime.today() - relativedelta(months=9)
        start = start_date.strftime("%Y-%m-%d")
    
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [col.split()[-1] if isinstance(col, str) else col for col in df.columns]
        df.columns = deduplicate_columns(df.columns)
        df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
        df.reset_index(inplace=True)
        
        # If we got no data or too little data, fall back to mock data
        if len(df) < 50:
            print(f"Warning: Insufficient real data for {symbol}, using mock data")
            df = create_mock_data(symbol)
    except Exception as e:
        print(f"Warning: Failed to fetch real data for {symbol}: {e}")
        print(f"Using mock data for {symbol}")
        df = create_mock_data(symbol)

    # Fetch shares outstanding information
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        shares_outstanding = info.get('sharesOutstanding', info.get('impliedSharesOutstanding', 0))
        if shares_outstanding == 0:
            # Fallback to reasonable estimates if data is unavailable
            fallback_shares = {
                "ASTS": 150000000,   # Approximate for ASTS
                "ASTX": 100000000,   # Approximate for ASTX  
                "XXRP": 80000000     # Approximate for XXRP
            }
            shares_outstanding = fallback_shares.get(symbol, 100000000)  # Default fallback
    except Exception as e:
        print(f"Warning: Could not fetch shares outstanding data for {symbol}: {e}")
        fallback_shares = {
            "ASTS": 150000000,   # Approximate for ASTS
            "ASTX": 100000000,   # Approximate for ASTX  
            "XXRP": 80000000     # Approximate for XXRP
        }
        shares_outstanding = fallback_shares.get(symbol, 100000000)  # Default fallback
    
    # Add shares outstanding as a column (same value for all rows since it doesn't change daily)
    df["Shares Outstanding"] = shares_outstanding

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
        "Volume": "sum",
        "Shares Outstanding": "first"  # Shares outstanding doesn't change, so use first value
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
    headers = ["Date", "Open", "High", "Low", "Close", "Volume", "Shares Outstanding"]
    rows = []
    for _, row in df.iterrows():
        row_data = [row["Date"].strftime("%Y-%m-%d"),
                    f"{row['Open']:.2f}", f"{row['High']:.2f}",
                    f"{row['Low']:.2f}", f"{row['Close']:.2f}",
                    f"{int(row['Volume']):,}",
                    f"{int(row['Shares Outstanding']):,}"]
        rows.append("<tr><td>" + "</td><td>".join(row_data) + "</td></tr>")
    return "<table><tr><th>" + "</th><th>".join(headers) + "</th></tr>\n" + "\n".join(rows) + "</table>"

def save_html(stocks_data, output_path):
    html = f"""
    <html>
    <head>
        <title>Multi-Stock Dashboard (ASTS, ASTX, XXRP)</title>
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
            .stock-section {{
                margin-bottom: 60px;
                border-bottom: 2px solid #444;
                padding-bottom: 40px;
            }}
            .stock-title {{
                text-align: center;
                color: #ffb347;
                font-size: 2em;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <h1 style='text-align:center; color: #ffb347;'>Multi-Stock Dashboard</h1>
        <p style='text-align:center; color: #aaa;'>ASTS, ASTX, and XXRP Stock Analysis</p>
    """
    
    for symbol, data in stocks_data.items():
        html += f"""
        <div class="stock-section">
            <h2 class="stock-title">{symbol} Dashboard</h2>
            {data['chart_html']}
            <h3 style='text-align:center;'>Weekly Chart</h3>
            {data['weekly_chart_html']}
            {data['pct_change_html']}
            {data['table_html']}
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Multi-stock dashboard saved to: {output_path}")

def main():
    symbols = ["ASTS", "ASTX", "XXRP"]
    stocks_data = {}
    
    for symbol in symbols:
        print(f"📊 Processing data for {symbol}...")
        try:
            df = fetch_data(symbol)
            chart_html = generate_plotly_chart(df)
            df_weekly = resample_weekly(df)
            weekly_chart_html = generate_plotly_chart(df_weekly)
            table_html = build_html_table(df)
            
            # Calculate percent change over period
            if len(df) > 0:
                first_close = df["Close"].iloc[0]
                last_close = df["Close"].iloc[-1]
                pct_change = ((last_close - first_close) / first_close) * 100
                pct_change_html = f"<h3 style='text-align:center;'>% Change Over Period: {pct_change:.2f}%</h3>"
            else:
                pct_change_html = "<h3 style='text-align:center;'>% Change Over Period: N/A</h3>"
            
            stocks_data[symbol] = {
                'chart_html': chart_html,
                'weekly_chart_html': weekly_chart_html,
                'table_html': table_html,
                'pct_change_html': pct_change_html
            }
            print(f"✅ {symbol} data processed successfully")
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            # Create placeholder data for failed stocks
            stocks_data[symbol] = {
                'chart_html': f"<p style='text-align:center; color:red;'>Failed to load data for {symbol}: {e}</p>",
                'weekly_chart_html': "",
                'table_html': f"<p style='text-align:center; color:red;'>No data available for {symbol}</p>",
                'pct_change_html': "<h3 style='text-align:center;'>% Change Over Period: N/A</h3>"
            }
    
    output_path = "asts.html"
    save_html(stocks_data, output_path)

if __name__ == "__main__":
    main()