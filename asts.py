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
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import base64
from io import BytesIO
import numpy as np

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
#git push -u origin mastergit push -u origin master
def create_mock_data():
    """Create mock data for testing when network is unavailable"""
    dates = pd.date_range(start='2024-10-01', end='2025-07-18', freq='D')
    dates = [d for d in dates if d.weekday() < 5]  # Only weekdays
    
    np.random.seed(42)  # For reproducible results
    n_days = len(dates)
    
    # Generate realistic stock price movements
    price_changes = np.random.normal(0, 0.02, n_days)
    price_changes[0] = 0  # Start with no change
    
    # Create cumulative price path starting at $30
    base_price = 30.0
    cumulative_changes = np.cumsum(price_changes)
    close_prices = base_price * np.exp(cumulative_changes)
    
    # Generate OHLV data
    opens = close_prices * np.random.normal(1.0, 0.005, n_days)
    highs = np.maximum(opens, close_prices) * np.random.normal(1.01, 0.01, n_days)
    lows = np.minimum(opens, close_prices) * np.random.normal(0.99, 0.01, n_days)
    volumes = np.random.randint(5000000, 50000000, n_days)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': close_prices,
        'Volume': volumes
    })
    
    return df

def fetch_data(start=None, end=None):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if start is None:
        start_date = datetime.today() - relativedelta(months=9)
        start = start_date.strftime("%Y-%m-%d")
    
    try:
        df = yf.download("ASTS", start=start, end=end, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [col.split()[-1] if isinstance(col, str) else col for col in df.columns]
        df.columns = deduplicate_columns(df.columns)
        df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
        df.reset_index(inplace=True)
        
        # If we got empty data, use mock data
        if df.empty:
            raise Exception("No data returned")
            
    except Exception as e:
        print(f"Warning: Could not fetch real data ({e}), using mock data")
        df = create_mock_data()

    # Fetch shares outstanding information
    try:
        ticker = yf.Ticker("ASTS")
        info = ticker.info
        shares_outstanding = info.get('sharesOutstanding', info.get('impliedSharesOutstanding', 0))
        if shares_outstanding == 0:
            # Fallback to a reasonable estimate if data is unavailable
            shares_outstanding = 150000000  # Approximate for ASTS as of recent data
    except Exception as e:
        print(f"Warning: Could not fetch shares outstanding data: {e}")
        shares_outstanding = 150000000  # Fallback value
    
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

def generate_seaborn_chart(df):
    """Generate Seaborn chart and return as base64 encoded image"""
    # Set the style for dark theme
    plt.style.use('dark_background')
    sns.set_palette("husl")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#111111')
    
    # Main price chart
    ax1.set_facecolor('#111111')
    
    # Plot candlestick data using line plots (Seaborn doesn't have native candlestick)
    ax1.plot(df["Date"], df["Close"], color='white', linewidth=2, label='Close Price')
    ax1.fill_between(df["Date"], df["Low"], df["High"], alpha=0.3, color='gray', label='High-Low Range')
    
    # Technical indicators
    if "SMA10" in df.columns and not df["SMA10"].isna().all():
        ax1.plot(df["Date"], df["SMA10"], color='orange', linewidth=1, label='SMA10')
    if "EMA20" in df.columns and not df["EMA20"].isna().all():
        ax1.plot(df["Date"], df["EMA20"], color='deepskyblue', linewidth=1, label='EMA20')
    if "UpperBB" in df.columns and not df["UpperBB"].isna().all():
        ax1.plot(df["Date"], df["UpperBB"], color='gray', linestyle='--', alpha=0.7, label='Upper BB')
    if "LowerBB" in df.columns and not df["LowerBB"].isna().all():
        ax1.plot(df["Date"], df["LowerBB"], color='gray', linestyle='--', alpha=0.7, label='Lower BB')
    if "VWAP" in df.columns and not df["VWAP"].isna().all():
        ax1.plot(df["Date"], df["VWAP"], color='white', linewidth=1, alpha=0.8, label='VWAP')
    
    ax1.set_title('ASTS Stock Price with Technical Indicators', color='white', fontsize=16)
    ax1.set_ylabel('Price ($)', color='white')
    ax1.tick_params(colors='white')
    ax1.legend(loc='upper left', frameon=False)
    ax1.grid(True, alpha=0.3)
    
    # Volume chart
    ax2.set_facecolor('#111111')
    ax2.bar(df["Date"], df["Volume"], color='royalblue', alpha=0.7)
    ax2.set_title('Trading Volume', color='white', fontsize=14)
    ax2.set_ylabel('Volume', color='white')
    ax2.set_xlabel('Date', color='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.3)
    
    # Format dates on x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Convert to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='#111111', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    buffer.close()
    plt.close()
    
    return image_base64

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

def save_html(chart_html, weekly_chart_html, seaborn_chart_b64, weekly_seaborn_chart_b64, table_html, output_path):
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
            .controls {{
                text-align: center;
                margin: 20px 0;
            }}
            .controls select {{
                padding: 8px 16px;
                background-color: #222;
                color: #eee;
                border: 1px solid #444;
                border-radius: 4px;
                font-size: 16px;
                font-family: monospace;
            }}
            .chart-container {{
                margin: 20px 0;
            }}
            .seaborn-chart {{
                display: none;
                text-align: center;
            }}
            .seaborn-chart img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
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
        <script>
            function toggleChartType() {{
                const chartType = document.getElementById('chartSelector').value;
                const plotlyCharts = document.querySelectorAll('.plotly-chart');
                const seabornCharts = document.querySelectorAll('.seaborn-chart');
                
                if (chartType === 'plotly') {{
                    plotlyCharts.forEach(chart => chart.style.display = 'block');
                    seabornCharts.forEach(chart => chart.style.display = 'none');
                }} else {{
                    plotlyCharts.forEach(chart => chart.style.display = 'none');
                    seabornCharts.forEach(chart => chart.style.display = 'block');
                }}
            }}
        </script>
    </head>
    <body>
        <div class="controls">
            <label for="chartSelector">Chart Library: </label>
            <select id="chartSelector" onchange="toggleChartType()">
                <option value="plotly" selected>Plotly (Interactive)</option>
                <option value="seaborn">Seaborn (Static)</option>
            </select>
        </div>
        
        <div class="chart-container">
            <h2 style='text-align:center;'>Daily Chart</h2>
            <div id="plotly-daily" class="plotly-chart">
                {chart_html}
            </div>
            <div id="seaborn-daily" class="seaborn-chart">
                <img src="data:image/png;base64,{seaborn_chart_b64}" alt="ASTS Daily Chart (Seaborn)" />
            </div>
        </div>
        
        <div class="chart-container">
            <h2 style='text-align:center;'>Weekly Chart</h2>
            <div id="plotly-weekly" class="plotly-chart">
                {weekly_chart_html}
            </div>
            <div id="seaborn-weekly" class="seaborn-chart">
                <img src="data:image/png;base64,{weekly_seaborn_chart_b64}" alt="ASTS Weekly Chart (Seaborn)" />
            </div>
        </div>
        
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
    seaborn_chart_b64 = generate_seaborn_chart(df)
    
    df_weekly = resample_weekly(df)
    weekly_chart_html = generate_plotly_chart(df_weekly)
    weekly_seaborn_chart_b64 = generate_seaborn_chart(df_weekly)
    
    table_html = build_html_table(df)
    output_path = "asts.html"

    # Calculate percent change over period
    first_close = df["Close"].iloc[0]
    last_close = df["Close"].iloc[-1]
    pct_change = ((last_close - first_close) / first_close) * 100
    pct_change_html = f"<h3 style='text-align:center;'>% Change Over Period: {pct_change:.2f}%</h3>"

    save_html(chart_html, weekly_chart_html, seaborn_chart_b64, weekly_seaborn_chart_b64, pct_change_html + table_html, output_path)

if __name__ == "__main__":
    main()