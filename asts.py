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
import base64
from io import BytesIO

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
    # Set the style for dark theme
    plt.style.use('dark_background')
    sns.set_palette("husl")
    
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#111111')
    
    # Main price chart
    ax1.set_facecolor('#111111')
    
    # Plot price lines
    ax1.plot(df["Date"], df["Close"], color='white', linewidth=2, label='Close Price')
    ax1.plot(df["Date"], df["SMA10"], color='orange', linewidth=1, label='SMA10')
    ax1.plot(df["Date"], df["EMA20"], color='deepskyblue', linewidth=1, label='EMA20')
    ax1.plot(df["Date"], df["UpperBB"], color='gray', linestyle='--', alpha=0.7, label='Upper BB')
    ax1.plot(df["Date"], df["LowerBB"], color='gray', linestyle='--', alpha=0.7, label='Lower BB')
    ax1.plot(df["Date"], df["VWAP"], color='white', linewidth=1, alpha=0.8, label='VWAP')
    
    # Fill area between Bollinger Bands
    ax1.fill_between(df["Date"], df["UpperBB"], df["LowerBB"], alpha=0.1, color='gray')
    
    # Add scaled indicators
    ax1.plot(df["Date"], df["RSI_scaled"], color='violet', linewidth=1, linestyle=':', label='RSI (scaled)')
    ax1.plot(df["Date"], df["MACD_scaled"], color='lightgreen', linewidth=1, linestyle='--', label='MACD (scaled)')
    ax1.plot(df["Date"], df["MACD_Signal_scaled"], color='orange', linewidth=1, linestyle='--', label='MACD Signal (scaled)')
    
    ax1.set_title('ASTS Stock Analysis', color='white', fontsize=16, pad=20)
    ax1.set_ylabel('Price ($)', color='white')
    ax1.legend(loc='upper left', framealpha=0.1)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(colors='white')
    
    # Volume chart
    ax2.set_facecolor('#111111')
    bars = ax2.bar(df["Date"], df["Volume"], color='royalblue', alpha=0.7)
    ax2.set_ylabel('Volume', color='white')
    ax2.set_xlabel('Date', color='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    # Convert to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', facecolor='#111111', edgecolor='none', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f'<img src="data:image/png;base64,{chart_base64}" style="width:100%; height:auto;">'

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

def save_html(chart_html, weekly_chart_html, seaborn_chart_html, weekly_seaborn_chart_html, table_html, output_path):
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
            .chart-controls {{
                text-align: center;
                margin: 20px 0;
            }}
            .chart-dropdown {{
                background-color: #222;
                color: #eee;
                border: 1px solid #444;
                padding: 8px 16px;
                font-size: 16px;
                border-radius: 4px;
                font-family: monospace;
            }}
            .chart-container {{
                margin: 20px 0;
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
            .hidden {{
                display: none;
            }}
        </style>
        <script>
            function toggleChartLibrary() {{
                var dropdown = document.getElementById('chartLibrary');
                var plotlyDaily = document.getElementById('plotly-daily-chart');
                var plotlyWeekly = document.getElementById('plotly-weekly-chart');
                var seabornDaily = document.getElementById('seaborn-daily-chart');
                var seabornWeekly = document.getElementById('seaborn-weekly-chart');
                
                if (dropdown.value === 'plotly') {{
                    plotlyDaily.classList.remove('hidden');
                    plotlyWeekly.classList.remove('hidden');
                    seabornDaily.classList.add('hidden');
                    seabornWeekly.classList.add('hidden');
                }} else {{
                    plotlyDaily.classList.add('hidden');
                    plotlyWeekly.classList.add('hidden');
                    seabornDaily.classList.remove('hidden');
                    seabornWeekly.classList.remove('hidden');
                }}
            }}
            
            // Initialize on page load
            window.onload = function() {{
                toggleChartLibrary();
            }}
        </script>
    </head>
    <body>
        <h1 style='text-align:center;'>ASTS Stock Dashboard</h1>
        
        <div class="chart-controls">
            <label for="chartLibrary">Chart Library: </label>
            <select id="chartLibrary" class="chart-dropdown" onchange="toggleChartLibrary()">
                <option value="plotly" selected>Plotly (Interactive)</option>
                <option value="seaborn">Seaborn (Static)</option>
            </select>
        </div>
        
        <div id="plotly-daily-chart" class="chart-container">
            <h2 style='text-align:center;'>Daily Chart (Plotly)</h2>
            {chart_html}
        </div>
        
        <div id="seaborn-daily-chart" class="chart-container hidden">
            <h2 style='text-align:center;'>Daily Chart (Seaborn)</h2>
            {seaborn_chart_html}
        </div>
        
        <div id="plotly-weekly-chart" class="chart-container">
            <h2 style='text-align:center;'>Weekly Chart (Plotly)</h2>
            {weekly_chart_html}
        </div>
        
        <div id="seaborn-weekly-chart" class="chart-container hidden">
            <h2 style='text-align:center;'>Weekly Chart (Seaborn)</h2>
            {weekly_seaborn_chart_html}
        </div>
        
        {table_html}
    </body>
    </html>
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Dashboard saved to: {{output_path}}")

def main():
    df = fetch_data()
    
    # Generate Plotly charts
    chart_html = generate_plotly_chart(df)
    df_weekly = resample_weekly(df)
    weekly_chart_html = generate_plotly_chart(df_weekly)
    
    # Generate Seaborn charts
    seaborn_chart_html = generate_seaborn_chart(df)
    weekly_seaborn_chart_html = generate_seaborn_chart(df_weekly)
    
    table_html = build_html_table(df)
    output_path = "asts.html"

    # Calculate percent change over period
    first_close = df["Close"].iloc[0]
    last_close = df["Close"].iloc[-1]
    pct_change = ((last_close - first_close) / first_close) * 100
    pct_change_html = f"<h3 style='text-align:center;'>% Change Over Period: {pct_change:.2f}%</h3>"

    save_html(chart_html, weekly_chart_html, seaborn_chart_html, weekly_seaborn_chart_html, pct_change_html + table_html, output_path)

if __name__ == "__main__":
    main()