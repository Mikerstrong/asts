from datetime import datetime
import schedule
import time
from services.stock_service import StockService
from utils.time_utils import is_within_market_hours

def fetch_stock_data():
    stock_service = StockService()
    if stock_service.is_market_open():
        stock_data = stock_service.fetch_stock_data()
        print(f"Fetched stock data: {stock_data}")
    else:
        print("The stock market is currently closed.")

def main():
    # Schedule the fetch_stock_data function to run every hour
    schedule.every().hour.at(":00").do(fetch_stock_data)

    print("Stock data fetcher is running...")
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()