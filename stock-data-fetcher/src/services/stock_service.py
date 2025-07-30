class StockService:
    def __init__(self, api_url):
        self.api_url = api_url

    def fetch_stock_data(self, symbol):
        import requests

        response = requests.get(f"{self.api_url}/stocks/{symbol}")
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

    def is_market_open(self):
        from datetime import datetime
        import pytz

        eastern = pytz.timezone('US/Eastern')
        current_time = datetime.now(eastern)
        return self.is_within_market_hours(current_time)

    def is_within_market_hours(self, current_time):
        market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= current_time <= market_close