from datetime import datetime, time

def get_current_time():
    return datetime.now()

def is_within_market_hours():
    current_time = get_current_time().time()
    market_open = time(9, 30)  # Market opens at 9:30 AM
    market_close = time(16, 0)  # Market closes at 4:00 PM
    return market_open <= current_time <= market_close