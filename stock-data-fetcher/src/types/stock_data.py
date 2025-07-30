class StockData:
    def __init__(self, symbol: str, price: float, volume: int):
        self.symbol = symbol
        self.price = price
        self.volume = volume

    def __repr__(self):
        return f"StockData(symbol={self.symbol}, price={self.price}, volume={self.volume})"