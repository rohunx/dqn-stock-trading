import yfinance as yf
import pandas as pd

def fetch_data(symbol: str, start: str = "2010-01-01", end: str = None) -> pd.DataFrame:
    data = yf.download(symbol, start=start, end=end)
    data.dropna(inplace=True)
    return data