from datetime import timezone, datetime
from pathlib import Path

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment
from alpaca.common.enums import Sort

import pandas as pd

API_KEY = ""
SECRET_KEY = ""

source_sectors = {
    "Information Technology": ["MSFT", "CSCO"],
    "Consumer Discretionary": ["AMZN", "EBAY"],
    "Consumer Staples": ["WMT", "COST"],
    "Financials": ["JPM", "BAC"],
    "Health Care": ["JNJ", "PFE"],
    "Energy": ["XOM", "CVX"]
}
source_symbols = [symbol for symbols in source_sectors.values() for symbol in symbols]
target_symbols = ["CRWD", "CHWY"]
all_symbols = source_symbols + target_symbols

def find_first_bar(client, symbol, timeframe):
    """
        Finds the date of the first available bar for a given symbol and timeframe.
    """
    # Only pull 1 bar no need for more
    # Sort so we get the earliest possible
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute if timeframe == '1m' else TimeFrame(5, TimeFrameUnit.Minute),
        start=datetime(2010, 1, 1, tzinfo=timezone.utc),
        end=datetime(2020, 1, 1, tzinfo=timezone.utc),
        feed=DataFeed.SIP,
        sort=Sort.ASC,
        limit=1  
    )

    bars = client.get_stock_bars(request)
    symbol_bars = bars.data.get(symbol, [])

    return symbol_bars[0].timestamp if symbol_bars else None

def next_month(dt):
    """
        Returns first day of the following month
    """
    year = dt.year + (dt.month == 12)
    month = dt.month % 12 + 1
    return datetime(year, month, 1, tzinfo=timezone.utc)

def fetch_symbol(client, symbol, start, end=datetime(2020, 2, 22, tzinfo=timezone.utc)):
    """
        Fetches the minute data for a given stock within a specified time range.
    """
    RAW_DIR = Path('raw')
    path = RAW_DIR / f"{symbol}.parquet"

    if path.exists():
        print("Symbol already downloaded")
        return

    chunks = []
    curr = start
    while curr < end:
        chunk_end = min(next_month(curr), end)

        request = StockBarsRequest(
            start=curr,
            end=chunk_end, 
            sort=Sort.ASC,
            feed=DataFeed.SIP,
            timeframe=TimeFrame.Minute,
            adjustment=Adjustment.RAW,
            symbol_or_symbols=symbol,
        )

        bars = client.get_stock_bars(request)
        df = bars.df

        if not df.empty:
            df = df.reset_index()   
            chunks.append(df)
            print(f"{symbol}: {curr:%Y-%m}")

        curr = chunk_end

    if not chunks:
        print("No data for {symbol}")

    df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["timestamp", "symbol"]).sort_values("timestamp").reset_index(drop=True)
    df.to_parquet(path, index=False)
    
def fetch_all(client, symbols, first_dates):
    """
        Fetches the 1 minute data for all symbols. 
    """
    for symbol in dict.fromkeys(symbols):
        fetch_symbol(client, symbol, first_dates[symbol])

client = StockHistoricalDataClient(
    api_key=API_KEY,
    secret_key=SECRET_KEY,
)

symbols_first_5m = {symbol: find_first_bar(client, symbol, '5m') for symbol in all_symbols}
symbols_first_1m = {symbol: find_first_bar(client, symbol, '1m') for symbol in all_symbols}

fetch_all(client, all_symbols, symbols_first_1m)