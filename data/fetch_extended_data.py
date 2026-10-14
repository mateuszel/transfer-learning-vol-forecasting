import yfinance as yf
import pandas as pd
from pathlib import Path

symbols = [
    "MSFT", "CSCO", "AMZN", "EBAY",
    "WMT", "COST", "JPM", "BAC",
    "JNJ", "PFE", "XOM", "CVX",
    "CRWD", "CHWY",
]

START_DATE = "2015-12-20"
END_DATE = "2020-02-22"  # yfinance end date is exclusive
OUTPUT_DIR = Path(__file__).resolve().parent / "raw" / "extended"
FRED_DTB3_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"


def _flatten_yahoo_columns(frame):
    """Return a single-level column index for a yfinance response."""
    result = frame.copy()
    if not isinstance(result.columns, pd.MultiIndex):
        return result

    result.columns = [
        column[0] if column[0] in {"Close", "Volume"} else column[-1]
        for column in result.columns
    ]
    return result


def normalize_yahoo_frame(frame, symbol):
    """Normalize one Yahoo Finance OHLCV response to daily stock data."""
    result = _flatten_yahoo_columns(frame)
    required = {"Close", "Volume"}
    missing = required.difference(result.columns)

    normalized = pd.DataFrame(
        {
            "date": pd.to_datetime(result.index).date,
            "symbol": symbol,
            "close": pd.to_numeric(result["Close"], errors="coerce").to_numpy(),
            "volume": pd.to_numeric(result["Volume"], errors="coerce").to_numpy(),
        }
    )
    return normalized.sort_values("date").reset_index(drop=True)


def normalize_yahoo_close_frame(frame, column_name):
    """Normalize one Yahoo Finance response containing only a close series."""
    result = _flatten_yahoo_columns(frame)

    return pd.DataFrame(
        {
            "date": pd.to_datetime(result.index).date,
            column_name: pd.to_numeric(result["Close"], errors="coerce").to_numpy(),
        }
    ).sort_values("date").reset_index(drop=True)


def fetch_yahoo_daily(symbol, start_date=START_DATE, end_date=END_DATE):
    """Download daily close and volume for one equity from Yahoo Finance."""
    raw = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    return normalize_yahoo_frame(raw, symbol)


def fetch_yahoo_close(symbol, output_name, start_date=START_DATE, end_date=END_DATE):
    """Download one daily close series from Yahoo Finance."""
    raw = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    return normalize_yahoo_close_frame(raw, output_name)


def normalize_fred_frame(raw):
    """Normalize a raw FRED DTB3 response to date and US3M columns."""
    date_column = "DATE" if "DATE" in raw.columns else "observation_date"

    result = raw.rename(columns={date_column: "date", "DTB3": "us3m"})
    result["date"] = pd.to_datetime(result["date"]).dt.date
    result["us3m"] = pd.to_numeric(result["us3m"], errors="coerce")
    return result[["date", "us3m"]].dropna(subset=["us3m"]).reset_index(drop=True)


def fetch_fred_dtb3(
    url=FRED_DTB3_URL,
    start_date=START_DATE,
    end_date=END_DATE,
):
    """Download the FRED DTB3 3-month Treasury bill rate."""
    result = normalize_fred_frame(pd.read_csv(url))
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    return result[(result["date"] >= start) & (result["date"] < end)].reset_index(drop=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stock_daily = pd.concat(
        [fetch_yahoo_daily(symbol) for symbol in symbols],
        ignore_index=True,
    )
    stock_daily.to_parquet(OUTPUT_DIR / "stock_daily.parquet", index=False)

    hsi = fetch_yahoo_close("^HSI", "hsi_close")
    hsi.to_parquet(OUTPUT_DIR / "hsi_daily.parquet", index=False)

    vix = fetch_yahoo_close("^VIX", "vix")
    vix.to_parquet(OUTPUT_DIR / "vix_daily.parquet", index=False)

    us3m = fetch_fred_dtb3()
    us3m.to_parquet(OUTPUT_DIR / "us3m_daily.parquet", index=False)


if __name__ == "__main__":
    main()

