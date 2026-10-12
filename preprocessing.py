import numpy as np
import pandas as pd
import exchange_calendars as xcals
from pathlib import Path

def load_bars(symbols, data_dir):
    """
    Inputs:
        symbols: list of symbol strings
        data_dir: path to the directory containing {SYMBOL}.parquet
    Output:
        dict[str, pd.DataFrame] - dictionary of DataFrames containing available 1 minute bars for each asset
        Columns: symbol, timestamp, open, close
        timestamp is in UTC time
    """
    datasets = {}
    for symbol in symbols:
        df = pd.read_parquet(Path(data_dir) / f"{symbol}.parquet")
        df = df[["symbol", "timestamp", "open", "close"]]
        datasets[symbol] = df

    return datasets

def build_trading_sessions(start_date, end_date, calendar_name="XNYS", timezone="America/New_York"):
    """
    Inputs:
        start_date: date of the first session
        end_date: date of the last session (inclusive)
        calendar_name: which trading calendar to use
        timezone: when adjusting timestamps, which timezone to use
    Outputs:
        pd.DataFrame with columns:
            open: session open datetime in the specified timezone
            close: session close datetime in the specified timezone
            minutes: session length in minutes
            date: day date of that session
    """

    # get the sessions
    cal = xcals.get_calendar(calendar_name)
    trading_sessions = cal.schedule.loc[start_date:end_date]
    trading_sessions = trading_sessions[["open", "close"]].copy().reset_index(drop=True)

    # convert timezones
    trading_sessions["open"] = trading_sessions["open"].dt.tz_convert(timezone)
    trading_sessions["close"] = trading_sessions["close"].dt.tz_convert(timezone)

    # compute session length in minutes
    trading_sessions["minutes"] = (trading_sessions["close"] - trading_sessions["open"]).dt.total_seconds()/60

    # add date column
    trading_sessions["date"] = trading_sessions["open"].dt.date

    return trading_sessions

def prepare_bars(datasets, trading_sessions, timezone="America/New_York"):
    """
    Inputs:
        datasets: dict[str, pd.DataFrame] - the output of load_bars(...) 
        trading_sessions: pd.DataFrame - output of build_trading_sessions(...)
        timezone: str - timezone to convert to
    Outputs:
        dict[str, pd.DataFrame] - this function first converts all bar timestamps for every assset into 
        chosen timezone, then matches the session date to filter out bars that are outside of the
        session for that day. So this filters out any overnight bars etc. keeps only the ones
        inside of sessions and adjusts timezones.
    """
    prepped_datasets = {}
    for symbol in datasets:
        df = datasets[symbol].copy()

        # adjust the timezone of the bar timestamps and create a date column
        df["timestamp"] = df["timestamp"].dt.tz_convert(timezone)
        df["date"] = df["timestamp"].dt.date

        # match bars to sessions, left join because we want to keep all bars, even if they dont match any sessions
        # but it's unlikely that a bar wouldn't match any session
        df = df.merge(
            trading_sessions[["date", "open", "close"]].rename(
                columns={
                    "open": "session_open",
                    "close": "session_close",
                }
            ),
            on="date",
            how="left",
        )

        # filter out bars outside of sessions, last bar has a timestamp of session_close - 1m
        df = df[(df["timestamp"] >= df["session_open"]) & (df["timestamp"] < df["session_close"])]

        # clean up columns and index
        df = df.drop(columns=["session_open", "session_close"])
        df = df.reset_index(drop=True)
        prepped_datasets[symbol] = df
    return prepped_datasets

def summarize_raw_coverage(datasets, trading_sessions):
    """
    Inputs: 
        datasets: dict[str, pd.DataFrame] - raw bar dataset with converted timestamps - output of prepare_bars(...)
        trading_sessions: pd.DataFrame - contains information about sessions, output of build_trading_sessions(...)
    Outputs:

    """
    asset_reports = []
    for symbol in datasets:
        df = datasets[symbol].copy()

        # duplicated bars in this asset
        duplicated_bars = df["timestamp"].duplicated().sum()

        # find the first and last observed date
        first_date = df["timestamp"].min().date()
        last_date = df["timestamp"].max().date()

        # find sessions for this asset (anything between first_bar and last_bar)
        sessions = trading_sessions[
            (trading_sessions["date"] >= first_date) & (trading_sessions["date"] <= last_date)
        ]

        # number of nonempty sessions this asset should have
        expected_sessions = len(sessions)

        # number of non empty sessions present in the data for the asset (containing at least 1 bar)
        observed_sessions = df["date"].nunique()

        # missing sessions
        missing_sessions = expected_sessions - observed_sessions

        # incomplete sessions (less bars than it should have)
        day_grouped = df.groupby("date").size().rename("count")
        merged = sessions.merge(day_grouped, on="date", how="inner") # inner because we dont want empty sessions
        incomplete_sessions = (merged["count"] < merged["minutes"]).sum()

        # expected number of bars (sum of bars for all sessions)
        expected_bars = sessions["minutes"].sum()

        # observed bars
        observed_bars = df["timestamp"].nunique()

        asset_reports.append({
            "symbol": symbol,
            "first_date": first_date,
            "last_date": last_date,
            "expected_sessions": expected_sessions,
            "observed_sessions": observed_sessions,
            "missing_sessions": missing_sessions,
            "incomplete_sessions": incomplete_sessions, 
            "expected_bars": expected_bars,
            "observed_bars": observed_bars,
            "duplicated_bars": duplicated_bars,
            "coverage_pct": observed_bars / expected_bars
        })
    return pd.DataFrame(asset_reports)

def compute_returns(bars, trading_sessions):
    """
    Inputs:
        bars: pd.DataFrame - one symbol's bar data, either filled or raw. With adjusted timezone like prepare_bars(...) output
        trading_sessions: pd.DataFrame - output of build_trading_sessions(...)
    Outputs:
        pd.Series that contains 5 minute log returns for this asset over the entire 5min grid including NaN's.
        Index is interval end and value is this interval's log return. 
        When computing a return we compare current interval's close price to previous interval's
        close price. If the interval is the first interval of the session use open price of the session
        as a reference instead. 
    """
    df = bars.copy()
    df = df.set_index("timestamp")

    # needed to get the appropriate trading sessions
    first_date = df.index.date.min()
    last_date = df.index.date.max()

    # tailor the sessions to the asset
    sessions = trading_sessions[
        (trading_sessions["date"] >= first_date) & (trading_sessions["date"] <= last_date)
    ]

    # now create the 5 - min grid
    # first index is the session open
    # next ones are just close times of following 5 min intervals
    endpoints = pd.DatetimeIndex([])
    for _, session in sessions.iterrows():
        grid = pd.date_range(
            start=session["open"],
            end=session["close"],
            freq="5min"
        )
        endpoints = endpoints.append(grid)

    minute_closes = df["close"].copy() # all bar close prices
    # their timestamps represent time when the bar opened
    # we want to know when they closed since we index intervals by the timestamp they ended
    minute_closes.index += pd.Timedelta(minutes=1)

    prices_5m = minute_closes.reindex(endpoints) # prices indexed by 5 min interval

    # we need to fill prices at the start of session wherever possible
    # thats because after shifting minute_closes index we can never have a bar with timestamp 9:30

    session_opens = pd.DatetimeIndex(sessions["open"])
    prices_5m.loc[session_opens] = df["open"].reindex(session_opens).to_numpy()

    # now we can compute the 5 minute log returns
    # we group because we dont want to compute returns across two sessions, only inside
    returns = np.log(prices_5m).groupby(prices_5m.index.date).diff()

    # each session starts with NaN (as it should), we dont return that
    returns.index.name = "interval_end"
    return returns[~returns.index.isin(session_opens)]

def summarize_missing_returns(returns):
    """
    Inputs:
        dict[str, pd.Series] - dictionary of returns per asset, dictionary of compute_returns() outputs
    Outputs:
        pd.Series - contains number of NaN's in returns per symbol
    """
    missing = {}

    for symbol, df in returns.items():
        missing[symbol] = df.isna().sum().sum()

    return pd.Series(missing, name="missing_returns").rename_axis("Symbol")
        
def count_valid_rolling_windows(returns, window):
    """
    Inputs:
        returns: dict[str, pd.Series] - dictionary of 5 minute log returns for different assets indexed by interval_end
        window: int - length of the rolling window (number of trading sessions)
    Outputs:
        pd.Series - number of windows of specified length without NaN's for every present asset
    """
    valid = {}

    for symbol in returns:
        df = returns[symbol].copy()

        # this gives us a pd.Series that says whether given session doesn't have any NaN's
        valid_sessions = df.notna().groupby(df.index.date).all()

        # now we count the number of valid rolling windows
        r_sum = valid_sessions.rolling(window).sum()
        count = (r_sum == window).sum()

        valid[symbol] = count

    return pd.Series(valid, name=f"valid_{window}windows").rename_axis("Symbol")

def calculate_endpoint_lags(bars, trading_sessions):
    """
    Inputs:
        bars: pd.DataFrame - bars of one asset prepared by prepare_bars(...)
        trading_sessions: pd.DataFrame - output of build_trading_sessions(...)
    Outputs:
        pd.Series - index - timestamp of missing 5 min return (eoI), values - lag in minutes to the previous same-session bar
    """
    returns = compute_returns(bars, trading_sessions) # this contains full grid
    df = bars.set_index("timestamp")

    # now we find last bars timestamp for each missing return
    missing_return_timestamps = returns[returns.isna()].index
    missing_bar_timestamps = missing_return_timestamps - pd.Timedelta(minutes=1)

    # check whether the closing bars corresponding to missing returns actually dont have close
    missing_bar_mask = df["close"].reindex(missing_bar_timestamps).isna().to_numpy()

    # only leave cases where the last bar of interval is missing
    missing_return_timestamps = missing_return_timestamps[missing_bar_mask]
    missing_bar_timestamps = missing_bar_timestamps[missing_bar_mask]

    # find which bars have close available
    available_bar_timestamps = df.index[df["close"].notna()]

    # now we can find for each missing bar timestamp the closest previous one
    previous_bar_timestamps = available_bar_timestamps.to_series().reindex(missing_bar_timestamps, method="ffill")

    # we now have to check if we didnt use different session when forward filling
    same_session_mask = previous_bar_timestamps.notna() & (previous_bar_timestamps.dt.date == missing_bar_timestamps.date)

    # now we find the lags and swap previous session fills for NaN's to make it more readable
    lags = ((missing_bar_timestamps - previous_bar_timestamps) / pd.Timedelta(minutes=1)).where(same_session_mask)

    lags.index = missing_return_timestamps
    lags.name = "lag_minutes"

    return lags

def fill_short_gaps(bars, trading_sessions, excluded_dates, limit=7):
    """
    Inputs:
        bars: pd.DataFrame - bars of one asset prepared by prepare_bars(...)
        trading_sessions: pd.DataFrame - output of build_trading_sessions(...)
        excluded_dates: dict - dates excluded from filling for the given asset
        limit: int - maximal lag that can be filled with this operation
    Outputs:
        pd.DataFrame - cotains bars of the asset after forward filling close that are withing the lag distance
        and omitting excluded dates
        n_filled: int - number of returns forward filled
    """
    df = bars.copy()
    symbol = df["symbol"].iloc[0]

    # first create expected 1 minute grid for bars
    first_bar_date = bars["timestamp"].min().date()
    last_bar_date = bars["timestamp"].max().date()
    sessions = trading_sessions[(trading_sessions["date"] >= first_bar_date) & (trading_sessions["date"] <= last_bar_date)]

    minute_grid = pd.DatetimeIndex([])
    for _, session in sessions.iterrows():
        grid = pd.date_range(
            start=session["open"],
            end=session["close"],
            freq="1min",
            inclusive="left" # drops right end explicitly
        )
        minute_grid = minute_grid.append(grid)

    # now we reindex the df
    df = df.set_index("timestamp")
    df = df.reindex(minute_grid)
    df["date"] = df.index.date # after reindexing some of the dates are NaN's
    df["symbol"] = symbol

    # create a mask for the excluded dates
    mask = df["date"].isin(excluded_dates)

    # now we forward fill using a dummy column
    # no forward filling across sessions
    df["close_filled"] = df["close"].copy()
    df.loc[~mask, "close_filled"] = df.loc[~mask, "close"].groupby(df.loc[~mask, "date"]).ffill(limit=limit)
    n_filled = (df["close"].isna() & df["close_filled"].notna()).sum()
    df["close"] = df["close_filled"]
    df = df.drop(columns=["close_filled"])

    return df.rename_axis("timestamp").reset_index(), n_filled