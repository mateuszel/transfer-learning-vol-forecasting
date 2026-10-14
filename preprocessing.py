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

def summarize_missing_sessions(returns):
    """
    Inputs:
        returns: dict[str, pd.Series] - dictionary of per asset returns, outputs of compute_returns(...)
    Outputs:
        pd.DataFrame - each row corresponds to one trading session of one symbol with columns:
        symbol, date, expected_returns, observed_returns, missing_returns, leading_missing_returns, longest_missing_run,
        first_missing, last_missing
        only sessions with at least one missing return are returned
    """
    summary = []
    for symbol in returns:
        df = returns[symbol].copy()
        # start by computing expected, observed and missing returns
        # since our series already has the full session 5 minute grid 
        # we can just count NaN's
        grouped = df.groupby(df.index.date).agg(expected_returns="size", observed_returns="count")
        grouped["missing_returns"] = grouped["expected_returns"] - grouped["observed_returns"]
        
        # filter so we only have sessions with at least one missing return
        grouped = grouped[grouped["missing_returns"] > 0]

        # add symbol as a column
        grouped["symbol"] = symbol
        res = grouped[["symbol", "expected_returns", "observed_returns", "missing_returns"]]

        # find first and last missing timestamp for every session with at least one NaN
        missing_times = df[df.isna()].index.to_series().reset_index(drop=True)
        grouped = missing_times.groupby(missing_times.dt.date).agg(min_nan="min", max_nan="max")
        grouped["min_nan"] = grouped["min_nan"].dt.time
        grouped["max_nan"] = grouped["max_nan"].dt.time

        # add these to the result
        res = res.join(grouped, how="inner")

        # now find the leading missing returns 
        leading_nans = df.isna().groupby(df.index.date).agg(leading_missing=lambda x: x.cumprod().sum())
        res = res.join(leading_nans, how="inner")

        # last we find the longest missing run for every session
        longest_runs = df.groupby(df.index.date).agg(
            lambda x: x.isna().groupby(x.notna().cumsum()).sum().max()
        ).rename("longest_nan_run")
        res = res.join(longest_runs, how="inner")

        summary.append(res)
    return pd.concat(summary)

def compute_daily_rv(returns):
    """
    Inputs: 
        returns: dict[str, pd.Series] - dictionary containing log returns per asset, outputs of compute_returns(...)
    Outputs:
        dict[str, pd.Series] - for each asset a series of daily RV (or its approximation) indexed by date
    """
    res = {}
    for symbol in returns:
        df = returns[symbol].copy()
        date_grouped = df.groupby(df.index.date)
        # first calculate the expected and observed number of returns per session
        returns_diagnosis = date_grouped.agg(expected_returns="size", observed_returns="count")

        # now compute scaled realized variance or NaN if no returns for a session
        rv = (df**2).groupby(df.index.date).sum(min_count=1) # if no observations it gives NaN
        scaled_rv = rv * returns_diagnosis["expected_returns"] / returns_diagnosis["observed_returns"].replace(0, np.nan)
        scaled_rv = scaled_rv.rename("rv").rename_axis("date")

        res[symbol] = scaled_rv

    return res

def build_basic_dataset(rvs):
    """
    Inputs:
        rvs: dict[str, pd.Series] - output of compute_daily_rv(...)
    Outputs: 
        pd.DataFrame - the basic dataset with columns - RV_t, RV_t,w RV_t,m, symbol, RV_t+1, 
        date - the date t
    """
    dataset = []
    for symbol in rvs:
        # compute rolling features
        rvt = rvs[symbol].copy()
        rvw = rvt.rolling(5).mean()
        rvm = rvt.rolling(22).mean()
        rv_next = rvt.shift(-1)

        # combine everything together
        df = pd.concat([rvt, rvw, rvm, rv_next], axis=1)
        df = df.reset_index(names="t")
        df["symbol"] = symbol
        df.columns = ["t", "rv_t", "rv_w", "rv_m", "rv_next", "symbol"]
        df = df[["t", "symbol", "rv_t", "rv_w", "rv_m", "rv_next"]]

        dataset.append(df)
        
    return pd.concat(dataset).reset_index(drop=True)

def load_extended_data(data_dir='raw/extended'):
    """
    Inputs:
        data_dir: string - relative path to the folder with extended data raw files
    Outputs:
        dict[str, pd.DataFrame] - dictionary of dataframes containing the extended data
    """
    data_dir = Path(data_dir)

    stock_daily = pd.read_parquet(data_dir / "stock_daily.parquet")
    hsi_daily = pd.read_parquet(data_dir / "hsi_daily.parquet")
    vix_daily = pd.read_parquet(data_dir / "vix_daily.parquet")
    us3m_daily = pd.read_parquet(data_dir / "us3m_daily.parquet")

    frames = {
        "stock_daily": stock_daily,
        "hsi_daily": hsi_daily,
        "vix_daily": vix_daily,
        "us3m_daily": us3m_daily,
    }

    for frame in frames.values():
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()

    stock_daily = stock_daily.sort_values(["symbol", "date"])
    hsi_daily = hsi_daily.sort_values("date")
    vix_daily = vix_daily.sort_values("date")
    us3m_daily = us3m_daily.sort_values("date")

    return {
        "stock_daily": stock_daily,
        "hsi_daily": hsi_daily,
        "vix_daily": vix_daily,
        "us3m_daily": us3m_daily,
    }

def summarize_stock_daily_coverage(stock_daily, ext_session_dates, ipo_dates):
    """
    Inputs:
        stock_daily: pd.DataFrame - daily stock data from the extended dataset
        ext_session_dates: pd.Series - expected NYSE session dates
        ipo_dates: dict[str, str] - IPO date for every target asset
    Outputs:
        pd.DataFrame - one row per asset with daily data coverage information
    """
    expected_dates = pd.DatetimeIndex(
        pd.to_datetime(ext_session_dates)
    ).normalize().unique()
    reports = []

    for symbol, df in stock_daily.groupby("symbol"):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        first_date = pd.Timestamp(ipo_dates[symbol]) if symbol in ipo_dates else expected_dates.min()
        active_dates = expected_dates[expected_dates >= first_date]
        active_data = df[df["date"].isin(active_dates)]

        observed_dates = pd.DatetimeIndex(active_data["date"].unique())
        missing_sessions = len(active_dates.difference(observed_dates))
        missing_close = active_data["close"].isna().sum()
        missing_volume = active_data["volume"].isna().sum()

        reports.append({
            "symbol": symbol,
            "role": "target" if symbol in ipo_dates else "source",
            "expected_sessions": len(active_dates),
            "observed_sessions": len(observed_dates),
            "missing_sessions": missing_sessions,
            "missing_close": missing_close,
            "missing_volume": missing_volume,
            "status": "OK" if (missing_sessions + missing_close + missing_volume) == 0 else "CHECK",
        })

    return pd.DataFrame(reports).sort_values("symbol").reset_index(drop=True)

def summarize_market_daily_coverage(ext_datasets, ext_session_dates, target_start):
    """
    Inputs:
        ext_datasets: dict[str, pd.DataFrame] - daily market datasets
        ext_session_dates: pd.Series - expected NYSE session dates
        target_start: date-like - first date of the target period
    Outputs:
        pd.DataFrame - one row per market dataset with missing date information
    """
    expected_dates = pd.DatetimeIndex(
        pd.to_datetime(ext_session_dates)
    ).normalize().unique()
    target_start = pd.Timestamp(target_start).normalize()
    source_dates = expected_dates[expected_dates < target_start]
    target_dates = expected_dates[expected_dates >= target_start]

    market_columns = {
        "hsi_daily": ("HSI", "hsi_close"),
        "vix_daily": ("VIX", "vix",),
        "us3m_daily": ("US3M", "us3m"),
    }
    reports = []

    for dataset_name, (source, value_column) in market_columns.items():
        df = ext_datasets[dataset_name].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        observed_dates = pd.DatetimeIndex(df["date"].drop_duplicates())

        source_missing = len(source_dates.difference(observed_dates))
        target_missing = len(target_dates.difference(observed_dates))
        null_values = df[value_column].isna().sum()

        reports.append({
            "source": source,
            "missing_source_period": source_missing,
            "missing_target_period": target_missing,
            "null_values": null_values,
        })

    return pd.DataFrame(reports)

def align_market_data(hsi_daily, vix_daily, us3m_daily, ext_session_dates):
    """
    Inputs:
        hsi_daily: pd.DataFrame - contains HSI values for available dates
        vix_daily: pd.DataFrame - contains VIX values for available dates
        us3m_daily: pd.DataFrame - contains US3M values for available dates
    Outputs:
        pd.DataFrame - data frame that has for each NYSE session filled and aligned values of the input columns
    """

    market = pd.DataFrame({"date": pd.to_datetime(ext_session_dates)})

    # first merge hsi_daily and forward fill
    market = market.merge(hsi_daily, on="date", how="left")
    market["hsi_close"] = market["hsi_close"].ffill()

    # now merge us3m daily and ffill
    market = market.merge(us3m_daily, on="date", how="left")
    market["us3m"] = market["us3m"].ffill()

    # and lastly merge VIX (it had no missing values so no ffill)
    market = market.merge(vix_daily, on="date", how="left")
    
    return market

def build_extended_features(stock_daily, market):
    """
    Inputs:
        stock_daily: pd.DataFrame - contains daily close and volume stock data, output of load_extended_data(...)
        market: pd.DataFrame - output of align_market_data(...)
    Outputs:
        pd.DataFrame - contains all extended features for the HAR model aligned to the NYSE trading session dates
    """
    stock_daily = stock_daily.copy()
    market = market.copy()

    stock_daily = stock_daily.sort_values(["symbol", "date"])
    market = market.sort_values("date")

    # first compute stock features
    stock_daily["log_dollar_volume"] = np.log(stock_daily["close"] * stock_daily["volume"])
    stock_daily["log_close"] = np.log(stock_daily["close"])

    # MOM
    stock_daily["MOM"] = (stock_daily.groupby("symbol")["log_close"].diff(5))

    # DVC
    stock_daily["DVC"] = (stock_daily.groupby("symbol")["log_dollar_volume"].diff(1))

    # market features
    market["hsi_sqr"] = (np.log(market["hsi_close"]).diff(1) ** 2)

    res = stock_daily.merge(
        market[["date", "hsi_sqr", "vix", "us3m"]],
        on="date",
        how="left",
        validate="many_to_one",
    )

    res = res.rename(columns={"date": "t"})
    res = res[["t", "symbol", "hsi_sqr", "vix", "us3m", "MOM", "DVC"]]

    res["t"] = res["t"].dt.date

    return res.sort_values(by=["symbol", "t"], ascending=[True, True]).reset_index(drop=True)