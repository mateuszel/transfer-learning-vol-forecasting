import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Preprocessing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this notebook we prepare the data for further experiments. The data that we have is raw data obtained using Alpaca. When fetching, we tried to get as much data as possible for the given assets, without getting into the COVID-19 range, as this period was very unpredictable. It might be interesting to include it later, to check how our methods work across market regimes.
    """)
    return


@app.cell
def imports():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import exchange_calendars as xcals

    return mo, np, pd, xcals


@app.cell
def universe_definition():
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
    return (all_symbols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alpaca returns more columns that we will actually need, thus we first load the datasets from corresponding Parquet files and drop the unneccessary columns.
    """)
    return


@app.cell
def dataset_loading(all_symbols, pd):
    datasets = {}

    for _symbol in all_symbols:
        _df = pd.read_parquet(f"data/raw/{_symbol}.parquet")
        _df = _df[["symbol", "timestamp", "open", "close", "volume"]]
        datasets[_symbol] = _df
    return (datasets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To make proper validation of the dataset easier we will now convert the timestamps from UTC to local NYC time.

    This will also make filtering of the dataset simpler, since similarly to Teller we will not be considering overnight returns.

    Following that we filter out any bars that are not inside the trading session. Any bar starting at 16:00 is also dropped. Notice that the timestamp of a given bar means the time it has opened.
    """)
    return


@app.cell
def trading_calendar(xcals):
    xnys = xcals.get_calendar("XNYS")
    trading_sessions = xnys.schedule.loc["2016-01-01":"2020-02-22"]
    trading_sessions = trading_sessions[["open", "close"]].copy().reset_index(drop=True)

    # convert to NYC time
    trading_sessions["open"] = trading_sessions["open"].dt.tz_convert("America/New_York")
    trading_sessions["close"] = trading_sessions["close"].dt.tz_convert("America/New_York")

    # calculate number of bars 
    trading_sessions["minutes"] = (trading_sessions["close"]-trading_sessions["open"]).dt.total_seconds()/60

    # format for easier access
    trading_sessions["date"] = trading_sessions["open"].dt.date
    trading_sessions
    return (trading_sessions,)


@app.cell
def time_conversion(all_symbols, datasets, trading_sessions):
    for _symbol in all_symbols:
        _df = datasets[_symbol].copy()
        _df["timestamp"] = _df["timestamp"].dt.tz_convert("America/New_York")
        _df["date"] = _df["timestamp"].dt.date

        _df = _df.merge(
            trading_sessions[["date", "open", "close"]].rename(
                columns={
                    "open": "session_open", 
                    "close": "session_close"}), 
            on="date", 
            how="left")

        _df = _df[(_df["timestamp"] >= _df["session_open"]) & (_df["timestamp"] < _df["session_close"])]

        _df = _df.drop(columns=["session_open", "session_close"])
        _df = _df.reset_index(drop=True)

        datasets[_symbol] = _df
    return


@app.cell
def _(datasets):
    datasets["MSFT"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After this initial preprocessing we can create data quality reports for the datasets.

    In particular we want to inspect coverage rate for each asset and find out the exact time range of the available data. The missing bars will be aggregated over trading sessions and 5 minute intervals.
    Afterwards, we will try to find out how many 5 minute returns we will be able to actually compute without imputing any data. This amount may be different across assets.
    """)
    return


@app.cell
def _(pd, trading_sessions):
    def summarise_asset(symbol, data):
        df = data[symbol].copy()

        # data range
        first_bar = df["timestamp"].min()
        last_bar = df["timestamp"].max()

        sessions = trading_sessions[(trading_sessions["date"] >= first_bar.date()) & (trading_sessions["date"] <= last_bar.date())]

        # duplicates
        duplicate_timestamps = df.duplicated(subset=["timestamp"]).sum()

        # session data
        actual_sessions = df["date"].nunique()

        expected_sessions = len(sessions)

        missing_sessions = expected_sessions - actual_sessions

        # incomplete_session = exists, but not enough bars
        day_grouped = df.groupby("date").size().rename("count")
        merged = sessions.merge(day_grouped, on="date", how="inner")

        incomplete_sessions = (merged["count"] < merged["minutes"]).sum()

        # 1m data

        expected_1m = int(sessions["minutes"].sum())
        actual_1m = df["timestamp"].nunique() # only unique bars
        coverage_1m_pct = actual_1m / expected_1m

        # 5m data

        expected_5m = expected_1m // 5 # this is an integer

        merged = sessions.merge(df, on="date", how="inner")
        session_open_present = (merged["timestamp"] == merged["open_x"]).groupby(merged["date"]).any()
        available_startpoints = session_open_present.sum()
        missing_startpoints = expected_sessions - available_startpoints
        startpoint_coverage_pct = available_startpoints / expected_sessions
        merged["bucket"] = (merged["timestamp"] - merged["open_x"]) // pd.Timedelta(minutes=5) + 1
        merged["endpoint"] = merged["open_x"] + merged["bucket"] * pd.Timedelta(minutes=5)
        merged["endpoint_flag"] = (merged["timestamp"] + pd.Timedelta(minutes=1) == merged["endpoint"])

        session_bucket_grid = [
            (session["date"], bucket) 
            for _, session in sessions.iterrows() 
            for bucket in range(1, int(session["minutes"]/5)+1)
        ]

        grouped = merged.groupby(["date", "bucket"]).agg(
            bars_count=("timestamp", "nunique"),
            endpoint_present=("endpoint_flag", "max")
        )

        grouped = grouped.reindex(pd.MultiIndex.from_tuples(session_bucket_grid, names=["date", "bucket"]), fill_value=False)
        grouped["start_present"] = grouped.groupby(level="date")["endpoint_present"].shift(1) # shift only within session not across them

        # now we check for the first bucket of each session
        first_bucket = grouped.index.get_level_values("bucket") == 1
        first_dates = grouped.index.get_level_values("date")[first_bucket]

        grouped.loc[first_bucket, "start_present"] = first_dates.map(session_open_present).fillna(False).to_numpy()

        missing_start = int((
            ~grouped["start_present"] & grouped["endpoint_present"]
        ).sum())

        missing_end = int((
            grouped["start_present"] & ~grouped["endpoint_present"]
        ).sum())

        missing_both = int((
            ~grouped["start_present"] & ~grouped["endpoint_present"]
        ).sum())

        missing_returns_5m = missing_start + missing_end + missing_both
        returns_coverage_pct = 1 - missing_returns_5m / expected_5m

        actual_5m = int((grouped["bars_count"] > 0).sum())
        complete_5m = int((grouped["bars_count"] == 5).sum())
        incomplete_5m = int(expected_5m - complete_5m)
        missing_endpoints_5m = int(expected_5m - grouped["endpoint_present"].sum())
        endpoint_coverage_pct = 1 - missing_endpoints_5m / expected_5m # also counts intervals that dont have any bars

        return {
            "symbol": symbol,
            "first_bar": first_bar,
            "last_bar": last_bar,
            "expected_sessions": expected_sessions,
            "actual_sessions": actual_sessions,
            "missing_sessions": missing_sessions,
            "incomplete_sessions": incomplete_sessions,
            "missing_startpoints": missing_startpoints,
            "startpoint_coverage_pct": startpoint_coverage_pct,
            "duplicate_timestamps": duplicate_timestamps,
            "expected_1m": expected_1m,
            "actual_1m": actual_1m,
            "coverage_1m_pct": coverage_1m_pct,
            "expected_5m": expected_5m,
            "actual_5m": actual_5m,
            "complete_5m": complete_5m,
            "incomplete_5m": incomplete_5m,
            "missing_endpoints_5m": missing_endpoints_5m,
            "endpoint_coverage_pct": endpoint_coverage_pct,
            "missing_start": missing_start,
            "missing_end": missing_end,
            "missing_both": missing_both,
            "missing_returns_5m": missing_returns_5m,
            "returns_coverage_pct": returns_coverage_pct
        }

    return (summarise_asset,)


@app.cell
def _(all_symbols, datasets, pd, summarise_asset):
    _quality_report = pd.DataFrame(
        [summarise_asset(symbol, datasets) for symbol in all_symbols]
    )

    _quality_report = _quality_report.set_index("symbol")
    _quality_report.drop(columns=["missing_end", "missing_both", "missing_returns_5m", "returns_coverage_pct", "missing_start"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At first glance our data looks solid and seems to have good coverage for almost all assets. The most worrying are our target datasets, who seem to have proportionally the most missing data. This is problematic, because data related to those assets is already quite limited.

    Now, we will inspect those missing candles in detail across assets. We want to find out how many 5-minute returns are impossible to compute. Only after that we will make decisions regarding imputing or dropping parts of the dataset.
    """)
    return


@app.cell
def _(np, pd, trading_sessions):
    def compute_5m_prices(symbol, data):
        df = data[symbol].copy()
        df = df.set_index("timestamp")
        first_bar = df.index.min()
        last_bar = df.index.max()

        sessions = trading_sessions[
            (trading_sessions["date"] >= first_bar.date()) & 
            (trading_sessions["date"] <= last_bar.date())]

        # first timestamp here is price at the beggining of the session
        # following ones are log returns of intervals ending at the timestamp
        expected_endpoints = pd.DatetimeIndex([])
        for _, session in sessions.iterrows():
            grid = pd.date_range(
                start=session["open"],
                end=session["close"],
                freq="5min"
            )
            expected_endpoints = expected_endpoints.append(grid)

        minute_closes = df["close"].copy()
        minute_closes.index += pd.Timedelta(minutes=1) # close prices indexed by the timestamp they close

        prices = minute_closes.reindex(expected_endpoints)
        return prices

    def compute_returns(symbol, data):
        df = data[symbol].copy()
        df = df.set_index("timestamp")
        first_bar = df.index.min()
        last_bar = df.index.max()

        sessions = trading_sessions[
            (trading_sessions["date"] >= first_bar.date()) & 
            (trading_sessions["date"] <= last_bar.date())]

        prices = compute_5m_prices(symbol, data)

        # all left to do is to replace some of the NaN's with corresponding open prices
        for _, session in sessions.iterrows():
            session_open = session["open"]
            if session_open in df.index:
                prices.loc[session_open] = df.loc[session_open, "open"]

        # now we are ready to calculate the returns
        returns = np.log(prices).groupby(prices.index.date).diff()
        session_opens = pd.DatetimeIndex(sessions["open"])
        returns = returns[~returns.index.isin(session_opens)] # we drop the Nulls at the beggining of each session

        return returns

    return compute_5m_prices, compute_returns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now compute all the possible log returns over 5 minute intervals. Below we can see how many missing values there is per asset per trading session. As expected before the most troubling ones are the missing returns of target assets.
    """)
    return


@app.cell
def _(all_symbols, compute_returns, datasets):
    returns = {symbol: compute_returns(symbol, datasets) for symbol in all_symbols}
    return (returns,)


@app.cell
def _(returns):
    returns["MSFT"]
    return


@app.cell
def _(pd, returns):
    _quality_report = []

    for symbol, r in returns.items():
        missing = r.isna().groupby(r.index.date).sum()

        _quality_report.append({
            "symbol": symbol,
            "complete": (missing == 0).sum(),
            "incomplete": (missing > 0).sum(),
            "missing_1or2": ((missing > 0) & (missing <= 2)).sum(),
            "missing_>_2": (missing > 2).sum()
        })

    pd.DataFrame(_quality_report).set_index("symbol")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For our HAR model we want to use 22 day rolling volatility as feature. Let's see how many of those windows we lose due to missing values.
    """)
    return


@app.cell
def _(returns):
    def missing_22(symbol):
        r = returns[symbol]
        daily_missing = r.isna().groupby(r.index.date).sum().to_frame("missing")
        daily_missing["complete"] = (daily_missing["missing"] == 0)
        daily_missing["valid_22"] = (daily_missing["complete"].rolling(22, min_periods=22).sum() == 22)
        return daily_missing["valid_22"].sum()

    return (missing_22,)


@app.cell
def _(all_symbols, missing_22):
    valid_22 = {_symbol: missing_22(_symbol) for _symbol in all_symbols}
    valid_22
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we can see the returns for some of the assets are completely unusable. In particular, for "CHWY" we wouldn't be able to construct a single 22 day rolling volatility value. For a lot of the assets, we have a decent coverage. Regardless of that, we will now precisely inspect source of those missing 5 minute returns and try to compute them again in a less strict way, which unfortunately may decrease the reliability of our data, but it's not possible to proceed any further without doing that.

    We start by inspecting for each asset the source of missing 5-minute returns. Since for computing the returns within an interval we don't need its insides, we only have to check how many NaN's are caused by missing start of the interval and how many are caused by missing the end of the interval.
    """)
    return


@app.cell
def _(all_symbols, datasets, pd, summarise_asset):
    _quality_report = pd.DataFrame(
        [summarise_asset(symbol, datasets) for symbol in all_symbols]
    )

    _quality_report = _quality_report.set_index("symbol")
    _quality_report[["missing_start", "missing_end", "missing_both", "missing_returns_5m", "returns_coverage_pct"]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One can notice that because of the method of computing the returns, imputing some of the missing_end's would have a great effect on missing_returns_5m. It might be reasonable to fill some of the missing prices using the closest previous price. However, we shouldn't just use any closest available price from past, as this may lead to huge errors if we sample from very far away. For that reason, we first inspect asset by asset how many of the missing_end values can be filled by going back in time by 1, 2 or 3 minutes and how many don't have any previous price available.
    """)
    return


@app.cell
def _(compute_5m_prices, datasets, np, pd, trading_sessions):
    def endpoint_lag(symbol, data):
        # for each missing endpoint we want to check how far away in the past is the closes available one
        prices = compute_5m_prices(symbol, data)
        sessions = trading_sessions[trading_sessions["date"].isin(prices.index.date)]
        session_opens = pd.DatetimeIndex(sessions["open"])
        # we dont want the NaN's associated with the first interval of a session
        missing_endpoints = prices[(prices.isna()) & ~prices.index.isin(session_opens)].index
        df = datasets[symbol].set_index("timestamp")

        # get_indexer finds the biggest index <= missing - 1m
        past_positions = df.index.get_indexer(missing_endpoints - pd.Timedelta(minutes=1), method='pad')
        matched_times = df.index.take(past_positions.clip(min=0))

        lags = pd.Series(
            (missing_endpoints - pd.Timedelta(minutes=1) - matched_times) / pd.Timedelta(minutes=1),
            index=missing_endpoints
        )

        # now we dont want to match across sessions - this would introduce overnight returns
        mask = (past_positions == -1) | (matched_times.date != missing_endpoints.date)
        lags[mask] = np.nan
        return lags

    return (endpoint_lag,)


@app.cell
def _(all_symbols, mo):
    endpoint_dropdown = mo.ui.dropdown(options=all_symbols, value=all_symbols[0])
    return (endpoint_dropdown,)


@app.cell
def _(datasets, endpoint_dropdown, endpoint_lag, mo):
    mo.vstack([
        endpoint_dropdown,
        endpoint_lag(endpoint_dropdown.value, datasets)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    MSFT: 2016-02-02 - two bars(lag2 and lag7 consecutive), 2016-02-22: about 60 minute missing block, 2018-05-02, 2018-05-03 missing almost entirely

    CSCO: same as above

    AMZN: same as above + some lag1 and very few lag2

    EBAY: same as AMZN

    WMT: 2016-02-02 same as above, 2019-08-12 missing 30minutes block, very few lag1

    COST: 2016-02-02 as above, 2018-05-02, 2018-05-03 missing entirely pretty much

    JPM: 2016-02-02 as above, 2019-08-12 as WMT

    BAC: 2016-02-02 as above, 2019-08-12 lag2 lag7, then lag3 and lag8

    JNJ: 2016-02-02 as above, 2019-08-12 as WMT, 2019-10-29 missing ~30 minutes at the end

    PFE: 2016-02-02 as above, 2019-08-12 as WMT, 2019-10-09 missing about an hour straight then lag5 and lag4 at the same day

    XOM: 2016-02-02 as above, 2019-08-12 as WMT

    CVX: 2016-02-02 as above, 2019-08-12 as WMT

    CRWD: 2019-06-12 (ipo) a bunch of NaN at the start some lag1 lag2 very few lag3 lag4

    CHWY: similar to above when it comes to NaNs, some lag1 lag2 and lag3 and lag4 but not much, 2019-08-12 some missing up to lag15
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Through a manual inspection of the above table we are able to see that there are maybe 4 problematic days with a lot of missing data. Other missing endpoints are can be filled using previous candles or the ones before and since there is not that many of them in total, it filling them using past data shouldn't make the data too unreliable. So for now we will fill the 'easy' ones and re-run the diagnostics to see what else can we do.
    """)
    return


@app.cell
def _(datasets, pd, trading_sessions):
    # based on manual analysis
    excluded_dates = {
        "MSFT": ["2016-02-02", "2016-02-22", "2018-05-02", "2018-05-03"],
        "CSCO": ["2016-02-02", "2016-02-22", "2018-05-02", "2018-05-03"],
        "AMZN": ["2016-02-02", "2016-02-22", "2018-05-02", "2018-05-03"],
        "EBAY": ["2016-02-02", "2016-02-22", "2018-05-02", "2018-05-03"],
        "WMT":  ["2016-02-02", "2019-08-12"],
        "COST": ["2016-02-02", "2018-05-02", "2018-05-03"],
        "JPM":  ["2016-02-02", "2019-08-12"],
        "BAC":  ["2016-02-02", "2019-08-12"],
        "JNJ":  ["2016-02-02", "2019-08-12", "2019-10-29"],
        "PFE":  ["2016-02-02", "2019-08-12", "2019-10-09"],
        "XOM":  ["2016-02-02", "2019-08-12"],
        "CVX":  ["2016-02-02", "2019-08-12"],
        "CRWD": ["2019-06-12"],
        "CHWY": ["2019-06-14", "2019-08-12"]
    }

    excluded_dates = {
        symbol: {pd.Timestamp(d).date() for d in dates}
        for symbol, dates in excluded_dates.items()
    }

    def fill_easy(symbol):
        df = datasets[symbol].copy()
        first_bar = df["timestamp"].min()
        last_bar = df["timestamp"].max()
        sessions = trading_sessions[(trading_sessions["date"] >= first_bar.date())  & (trading_sessions["date"] <= last_bar.date())]

        minute_grid = pd.DatetimeIndex([])
        for _, session in sessions.iterrows():
            grid = pd.date_range(
                start=session["open"],
                end=session["close"],
                freq="1min",
                inclusive='left'
            )
            minute_grid = minute_grid.append(grid)
        df = df.set_index("timestamp")
        df = df.reindex(minute_grid)

        excluded = excluded_dates[symbol]
        mask = pd.Index(df.index.date).isin(excluded)

        df["close_filled"] = df["close"]
        df.loc[~mask, "close_filled"] = df.loc[~mask, "close"].groupby(df.loc[~mask].index.date).ffill(limit=7)
        df["close"] = df["close_filled"]
        df = df.drop(columns=["close_filled"])
        return df.rename_axis("timestamp").reset_index()

    return (fill_easy,)


@app.cell
def _(all_symbols, fill_easy):
    datasets_filled = {_symbol: fill_easy(_symbol) for _symbol in all_symbols}
    return (datasets_filled,)


@app.cell
def _(all_symbols, compute_returns, datasets_filled):
    returns_filled = {_symbol: compute_returns(_symbol, datasets_filled) for _symbol in all_symbols}
    return (returns_filled,)


@app.cell
def _(pd, returns, returns_filled):
    def _missing_return_counts(return_data):
        return pd.Series(
            {
                _symbol: int(_returns.isna().sum())
                for _symbol, _returns in return_data.items()
            },
            name="missing_returns",
        )

    _quality_report = pd.DataFrame({
        "missing_returns_raw": _missing_return_counts(returns),
        "missing_returns_filled": _missing_return_counts(returns_filled),
    })
    _quality_report["missing_returns_reduction"] = (
        _quality_report["missing_returns_raw"]
        - _quality_report["missing_returns_filled"]
    )
    _quality_report

    return


if __name__ == "__main__":
    app.run()
