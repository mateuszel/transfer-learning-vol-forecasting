import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Selection and Preprocessing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this notebook we prepare the data for further experiments. The data that we have is raw 1 minute OHLCV data obtained using Alpaca. When fetching, we tried to get as much data as possible for the given assets, without getting into the COVID-19 range, as this period was very unpredictable. It might be interesting to include it later, to check how our methods work across market regimes.
    Throughout the notebook we try to consistently use the following conventions:
    - timestamps represent the beggining of each bar and are converted into NYC timezone
    - we restrict the data to only cover regular NYSE sessions
    - overnight returns are not included or analyzed

    The goal is to first inspect the quality of raw data and then process data wherever needed, in order to obtain as much data as possible for future forecasting and validation. We put extra efforts into retrieving missing data for target variables as the available information for those assets is already severly limited.

    Most of the source code used in this notebook comes from ``preprocessing.py`` file as the notebook is supposed to only present and highlight the decision making process and key results regarding data preprocessing.

    The whole run of the notebook shouldn't take more than a minute, so don't be afraid to run it if needed.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path

    from preprocessing import (
        load_bars,
        build_trading_sessions,
        prepare_bars,
        summarize_raw_coverage,
        compute_returns,
        summarize_missing_returns,
        count_valid_rolling_windows,
        calculate_endpoint_lags,
        fill_short_gaps,
        summarize_missing_sessions,
        compute_daily_rv,
        build_basic_dataset,
        load_extended_data,
        summarize_stock_daily_coverage,
        summarize_market_daily_coverage,
        align_market_data,
        build_extended_features,
    )


    return (
        Path,
        align_market_data,
        build_basic_dataset,
        build_extended_features,
        build_trading_sessions,
        calculate_endpoint_lags,
        compute_daily_rv,
        compute_returns,
        count_valid_rolling_windows,
        fill_short_gaps,
        load_bars,
        load_extended_data,
        mo,
        pd,
        prepare_bars,
        summarize_market_daily_coverage,
        summarize_missing_returns,
        summarize_missing_sessions,
        summarize_raw_coverage,
        summarize_stock_daily_coverage,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data loading, initial preprocessing and quality report

    In this section we do the most basic processing of the data, such as timestamps conversions and inspect the overall quality.
    """)
    return


@app.cell(hide_code=True)
def _():
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
    return all_symbols, source_sectors


@app.cell(hide_code=True)
def _(mo, source_sectors):
    sectors_table = mo.md(
        "| Sector | Source stocks |\n"
        "|---|---|\n"
        + "\n".join(
            f"| {sector} | {', '.join(symbols)} |"
            for sector, symbols in source_sectors.items()
        )
    )
    return (sectors_table,)


@app.cell(hide_code=True)
def _(mo, sectors_table):
    mo.md(rf"""
    First, the definition of the universe of stocks we will be working with.

    As the project required, we selected 12 mature stocks with rich history from 6 different sectors to serve as our source for transfer:

    {sectors_table}

    For all of the above assets we fetch data from **January 1st, 2016** to **February 21st, 2020**, both ends inclusive. The choice of this range is not random. Our data provider - Alpaca - only offers data up to 2016, so we go back in history as much as we can. The right boundary is related to the aforementioned problem with COVID-19 pandemic, as it's one of the latest dates before extreme changes in the markets. 

    Additionally, we've chosen 2 target assets: **CHWY** and **CRWD**. Their choice was also dependent on the available data. Their IPO's happened respectively at **June 14th, 2019** and **June 12th, 2019**, which means that with some luck we should be able to obtain around 150 sessions of data for each of those assets, a number sufficient for testing the behavior of models depending on target dataset size.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we proceed to downloading and then loading our datasets into the notebook. Below one can see the initial format of the datatables that were loaded for a chosen asset.
    """)
    return


@app.cell(hide_code=True)
def _(all_symbols, load_bars):
    raw_datasets = load_bars(all_symbols, 'data/raw')
    return (raw_datasets,)


@app.cell(hide_code=True)
def _(all_symbols, mo):
    raw_dataset_dropdown = mo.ui.dropdown(options=all_symbols, value=all_symbols[0])
    return (raw_dataset_dropdown,)


@app.cell(hide_code=True)
def _(mo, raw_dataset_dropdown, raw_datasets):
    mo.vstack([raw_dataset_dropdown, raw_datasets[raw_dataset_dropdown.value]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We only select columns related to timestamps, symbols and open / close prices. The column containing volumes is dropped on purpose, as it will not be needed for our experiments.

    At this point the timestamps are expressed in the UTC timezone. To make it easier to align the data with the regular NYSE trading sessions we now convert all of the timestamps to NYC timezone.

    Moreover, as we don't want to include any overnight returns, we filter out any bars that are not inside regular trading sessions. For that we first need to fetch the information about all considered sessions. This data can be seen in the table below:
    """)
    return


@app.cell(hide_code=True)
def _(build_trading_sessions):
    trading_sessions = build_trading_sessions("2016-01-01", "2020-02-21")
    trading_sessions
    return (trading_sessions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The datasets for every asset after aligning timestamps and filtering out any bars outside of regular NYSE trading sessions can be inspected below. It's important to notice that the timestamps represent the exact time each bar opened.
    """)
    return


@app.cell(hide_code=True)
def _(all_symbols, mo, prepare_bars, raw_datasets, trading_sessions):
    datasets = prepare_bars(raw_datasets, trading_sessions)
    datasets_dropdown = mo.ui.dropdown(options=all_symbols, value=all_symbols[0])
    return datasets, datasets_dropdown


@app.cell(hide_code=True)
def _(datasets, datasets_dropdown, mo):
    mo.vstack([datasets_dropdown, datasets[datasets_dropdown.value]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now perform the first quality check: how much of the expected intraday bar data is present before any imputation. This report describes the coverage at a one-minute bar level. It doesn't measure the number of missing five-minute returns.
    """)
    return


@app.cell(hide_code=True)
def _(datasets, summarize_raw_coverage, trading_sessions):
    raw_coverage = summarize_raw_coverage(
        datasets=datasets,
        trading_sessions=trading_sessions,
    )
    raw_coverage
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At a first glance the raw data coverage report looks promising. The most worrying values are the ones related to target assets, as they have the lowest coverage rate. This can become problematic considering that we already have very little information about those two stocks.

    However, this one-minute coverage report is not sufficient to determine the usability of the data for our model, as we will be using five-minute returns. Thus, we will now compute the five-minute log returns for all assets and check how many of them we are missing.

    First let's take a look at the computed returns below. Here the interval end timestamp, represents the exact time that the respective five-minute interval used for computing returns closed. The formula that we use for computing the returns is as follows:

    Assume that we want to calculate the returns over a given five-minute interval. Let $P_t$ denote the closing price of the last bar within this interval and $P_{t-1}$ denote the closing price of the last bar within the previous interval. Then
    \[
        r_t = \log\frac{P_t}{P_{t-1}}.
    \]
    If the considered interval is the first one of the trading session we use the trading sessions's first bar's opening price instead of $P_{t-1}$.
    """)
    return


@app.cell(hide_code=True)
def _(compute_returns, datasets, trading_sessions):
    returns = {
        symbol: compute_returns(bars, trading_sessions)
        for symbol, bars in datasets.items()
    }
    return (returns,)


@app.cell(hide_code=True)
def _(all_symbols, mo):
    returns_dropdown = mo.ui.dropdown(options=all_symbols, value=all_symbols[0])
    return (returns_dropdown,)


@app.cell(hide_code=True)
def _(mo, returns, returns_dropdown):
    mo.vstack([returns_dropdown, returns[returns_dropdown.value]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Not let's compare the numbers of missing returns across assets.
    """)
    return


@app.cell(hide_code=True)
def _(returns, summarize_missing_returns):
    raw_missing_returns = summarize_missing_returns(returns)
    raw_missing_returns
    return (raw_missing_returns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Given how long is the history for our source assets most of them aren't worrying. The one standing out is Costco. As we were rightfully worried before, both target assets miss a lot of values. This is a big issue, because in our models we intend to use among others, the 22 day rolling volatility. This means that with so many missing returns we might not be able to compute enough features.

    To see this better we can count how many windows we would be able to construct for each asset without any imputing:
    """)
    return


@app.cell(hide_code=True)
def _(count_valid_rolling_windows, pd, returns):
    pd.concat([count_valid_rolling_windows(returns, _window) for _window in [1, 5, 22]], axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we can see the returns for some of the assets are completely unusable. In particular, for CHWY we wouldn't be able to construct a single 22 day rolling volatility value. For a lot of the assets, we have a decent coverage.

    First, notice that a missing return value can have three sources:
    1. Interval's close price is missing
    2. Previous interval's close price is missing
    3. Session open price is missing

    We can see that if we would impute some of the closing prices at the ends of intervals, we would effectively take care of two sources (1, 2) of missing returns values. That's why we will begin with this approach.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forward filling of missing data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The idea behind imputing the interval endpoints is very simple. We will try to just forward fill the missing values, but in a controled way. Ideally, we would like to forward fill not further away than three minutes. This limit is just a guess of what makes sense and hopefully most of the missing bars can be imputed using the bars directly before them. Moreover, we don't want to forward fill across different sessions, because data points from two different sessions would be very far away from each other and may introduce some of the overnight returns.

    In case it's not clear, we don't want the data points to be too far away from each other, because it affects the data more than it does when we use adjacent points, and our goal is to create a dataset that is as complete as possible and as similar to reality as possible.
    """)
    return


@app.cell(hide_code=True)
def _(all_symbols, mo):
    endpoint_dropdown = mo.ui.dropdown(options=all_symbols, value=all_symbols[0])
    return (endpoint_dropdown,)


@app.cell(hide_code=True)
def _(
    calculate_endpoint_lags,
    datasets,
    endpoint_dropdown,
    mo,
    trading_sessions,
):
    mo.vstack([
        endpoint_dropdown,
        calculate_endpoint_lags(datasets[endpoint_dropdown.value], trading_sessions)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The above table shows two types of missing data:

    - Short isolated gaps, typically one or two missing bars, visible as small lags of 1 or 2 minutes
    - Long consecutive gaps, where the lag increases by approximately five minutes for each missing interval. These indicate extended periods with no available bars rather than isolated observations.

    The most important cases:
    - **2016-02-02:** short gaps affecting many assets
    - **2016-02-22:** an approximately one-hour block missing for many assets
    - **2018-05-02** and **2018-05-03:** nearly complete loss of the trading session for selected assets
    - **2019-08-12:** approximately 30-minute gaps for several assets
    - **CRWD and CHWY IPO dates:** missing observations at the beggining of the session

    Forward filling those dates could introduce unreliable prices, hence these dates are for now excluded from forward filling.

    | Asset | Dates excluded from forward fill |
    |---|---|
    | MSFT | 2016-02-02, 2016-02-22, 2018-05-02, 2018-05-03 |
    | CSCO | 2016-02-02, 2016-02-22, 2018-05-02, 2018-05-03 |
    | AMZN | 2016-02-02, 2016-02-22, 2018-05-02, 2018-05-03 |
    | EBAY | 2016-02-02, 2016-02-22, 2018-05-02, 2018-05-03 |
    | WMT | 2016-02-02, 2019-08-12 |
    | COST | 2016-02-02, 2018-05-02, 2018-05-03 |
    | JPM | 2016-02-02, 2019-08-12 |
    | BAC | 2016-02-02, 2019-08-12 |
    | JNJ | 2016-02-02, 2019-08-12, 2019-10-29 |
    | PFE | 2016-02-02, 2019-08-12, 2019-10-09 |
    | XOM | 2016-02-02, 2019-08-12 |
    | CVX | 2016-02-02, 2019-08-12 |
    | CRWD | 2019-06-12 |
    | CHWY | 2019-06-14, 2019-08-12 |

    Apart from these dates we will use forward fill with maximum lag of 7 minutes to imput some of the missing data. While this choice of maximum lag might seem risky, there were very few missing values with 7 minute lag, and imputing them shouldn't have a great impact on the overall quality of the resulting dataset.

    In the table below we can see how many values were imputed for each asset and what was the reduction in missing returns.
    """)
    return


@app.cell(hide_code=True)
def _(pd):
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
    return (excluded_dates,)


@app.cell(hide_code=True)
def _(
    all_symbols,
    compute_returns,
    datasets,
    excluded_dates,
    fill_short_gaps,
    trading_sessions,
):
    datasets_filled = {_symbol: fill_short_gaps(datasets[_symbol], trading_sessions, excluded_dates[_symbol])[0] for _symbol in all_symbols}

    n_fills = {_symbol: fill_short_gaps(datasets[_symbol], trading_sessions, excluded_dates[_symbol])[1] for _symbol in all_symbols}

    returns_filled = {_symbol: compute_returns(datasets_filled[_symbol], trading_sessions) for _symbol in all_symbols}
    return datasets_filled, n_fills, returns_filled


@app.cell(hide_code=True)
def _(
    n_fills,
    pd,
    raw_missing_returns,
    returns_filled,
    summarize_missing_returns,
):
    filled_missing_returns = summarize_missing_returns(returns_filled)

    missing_returns_comparison = pd.concat([raw_missing_returns.rename("missing_returns_raw"), filled_missing_returns.rename("missing_returns_filled")], axis=1)

    missing_returns_comparison["missing_returns_reduction"] = (missing_returns_comparison["missing_returns_raw"] - missing_returns_comparison["missing_returns_filled"])

    imputed_close_cnt = pd.Series(n_fills, name="imputed_close_values")
    missing_returns_comparison["imputed_close_values"] = imputed_close_cnt

    missing_returns_comparison
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we can see in the above table, the imputation had a great effect on some of the asset, including the target assets. It didn't have an effect on some of the other assets though, this indicates that they are affected mostly or only by the missing values occuring on excluded dates.

    Let's also look at how many rolling features are we able to construct with the available data now:
    """)
    return


@app.cell(hide_code=True)
def _(count_valid_rolling_windows, pd, returns_filled):
    pd.concat([count_valid_rolling_windows(returns_filled, _window) for _window in [1, 5, 22]], axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first controlled endpoint fill is now complete. We do not want to fill every remaining missing value automatically. We now inspect what is still missing and separate the remaining cases into sessions that should be excluded, small source gaps that may be handled at the daily realized-variance level, and target-specific missingness that requires explicit treatment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Further inspection of the missing data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The table below makes it easier to see the structure of remaining missing returns per session per asset.
    """)
    return


@app.cell(hide_code=True)
def _(returns_filled, summarize_missing_sessions):
    summarize_missing_sessions(returns_filled)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By manual inspection of the above table we are able to come up with ideas for final adjustments in our data, but first let's explain how exactly will the returns be used in feature engineering for our experiments.

    The problem we are tackling is forecasting the realized variance $RV_{t+1}$ with information available up to and including time $t$, where time $t$ represents a trading session or its date.

    For a single trading session at time $t$ we define the daily realized variance as:

    \[
    RV_t^{(n)} = \sum_{i=1}^n r_{t, i}^2,
    \]

    where $n$ represents the number of intervals we compute the log returns over within one session. Intuitively, as $n$ gets larger we sample the returns at a higher frequency. Turns out that $RV_t^{(n)}$ is a consistent estimator of the actual variance of the price over one session.

    Then in order to forecast $Y_t = RV_{t+1}$ we will use, among others, $X_t = (RV_{t, d}, RV_{t, w}, RV_{t, m})$, where:

    \[
        RV_{t, d} \quad \text{represents the previous day's realized variance},
    \]
    \[
        RV_{t, w} =\frac{1}{5}\sum_{i=0}^4RV_{t-i, d} \quad \text{represents the weekly realized variance and }
    \]
    \[
        RV_{t, m} = \frac{1}{22}\sum_{i=0}^{21} RV_{t-i, d} \quad \text{represents the monthly realized variance.}
    \]

    Having seen that we can define our strategy for final adjustments in our data:

    1. If we are missing only a few returns from the session, let's say $k$ returns, then it might makes sense to just scale the realized variance $\tilde{RV}_t = \frac{n}{n-k}\sum_{j\in A_r}r_{t, j}^2$. If $k$ is sufficiently small this method shouldn't affect the data too much, especially if it isn't used too frequently.
    2. Ignore the entire 2018-05-02 and 2018-05-03 sessions, as they have no available returns at all. Imputing those gaps would be highly inaccurate.
    3. For target assets drop the dates of their IPO's respectively as they miss about half of session of data, due to trading starting during the day rather than at the beginning of the session.

    So now the only thing that we will do is drop the respective IPO dates. After that we will continue to construct the final dataset, by using the approach described in 1 whenever needed. As it can be seen in data, there are very few sessions for some assets that have a significant number of missing returns, so using the $RV$ scaling shouldn't decrease the overall quality of data that much and let's us create more usable feature values.

    The final returns per asset can be inspected in the table below:
    """)
    return


@app.cell(hide_code=True)
def _(compute_returns, datasets_filled, pd, trading_sessions):
    _ipo_dates = {
        "CRWD": {pd.Timestamp("2019-06-12").date()},
        "CHWY": {pd.Timestamp("2019-06-14").date()},
    }

    datasets_final = {
        symbol: df.loc[
            ~pd.to_datetime(df["date"]).dt.date.isin(_ipo_dates.get(symbol, set()))
        ].copy()
        for symbol, df in datasets_filled.items()
    }

    returns_final = {symbol: compute_returns(bars, trading_sessions) for symbol, bars in datasets_final.items()}
    return (returns_final,)


@app.cell(hide_code=True)
def _(all_symbols, mo):
    final_returns_dropdown = mo.ui.dropdown(options=all_symbols, value=all_symbols[0])
    return (final_returns_dropdown,)


@app.cell(hide_code=True)
def _(final_returns_dropdown, mo, returns_final):
    mo.vstack([final_returns_dropdown, returns_final[final_returns_dropdown.value]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final dataset construction

    In this, final, section of this notebook we prepare and document the actual datasets that will be used for any later experiments. We begin by constructing for each asset and possible date the following observation:
    \[
        (X_t,Y_t) = (RV_{t, d}, RV_{t, w}, RV_{t, m}, RV_{t+1, d}).
    \]

    At the same time we keep track of the last date used for constructing the rolling features, which is $t$. This will make it easier to avoid any data leakage in the future.

    The basic dataset can be seen below:
    """)
    return


@app.cell(hide_code=True)
def _(compute_daily_rv, returns_final):
    rvs = compute_daily_rv(returns_final)
    return (rvs,)


@app.cell(hide_code=True)
def _(build_basic_dataset, rvs):
    df = build_basic_dataset(rvs)
    df
    return (df,)


@app.cell(hide_code=True)
def _(df):
    _basic_columns = ["t", "rv_t", "rv_w", "rv_m", "rv_next"]
    _complete_df = df.dropna(subset=_basic_columns)

    _summary = (
        _complete_df.groupby("symbol")
        .agg(
            first_day=("t", "min"),
            last_day=("t", "max"),
            observations_without_nan=("t", "size"),
        )
        .reset_index()
    )

    _columns = ["symbol", "first_day", "last_day", "observations_without_nan"]
    _header = "| " + " | ".join(_columns) + " |"
    _separator = "| " + " | ".join(["---"] * len(_columns)) + " |"
    _rows = "\n".join(
        "| " + " | ".join(str(row[column]) for column in _columns) + " |"
        for _, row in _summary.iterrows()
    )

    basic_dataset_summary_table = "\n".join([_header, _separator, _rows])
    return (basic_dataset_summary_table,)


@app.cell(hide_code=True)
def _(basic_dataset_summary_table, mo):
    mo.md(f"""
    The table below summarizes how many observations that can be used in the HAR model we have managed to obtain through our work in this notebook. As we can see most of the observations were retained and we now have a quite complete dataset. 

    {basic_dataset_summary_table}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extended HAR model features

    The features that we have already computed aren't the only ones used in HAR model, they are just the most basic ones. There is a number of different features one can include in the extended HAR model, but for simplicity we will only add five more features:

    1. **5-day-momentum** given by: $MOM_t = \log C_t - \log C_{t-5}$, where $C_t$ represents the closing price on day $t$.
    2. **dollar volume change** given by: $DV_t = \log(C_t V_t) - \log (C_{t-1}V_{t-1})$, where $V_t$ is the trading volume on day $t$.
    3. **CBOE Volatility Index** VIX
    4. **3-month Treasury Bill rate** US3M
    5. **Hang Seng squared return** given by: $HSI_t = (\log H_t - \log H_{t-1})^2$

    The data needed to construct those values will now be downloaded and loaded into the notebook. Afterwards we will quickly inspect whether there are any missing values, which hopefully there are not, and construct the final dataset.
    """)
    return


@app.cell(hide_code=True)
def _(build_trading_sessions, load_extended_data):
    ext_datasets = load_extended_data("data/raw/extended")

    stock_daily = ext_datasets["stock_daily"].reset_index(drop=True)
    hsi_daily = ext_datasets["hsi_daily"]
    vix_daily = ext_datasets["vix_daily"]
    us3m_daily = ext_datasets["us3m_daily"]

    ext_session_dates = build_trading_sessions(
        "2015-12-21",
        "2020-02-21",
    )["date"]
    return (
        ext_datasets,
        ext_session_dates,
        hsi_daily,
        stock_daily,
        us3m_daily,
        vix_daily,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When it comes to data needed for the extended features we loaded:
    1. The daily close prices of our stocks
    2. The daily Hang Seng close prices
    3. The daily VIX
    4. The daily US3M

    The data loaded is from **2015-12-21** to **2020-02-21** inclusive, which is a bit broader time range compared to the original datasets, but since some of the extended features use information from the past, we didn't want to lose any of previously prepared data unneccessarily.

    Below we can quickly inspect all of the loaded datasets:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    extended_dropdown = mo.ui.dropdown(options=["stock_daily", "hsi_daily", "vix_daily", "us3m_daily"], value="stock_daily")
    return (extended_dropdown,)


@app.cell(hide_code=True)
def _(ext_datasets, extended_dropdown, mo):
    mo.vstack([extended_dropdown, ext_datasets[extended_dropdown.value].reset_index(drop=True)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's summarize the completness of the extended data.
    """)
    return


@app.cell(hide_code=True)
def _(pd):
    ipo_dates = {
        "CRWD": pd.Timestamp("2019-06-12"),
        "CHWY": pd.Timestamp("2019-06-14"),
    }
    return (ipo_dates,)


@app.cell(hide_code=True)
def _(
    ext_session_dates,
    ipo_dates,
    stock_daily,
    summarize_stock_daily_coverage,
):
    stock_daily_summary = summarize_stock_daily_coverage(
        stock_daily,
        ext_session_dates,
        ipo_dates,
    )

    stock_daily_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we can see above the daily close and volume data is complete, so we can use without any transformations.
    """)
    return


@app.cell(hide_code=True)
def _(
    ext_session_dates,
    hsi_daily,
    summarize_market_daily_coverage,
    us3m_daily,
    vix_daily,
):
    _market_datasets = {
        "hsi_daily": hsi_daily,
        "vix_daily": vix_daily,
        "us3m_daily": us3m_daily,
    }

    market_daily_summary = summarize_market_daily_coverage(
        _market_datasets,
        ext_session_dates,
        "2019-06-12",
    )

    market_daily_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we can see that both HSI and US3M contain some missing observations.
    Since the missing US3M observations are very limited and mainly reflect days without a published observation, we will forward-fill this series. This is reasonable because US3M is a slowly changing interest-rate level, and the latest available rate is a decent estimate of the rate applicable on the missing date.

    The missing HSI observations are caused by the misalignment between Hong Kong and NYSE trading calendars. There are NYSE trading days on which the Hong Kong exchange was closed. For these dates, we will align the latest available HSI closing price to each NYSE session. This avoids introducing information from the future and preserves all NYSE observations.

    Since the HSI feature is based on returns, we will first align the HSI closing prices and then calculate the squared returns. As a result, a NYSE session without a new HSI observation produces no new HSI price movement, while the next available HSI observation captures the accumulated change since the previous Hong Kong trading session.

    After alignment this is how our data looks like:
    """)
    return


@app.cell(hide_code=True)
def _(align_market_data, ext_session_dates, hsi_daily, us3m_daily, vix_daily):
    filled_ext_data = align_market_data(hsi_daily, vix_daily, us3m_daily, ext_session_dates)
    filled_ext_data
    return (filled_ext_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The only thing left to do is to compute the feature values using the above presented data and align them with the previously computed $RV$ features. Then our dataset is ready to use.

    The final dataset can be seen below, along with some information about this data.
    """)
    return


@app.cell(hide_code=True)
def _(build_extended_features, filled_ext_data, stock_daily):
    extended_features = build_extended_features(stock_daily, filled_ext_data)
    return (extended_features,)


@app.cell(hide_code=True)
def _(df, extended_features):
    final_dataset = df.merge(
        extended_features,
        on=["symbol", "t"],
        how="left",
        validate="one_to_one",
    )
    final_dataset
    return (final_dataset,)


@app.cell(hide_code=True)
def _(final_dataset, pd):
    _required_columns = [
        "rv_t",
        "rv_w",
        "rv_m",
        "rv_next",
        "hsi_sqr",
        "vix",
        "us3m",
        "MOM",
        "DVC",
    ]

    _final_complete = final_dataset.dropna(subset=_required_columns)

    final_dataset_summary = pd.DataFrame({
        "rows_total": [len(final_dataset)],
        "complete_rows": [len(_final_complete)],
        "assets": [final_dataset["symbol"].nunique()],
        "first_day": [final_dataset["t"].min()],
        "last_day": [final_dataset["t"].max()],
    })

    final_dataset_summary
    return


@app.cell(hide_code=True)
def _(Path, final_dataset):
    final_dataset_path = Path(
        "data/processed/har_extended_dataset.parquet"
    )
    final_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    final_dataset.to_parquet(final_dataset_path, index=False)
    return


if __name__ == "__main__":
    app.run()
