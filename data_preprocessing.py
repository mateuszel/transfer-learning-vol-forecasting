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
    """)
    return


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import pandas as pd

    from preprocessing import (
        load_bars, 
        build_trading_sessions, 
        prepare_bars, 
        summarize_raw_coverage, 
        compute_returns,
        summarize_missing_returns, 
        count_valid_rolling_windows, 
        calculate_endpoint_lags, 
        fill_short_gaps
    )

    return (
        build_trading_sessions,
        calculate_endpoint_lags,
        compute_returns,
        count_valid_rolling_windows,
        fill_short_gaps,
        load_bars,
        mo,
        pd,
        prepare_bars,
        summarize_missing_returns,
        summarize_raw_coverage,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data loading, initial preprocessing and quality report

    In this section we do the most basic processing of the data, such as timestamps conversions and inspect the overall quality.
    """)
    return


@app.cell(hide_code=True)
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
def dataset_loading(all_symbols, load_bars):
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
def quality_introduction(mo):
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
def raw_returns_sample(mo, returns, returns_dropdown):
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
    return n_fills, returns_filled


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


@app.cell
def _(count_valid_rolling_windows, pd, returns_filled):
    pd.concat([count_valid_rolling_windows(returns_filled, _window) for _window in [1, 5, 22]], axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first controlled endpoint fill is now complete. We do not want to fill every remaining missing value automatically. We now inspect what is still missing and separate the remaining cases into sessions that should be excluded, small source gaps that may be handled at the daily realized-variance level, and target-specific missingness that requires explicit treatment.
    """)
    return


if __name__ == "__main__":
    app.run()
