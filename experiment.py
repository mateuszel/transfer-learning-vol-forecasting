import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experiment
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import statsmodels.api as sm

    from modelling import (
        get_source_only_training_data,
        get_target_only_training_data,
        get_naive_pooling_training_data,
        get_next_session_date,
        get_previous_session_date,
        fit_har_model,
        predict_har_model,
        run_rolling_predictions,
        compute_mse,
        compute_qlike,
        compute_prediction_metrics,
        plot_one_method_predictions,
        compare_metric_tables,
        compare_feature_set_metrics,
    )

    from preprocessing import build_trading_sessions

    return (
        build_trading_sessions,
        compare_feature_set_metrics,
        compare_metric_tables,
        compute_prediction_metrics,
        mo,
        pd,
        plot_one_method_predictions,
        run_rolling_predictions,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## HAR models
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We start by introducing the standard and extended HAR models that will be main statitical models used throughout our experiments.

    Suppose we are at time $t$ and that we managed to obtain the 5-minute log returns for that day. Let $r_{t, i}$ denote the $i$-th 5-minute log return of the trading session that happened at time $t$, where $i=1,\dots,n$ and $n$ is the number of 5-minute intervals within that session. Usually $n=78$ or $n=42$.

    The set of standard HAR predictors consists of:
    - $RV_{t, d} = \sum_{i=1}^n r_{t, i}^2$
    - $RV_{t, w} = \frac{1}{5}\sum_{i=0}^4 RV_{t-i, d}$
    - $RV_{t, m} = \frac{1}{22}\sum_{i=0}^{21} RV_{t-i, d}$

    For the standard HAR model we assume that:
    \[
        RV_{t+1, d} = \beta_0 + \beta_d RV_{t, d} + \beta_w RV_{t, w} + \beta_m RV_{t, m} + \varepsilon_{t+1},
    \]
    where $\varepsilon_{t+1}$ represents random noise. We don't assume that the random variables representing the noise are independent or identically distributed, that wouldn't be true for most financial time series.

    The standard HAR model is motivated by the presence of market participants operating over different time horizons. This motivates the use of daily, weekly, and monthly realized variance components, each capturing information about volatility over a different time scale. By combining these components, the model can account for volatility persistence and approximate the slowly decaying autocorrelation observed in realized variance.

    While the above model relies only on historical realized variance, additional financial and macroeconomic variables may contain useful information for predicting future volatility. Because of that we will also consider the extended HAR model which augments the standard predictor set by adding five additional features:

    - $MOM_t$
    - $DV_t$
    - $VIX_t$
    - $US3M_t$
    - $HSI_t$

    A more detailed description of those features can be found in the ``data_preprocessing.py`` notebook. It's also important to notice that the extended predictor set can be modified and tailored to specific usage. The primary reason for selecting this specific set of extended predictors was data availability.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use twelve mature stocks as source assets and two IPO stocks as target assets. The source data covers the period from January 2016 to February 21, 2020. The target data starts around mid-June 2019, following the IPOs of CHWY and CRWD, and ends on February 21, 2020. This period was chosen to avoid the COVID-19 market disruption and because data before 2016 was not available.

    The original data consists of one-minute OHLCV observations. After controlled gap filling and preprocessing, we constructed five-minute log returns and the features used by the extended HAR models. The full preprocessing procedure, including the missing-data analysis, is documented in `data_preprocessing.py`.

    The dataset loaded below is the most complete version obtained, although some missing values remain. We retain the remaining missing values for now, since some of the methods used later, such as Dynamic Time Warping (DTW), rely on the temporal structure of the data. It is therefore important to preserve information about where observations are missing rather than treating the remaining observations as a continuous sequence.
    """)
    return


@app.cell(hide_code=True)
def _(build_trading_sessions, pd):
    extended_df = pd.read_parquet("data/processed/har_extended_dataset.parquet")
    extended_source_df = extended_df[~extended_df["symbol"].isin(["CHWY", "CRWD"])]
    extended_target_df = extended_df[extended_df["symbol"].isin(["CHWY", "CRWD"])]
    trading_sessions = build_trading_sessions("2016-01-04", "2020-02-21")

    extended_df
    return extended_source_df, extended_target_df, trading_sessions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Benchmark Models
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The benchmark experiments aim to assess the severity of the cold-start forecasting problem and establish how different ways of incorporating source information affect predictive performance. We consider Target Only, Source Only, Naive Pooling, and DTW-based transfer, largely following Teller et al. (2026). Beyond providing reference results for subsequent domain adaptation methods, these experiments motivate a closer investigation of whether similarity-based source selection reduces distributional differences and improves the stability of predictive relationships across assets.

    Since our research is primarily focused on transfer of information between sources and target we will for now only focus on HAR and HAR-EXT models. The main reasons for that:
    - HAR models are known to perform reasonably well in terms of forecasting realized variance
    - Simple architecture of the model allows good interpretability of the results and easy diagnostics
    - Low computational cost

    Having that in mind, the benchmark focuses on already known techniques for usage of the available data to train HAR models. Results of those techniques can later be compared to the ones obtained through domain adaptation methods.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluation Setup

    As it was mentioned above we want to check how different methods of incorporating source information into the training dataset affect predictive performance.

    To ensure a fair comparison, all benchmark models will be evaluated over the same out-of-sample period, consisting of the final 50 trading sessions with available realized variance labels for each target asset. We initially consider $n_T = 40$ target observations preceding this evaluation period.

    During the evaluation, the target training sample is expanded as new observations become available, while its starting point remains fixed. The HAR coefficients are reestimated every 5 trading sessions using all observations available for the respective training method at that point in time.

    Forecasting performance will be evaluated using the following metrics:
    1. QLIKE
    2. MSE
    3. Relative Loss and Negative Transfer Gap which compare selected model's performance to the benchmark Target Only based model.

    Performance will be assessed over the entire 50-session out-of-sample period. Additionally, we will retain individual forecast losses and cumulative average losses to examine the stability of predictive performance over time. This ensures that all methods are evaluated over identical forecasting dates rather than different market periods.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Target Only
    """)
    return


@app.cell(hide_code=True)
def _(
    compare_feature_set_metrics,
    compute_prediction_metrics,
    extended_source_df,
    extended_target_df,
    run_rolling_predictions,
    trading_sessions,
):
    # running predictions, computing metrics 
    target_only_crwd_std = run_rolling_predictions(
        mode="target_only",
        symbol="CRWD",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=False,
        initial_nt=40,
        refit_freq=5,
    )

    target_only_chwy_std = run_rolling_predictions(
        mode="target_only",
        symbol="CHWY",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=False,
        initial_nt=40,
        refit_freq=5,
    )

    target_only_crwd_ext = run_rolling_predictions(
        mode="target_only",
        symbol="CRWD",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=True,
        initial_nt=40,
        refit_freq=5,
    )

    target_only_chwy_ext = run_rolling_predictions(
        mode="target_only",
        symbol="CHWY",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=True,
        initial_nt=40,
        refit_freq=5,
    )

    to_crwd_std_overall, to_crwd_std_daily = compute_prediction_metrics(target_only_crwd_std)
    to_crwd_ext_overall, to_crwd_ext_daily = compute_prediction_metrics(target_only_crwd_ext)

    to_chwy_std_overall, to_chwy_std_daily = compute_prediction_metrics(target_only_chwy_std)
    to_chwy_ext_overall, to_chwy_ext_daily = compute_prediction_metrics(target_only_chwy_ext)

    to_crwd_overall = compare_feature_set_metrics(
        to_crwd_std_overall,
        to_crwd_ext_overall,
        "Target Only",
    )
    to_chwy_overall = compare_feature_set_metrics(
        to_chwy_std_overall,
        to_chwy_ext_overall,
        "Target Only",
    )
    return (
        target_only_chwy_ext,
        target_only_chwy_std,
        target_only_crwd_ext,
        target_only_crwd_std,
        to_chwy_ext_overall,
        to_chwy_std_overall,
        to_crwd_ext_overall,
        to_crwd_std_overall,
    )


@app.cell(hide_code=True)
def _(
    mo,
    plot_one_method_predictions,
    target_only_chwy_ext,
    target_only_chwy_std,
    target_only_crwd_ext,
    target_only_crwd_std,
):
    _chart = plot_one_method_predictions(target_only_chwy_std, target_only_chwy_ext, target_only_crwd_std, target_only_crwd_ext, "Target Only")
    mo.hstack([_chart], justify="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Source Only
    """)
    return


@app.cell(hide_code=True)
def _(
    compare_feature_set_metrics,
    compare_metric_tables,
    compute_prediction_metrics,
    extended_source_df,
    extended_target_df,
    run_rolling_predictions,
    to_chwy_ext_overall,
    to_chwy_std_overall,
    to_crwd_ext_overall,
    to_crwd_std_overall,
    trading_sessions,
):
    # running predictions, computing metrics, and comparing with Target Only
    source_only_crwd_std = run_rolling_predictions(
        mode="source_only",
        symbol="CRWD",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=False,
        initial_nt=40,
        refit_freq=5,
    )

    source_only_chwy_std = run_rolling_predictions(
        mode="source_only",
        symbol="CHWY",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=False,
        initial_nt=40,
        refit_freq=5,
    )

    source_only_crwd_ext = run_rolling_predictions(
        mode="source_only",
        symbol="CRWD",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=True,
        initial_nt=40,
        refit_freq=5,
    )

    source_only_chwy_ext = run_rolling_predictions(
        mode="source_only",
        symbol="CHWY",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=True,
        initial_nt=40,
        refit_freq=5,
    )

    so_crwd_std_overall, so_crwd_std_daily = compute_prediction_metrics(source_only_crwd_std)
    so_crwd_ext_overall, so_crwd_ext_daily = compute_prediction_metrics(source_only_crwd_ext)

    so_chwy_std_overall, so_chwy_std_daily = compute_prediction_metrics(source_only_chwy_std)
    so_chwy_ext_overall, so_chwy_ext_daily = compute_prediction_metrics(source_only_chwy_ext)

    so_crwd_overall = compare_feature_set_metrics(so_crwd_std_overall, so_crwd_ext_overall, "Source Only")
    so_chwy_overall = compare_feature_set_metrics(so_chwy_std_overall, so_chwy_ext_overall, "Source Only")

    so_crwd_std_vs_to_overall = compare_metric_tables(so_crwd_std_overall, to_crwd_std_overall, "Source Only STD", "Target Only STD")
    so_crwd_ext_vs_to_overall = compare_metric_tables(so_crwd_ext_overall, to_crwd_ext_overall, "Source Only EXT", "Target Only EXT")

    so_chwy_std_vs_to_overall = compare_metric_tables(so_chwy_std_overall, to_chwy_std_overall, "Source Only STD", "Target Only STD")
    so_chwy_ext_vs_to_overall = compare_metric_tables(so_chwy_ext_overall, to_chwy_ext_overall, "Source Only EXT", "Target Only EXT")
    return (
        source_only_chwy_ext,
        source_only_chwy_std,
        source_only_crwd_ext,
        source_only_crwd_std,
    )


@app.cell(hide_code=True)
def _(
    mo,
    plot_one_method_predictions,
    source_only_chwy_ext,
    source_only_chwy_std,
    source_only_crwd_ext,
    source_only_crwd_std,
):
    _chart = plot_one_method_predictions(source_only_chwy_std, source_only_chwy_ext, source_only_crwd_std, source_only_crwd_ext, "Source Only")
    mo.hstack([_chart], justify="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Naive Pooling
    """)
    return


@app.cell(hide_code=True)
def _(
    compare_feature_set_metrics,
    compare_metric_tables,
    compute_prediction_metrics,
    extended_source_df,
    extended_target_df,
    run_rolling_predictions,
    to_chwy_ext_overall,
    to_chwy_std_overall,
    to_crwd_ext_overall,
    to_crwd_std_overall,
    trading_sessions,
):
    # running predictions, computing metrics, and comparing with Target Only
    naive_crwd_std = run_rolling_predictions(
        mode="naive_pooling",
        symbol="CRWD",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=False,
        initial_nt=40,
        refit_freq=5,
    )

    naive_chwy_std = run_rolling_predictions(
        mode="naive_pooling",
        symbol="CHWY",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=False,
        initial_nt=40,
        refit_freq=5,
    )

    naive_crwd_ext = run_rolling_predictions(
        mode="naive_pooling",
        symbol="CRWD",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=True,
        initial_nt=40,
        refit_freq=5,
    )

    naive_chwy_ext = run_rolling_predictions(
        mode="naive_pooling",
        symbol="CHWY",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=True,
        initial_nt=40,
        refit_freq=5,
    )

    naive_crwd_std_overall, naive_crwd_std_daily = compute_prediction_metrics(naive_crwd_std)
    naive_crwd_ext_overall, naive_crwd_ext_daily = compute_prediction_metrics(naive_crwd_ext)

    naive_chwy_std_overall, naive_chwy_std_daily = compute_prediction_metrics(naive_chwy_std)
    naive_chwy_ext_overall, naive_chwy_ext_daily = compute_prediction_metrics(naive_chwy_ext)

    naive_crwd_overall = compare_feature_set_metrics(naive_crwd_std_overall, naive_crwd_ext_overall, "Naive Pooling")
    naive_chwy_overall = compare_feature_set_metrics(naive_chwy_std_overall, naive_chwy_ext_overall, "Naive Pooling")

    naive_crwd_std_vs_to_overall = compare_metric_tables(
        naive_crwd_std_overall, to_crwd_std_overall,
        "Naive Pooling STD", "Target Only STD"
    )

    naive_crwd_ext_vs_to_overall = compare_metric_tables(
        naive_crwd_ext_overall, to_crwd_ext_overall,
        "Naive Pooling EXT", "Target Only EXT"
    )

    naive_chwy_std_vs_to_overall = compare_metric_tables(
        naive_chwy_std_overall, to_chwy_std_overall,
        "Naive Pooling STD", "Target Only STD"
    )

    naive_chwy_ext_vs_to_overall = compare_metric_tables(
        naive_chwy_ext_overall, to_chwy_ext_overall,
        "Naive Pooling EXT", "Target Only EXT"
    )
    return naive_chwy_ext, naive_chwy_std, naive_crwd_ext, naive_crwd_std


@app.cell(hide_code=True)
def _(
    mo,
    naive_chwy_ext,
    naive_chwy_std,
    naive_crwd_ext,
    naive_crwd_std,
    plot_one_method_predictions,
):
    _chart = plot_one_method_predictions(naive_chwy_std, naive_chwy_ext, naive_crwd_std, naive_crwd_ext, "Naive Pooling")
    mo.hstack([_chart], justify="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### DTW-based transfer
    """)
    return


@app.cell(hide_code=True)
def _(
    compare_feature_set_metrics,
    compare_metric_tables,
    compute_prediction_metrics,
    extended_source_df,
    extended_target_df,
    run_rolling_predictions,
    to_chwy_ext_overall,
    to_chwy_std_overall,
    to_crwd_ext_overall,
    to_crwd_std_overall,
    trading_sessions,
):
    # running predictions, computing metrics, and comparing with Target Only
    dtw_crwd_std = run_rolling_predictions(
        mode="dtw",
        symbol="CRWD",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=False,
        initial_nt=40,
        refit_freq=5,
        m=22,
        dtw_features=["rv_t", "rv_w", "rv_m"],
        percentile=25,
    )

    dtw_chwy_std = run_rolling_predictions(
        mode="dtw",
        symbol="CHWY",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=False,
        initial_nt=40,
        refit_freq=5,
        m=22,
        dtw_features=["rv_t", "rv_w", "rv_m"],
        percentile=25,
    )

    dtw_crwd_ext = run_rolling_predictions(
        mode="dtw",
        symbol="CRWD",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=True,
        initial_nt=40,
        refit_freq=5,
        m=22,
        dtw_features=["rv_t", "rv_w", "rv_m"],
        percentile=25,
    )

    dtw_chwy_ext = run_rolling_predictions(
        mode="dtw",
        symbol="CHWY",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=True,
        initial_nt=40,
        refit_freq=5,
        m=22,
        dtw_features=["rv_t", "rv_w", "rv_m"],
        percentile=25,
    )

    dtw_crwd_std_overall, dtw_crwd_std_daily = compute_prediction_metrics(dtw_crwd_std)
    dtw_crwd_ext_overall, dtw_crwd_ext_daily = compute_prediction_metrics(dtw_crwd_ext)

    dtw_chwy_std_overall, dtw_chwy_std_daily = compute_prediction_metrics(dtw_chwy_std)
    dtw_chwy_ext_overall, dtw_chwy_ext_daily = compute_prediction_metrics(dtw_chwy_ext)

    dtw_crwd_overall = compare_feature_set_metrics(
        dtw_crwd_std_overall,
        dtw_crwd_ext_overall,
        "DTW",
    )
    dtw_chwy_overall = compare_feature_set_metrics(
        dtw_chwy_std_overall,
        dtw_chwy_ext_overall,
        "DTW",
    )

    dtw_crwd_std_vs_to_overall = compare_metric_tables(
        dtw_crwd_std_overall,
        to_crwd_std_overall,
        "DTW STD",
        "Target Only STD",
    )
    dtw_crwd_ext_vs_to_overall = compare_metric_tables(
        dtw_crwd_ext_overall,
        to_crwd_ext_overall,
        "DTW EXT",
        "Target Only EXT",
    )
    dtw_chwy_std_vs_to_overall = compare_metric_tables(
        dtw_chwy_std_overall,
        to_chwy_std_overall,
        "DTW STD",
        "Target Only STD",
    )
    dtw_chwy_ext_vs_to_overall = compare_metric_tables(
        dtw_chwy_ext_overall,
        to_chwy_ext_overall,
        "DTW EXT",
        "Target Only EXT",
    )
    return dtw_chwy_ext, dtw_chwy_std, dtw_crwd_ext, dtw_crwd_std


@app.cell(hide_code=True)
def _(
    dtw_chwy_ext,
    dtw_chwy_std,
    dtw_crwd_ext,
    dtw_crwd_std,
    mo,
    plot_one_method_predictions,
):
    _chart = plot_one_method_predictions(
        dtw_chwy_std,
        dtw_chwy_ext,
        dtw_crwd_std,
        dtw_crwd_ext,
        "DTW",
    )

    mo.hstack([_chart], justify="center")
    return


if __name__ == "__main__":
    app.run()
