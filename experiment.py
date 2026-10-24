import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Experiment**
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import statsmodels.api as sm
    import altair as alt
    import importlib
    import modelling

    importlib.reload(modelling)

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
        run_benchmark_grid,
        compute_distribution_statistics,
        get_no_label_dtw,
        compute_monte_carlo,
        compute_label_distribution_statistics,
        compute_label_monte_carlo,
        compute_conditional_shift_statistics
    )

    from preprocessing import build_trading_sessions

    return (
        alt,
        build_trading_sessions,
        compare_feature_set_metrics,
        compare_metric_tables,
        compute_conditional_shift_statistics,
        compute_distribution_statistics,
        compute_label_distribution_statistics,
        compute_label_monte_carlo,
        compute_monte_carlo,
        compute_prediction_metrics,
        mo,
        pd,
        plot_one_method_predictions,
        run_benchmark_grid,
        run_rolling_predictions,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **1. HAR models**
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
    ## **2. Data**
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
    return extended_source_df, extended_target_df, trading_sessions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **3. Benchmark Models**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The benchmark experiments aim to assess the severity of the cold-start forecasting problem and establish how different ways of incorporating source information affect predictive performance. We consider Target Only, Source Only and Naive Pooling, largely following Teller et al. (2026). Beyond providing reference results for subsequent domain adaptation methods, these experiments motivate a closer investigation of whether similarity-based source selection reduces distributional differences and improves the stability of predictive relationships across assets.

    Since our research is primarily focused on transfer of information between sources and target we will for now only focus on HAR and HAR-EXT models. The main reasons for that:
    - HAR models are known to perform reasonably well in terms of forecasting realized variance
    - Simple architecture of the model allows good interpretability of the results and easy diagnostics
    - Low computational cost

    Having that in mind, the benchmark focuses on already known techniques for usage of the available data to train HAR models. Results of those techniques can later be compared to the ones obtained through domain adaptation methods.

    The four following methods will be presented now:

    | Method | Training data | Role |
    |---|---|---|
    | Target Only | Target history | Cold-start baseline |
    | Source Only | Source assets | Source-transfer control |
    | Naive Pooling | Source + target | Indiscriminate transfer |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **3.1 Evaluation Setup**

    As it was mentioned above we want to check how different methods of incorporating source information into the training dataset affect predictive performance.

    To ensure a fair comparison, all benchmark models will be evaluated over the same out-of-sample period, consisting of the final 50 trading sessions with available realized variance labels for each target asset. The number of target observations preceding this evaluation period will be denoted as $n_T$.

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
    ### **3.2 Target Only (TO)**

    This method is the simplest benchmark of the ones presented. It only uses limited labeled target data in order to train the HAR model. All of the following methods will be in particular compared to the TO method.

    Below we can see four plots presenting the values predicted by the model compared to true values. While this figure doesn't provide any useful quantitative information, it's useful to see how the baseline model performs.

    As we can see it is rather inaccurate, which is not surprising given the scarcity of training data.
    In the left column we see the performance of the HAR model that only uses the standard predictors (HAR-STD). It's easy to notice that this model doesn't really capture the dynamic changes in realized variance over time.

    On the other hand, the HAR model that uses extended feature set (HAR-EXT) is too unstable. That is probably caused by the small size of dataset combined with a relatively large number of predictors used.
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
        initial_nt=120,
        refit_freq=5,
    )

    target_only_chwy_std = run_rolling_predictions(
        mode="target_only",
        symbol="CHWY",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=False,
        initial_nt=120,
        refit_freq=5,
    )

    target_only_crwd_ext = run_rolling_predictions(
        mode="target_only",
        symbol="CRWD",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=True,
        initial_nt=120,
        refit_freq=5,
    )

    target_only_chwy_ext = run_rolling_predictions(
        mode="target_only",
        symbol="CHWY",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=True,
        initial_nt=120,
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
    ### **3.3 Source Only (SO)**

    Contrary to TO, this method doesn't use target data at all. Instead it uses all the possible source data for training the model.

    Once again we can take a look at the plots presenting predicted values and true values of the realized variance. With this method we can immediately notice that both HAR-STD and HAR-EXT don't capture the dynamic changes in realized variance. That's likely caused by the fact that the source dataset is large, but doesn't contain assets that would provide good enough information about predicting RV for a new issue.
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
        initial_nt=120,
        refit_freq=5,
    )

    source_only_chwy_std = run_rolling_predictions(
        mode="source_only",
        symbol="CHWY",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=False,
        initial_nt=120,
        refit_freq=5,
    )

    source_only_crwd_ext = run_rolling_predictions(
        mode="source_only",
        symbol="CRWD",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=True,
        initial_nt=120,
        refit_freq=5,
    )

    source_only_chwy_ext = run_rolling_predictions(
        mode="source_only",
        symbol="CHWY",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=True,
        initial_nt=120,
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
    ### **3.4 Naive Pooling (NP)**

    The NP method pools all available source data and all available target data. This construction can lead to using irrelevant data for training that might decrease models performance - this is knon as Negative Transfer. This method is used as one of the benchmark models primarily to establish whether Negative Transfer occurs in our specific problem.

    In this case we don't present the plot of predictions vs true values, as it's not very informative. We will however inspect the performance of all three methods discussed up to this point.
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
        initial_nt=120,
        refit_freq=5,
    )

    naive_chwy_std = run_rolling_predictions(
        mode="naive_pooling",
        symbol="CHWY",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=False,
        initial_nt=120,
        refit_freq=5,
    )

    naive_crwd_ext = run_rolling_predictions(
        mode="naive_pooling",
        symbol="CRWD",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=True,
        initial_nt=120,
        refit_freq=5,
    )

    naive_chwy_ext = run_rolling_predictions(
        mode="naive_pooling",
        symbol="CHWY",
        session_dates=trading_sessions,
        extended_source_df=extended_source_df,
        extended_target_df=extended_target_df,
        extended=True,
        initial_nt=120,
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **3.5 Impact of Target History on Transfer Performance**

    In this section we will try to figure out how does the amount of available target data affect the benefits and risk of source transfer.

    In order to do that we will use the exact same evaluation setup that was described before, but we will re-run the evaluation for each model using different values $n_T\in\{40, 60, 80, 100, 120\}$.

    This will allow us to see if the negative transfer occurs and how is it related to the amount of target data available for training.

    Following Teller et al. (2026) we can expect the negative transfer to occur at some point. At the same time, it shouldn't be surprising if the models using source data for training (in an uncontrolled way) outperform the TO model. That's because the values $n_T$ that will be tested are quite small, so resulting TO models might be so bad, that even feeding them random source data will make them perform better and more stable.
    """)
    return


@app.cell(hide_code=True, expand_output=True)
def _(
    alt,
    extended_source_df,
    extended_target_df,
    mo,
    pd,
    run_benchmark_grid,
    trading_sessions,
):
    benchmark_grid = run_benchmark_grid(trading_sessions, extended_source_df, extended_target_df, (40, 60, 80, 100, 120), ("CHWY", "CRWD"), 5, 50)

    _plot_data = benchmark_grid[
        benchmark_grid["method"].isin(
            [
                "Source Only",
                "Naive Pooling",
            ]
        )
    ].copy()

    _plot_data = _plot_data.melt(
        id_vars=[
            "symbol",
            "nT",
            "features",
            "method",
        ],
        value_vars=[
            "relative_mse",
            "relative_qlike",
        ],
        var_name="metric",
        value_name="relative_loss",
    )

    _plot_data["metric"] = _plot_data["metric"].map(
        {
            "relative_mse": "MSE",
            "relative_qlike": "QLIKE",
        }
    )

    _plot_data["feature_label"] = _plot_data["features"].map(
        {
            "STD": "STD (solid line)",
            "EXT": "EXT (dashed line)",
        }
    )

    _x_domains = {
        "MSE": [
            _plot_data[
                _plot_data["metric"] == "MSE"
            ]["nT"].min(),
            _plot_data[
                _plot_data["metric"] == "MSE"
            ]["nT"].max(),
        ],
        "QLIKE": [
            _plot_data[
                _plot_data["metric"] == "QLIKE"
            ]["nT"].min(),
            _plot_data[
                _plot_data["metric"] == "QLIKE"
            ]["nT"].max(),
        ],
    }

    _y_domains = {}

    for _symbol in ["CHWY", "CRWD"]:
        _symbol_data = _plot_data[
            _plot_data["symbol"] == _symbol
        ]

        _symbol_max = max(
            1.0,
            _symbol_data["relative_loss"].max(),
        )

        _y_domains[_symbol] = [
            0,
            _symbol_max * 1.05,
        ]


    def _make_relative_loss_panel(
        symbol,
        metric,
        show_legend=False,
    ):
        _panel_data = _plot_data[
            (_plot_data["symbol"] == symbol)
            & (_plot_data["metric"] == metric)
        ].copy()

        _method_legend = (
            alt.Legend(
                title="Training method",
                orient="top",
                direction="horizontal",
            )
            if show_legend
            else None
        )

        _features_legend = (
            alt.Legend(
                title="Feature set / line style",
                orient="top",
                direction="horizontal",
            )
            if show_legend
            else None
        )

        _y_title = (
            "Relative loss"
            if metric == "MSE"
            else None
        )

        _x_title = (
            "Target observations (nT)"
            if symbol == "CRWD"
            else None
        )

        _lines = (
            alt.Chart(_panel_data)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "nT:Q",
                    title=_x_title,
                    scale=alt.Scale(
                        domain=_x_domains[metric],
                        nice=False,
                    ),
                    axis=alt.Axis(
                        values=[40, 60, 80, 100, 120],
                    ),
                ),
                y=alt.Y(
                    "relative_loss:Q",
                    title=_y_title,
                    scale=alt.Scale(
                        domain=_y_domains[symbol],
                        nice=False,
                    ),
                ),
                color=alt.Color(
                    "method:N",
                    title="Training method",
                    scale=alt.Scale(
                        domain=[
                            "Source Only",
                            "Naive Pooling",
                        ],
                        range=[
                            "steelblue",
                            "firebrick",
                        ],
                    ),
                    legend=_method_legend,
                ),
                strokeDash=alt.StrokeDash(
                    "feature_label:N",
                    title="Feature set / line style",
                    scale=alt.Scale(
                        domain=[
                            "STD (solid line)",
                            "EXT (dashed line)",
                        ],
                        range=[
                            [1, 0],
                            [6, 3],
                        ],
                    ),
                    legend=_features_legend,
                ),
                tooltip=[
                    alt.Tooltip(
                        "symbol:N",
                        title="Asset",
                    ),
                    alt.Tooltip(
                        "metric:N",
                        title="Metric",
                    ),
                    alt.Tooltip(
                        "method:N",
                        title="Training method",
                    ),
                    alt.Tooltip(
                        "features:N",
                        title="Features",
                    ),
                    alt.Tooltip(
                        "nT:Q",
                        title="nT",
                    ),
                    alt.Tooltip(
                        "relative_loss:Q",
                        title="Relative loss",
                        format=".3f",
                    ),
                ],
            )
            .properties(
                title=f"{symbol} - {metric}",
                width=420,
                height=260,
            )
        )

        _reference_line = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "relative_loss": [1.0],
                    }
                )
            )
            .mark_rule(
                color="gray",
                strokeDash=[5, 3],
            )
            .encode(
                y=alt.Y(
                    "relative_loss:Q",
                    scale=alt.Scale(
                        domain=_y_domains[symbol],
                        nice=False,
                    ),
                )
            )
        )

        return alt.layer(
            _lines,
            _reference_line,
        )


    _chwy_mse = _make_relative_loss_panel(
        "CHWY",
        "MSE",
        show_legend=True,
    )

    _chwy_qlike = _make_relative_loss_panel(
        "CHWY",
        "QLIKE",
    )

    _crwd_mse = _make_relative_loss_panel(
        "CRWD",
        "MSE",
    )

    _crwd_qlike = _make_relative_loss_panel(
        "CRWD",
        "QLIKE",
    )

    _top_row = (
        alt.hconcat(
            _chwy_mse,
            _chwy_qlike,
        )
        .resolve_scale(
            x="shared",
            y="shared",
        )
    )

    _bottom_row = (
        alt.hconcat(
            _crwd_mse,
            _crwd_qlike,
        )
        .resolve_scale(
            x="shared",
            y="shared",
        )
    )

    _relative_loss_chart = (
        alt.vconcat(
            _top_row,
            _bottom_row,
        )
        .resolve_scale(
            x="independent",
            y="independent",
        )
    )

    mo.hstack(
        [_relative_loss_chart],
        justify="center",
    )
    return (benchmark_grid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The above chart presents the results of the experiment.

    For CHWY we can clearly notice that the benefits associated with the transfer are getting smaller as $n_T$ grows. The relative MSE stays under $1$, which indicates no negative transfer for this metric. When it comes to QLIKE, we can see for the extended HAR models that there is negative transfer, while the QLIKE of standard HAR models balances near the boundary.

    For CRWD the results stay relatively stable across different values of $n_T$, but show some growth in terms of MSE. However, we can see negative transfer regardless of $n_T$ for the extended source model.

    Even though the above results don't explicitly show negative transfer as a serious problem, they are convincing enough to follow this experiment up with different methods of transfer learning, that potentially mitigative negative transfer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **4. DTW-based transfer**

    We now introduce an alternative method of using source data for training that as shown in Teller et al. (2026) might mitigate negative transfer. First we begin by explaining what is DTW, then we will clarify how exactly is it used in our experiment.

    Dynamic Time Warping (DTW) is a measure used to compare time series that may exhibit similar patterns at different speeds or with temporal shifts. Unlike Euclidean distance it allows for nonlinear alignment of the observations, making it more flexible when comparing dynamics.

    In the context of volatility forecasting we will use DTW to identify historical periods of source assets that resemble recent behavior of the target asset.

    Now let's go through the algorithm we use to actually apply DTW in our problem. Imagine we are at time $t$ and we want to refit our model and then forecast $RV_{t+1}$.
    1. Separate the $m$ most recent target observations, including the observation at time $t$.
    2. If any of the observations from this interval is missing - we can't perform the DTW as it requires no missing values.
    3. For every source asset split its history into intervals of length exactly $m$ starting at the end, so if the number of observations for the asset is not divisible by $m$ we omit the observations that are the oldest.
    4. Discard any intervals that contain missing values.
    5. For every source interval compute the DTW distance between the interval and the target interval.
    6. Select source intervals whose distance from the target interval is in the $\varepsilon$'th percentile.
    7. Combine data from the selected source intervals with all available target data, which gives us our training set.

    By default our algorithm uses $m=22$. This value hasn't been tuned, but it was inspired by Teller et al. (2026).

    Lastly, it's important to notice that DTW identifies similarities between historical volatility paths, but on its own it doesn't guarantee similarity of predictive relationships between source and target. Examining what exact similarities does DTW identify and how they affect domain adaptation methods will be discussed in later sections.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **4.1 Does selecting only similar source observations improve forecasting performance relative to indiscriminate pooling?**

    In this section we will ask whether using the above DTW algorithm can improve the predictive performance compared to previously presented methods. In particular we are interested in checking if some of the signs indicating negative transfer are mitigated when DTW is applied.

    Knowing the idea behind DTW and how it measures distance it is reasonable to expect a better predictive performance than using source only data or naively pooled data for training.

    For the experiment we will use the same evaluation setup as before with test period being 50 sessions long and reestimating the models every 5 sessions. Intially we will only use $n_T=100$, as this is the smallest value (out of the tested ones) that offers stability of all TO-HAR-EXT models at the beginning of the test period. Moreover, we will consider models using DTW with three different $\varepsilon\in\{25, 50, 75\}$ in order to see how the performance depends on how selective we are when choosing source intervals. Again we will use relative QLIKE and MSE as our metrics. Lastly, it is important to note that for calculation of DTW distance only standard HAR predictors will be used, even though extended feature set may be included in the final model.
    """)
    return


@app.cell(hide_code=True, expand_output=True)
def _(
    benchmark_grid,
    compute_prediction_metrics,
    extended_source_df,
    extended_target_df,
    mo,
    pd,
    run_rolling_predictions,
    trading_sessions,
):
    _dtw_results = []

    for _symbol in ["CRWD", "CHWY"]:
        for _extended, _features in [(False, "STD"), (True, "EXT")]:
            for _percentile in [25, 50, 75]:
                _predictions = run_rolling_predictions(
                    mode="dtw",
                    symbol=_symbol,
                    session_dates=trading_sessions,
                    extended_source_df=extended_source_df,
                    extended_target_df=extended_target_df,
                    extended=_extended,
                    initial_nt=100,
                    refit_freq=5,
                    test_size=50,
                    m=22,
                    dtw_features=["rv_t", "rv_w", "rv_m"],
                    percentile=_percentile
                )

                _overall_metrics, _daily_metric = compute_prediction_metrics(_predictions)
                _row = _overall_metrics.iloc[0]

                _dtw_results.append({
                    "symbol": _symbol,
                    "features": _features,
                    "method": f"DTW-{_percentile}",
                    "mse": _row["mse"],
                    "qlike": _row["qlike"]
                })

    _dtw = pd.DataFrame(_dtw_results)

    _benchmark_grid = benchmark_grid[benchmark_grid["nT"] == 100][["symbol", "features", "method", "mse", "qlike"]].copy()

    _benchmark = pd.concat([_dtw, _benchmark_grid], ignore_index=True)

    _target_metrics = _benchmark[_benchmark["method"] == "Target Only"][["symbol", "features", "mse", "qlike"]].rename(columns={"mse": "target_mse", "qlike": "target_qlike"})

    _benchmark = _benchmark.merge(_target_metrics, on=["symbol", "features"], how="left")
    _benchmark["relative_mse"] = _benchmark["mse"] / _benchmark["target_mse"]
    _benchmark["relative_qlike"] = _benchmark["qlike"] / _benchmark["target_qlike"]

    _method_order = ["Target Only", "Source Only", "Naive Pooling", "DTW-25", "DTW-50", "DTW-75"]
    def _make_relative_loss_table(features):
        _data = _benchmark[
            _benchmark["features"] == features
        ].copy()

        _data["method"] = pd.Categorical(
            _data["method"],
            categories=_method_order,
            ordered=True,
        )

        _data = _data.sort_values("method")

        _table = _data.pivot(
            index="method",
            columns="symbol",
            values=[
                "relative_mse",
                "relative_qlike",
            ],
        )

        return pd.DataFrame(
            {
                "Method": _method_order,
                "CHWY Rel. MSE": [
                    _table.loc[
                        _method,
                        ("relative_mse", "CHWY")
                    ]
                    for _method in _method_order
                ],
                "CHWY Rel. QLIKE": [
                    _table.loc[
                        _method,
                        ("relative_qlike", "CHWY")
                    ]
                    for _method in _method_order
                ],
                "CRWD Rel. MSE": [
                    _table.loc[
                        _method,
                        ("relative_mse", "CRWD")
                    ]
                    for _method in _method_order
                ],
                "CRWD Rel. QLIKE": [
                    _table.loc[
                        _method,
                        ("relative_qlike", "CRWD")
                    ]
                    for _method in _method_order
                ],
            }
        ).set_index("Method")

    def _table_with_min_highlights(dataframe):
        _display_dataframe = dataframe.round(3)
        _column_minimums = _display_dataframe.min(
            numeric_only=True
        )

        def _style_cell(
            row_id,
            column_name,
            value,
        ):
            if column_name not in _column_minimums:
                return {}

            if pd.isna(value):
                return {}

            if value == _column_minimums[column_name]:
                return {
                    "backgroundColor": "#d9f2d9",
                    "color": "#14532d",
                    "fontWeight": "bold",
                }

            return {}

        return mo.ui.table(
            data=_display_dataframe,
            selection=None,
            style_cell=_style_cell,
        )

    _std_table = _make_relative_loss_table("STD")
    _ext_table = _make_relative_loss_table("EXT")

    mo.vstack(
        [
            mo.md("#### **Table 1 - HAR-STD Forecasting Performance**"),
            _table_with_min_highlights(_std_table),
            mo.md("#### **Table 2 - HAR-EXT Forecasting Performance**"),
            _table_with_min_highlights(_ext_table),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we can see in the above tables models using DTW achieve the best forecasting performance across every metric and asset.

    For HAR-STD, DTW achieves slightly lower MSE and QLIKE for both target assets. Similar improvements are observed for HAR-EXT.

    An interesting case arises for CHWY under HAR-EXT, where Naive Pooling exhibits negative transfer in terms of QLIKE, while all DTW variants reduce the relative loss below one. This suggests that similarity-based source selection can mitigate negative transfer in certain settings.

    Overall, the results support the usefullness of DTW source transfer, but the relatively small imporvements and variation across selection thresholds raise an important question: Why does DTW improve forecasting? We investigate the question in the following section.

    Lastly, it's important to remember that there is no guarantee for the results' statistical significance. In particular, the test consists of only 50 samples, so the results may still be noisy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **5. What does DTW actually change?**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **5.1 Marginal covariate shift**

    We first want to examine if covariates in the intervals selected from the source using DTW have similar unconditional distribution to target covariates. In total two experiments will be run regarding this matter. Similarly to before we will only compare distributions of the standard HAR covariates, so $X=(RV_d, RV_w, RV_m)$.

    Recall from earlier sections the construction of our evaluation setup and how it depends on the value $n_T$. We will try to check if DTW mitigates the distribution shift both in-sample and out-of-sample. Notice that in this experiment the value of $n_T$ doesn't matter as long as it is greater than or equal to $22$, that's because the test size is fixed at $50$ and we don't actually use any target observations apart from the $22$ ones directly preceeding test period. Let us denote:
    \[
        X_{all} \text{ - contains all source observations from all assets that fall into training set,}
    \]
    \[
        X_{DTW} \text{ - contains source observations selected by DTW with } m=22 \text{ from } X_{all},
    \]
    \[
        X_{local} \text{ - contains last } m=22 \text{ target observations, so the ones used as a reference for DTW},
    \]
    \[
        X_{OOS} \text{ - contains all of the test target observations available}.
    \]
    For a given distribution discrepancy measure $\mathcal D$ we will try to verify the following hypotheses:
    1. $\mathcal D(P(X_{DTW}), P(X_{local})) < \mathcal D(P(X_{all}), P(X_{local}))$
    2. $\mathcal D(P(X_{DTW}), P(X_{OOS})) < \mathcal D(P(X_{all}), P(X_{OOS}))$

    For this, and any future experiments examining distribution changes we will use three metrics, that will now be briefly introduced.
    1. Mean discrepancy given by: $D_\mu(S, T) = ||\hat\mu_S - \hat\mu_T||_2$- a very simple metric that compares only the first moments of the distributions and is easy to interpret. However, it doesn't take into the account the shape of the distributions outside of their means.
    2. Covariance discrepancy given by: $D_\Sigma(S,T) = ||\hat\Sigma_S - \hat\Sigma_T||_F$, which measures the similarity of variances of HAR predictors and similarity of their codependence.
    3. Maximum Mean Discrepancy (MMD) - a kernel-based measure that compares distributions beyond their first two moments. For two samples $S$ and $T$, its squared empirical value is defined as:

    $$\widehat{\operatorname{MMD}}^2(S,T)=\frac{1}{N_S^2}\sum_{i,j=1}^{N_S}k(x_{S,i},x_{S,j})+\frac{1}{N_T^2}\sum_{i,j=1}^{N_T}k(x_{T,i},x_{T,j})-\frac{2}{N_SN_T}\sum_{i=1}^{N_S}\sum_{j=1}^{N_T}k(x_{S,i},x_{T,j}),$$

    where $N_S$ and $N_T$ denote sample sizes and $k$ is a kernel function.

    We use the Gaussian RBF kernel:

    $$k(x,y)=\exp\left(-\frac{||x-y||_2^2}{2\sigma^2}\right),$$

    where the bandwidth $\sigma$ is set to the median Euclidean distance between observations from $X_{all}$ and $X_{local}$ and kept fixed across all comparisons. This kernel allows MMD to capture differences in the overall shape of multivariate distributions, although its results are sensitive to bandwidth selection and small sample sizes.

    Together, these metrics provide a complementary assessment of distributional similarity. Mean and covariance discrepancies offer simple, interpretable diagnostics, while MMD captures more complex differences that the first two moments may miss.
    """)
    return


@app.cell(hide_code=True)
def _(
    compute_distribution_statistics,
    extended_source_df,
    extended_target_df,
    mo,
    pd,
    trading_sessions,
):
    _dtw_features = ["rv_t", "rv_w", "rv_m"]
    _m = 22
    _symbols = ["CHWY", "CRWD"]
    _percentiles = [25, 50, 75]

    _distribution_results = []

    for _symbol in _symbols:
        for _percentile in _percentiles:
            _distribution_results.append(
                compute_distribution_statistics(
                    extended_source_df=extended_source_df,
                    extended_target_df=extended_target_df,
                    session_dates=trading_sessions,
                    symbol=_symbol,
                    dtw_features=_dtw_features,
                    m=_m,
                    percentile=_percentile,
                )
            )

    distribution_results = pd.concat(
        _distribution_results,
        ignore_index=True,
    )


    def make_relative_distribution_table(results, symbol):
        _symbol_results = results[
            results["symbol"] == symbol
        ]

        _dtw_local = _symbol_results[
            _symbol_results["comparison"] == "DTW vs Local"
        ].set_index("percentile")
        _all_local = _symbol_results[
            _symbol_results["comparison"] == "All vs Local"
        ].set_index("percentile")
        _dtw_oos = _symbol_results[
            _symbol_results["comparison"] == "DTW vs OOS"
        ].set_index("percentile")
        _all_oos = _symbol_results[
            _symbol_results["comparison"] == "All vs OOS"
        ].set_index("percentile")

        _rows = []

        for _percentile in _percentiles:
            for _comparison, _dtw, _all in [
                ("Local", _dtw_local, _all_local),
                ("OOS", _dtw_oos, _all_oos),
            ]:
                _rows.append(
                    {
                        "percentile": _percentile,
                        "comparison": _comparison,
                        "relative_mean_discrepancy": (
                            _dtw.loc[_percentile, "mean_discrepancy"]
                            / _all.loc[_percentile, "mean_discrepancy"]
                        ),
                        "relative_covariance_discrepancy": (
                            _dtw.loc[_percentile, "covariance_discrepancy"]
                            / _all.loc[_percentile, "covariance_discrepancy"]
                        ),
                        "relative_mmd": (
                            _dtw.loc[_percentile, "mmd"]
                            / _all.loc[_percentile, "mmd"]
                        ),
                    }
                )

        return pd.DataFrame(_rows).round(3)


    _chwy_relative_results = make_relative_distribution_table(distribution_results, "CHWY")
    _crwd_relative_results = make_relative_distribution_table(distribution_results, "CRWD")

    mo.vstack(
        [
            mo.md("#### **Table 3 - Summary of distribution discrepancy metrics for CHWY**"),
            mo.ui.table(_chwy_relative_results, selection=None),
            mo.md("#### **Table 4 - Summary of distribution discrepancy metrics for CRWD**"),
            mo.ui.table(_crwd_relative_results, selection=None),
        ]
    )
    return (make_relative_distribution_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The tables report DTW discrepancies relative to all available source observations. Values below one indicate that DTW achieves a smaller discrepancy, while values above one indicate a larger discrepancy. `Local` refers to the DTW reference window, and `OOS` to the out-of-sample target period.

    What we can notice first is that all of the values are smaller than one. That means that in general the sample created using the DTW distance is closer to the target sample in terms of similarity of their distribution, regardless of the version of the target sample and percentile.

    Secondly, we can see that smaller values of $\varepsilon$ result in smaller discrepancies in most cases, apart from the CRWD local discrepancy. This suggests that testing smaller values of $\varepsilon$ might be reasonable.

    Moreover, the DTW distance approximates the `OOS` sample better than the `Local` one. This is surprising, as one might expect the $X_{all}$ distribution to be closer to the $X_{OOS}$ than to $X_{local}$, which would make the relative discrepancy of $X_{DTW}$ bigger. That isn't the case though, and an explanation of this phenomenon will not be examined at this time.

    Lastly, we can see that the difference between discrepancies across assets is significant. This leads to questions regarding the generalization of the DTW method for a broader range of the assets. This might however not be caused by the method itself, but rather by the selected universe of source assets, as they might just be more suited to CHWY than CRWD, allowing for better DTW based selection.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we will answer a question regarding significancy of the above results. It could be possible that the reduction of the distribution isn't a result of the way DTW selects samples, but is caused by the reduction of the sample size.

    So now we will compare the discrepancy of DTW selected sample to discrepancy of a randomly selected sample of the same size. For simplicity, we will focus on comparing to the `OOS` sample and we will only do that for $\varepsilon=25$. The rest of the setup stays exactly the same as previously.

    For each target asset we first construct the set of intervals selected by DTW-25, suppose there is $K$ of these intervals. We then randomly (without replacement) choose $K$ intervals out of all possible source intervals and compute the relative discrepancies similarly to previous experiment.

    The last step will be repeated $B=350$ times and we will report the means, 5th and 95th percentiles and the percent of results where discrepancy was less than or equal to the DTW one.
    """)
    return


@app.cell(hide_code=True)
def _(
    compute_monte_carlo,
    extended_source_df,
    extended_target_df,
    mo,
    pd,
    trading_sessions,
):
    _monte_carlo_results = []

    for _symbol in ["CRWD", "CHWY"]:
        _monte_carlo_results.append(compute_monte_carlo(extended_source_df, extended_target_df, trading_sessions, _symbol, ["rv_t", "rv_w", "rv_m"], 22, 25))

    monte_carlo_results = pd.concat(_monte_carlo_results, ignore_index=True).drop(columns=["n_repetitions", "n_intervals"])

    mo.vstack([
        mo.md("#### **Table 5 - Marginal Covariate Discrepancy: DTW vs Random Source Selection**"),
        monte_carlo_results
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we can see above DTW-based source selection consistently reduces marginal covariate discrepancy relative to both the complete source pool and randomly selected subsets of equal size. This effect can be observed for both target assets and all considered metrics. So, we can say that the reduction of discrepancies across metrics is not caused by the smaller sample size.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **5.2 Marginal label shift**

    So far we have managed to show that DTW-based source selection reduces the discrepancy between marignal covariate distributions. Since when selecting the intervals for transfer, DTW takes the covariate values into account it is not that surprising. Now we will tackle a more interesting question:

    Does DTW-based source selection also reduce the discrepancy between source and target label distribution?

    Since DTW doesn't use the labels for selection it wouldn't be unexpected if it didn't reduce the label distribution shift. On the other hand, if the discrepancies get smaller, that would indicate that similarity in historical volatility patterns translates into similarity in the distribution of future realized volatility.

    The experiment follows the same setup as the marginal covariate analysis, except that we now compare the distributions of $Y=RV_{t+1}$ rather than the HAR predictors. We retain the same DTW-selected source intervals and evaluate the mean, variance, and MMD discrepancies relative to the complete source dataset.

    For each target asset, we consider the same selection thresholds $\varepsilon\in\{25,50,75\}$ and compare the selected source labels with both local and out-of-sample target labels. The local sample contains 21 known labels corresponding to the first 21 observations of the 22-day DTW reference window. The final label is excluded because its realization is not yet available at the selection date. Out-of-sample labels are used exclusively for retrospective evaluation and do not influence DTW selection or kernel bandwidth estimation.
    """)
    return


@app.cell(hide_code=True)
def _(
    compute_label_distribution_statistics,
    extended_source_df,
    extended_target_df,
    make_relative_distribution_table,
    mo,
    pd,
    trading_sessions,
):
    _label_distribution_results = []

    for _symbol in ["CHWY", "CRWD"]:
        for _percentile in [25, 50, 75]:
            _label_distribution_results.append(
                compute_label_distribution_statistics(
                    extended_source_df=extended_source_df,
                    extended_target_df=extended_target_df,
                    session_dates=trading_sessions,
                    symbol=_symbol,
                    dtw_features=["rv_t", "rv_w", "rv_m"],
                    m=22,
                    percentile=_percentile,
                )
            )

    label_distribution_results = pd.concat(
        _label_distribution_results,
        ignore_index=True,
    )

    _chwy_label_relative_results = make_relative_distribution_table(
        label_distribution_results,
        "CHWY",
    )

    _crwd_label_relative_results = make_relative_distribution_table(
        label_distribution_results,
        "CRWD",
    )

    mo.vstack(
        [
            mo.md("#### **Table 6 - Label Distribution Discrepancy: CHWY**"),
            mo.ui.table(_chwy_label_relative_results, selection=None),
            mo.md("#### **Table 7 - Label Distribution Discrepancy: CRWD**"),
            mo.ui.table(_crwd_label_relative_results, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The results suggests that DTW-based source selection reduces marginal lebale distribution discrepancies for both target assets, despite using only covariates during selection. Every reported value is below one and the largest improvements are generally observed for the most selective variant of DTW.

    In particular, for the out-of-sample target distributions, DTW-25 reduces MMD by approximately 20% for CHWY and 14% for CRWD. This suggests that similarity in historical volatility patterns may also translate into similarity in the distributions of future realized volatility. However, it does not necessarily imply that the conditional predictive relationships between covariates and labels are preserved.

    As in the previous experiment, we cannot yet determine whether these improvements are specific to DTW-based selection or could also arise from selecting a smaller subset of source observations. To investigate this, we repeat the Monte Carlo randomization experiment, comparing DTW-25 with randomly selected sets containing the same number of source intervals.
    """)
    return


@app.cell(hide_code=True)
def _(
    compute_label_monte_carlo,
    extended_source_df,
    extended_target_df,
    mo,
    pd,
    trading_sessions,
):
    _label_monte_carlo_results = []

    for _symbol in ["CHWY", "CRWD"]:
        _label_monte_carlo_results.append(
            compute_label_monte_carlo(
                extended_source_df=extended_source_df,
                extended_target_df=extended_target_df,
                session_dates=trading_sessions,
                symbol=_symbol,
                dtw_features=["rv_t", "rv_w", "rv_m"],
                m=22,
                percentile=25,
            )
        )

    label_monte_carlo_results = pd.concat(_label_monte_carlo_results, ignore_index=True).drop(columns=["n_repetitions", "n_intervals"])
    mo.vstack(
        [
            mo.md("#### **Table 8 - Label Shift: DTW vs Random Source Selection**"),
            mo.ui.table(label_monte_carlo_results, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Monte Carlo results confirm that DTW-25 achieves lower marginal label discrepancies than randomly selected source subsets of equal size. For both assets and all three metrics, none of the random selections performed better than DTW. This suggests that the observed improvement is attributable to similarity-based selection rather than merely reducing the sample size.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **5.3 Predictive conditional shift**
    We have managed to empirically show that DTW-based source selections leads to decrease in both marginal covariate distribution shift and label distribution shift. However, none of these results imply that the predictive relationship between covariates and labels is stable across source and target when using DTW. In other words, the predictive mechanism in target domain may use similar values of features, but work completely different, even after DTW-based source selection.

    More formally we want to check if
    \[
     P_S(Y|X)\approx P_T(Y|X),
    \]
    and how the relationship changes when source is selected using the DTW distance.

    Full conditional distributions are difficult to estimate reliably with few observations, especially in the case of regression. Because of that we will treat discrepancies in the conditional expectation functions across target and source as a proxy for discrepancies in their distributions.

    We will train three HAR models using different training datasets similarly to previous approaches:
    1. HAR-All will be trained on all of the available source data preceeding test period,
    2. HAR-DTW will be trained on pooled source observations selected by DTW, without using target observations
    3. HAR-Target will only use available target observations

    Assuming that the relationship between X and Y on both source and target datasets is described correctly by the linear HAR model, evaluating the trained models on given observations should yield good estimation of the conditional expectation.

    Let $m_S(x)$, $m_{S_{\mathrm{DTW}}}(x)$, and $m_T(x)$ denote the conditional mean functions associated with the original source distribution, DTW-selected source distribution, and target distribution, respectively.

    Their corresponding HAR-based estimates are denoted by $\hat m_S(x)$, $\hat m_{S_{\mathrm{DTW}}}(x)$, and $\hat m_T(x)$.

    To quantify the differences between these conditional mean functions, we evaluate all three fitted models on the same set of target covariates $X_1,\ldots,X_n$ and compute the root mean squared discrepancy:

    $$
    D(A,T)=\sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(\hat m_A(X_i)-\hat m_T(X_i)\right)^2},
    $$

    where $A\in\{S,S_{\mathrm{DTW}}\}$. Thus, we measure the discrepancy between the target conditional mean estimate and each of the two source conditional mean estimates using identical covariate observations.

    The choice of target covariates is motivated by the relationship between conditional mean differences and forecasting risk. Under squared-error loss, the excess target risk of using the true source conditional mean instead of the optimal target conditional mean satisfies

    $$
    R_T(m_A)-R_T(m_T)
    =
    E_T\left[(m_A(X)-m_T(X))^2\right].
    $$

    Here, $R_T(f)=E_T[(Y-f(X))^2]$ denotes the expected squared prediction error in the target domain. Therefore, differences between the true conditional mean functions, averaged over the target covariate distribution, quantify the additional target prediction error associated with using the source conditional mean instead of the optimal target conditional mean.

    This motivates our empirical discrepancy measure, although the identity applies to the true conditional means, whereas our experiment uses their HAR-based estimates.

    Finally, we quantify the effect of DTW selection through the relative discrepancy:

    $$
    D_{\mathrm{rel}}=\frac{D(S_{\mathrm{DTW}},T)}{D(S,T)}.
    $$

    Values below one indicate that DTW selection reduces the discrepancy between the estimated conditional mean functions, whereas values above one indicate an increase.

    Importantly, this analysis only examines the first conditional moment under a linear HAR specification. It does not establish equality of the full conditional distributions or verify the covariate shift assumption.
    """)
    return


@app.cell(hide_code=True)
def _(
    compute_conditional_shift_statistics,
    extended_source_df,
    extended_target_df,
    mo,
    pd,
    trading_sessions,
):
    _conditional_results = []

    for _symbol in ["CHWY", "CRWD"]:
        for _extended, _features in [
            (False, "STD"),
            (True, "EXT"),
        ]:
            for _percentile in [25, 50, 75]:
                _conditional_results.append(
                    compute_conditional_shift_statistics(
                        extended_source_df=extended_source_df,
                        extended_target_df=extended_target_df,
                        session_dates=trading_sessions,
                        symbol=_symbol,
                        dtw_features=[
                            "rv_t",
                            "rv_w",
                            "rv_m",
                        ],
                        m=22,
                        percentile=_percentile,
                        extended=_extended,
                        initial_nt=100,
                    )
                )

    conditional_shift_results = pd.concat(
        _conditional_results,
        ignore_index=True,
    )

    mo.vstack([
        mo.md("#### Table 9 - Predictive conditional shift"),
        conditional_shift_results
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The results indicate that DTW-based source selection does not improve the agreement between the estimated source and target conditional mean functions. For both CHWY and CRWD, the relative discrepancy exceeds one, with larger deviations under more restrictive DTW selection.

    This contrasts with the previous findings of improved marginal alignment and forecasting performance. One possible explanation is that the target-only HAR model, estimated from a limited historical sample, is an imperfect approximation of the underlying target conditional mean. Consequently, greater agreement with this reference model does not necessarily imply better out-of-sample predictions.

    These findings suggest that improved marginal alignment and forecasting performance need not coincide with greater agreement between estimated conditional mean functions. However, they do not establish that DTW increases the true predictive conditional distribution shift.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **5.4 Reverse conditional shift**
    """)
    return


if __name__ == "__main__":
    app.run()
