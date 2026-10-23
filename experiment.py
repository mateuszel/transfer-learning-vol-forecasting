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
        run_benchmark_grid
    )

    from preprocessing import build_trading_sessions

    return (
        alt,
        build_trading_sessions,
        compare_feature_set_metrics,
        compare_metric_tables,
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

    For the experiment we will use the same evaluation setup as before with test period being 50 sessions long and reestimating the models every 5 sessions. Intially we will only use $n_T=100$, as this is the smallest value (out of the tested ones) that offers stability of all TO-HAR-EXT models at the beginning of the test period. Moreover, we will consider models using DTW with three different $\varepsilon\in\{25, 50, 75\}$ in order to see how the performance depends on how selective we are when choosing source intervals. Again we will use relative QLIKE and MSE as our metrics.
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


if __name__ == "__main__":
    app.run()
