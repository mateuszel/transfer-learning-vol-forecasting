import pandas as pd
import statsmodels.api as sm
import numpy as np
import altair as alt

def get_next_session_date(current_date, session_dates):
    """
    Inputs:
        current_date: Date - date of the current trading session
        session_dates: pd.DataFrame - contains dates of all trading sessions,
        in the format returned by build_trading_sessions()
    Outputs:
        Date or None - date of the first trading session after current_date,
        or None if current_date is the last available session
    """
    current_date = pd.Timestamp(current_date).date()
    dates = pd.to_datetime(session_dates["date"]).dt.date
    next_dates = dates[dates > current_date]

    if next_dates.empty:
        return None

    return next_dates.min()

def get_previous_session_date(current_date, session_dates):
    """
    Inputs:
        current_date: Date - date of the current trading session
        session_dates: pd.DataFrame - contains dates of all trading sessions,
        in the format returned by build_trading_sessions()
    Outputs:
        Date or None - date of the last trading session before current_date,
        or None if current_date is the first available session
    """

    current_date = pd.Timestamp(current_date).date()
    dates = pd.to_datetime(session_dates["date"]).dt.date
    previous_dates = dates[dates < current_date]

    if previous_dates.empty:
        return None

    return previous_dates.max()

def get_target_only_training_data(symbol, nt, extended_target_df, extended):
    """
    Inputs:
        symbol: string - name of the symbol for which we want to get target only training data
        nt: int - number of training observations
        extended_target_df: pd.DataFrame - contains all observations for both target assets
        extended: bool - True if the returned object should contain extended HAR features
    Outputs:
        pd.DataFrame - contains first nt observations for the given target asset, with NaN's dropped
        additionally a column for a constant is added
        we choose nt-1 because this also contains labels so if we got all nt we would leak information about nt+1 label
        pd.Series - contains labels for the observations from the other DataFrame
    """

    df = extended_target_df.copy()

    # filter out the other symbol
    df = df[df["symbol"] == symbol]

    # filter to only have first nt observations
    # data is assumed to be sorted by date, hence we can just do this
    df = df.head(nt-1)

    # get rid of the NaN's
    df = df.dropna()

    # select columns
    df = df.drop(columns=["t", "symbol"])

    # if we dont want extended features drop them
    if not extended:
        df = df.drop(columns=["vix", "us3m", "MOM", "DVC", "hsi_sqr"])

    df = sm.add_constant(df)
    
    return df.drop(columns=["rv_next"]), df["rv_next"]
    
def get_source_only_training_data(forecast_date, session_dates, extended_source_df, extended):
    """
    Inputs:
        forecast_date: Date - date of the closest forecast, we only take prior to this point
        session_dates: pd.DataFrame - contains dates of all trading sessions,
        in the format returned by build_trading_sessions()
        extended_source_df: pd.DataFrame - dataframe of all source data
        extended: bool - True if we want the extended HAR features
    Outputs:
        pd.DataFrame - contains all source observations up to time forecast_date-2, we choose -2 because
        the labels would leak the future otherwise
        pd.Series - contains labels (next day's RVs) for every observation
    """

    df = extended_source_df.copy()

    # filter out the dates
    # last observation picked for every symbol should be no later than forecast_day - 2 days
    # because the label of that observation is from the day ahead
    cutoff_date = get_previous_session_date(forecast_date, session_dates)
    df = df[df["t"] < cutoff_date]

    # get rid of the NaN's
    df = df.dropna()

    # select columns
    df = df.drop(columns=["t", "symbol"])

    # if we dont want extended features drop them
    if not extended:
        df = df.drop(columns=["vix", "us3m", "MOM", "DVC", "hsi_sqr"])

    df = sm.add_constant(df)
    
    return df.drop(columns=["rv_next"]), df["rv_next"]

def get_naive_pooling_training_data(symbol, forecast_date, session_dates, nt, extended_source_df, extended_target_df, extended):
    """
    Inputs:
        symbol: string - name of the symbol for which we want to get target only training data
        forecast_date: Date - date of the closest forecast, we only take prior to this point
        nt: int - number of target training observations
        session_dates: pd.DataFrame - contains dates of all trading sessions,
        in the format returned by build_trading_sessions()
        extended_source_df: pd.DataFrame - dataframe of all source data
        extended_target_df: pd.DataFrame - contains all observations for both target assets
        extended: bool - True if we want the extended HAR features
    Outputs:
        pd.DataFrame - pooled source and target training data without any information about forecast_date data
        pd.Series - labels for the above training dataset
    """

    # get the source only and target only datasets
    X_target, y_target = get_target_only_training_data(symbol, nt, extended_target_df, extended)
    X_source, y_source = get_source_only_training_data(forecast_date, session_dates, extended_source_df, extended)

    # combine the two datasets, they should have the exact same columns
    X_train = pd.concat([X_target, X_source], ignore_index=True)
    y_train = pd.concat([y_target, y_source], ignore_index=True)

    return X_train, y_train

def fit_har_model(X_train, y_train):
    """
    Inputs:
        X_train: pd.DataFrame - training dataset, with constant column. output of get_target_only_training_data(),
        get_source_only_training_data(), get_naive_pooling_training_data()
        y_train: pd.Series - labels for the X_train dataset
    Outputs:
        RegressionResultsWrapper - fitted OLS model using statsmodels.api
    """

    return sm.OLS(y_train, X_train).fit()

def predict_har_model(model, X_test, test_dates, session_dates):
    """
    Inputs:
        model: RegressionResultsWrapper - output of fir_har_model(X_train, y_train)
        X_test: pd.DataFrame - contains observations for forecast without their dates and without constant
        test_dates: pd.Series - contains dates of the observations we are using to predict next days RV
        session_dates: output of build_trading_sessions()
    Outputs:
        pd.Series - predictions made by the model, index of each result is the date of the session following
        the time of the X_test date
    """

    # get the dates of the predictions
    next_dates = [get_next_session_date(date, session_dates) for date in test_dates]

    # predict on the set
    predictions = model.predict(sm.add_constant(X_test, has_constant="add"))

    return pd.Series(predictions.to_numpy(), index=pd.DatetimeIndex(next_dates), name="rv_pred")

def run_rolling_predictions(mode, symbol, session_dates, extended_source_df, extended_target_df, extended, initial_nt=40, refit_freq=5):
    """
    Inputs:
        mode: string - "target_only" or "source_only" or "naive_pooling" 
        symbol: string - symbol of the target we want to predict
        session_dates: pd.DataFrame - output of build_trading_sessions(...)
        extended_source_df: pd.DataFrame - dataframe of all source data
        extended_target_df: pd.DataFrame - contains all observations for both target assets
        extended: bool - True if we want the extended HAR features
        initial_nt: int - how many trading sessions of the target we allow for training, first prediction is based on
        the data from exactly initial_nt'th session
        refit_freq: int - tells us how often we refit the model to include new data, 
    Outputs:
        pd.DataFrame - contains prediction and diagnostics data, exact columns should be studied below
    """
    # store all predictions here along with additional information
    predictions = []

    # get the target data and dates available for the target
    target = extended_target_df[extended_target_df["symbol"] == symbol].sort_values("t").reset_index(drop=True)
    target_dates = target["t"].tolist()

    # the first prediction we want to make is based on the data for the initial_nt'th session
    # so first date is target_dates[initial_nt-1]
    # for the training we use nt-1 first trading sessions
    
    refit_id = 0 # used to track when we are using which model over time 

    for forecast_step, current_date in enumerate(target_dates[initial_nt-1: ]):

        # we make forecast for this date
        forecast_date = get_next_session_date(current_date, session_dates)

        if forecast_date is None: 
            break

        # number of target sessions we can use without leaking from the future
        nt = initial_nt + forecast_step

        if forecast_step % refit_freq == 0:
            # prepare data if refit needed using leakage free methods
            if mode == "target_only":
                X_train, y_train = get_target_only_training_data(symbol, nt, extended_target_df, extended)
            elif mode == "source_only":
                X_train, y_train = get_source_only_training_data(forecast_date, session_dates, extended_source_df, extended)
            elif mode == "naive_pooling":
                X_train, y_train = get_naive_pooling_training_data(symbol, forecast_date, session_dates, nt, extended_source_df, extended_target_df, extended)
        
            # refit
            model = fit_har_model(X_train, y_train)
            # both below mostly just for diagnostics
            refit_id += 1
            n_train = len(y_train)

        # get current row in order to make a prediction
        current_row = target.iloc[[initial_nt-1 + forecast_step]]

        # prepare the correct columns
        X_test = current_row.drop(columns=["t", "symbol", "rv_next"])

        if not extended:
            X_test = X_test.drop(columns=["vix", "us3m", "MOM", "DVC", "hsi_sqr"])

        y_pred = predict_har_model(model, X_test, current_row["t"], session_dates)

        predictions.append({
            "symbol": symbol,
            "current_date": current_date,
            "forecast_date": forecast_date,
            "y_true": current_row["rv_next"].iloc[0],
            "y_pred": max(1e-8, y_pred.iloc[0]),
            "nt": nt,
            "refit_id": refit_id,
            "n_train": n_train,
            "clipped": y_pred.iloc[0] < 1e-8
        })

    return pd.DataFrame(predictions)

def compute_mse(y_true, y_pred):
    """
    Inputs:
        y_true: pd.Series - true values of RV
        y_pred: pd.Series - predicted values of RV
    Outputs:
        float: the MSE of the predictions
    """
    return ((y_true-y_pred)**2).mean()

def compute_qlike(y_true, y_pred):
    """
        Inputs:
            y_true: pd.Series - true values of RV
            y_pred: pd.Series - predicted values of RV
        Outputs:
            float: the QLIKE of the predictions
        """
    return (y_true / y_pred - np.log(y_true / y_pred) - 1).mean()

def compute_prediction_metrics(predictions, window_size=20):
    """
    Inputs:
        predictions: pd.DataFrame - result of run_rolling_predictions(...)
        window_size: int - length of one window that we compute metrics on
    Outputs:
        pd.DataFrame - one row dataframe that aggregates metrics over the entire evaluation period (all predictions)
        pd.DataFrame - each row defines metric for one evaluation window
    """
    df = predictions.copy()

    # compute squared error and QLIKE for every prediction
    df["sq_error"] = (df["y_true"] - df["y_pred"]) ** 2
    df["qlike"] = (df["y_true"] / df["y_pred"] - np.log(df["y_true"] / df["y_pred"]) - 1)

    # first df to return
    overall_result = pd.DataFrame([{
        "symbol": df["symbol"].loc[0],
        "n_forecasts": len(df),
        "mse": compute_mse(df["y_true"], df["y_pred"]),
        "qlike": compute_qlike(df["y_true"], df["y_pred"])
    }])

    # get the 'windowed' df to return

    # first set window_id for every observation
    df["window_id"] = [i // window_size + 1 for i in range(len(df))]

    # for each window compute the mse and qlike
    grouped = df.groupby("window_id").apply(
        lambda g: pd.Series({
            "mse": compute_mse(g["y_true"], g["y_pred"]),
            "qlike": compute_qlike(g["y_true"], g["y_pred"]),
            "n_forecasts": len(g)
        })
    )
    grouped = grouped[grouped["n_forecasts"] == window_size].reset_index()
    grouped["symbol"] = df["symbol"].loc[0]
    
    return overall_result, grouped

def plot_one_method_predictions(predictions_target1_std, predictions_target1_ext, predictions_target2_std, predictions_target2_ext, method):
    """
    Inputs:
        predictions_target1_std: pd.DataFrame - standard HAR predictions for target 1
        predictions_target1_ext: pd.DataFrame - extended HAR predictions for target 1
        predictions_target2_std: pd.DataFrame - standard HAR predictions for target 2
        predictions_target2_ext: pd.DataFrame - extended HAR predictions for target 2
        method: string - training method used to generate the predictions
    Outputs:
        alt.Chart - four plots comparing true values with STD and EXT predictions
    """

    target1 = predictions_target1_std[["forecast_date", "y_true", "y_pred"]].assign(target=predictions_target1_std["symbol"].iloc[0], model="STD")
    target1_ext = predictions_target1_ext[["forecast_date", "y_true", "y_pred"]].assign(target=predictions_target1_ext["symbol"].iloc[0], model="EXT")
    target2 = predictions_target2_std[["forecast_date", "y_true", "y_pred"]].assign(target=predictions_target2_std["symbol"].iloc[0], model="STD")
    target2_ext = predictions_target2_ext[["forecast_date", "y_true", "y_pred"]].assign(target=predictions_target2_ext["symbol"].iloc[0], model="EXT")

    raw_data = pd.concat([target1, target1_ext, target2, target2_ext], ignore_index=True)
    raw_data["forecast_date"] = pd.to_datetime(raw_data["forecast_date"])

    true_data = raw_data[["forecast_date", "target", "model", "y_true"]].rename(columns={"y_true": "value"}).assign(series="True RV")
    prediction_data = raw_data[["forecast_date", "target", "model", "y_pred"]].rename(columns={"y_pred": "value"}).assign(series=raw_data["model"].values)

    plot_data = pd.concat([true_data, prediction_data], ignore_index=True)

    return (
        alt.Chart(plot_data)
        .mark_line()
        .encode(
            x=alt.X("forecast_date:T", title="Forecast date"),
            y=alt.Y("value:Q", title="Realized variance"),
            color=alt.Color(
                "series:N",
                title=None,
                sort=["True RV", "STD", "EXT"],
                scale=alt.Scale(
                    domain=["True RV", "STD", "EXT"],
                    range=["gray", "steelblue", "firebrick"],
                ),
                legend=alt.Legend(
                    orient="top",
                    direction="horizontal",
                    columns=3,
                ),
            ),
            strokeDash=alt.StrokeDash(
                "series:N",
                title=None,
                sort=["True RV", "STD", "EXT"],
                scale=alt.Scale(
                    domain=["True RV", "STD", "EXT"],
                    range=[[5, 3], [1, 0], [1, 0]],
                ),
                legend=None,
            ),
            row=alt.Row("target:N", title=None),
            column=alt.Column("model:N", sort=["STD", "EXT"], title=None),
            tooltip=[
                "forecast_date:T",
                "target:N",
                "series:N",
                alt.Tooltip("value:Q", format=".6f"),
            ],
        )
        .properties(title=f"{method} predictions", width=350, height=220)
        .resolve_scale(x="shared", y="shared")
        .configure_title(anchor="middle")
    )

def compare_metric_tables(model_metrics, benchmark_metrics, method_name, benchmark_name):
    """
    Inputs:
        model_metrics: pd.DataFrame) - desired model metrics computed by compute_prediction_metric(...)
        benchmark_metrics: pd.DataFrame) - benchmark model metrics computed by compute_prediction_metric(...)
        method_name: string - identifier / name of the method
        benchmark_name: string - identifier / name of the benchmark
    Outputs:
        pd.DataFrame - comparison of the two models with columns:
        symbol, window_id  (if not none), method, benchmark, model_mse, benchmark_mse,
        model_qlike, benchmark_qlike, relative_mse, relative_qlike, 
        mse_gap, qlike_gap
    """
    # clean up dataframes and extract some data
    symbol = model_metrics["symbol"].loc[0]
    is_windowed = "window_id" in model_metrics.columns
    
    model_metrics = model_metrics.rename(columns={
            "mse": f"{method_name}-MSE",
            "qlike": f"{method_name}-QLIKE",
            "n_forecasts": f"{method_name}-n_forecasts"
        })
    model_metrics = model_metrics.drop(columns=["symbol"])

    benchmark_metrics = benchmark_metrics.rename(columns={
            "mse": f"{benchmark_name}-MSE",
            "qlike": f"{benchmark_name}-QLIKE",
            "n_forecasts": f"{benchmark_name}-n_forecasts"
        })
    benchmark_metrics = benchmark_metrics.drop(columns=["symbol"])
    
    # check whether we got the overall_results or window_results
    if is_windowed:
        # window results
        merged = model_metrics.merge(benchmark_metrics, on="window_id")
    else:
        # overall_results
        merged = pd.concat([model_metrics, benchmark_metrics], axis=1)

    # compute the rest of the metrics
    merged["relative_mse"] = merged[f"{method_name}-MSE"] / merged[f"{benchmark_name}-MSE"]
    merged["mse_gap"] = merged[f"{method_name}-MSE"] - merged[f"{benchmark_name}-MSE"]

    merged["relative_qlike"] = merged[f"{method_name}-QLIKE"] / merged[f"{benchmark_name}-QLIKE"]
    merged["qlike_gap"] = merged[f"{method_name}-QLIKE"] - merged[f"{benchmark_name}-QLIKE"]

    merged["symbol"] = symbol
    merged["method"] = method_name
    merged["benchmark"] = benchmark_name
    
    if "window_id" not in merged.columns:
        return merged[
            [
                "symbol",
                "method",
                "benchmark",
                f"{method_name}-n_forecasts",
                f"{method_name}-MSE",
                f"{benchmark_name}-MSE",
                "relative_mse",
                "mse_gap",
                f"{method_name}-QLIKE",
                f"{benchmark_name}-QLIKE",
                "relative_qlike",
                "qlike_gap",
            ]
        ]
    return merged[
        [
            "symbol",
            "window_id",
            "method",
            "benchmark",
            f"{method_name}-n_forecasts",
            f"{benchmark_name}-n_forecasts",
            f"{method_name}-MSE",
            f"{benchmark_name}-MSE",
            "relative_mse",
            "mse_gap",
            f"{method_name}-QLIKE",
            f"{benchmark_name}-QLIKE",
            "relative_qlike",
            "qlike_gap",
        ]
    ]

def compare_feature_set_metrics(std_metrics, ext_metrics, method_name):
    """
    Inputs:
        std_metrics: pd.DataFrame - metrics for a model with standard feature set output of compute_predictions_metrics(...)
        ext_metrics: pd.DataFrame - metrics for a model with extended feature set output of compute_predictions_metrics(...)
        method_name: string - name of the method used for both above models
    Outputs:
        pd.DataFrame - comparison of two models with different feature spaces using the same methods with columns
        window_id, EXT-n_forecasts, STD-n_forecasts (those 3 only if a windowed metrics are passed)
        symbol, method, STD-MSE, EXT-MSE, relative_mse, mse_gap, STD-QLIKE, EXT-QLIKE, relative_qlike, qlike_gap
    """
    is_windowed = "window_id" in std_metrics.columns
    symbol = std_metrics["symbol"].iloc[0]
    
    # clean up data frames and rename columns
    std_metrics = std_metrics.rename(columns={
        "mse": "STD-MSE",
        "qlike": "STD-QLIKE",
        "n_forecasts": "STD-n_forecasts"
    })
    std_metrics = std_metrics.drop(columns=["symbol"])

    ext_metrics = ext_metrics.rename(columns={
        "mse": "EXT-MSE",
        "qlike": "EXT-QLIKE",
        "n_forecasts": "EXT-n_forecasts"
    })
    ext_metrics = ext_metrics.drop(columns=["symbol"])

    # check if there is window id
    if is_windowed:
        merged = std_metrics.merge(ext_metrics, on="window_id")
    else:
        merged = pd.concat([std_metrics, ext_metrics], axis=1)

    # compute rest of the metrics
    
    merged["relative_mse"] = merged["EXT-MSE"] / merged["STD-MSE"]
    merged["mse_gap"] = merged["EXT-MSE"] - merged["STD-MSE"]

    merged["relative_qlike"] = merged["EXT-QLIKE"] / merged["STD-QLIKE"]
    merged["qlike_gap"] = merged["EXT-QLIKE"] - merged["STD-QLIKE"]

    merged["symbol"] = symbol
    merged["method"] = method_name
    
    if "window_id" not in merged.columns:
        return merged[
            [
                "symbol",
                "method",
                "STD-n_forecasts",
                "STD-MSE",
                "EXT-MSE",
                "relative_mse",
                "mse_gap",
                "STD-QLIKE",
                "EXT-QLIKE",
                "relative_qlike",
                "qlike_gap",
            ]
        ]
    return merged[
        [
            "symbol",
            "window_id",
            "method",
            "STD-n_forecasts",
            "EXT-n_forecasts",
            "STD-MSE",
            "EXT-MSE",
            "relative_mse",
            "mse_gap",
            "STD-QLIKE",
            "EXT-QLIKE",
            "relative_qlike",
            "qlike_gap",
        ]
    ]

