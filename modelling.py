import pandas as pd
import statsmodels.api as sm
import numpy as np
import altair as alt
from tslearn.metrics import dtw
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import cdist

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

def get_target_only_training_data(symbol, nt, extended_target_df, extended, start_idx):
    """
    Inputs:
        symbol: string - name of the symbol for which we want to get target only training data
        nt: int - number of training sessions (observations) before dropping NaN's
        extended_target_df: pd.DataFrame - contains all observations for both target assets
        extended: bool - True if the returned object should contain extended HAR features
        start_idx: int - Index of the first observation we allow for training 
    Outputs:
        pd.DataFrame - contains first nt observations for the given target asset starting at start_idx, with NaN's dropped
        additionally a column for a constant is added
        we choose nt-1 because this also contains labels so if we got all nt we would leak information about nt+1 label
        pd.Series - contains labels for the observations from the other DataFrame
    """

    df = extended_target_df.copy()

    # filter out the other symbol
    df = df[df["symbol"] == symbol]

    # filter to only have first nt observations
    # data is assumed to be sorted by date, hence we can just do this
    df = df.iloc[start_idx:start_idx+nt]

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

def get_naive_pooling_training_data(symbol, forecast_date, session_dates, nt, extended_source_df, extended_target_df, extended, start_idx):
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
        start_idx: int - Index of the first observation we allow for training
    Outputs:
        pd.DataFrame - pooled source and target training data without any information about forecast_date data
        pd.Series - labels for the above training dataset
    """

    # get the source only and target only datasets
    X_target, y_target = get_target_only_training_data(symbol, nt, extended_target_df, extended, start_idx)
    X_source, y_source = get_source_only_training_data(forecast_date, session_dates, extended_source_df, extended)

    # combine the two datasets, they should have the exact same columns
    X_train = pd.concat([X_target, X_source], ignore_index=True)
    y_train = pd.concat([y_target, y_source], ignore_index=True)

    return X_train, y_train

def fit_har_model(X_train, y_train):
    """
    Inputs:
        X_train: pd.DataFrame - training dataset, with constant column. output of get_target_only_training_data(),
        get_source_only_training_data(), get_naive_pooling_training_data(), get_dtw_training_data()
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

def run_rolling_predictions(mode, symbol, session_dates, extended_source_df, extended_target_df, extended, initial_nt=40, refit_freq=5, test_size=50, m=22, dtw_features=["rv_t", "rv_w", "rv_m"], percentile=25):
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
        m: int - interval size for DTW if needed
        percentile: int - percentile of the closest DTW observations to add to training set
    Outputs:
        pd.DataFrame - contains prediction and diagnostics data, exact columns should be studied below
    """
    # store all predictions here along with additional information
    predictions = []

    # get the target data and dates available for the target
    target = extended_target_df[extended_target_df["symbol"] == symbol].sort_values("t").dropna(subset=["rv_next"]).reset_index(drop=True)
    target_dates = target["t"].tolist()

    # we need to find the range of the observations we will be working with first
    first_test_idx = len(target) - test_size # this is the observation used for the first forecast
    start_idx = first_test_idx - initial_nt # first target training observation index, we use [start_idx, start_idx+nt-1) training
    
    refit_id = 0 # used to track when we are using which model over time 

    for forecast_step, current_date in enumerate(target_dates[first_test_idx:]):
        # we make forecast for this date
        forecast_date = get_next_session_date(current_date, session_dates)

        if forecast_date is None: 
            break

        # number of target sessions we can use without leaking from the future
        nt = initial_nt + forecast_step

        if forecast_step % refit_freq == 0:
            # prepare data if refit needed using leakage free methods
            if mode == "target_only":
                X_train, y_train = get_target_only_training_data(symbol, nt, extended_target_df, extended, start_idx)
            elif mode == "source_only":
                X_train, y_train = get_source_only_training_data(forecast_date, session_dates, extended_source_df, extended)
            elif mode == "naive_pooling":
                X_train, y_train = get_naive_pooling_training_data(symbol, forecast_date, session_dates, nt, extended_source_df, extended_target_df, extended, start_idx)
            elif mode == "dtw":
                dtw_target = get_target_for_dtw(target, first_test_idx+forecast_step, dtw_features, m)
                dtw_source = get_source_for_dtw(forecast_date, session_dates, extended_source_df)

                target_interval, og_intervals, dtw_intervals = construct_intervals(dtw_target, dtw_source, extended, dtw_features, m)
                selected_ids = compute_dtw(target_interval, dtw_intervals, percentile)

                X_train, y_train = get_dtw_training_data(symbol, nt, extended_target_df, extended, start_idx, og_intervals, selected_ids)

            # refit
            model = fit_har_model(X_train, y_train)
            # both below mostly just for diagnostics
            refit_id += 1
            n_train = len(y_train)

        # get current row in order to make a prediction
        current_row = target.iloc[[first_test_idx + forecast_step]]

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
            "initial_nt": initial_nt,
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

def compute_prediction_metrics(predictions):
    """
    Inputs:
        predictions: pd.DataFrame - result of run_rolling_predictions(...)
    Outputs:
        pd.DataFrame - one row dataframe that aggregates metrics over the entire evaluation period (all predictions)
        pd.DataFrame - each row defines metric for one evaluation window
    """
    df = predictions.sort_values("forecast_date").copy()

    # compute squared error and QLIKE for every prediction
    df["mse"] = (df["y_true"] - df["y_pred"]) ** 2
    df["qlike"] = (df["y_true"] / df["y_pred"] - np.log(df["y_true"] / df["y_pred"]) - 1)

    # comptue cumulative losses per metric over the prediction period
    df["n_forecasts"] = np.arange(1, len(df)+1)
    df["cumulative_mse"] = df["mse"].cumsum() / df["n_forecasts"]
    df["cumulative_qlike"] = df["qlike"].cumsum() / df["n_forecasts"]

    # get the neccessary columns only
    df = df[["symbol", "forecast_date", "n_forecasts", "mse", "qlike", "cumulative_mse", "cumulative_qlike"]].reset_index(drop=True)

    # overall metrics from the entire period
    overall_metrics = pd.DataFrame([{
        "symbol": df["symbol"].iloc[0],
        "n_forecasts": len(df),
        "mse": df["mse"].mean(),
        "qlike": df["qlike"].mean()
    }])

    return overall_metrics, df

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
        symbol, method, benchmark, model_mse, benchmark_mse,
        model_qlike, benchmark_qlike, relative_mse, relative_qlike, 
        mse_gap, qlike_gap
    """
    # clean up dataframes and extract some data
    symbol = model_metrics["symbol"].loc[0]
    
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
    
    merged = pd.concat([model_metrics, benchmark_metrics], axis=1)

    # compute the rest of the metrics
    merged["relative_mse"] = merged[f"{method_name}-MSE"] / merged[f"{benchmark_name}-MSE"]
    merged["mse_gap"] = merged[f"{method_name}-MSE"] - merged[f"{benchmark_name}-MSE"]

    merged["relative_qlike"] = merged[f"{method_name}-QLIKE"] / merged[f"{benchmark_name}-QLIKE"]
    merged["qlike_gap"] = merged[f"{method_name}-QLIKE"] - merged[f"{benchmark_name}-QLIKE"]

    merged["symbol"] = symbol
    merged["method"] = method_name
    merged["benchmark"] = benchmark_name
    
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

def get_target_for_dtw(target, current_index, dtw_features, m):
    """
    Inputs:
        target: pd.DataFrame - observations for target filtered by symbol, sorted by date no index
        current_index: int - the index of the observation used for next prediction
        dtw_featues: list - which features we will use for DTW
        m: int - length of each interval in DTW
    Outputs:
        pd.DataFrame: Dataframe with the target interval for the DTW of length m. 
        If a complete interval of length m can't be created then None
    """

    # access curent observation and all past observations
    target_history = target.iloc[:current_index].tail(m-1) # most recent m-1 observations
    current_row = target.iloc[[current_index]] # mth observation

    target_interval = pd.concat([target_history, current_row], ignore_index=True)

    if len(target_interval) != m:
        return None
    if target_interval[dtw_features].isna().any().any():
        return None

    return target_interval.drop(columns=["symbol", "t"])

def get_source_for_dtw(forecast_date, session_dates, extended_source_df):
    """
    Inputs:
        forecast_date: Date - date of the closest forecast, we only take prior to this point
        session_dates: pd.DataFrame - contains dates of all trading sessions,
        in the format returned by build_trading_sessions()
        extended_source_df: pd.DataFrame - dataframe of all source data
    Outputs:
        pd.DataFrame - all source data eligible for creation of intervals
    """

    cutoff_date = get_previous_session_date(forecast_date, session_dates)
    return extended_source_df[extended_source_df["t"] < cutoff_date].copy().reset_index(drop=True).drop(columns=["t"])

def construct_intervals(dtw_target, dtw_source, extended, dtw_features, m):
    """
    Inputs:
        dtw_target: pd.DataFrame - output of get_target_for_dtw(...), target interval
        dtw_source: pd.DataFrame - output of get_source_for_dtw(...), all data for source
        extended: bool - whether extended HAR features should be included in the final dataset
        dtw_featues: list - which features should be considered in when computing DTW distance
        m: int - length of a single interval
    Outputs:
        pd.DataFrame - target interval ready for DTW with only the dtw_features columns
        dict[int, pd.DataFrame] - dictionary window_id -> source_interval with all training features  and label, with symbol column
        dict[int, pd.DataFrame] - dictionary window_id -> source_dtw_interval with only dtw_features as columns, without labels
    """
    #safeguard in case we cant construct the target interval for dtw
    if dtw_target is None:
        raise Exception("Not enough non-NaN observations for DTW. Try bigger nt")
    
    # first prepare target data for return
    target_interval = dtw_target[dtw_features]

    # now prepare source
    if extended:
        source_df = dtw_source.copy()
    else:
        source_df = dtw_source.drop(columns=["hsi_sqr", "vix", "us3m", "MOM", "DVC"])
    
    # group source by symbol and sort by date again within the symbol to be safe
    grouped = source_df.groupby("symbol")

    dtw_intervals = {} # intervals prepped for dtw along with their id
    og_intervals = {} # id -> original interval (to restore all features)
    window_id = 0 # global window counter for retrieval
    
    for symbol, asset_df in grouped:
        # drop uneven data from the begining
        remainder = len(asset_df) % m # this many rows from the start we have to drop
        asset_df = asset_df.iloc[remainder:].reset_index(drop=True)

        # create intervals
        for start in range(0, len(asset_df), m):
            original_interval = asset_df.iloc[start:start+m].copy()

            # not possible to happen hopefully but still to be safe
            if len(original_interval) < m:
                continue

            # we have something missing - gets discarded
            if original_interval.isna().any().any():
                continue

            # we have full complete interval
            dtw_intervals[window_id] = original_interval[dtw_features].copy()
            og_intervals[window_id] = original_interval.copy()

            # update window counter
            window_id += 1

    return target_interval, og_intervals, dtw_intervals

def compute_dtw(target_interval, dtw_intervals, percentile):
    """
    Inputs:
        target_interval: pd.DataFrame - pd.DataFrame - target interval ready for DTW with only the dtw_features columns, output of construct_intervals(...)
        dtw_intervals: dict[int, pd.DataFrame] - dict[int, pd.DataFrame] - dictionary window_id -> source_dtw_interval with only dtw_features as columns, without labels, output of construct_intervals(...)
    Outputs:
        list - list of window_id's that are in the specified percentile
    """
    # compute all distances
    distances = {
        window_id: dtw(target_interval.to_numpy(), source_interval.to_numpy())
        for window_id, source_interval in dtw_intervals.items()
    }

    # find the threshold corresponding to the percentile
    threshold = np.percentile(list(distances.values()), percentile)

    return [window_id for window_id, distance in distances.items() if distance <= threshold]

def get_dtw_training_data(symbol, nt, extended_target_df, extended, start_idx, og_intervals, selected_ids):
    """
    Inputs:
        symbol, nt, extended_target_df, extended, start_idx - same inputs as for get_target_only_training_data(...)
        og_intervals: dict[int, pd.DataFrame] - dictionary window_id -> source_interval with all training features  and label, with symbol column
        selected_ids: list - output of compute_dtw(...) contains window_id's of all selected intervals
    Outputs:
    
    """
    # get target only data
    X_target, y_target = get_target_only_training_data(symbol, nt, extended_target_df, extended, start_idx) # contains constant

    # get dtw source data
    source_df = pd.concat([og_intervals[window_id] for window_id in selected_ids]).reset_index(drop=True)

    # drop symbol column and add constant column
    source_df = source_df.drop(columns=["symbol"])
    source_df = sm.add_constant(source_df)

    # split into X and y
    X_train = pd.concat([source_df.drop(columns=["rv_next"]), X_target]).reset_index(drop=True)
    y_train = pd.concat([source_df["rv_next"], y_target]).reset_index(drop=True)
    
    return X_train, y_train

def run_benchmark_grid(session_dates, extended_source_df, extended_target_df, nt_grid=(40, 60, 80, 100, 120), symbols=("CHWY", "CRWD"), refit_freq=5, test_size=50):
    """
    Inputs:
        session_dates: pd.DataFrame - contains dates of all trading sessions,
        in the format returned by build_trading_sessions()
        extended_source_df: pd.DataFrame - dataframe containing all source
        asset observations
        extended_target_df: pd.DataFrame - dataframe containing observations
        for all target assets
        nt_grid: iterable[int] - numbers of target observations available at
        the beginning of the evaluation period
        symbols: iterable[str] - symbols of the target assets to evaluate
        refit_freq: int - number of trading sessions between model refits
        test_size: int - number of out-of-sample forecasts for each target
    Outputs:
        pd.DataFrame - long-format benchmark results with one row for each
        combination of target asset, target-history length, feature set and
        training method. The dataframe contains raw MSE and QLIKE together
        with their relative values compared to Target Only using the same
        target asset, target-history length and feature set.
    """
    _results = []
    _methods = {
        "Target Only": "target_only",
        "Source Only": "source_only",
        "Naive Pooling": "naive_pooling",
    }

    for symbol in symbols:
        for nt in nt_grid:
            for extended, features in [(False, "STD"), (True, "EXT")]:
                _metrics = {}
                for method_name, mode in _methods.items():
                    _predictions = run_rolling_predictions(
                        mode=mode,
                        symbol=symbol,
                        session_dates=session_dates,
                        extended_source_df=extended_source_df,
                        extended_target_df=extended_target_df,
                        extended=extended,
                        initial_nt=nt,
                        refit_freq=refit_freq,
                        test_size=test_size,
                    )

                    _overall_metrics, _daily_metrics = compute_prediction_metrics(
                        _predictions
                    )
                    _metrics[method_name] = _overall_metrics.iloc[0]

                _target_mse = _metrics["Target Only"]["mse"]
                _target_qlike = _metrics["Target Only"]["qlike"]

                for method_name, _method_metrics in _metrics.items():
                    _results.append(
                        {
                            "symbol": symbol,
                            "nT": nt,
                            "features": features,
                            "method": method_name,
                            "mse": _method_metrics["mse"],
                            "qlike": _method_metrics["qlike"],
                            "relative_mse": (
                                _method_metrics["mse"] / _target_mse
                            ),
                            "relative_qlike": (
                                _method_metrics["qlike"] / _target_qlike
                            ),
                        }
                    )

    return pd.DataFrame(
        _results,
        columns=[
            "symbol",
            "nT",
            "features",
            "method",
            "mse",
            "qlike",
            "relative_mse",
            "relative_qlike",
        ],
    )

def get_no_label_dtw(extended_source_df, extended_target_df, trading_sessions, symbol, dtw_features, m, percentile):
    """
    Inputs:
        extended_source_df: pd.DataFrame - dataframe of all source data
        extended_target_df: pd.DataFrame - contains all observations for both target assets  
        trading_sessions: pd.DataFrame - contains dates of all trading sessions,
        in the format returned by build_trading_sessions(...)
        symbol: string - name of the symbol for which we want to get target only training data
        dtw_featues: list - which features should be considered in when computing DTW distance
        m: int - length of a single interval
        percentile: int - percentile of the closest DTW observations to add to training set
    Outputs:
        X_all: pd.DataFrame - output of get_source_for_dtw(...) - contains all source training data
        X_dtw: pd.DataFrame - constructed from construct_intervals(...) contains all training data selected by dtw
        X_local: pd.DataFrame - contains last 22 target observations before the test period, used as a reference for dtw
        contains the first observation from test aswell 
        X_oos: pd.DataFrame - contains all of test target observations (50)
    """
    # get the target interval of length m
    target = extended_target_df[extended_target_df["symbol"] == symbol].dropna(subset='rv_next').reset_index(drop=True)

    first_test_idx = len(target) - 50 # features from this index are used in forecast of the first oos test
    current_date = target.loc[first_test_idx, "t"]
    first_forecast_date = get_next_session_date(current_date, trading_sessions) # the date of the first forecast

    X_local = get_target_for_dtw(target, first_test_idx, dtw_features, m)
    X_oos = target.iloc[first_test_idx:][dtw_features]
    X_all = get_source_for_dtw(first_forecast_date, trading_sessions, extended_source_df)

    X_local, og_intervals, dtw_intervals = construct_intervals(X_local, X_all, False, dtw_features, m)
    selected_ids = compute_dtw(X_local, dtw_intervals, percentile)
    X_dtw = pd.concat([og_intervals[_id] for _id in selected_ids]).reset_index(drop=True)[dtw_features]
    X_all = X_all[dtw_features]

    return X_all, X_dtw, X_local, X_oos

def mean_discrepancy(df1, df2):
    """
    Inputs:
        df1: pd.DataFrame - contains observations sampled from the first distribution that have already been filtered in terms of dates
        and only has columns rv_t, rv_w, rv_m
        df2: pd.DataFrame - same as above, but for the other distribution
    Outputs:
        float: The mean discrepancy value
    """
    return np.sqrt(np.sum((df1.mean() - df2.mean())**2))

def covariance_discrepancy(df1, df2):
    """
    Inputs:
        df1: pd.DataFrame - contains observations sampled from the first distribution that have already been filtered in terms of dates
        and only has columns rv_t, rv_w, rv_m
        df2: pd.DataFrame - same as above, but for the other distribution
    Outputs:
        float: The covariance discrepancy value
    """
    return np.sqrt(np.sum((df2.cov() - df1.cov())**2))

def mmd(df1, df2, gamma):
    """
    Inputs:
        df1: pd.DataFrame - contains observations sampled from the first distribution that have already been filtered in terms of dates
        and only has columns rv_t, rv_w, rv_m
        df2: pd.DataFrame - same as above, but for the other distribution
    Outputs:
        float: Maximum Mean Discrepancy value
    """
    X = np.asarray(df1.dropna())
    Y = np.asarray(df2.dropna())

    K_xx = rbf_kernel(X, X, gamma=gamma)
    K_yy = rbf_kernel(Y, Y, gamma=gamma)
    K_xy = rbf_kernel(X, Y, gamma=gamma)

    mmd2 = K_xx.mean() + K_yy.mean() - 2*K_xy.mean()
    return np.sqrt(max(mmd2, 0))

def compute_rbf_bandwidth(X_local, X_all):
    """
    Inputs:
        X_all: pd.DataFrame - output of get_source_for_dtw(...) - contains all source training data
    Outputs:
        X_local: pd.DataFrame - contains last 22 target observations before the test period, used as a reference for dtw
        contains the first observation from test aswell
    """
    X_local = X_local.dropna().to_numpy()
    X_all = X_all.dropna().to_numpy()
    distances = cdist(X_local, X_all, metric="euclidean")

    return 1 / (2*np.median(distances)**2)

def compute_distribution_statistics(extended_source_df, extended_target_df, session_dates, symbol, dtw_features, m, percentile):
    """
    Inputs:
        extended_source_df: pd.DataFrame - dataframe of all source data
        extended_target_df: pd.DataFrame - dataframe of all target data
        session_dates: pd.DataFrame - dates of all trading sessions
        symbol: string - name of the target asset
        dtw_features: list - features used to compute DTW distances
        m: int - length of a single DTW interval
        percentile: int - percentile of the closest DTW intervals
    Outputs:
        pd.DataFrame - distribution statistics for all comparisons
    """
    # get the data for the distribution statistics
    X_all, X_dtw, X_local, X_oos = get_no_label_dtw(
        extended_source_df,
        extended_target_df,
        session_dates,
        symbol,
        dtw_features,
        m,
        percentile,
    )

    gamma = compute_rbf_bandwidth(X_local, X_all)

    # use complete rows for all metrics
    X_all = X_all.dropna()
    X_dtw = X_dtw.dropna()
    X_local = X_local.dropna()
    X_oos = X_oos.dropna()

    comparisons = [
        ("DTW vs Local", X_dtw, X_local),
        ("All vs Local", X_all, X_local),
        ("DTW vs OOS", X_dtw, X_oos),
        ("All vs OOS", X_all, X_oos),
    ]

    results = []

    for comparison, df1, df2 in comparisons:
        results.append(
            {
                "symbol": symbol,
                "percentile": percentile,
                "comparison": comparison,
                "mean_discrepancy": mean_discrepancy(df1, df2),
                "covariance_discrepancy": covariance_discrepancy(df1, df2),
                "mmd": mmd(df1, df2, gamma),
                "gamma": gamma,
                "n_first": len(df1),
                "n_second": len(df2),
            }
        )

    return pd.DataFrame(results)
