import darts.metrics
import pandas as pd
from ts_models import *
from RPI_Forecaster import Subcomponent
from darts.models import ExponentialSmoothing, FFT, AutoARIMA, RNNModel, Theta
from darts.dataprocessing.transformers import Scaler
from darts.utils.statistics import plot_acf
import warnings
import logging
import math
import ast
import xlwings as xw
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)



workstation = "MAC"
if workstation == "PC":
    subcomponents = r"C:\Users\joshb\Desktop\RPI Forecasting\RPI_Subcomponents.xlsx"

if workstation == "MAC":
    subcomponents = r"/Users/Josh/Desktop/RPI_Forecaster/RPI_Subcomponents.xlsx"
descriptions = pd.read_excel(subcomponents)['Description']
subgroups = pd.read_excel(subcomponents)['SubGroup']
hist_data = []
forecast_df = pd.DataFrame([], columns=descriptions)
hist_df = pd.DataFrame([], columns=descriptions)

def retrieve_pickles(codes):
    for idx, code in enumerate(codes[:]):
        code = Subcomponent(codes.iloc[idx], descriptions.iloc[idx], subgroups.iloc[idx])
        if workstation == "PC":
            fname = r"C:\Users\joshb\Desktop\RPI Forecasting\Pickles\filename_{:02d}.pkl".format(idx + 1)
        if workstation == "MAC":
            fname = r"/Users/Josh/Desktop/RPI_Forecaster/Pickles/filename_{:02d}.pkl".format(idx + 1)

        data = pd.read_pickle(fname)
        #
        # prediction, model, mom_forecasts, selected_models = code.forecast(data, idx)

        # update the first forecast mom versus last available index value
        hist_data.append(data)

        if idx == 0:
            dates = data['date']
            hist_df = pd.DataFrame(data['value'])
            hist_df.columns = [code.description]
            hist_df.index = dates

        else:
            new_dates = data['date']
            s = pd.DataFrame(data['value'])
            s.columns = [code.description]
            s.index = new_dates
            hist_df = pd.concat([hist_df, s], axis=1)


    return hist_df


def create_forecast_matrix(hist_df, description, models):


    for desc in descriptions:

        errors_df = pd.DataFrame([])
        train = hist_df[desc][:-24].pct_change().dropna()
        train_ts = TimeSeries.from_series(train)
        val = hist_df[desc][-25:].pct_change()
        val = val[1:]
        val_ts = TimeSeries.from_series(val)
        series = hist_df[desc].pct_change().dropna()
        series_ts = TimeSeries.from_series(series)

        for model in models.keys():
            print(model)
            if model == "SARIMAX": # specific model calibration for SARIMAX model
                forecasts, params = SARIMAX(train)
                params = "SARIMAX "+"p:"+str(params['p'])+", d:"+str(params['d'])+", q:"+str(params['q'])
            elif model == "RNN":
                my_model = RNNModel(model="LSTM",
                                    hidden_dim=20,
                                    dropout=0,
                                    batch_size=16,
                                    n_epochs=300,
                                    optimizer_kwargs={"lr": 1e-3},
                                    model_name="LSTM",
                                    log_tensorboard=True,
                                    random_state=42,
                                    training_length=20,
                                    input_chunk_length=14,
                                    force_reset=True,
                                    save_checkpoints=True)

                transformer = Scaler()
                train_transformed = transformer.fit_transform(train_ts)
                val_transformed = transformer.transform(val_ts)
                series_transformed = transformer.transform(series_ts)
                my_model.fit(
                    train_transformed,
                    val_series=val_transformed,
                    verbose=True,
                )
                forecasts = my_model.predict(n=24)
                forecasts = transformer.inverse_transform(forecasts)
                params = model
                forecasts = TimeSeries.pd_series(forecasts)

            else: # fit and forecast using selected Darts models
                models[model].fit(series=train_ts)
                forecasts = models[model].predict(24)
                params = model
                forecasts = TimeSeries.pd_series(forecasts)

            errors = forecasts - val
            errors_df[params] = errors

        print(desc)
        print(errors_df)
        fname = r'/Users/Josh/Desktop/RPI_Forecaster/Model Selection/'+str(desc)+'.csv'
        errors_df.to_csv(fname)

def MAPE(Y_actual,Y_Predicted):
    # Y_actual.replace(to_replace=0, value=0.0000001, inplace=True)
    print('here')
    print(np.abs(Y_actual - Y_Predicted))

    mape = np.mean(np.abs(Y_actual - Y_Predicted)/(Y_actual))*100
    return mape

def mean_bias_error(true, pred):
    true.replace(to_replace=0, value=0.0000001, inplace=True)
    mbe_loss = np.sum(true - pred)/true.size
    return mbe_loss

def RSME(y_actual, y_predicted):
    MSE = np.square(np.subtract(y_actual, y_predicted)).mean()
    RMSE = math.sqrt(MSE)
    return RMSE

def get_seasonality(hist_df):

    seasonality_df = pd.DataFrame([])

    for desc in [' Furniture']:

        train = TimeSeries.from_series(hist_df[desc][-145:-25].pct_change().dropna())
        if desc == ' Furniture':
            plot_acf(train,24)
        seas = check_seasonality(train, max_lag=12, m=6)
        seasonality_df.loc[desc, 'Is seasonal?'] = seas[0]
        seasonality_df.loc[desc, 'Seasonality Period'] = seas[1]

    return seasonality_df

def select_model(hist_df):

    metrics_df = pd.DataFrame([], columns=['RMSE'])
    for desc in descriptions:

        df = pd.read_csv(r'/Users/Josh/Desktop/RPI_Forecaster/Model Selection/'+str(desc)+'.csv')
        val = hist_df[desc][-25:].pct_change()
        val = val[1:]
        train = hist_df[desc][:-25].pct_change().dropna()
        rmse_dict = {}
        df.set_index(pd.to_datetime(df.iloc[:, 0]), inplace=True)
        for model in df.columns[1:]:
            pred = df[model]
            pred_ts = TimeSeries.from_series(pred)
            val_ts = TimeSeries.from_series(val)
            rmse = darts.metrics.rmse(actual_series=val_ts, pred_series=pred_ts)
            rmse_dict[model] = round(rmse, 5)
        metrics_df.loc[desc] = rmse_dict
    metrics_df.to_csv(r'/Users/Josh/Desktop/RPI_Forecaster/Model Selection/RMSE_results.csv')

def create_fwd_looking_forecasts():

    metrics_df = pd.read_csv(r'/Users/Josh/Desktop/RPI_Forecaster/Model Selection/RMSE_results.csv', index_col=0)
    best_models = {}
    for desc in descriptions:
        series = hist_df[desc].pct_change().dropna()
        series_ts = TimeSeries.from_series(series)
        i = metrics_df.loc[desc, 'RMSE']
        min_rmse = min(ast.literal_eval(i).values())
        best_model = [key for key in ast.literal_eval(i) if ast.literal_eval(i)[key] == min_rmse][0]
        best_models[desc] = best_model

        if "SARIMAX" in best_model:  # specific model calibration for SARIMAX model
            p = int(best_model[10])
            d = int(best_model[15])
            q = int(best_model[20])
            forecasts = SARIMAX_known(series, p, d, q)

        elif best_model == "RNN":
            my_model = RNNModel(model="LSTM",
                                hidden_dim=20,
                                dropout=0,
                                batch_size=16,
                                n_epochs=150,
                                optimizer_kwargs={"lr": 1e-3},
                                model_name="LSTM",
                                log_tensorboard=True,
                                random_state=42,
                                training_length=20,
                                input_chunk_length=14,
                                force_reset=True,
                                save_checkpoints=True)

            transformer = Scaler()
            series_transformed = transformer.fit_transform(series_ts)
            my_model.fit(
                series_transformed,
                verbose=True,
            )
            forecasts = my_model.predict(n=25)
            forecasts = transformer.inverse_transform(forecasts)
            forecasts = TimeSeries.pd_series(forecasts)

        else:  # fit and forecast using selected Darts models
            models[best_model].fit(series=series_ts)
            forecasts = models[best_model].predict(25)
            forecasts = TimeSeries.pd_series(forecasts)

        month = str(forecasts.index[0])
        fname = r'/Users/Josh/Desktop/RPI_Forecaster/Forward Looking Forecasts/'+month+'/'+str(desc)+'.csv'
        forecasts.to_csv(fname)

    pass

def update_xlsm_forecast_tab():

    descriptions = pd.read_excel(subcomponents)['Description']
    metrics_df = pd.read_csv(r'/Users/Josh/Desktop/RPI_Forecaster/Model Selection/RMSE_results.csv', index_col=0)
    best_models = {}
    best_forecasts_df = pd.DataFrame(columns=descriptions)
    for desc in descriptions:
        i = metrics_df.loc[desc, 'RMSE']
        min_rmse = min(ast.literal_eval(i).values())
        forecast_df = pd.read_csv(r'/Users/Josh/Desktop/RPI_Forecaster/Forward Looking Forecasts/'+str(desc)+'.csv', index_col=0)
        best_model = [key for key in ast.literal_eval(i) if ast.literal_eval(i)[key] == min_rmse]
        best_models[desc] = best_model
        best_forecasts_df[desc] = forecast_df[best_model]

    best_forecasts_df.index = forecast_df.index
    print(best_forecasts_df)
    # xw.Book("RPI_Forecaster.xlsm").set_mock_caller()
    # wb = xw.Book.caller()
    # mom_sheet = wb.sheets['MoM Forecasts']
    # mom_sheet.range('A1').value = forecast_df


hist_df = retrieve_pickles(codes=pd.read_excel(subcomponents)['Code_TS'])
models = {'SARIMAX': SARIMAX, 'Exponential Smoothing': ExponentialSmoothing(), 'Auto Arima': AutoARIMA(),
          'RNN': RNNModel(input_chunk_length=14)}
# create_forecast_matrix(hist_df, description=descriptions, models=models)
get_seasonality(hist_df)
# select_model(hist_df)
# create_fwd_looking_forecasts()
# update_xlsm_forecast_tab()