import pandas as pd
from ts_models import *
from RPI_Forecaster import Subcomponent
from darts.models import ExponentialSmoothing, FFT, AutoARIMA, RNNModel, Theta
from darts.dataprocessing.transformers import Scaler
import warnings
import logging
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


hist_df = retrieve_pickles(codes=pd.read_excel(subcomponents)['Code_TS'])
print(hist_df)
models = {'SARIMAX': SARIMAX, 'Exponential Smoothing': ExponentialSmoothing(), 'Auto Arima': AutoARIMA(),
          'RNN': RNNModel(input_chunk_length=14)}
create_forecast_matrix(hist_df, description=descriptions, models=models)