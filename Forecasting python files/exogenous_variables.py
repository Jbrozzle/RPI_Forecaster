import pandas as pd
import requests
import pprint
import darts
from tst_forecasting import *
from darts import TimeSeries

fname = r'/Users/Josh/Downloads/series-181122 (1).csv'
df = pd.read_csv(fname, index_col=0)

hist_df = retrieve_pickles(codes=pd.read_excel(subcomponents)['Code_TS'])
food_df = hist_df.iloc[:, :43]
food_df = food_df.loc['2009-07-01':]
food_df.index = pd.to_datetime(food_df.index)
# food_df.index = food_df.index.shift(6)
df = pd.DataFrame(df[76:])
df.index = pd.to_datetime(df.index)
df.index = df.index.shift(6, freq="MS")
print(df.head())
exog_df = TimeSeries.from_series(df)
print(food_df.tail())

def create_with_without_exog():

    # metrics_df = pd.read_csv(r'/Users/Josh/Desktop/RPI_Forecaster/Model Selection/RMSE_results.csv', index_col=0)
    best_models = {}

    for desc in descriptions:

        train = food_df[desc][:-6].pct_change().dropna()
        train_ts = TimeSeries.from_series(train)
        val = food_df[desc][-7:].pct_change()
        val_ts = TimeSeries.from_series(val[1:])
        series = food_df[desc].pct_change().dropna()
        series_ts = TimeSeries.from_series(series)

        # i = metrics_df.loc[desc, 'RMSE']
        # min_rmse = min(ast.literal_eval(i).values())
        # best_model = [key for key in ast.literal_eval(i) if ast.literal_eval(i)[key] == min_rmse][0]
        # best_models[desc] = best_model
        #
        # best_model = "RNN"

        no_exog_model = RNNModel(model="LSTM",
                            hidden_dim=20,
                            dropout=0,
                            batch_size=16,
                            n_epochs=150,
                            optimizer_kwargs={"lr": 1e-3},
                            model_name="LSTM",
                            log_tensorboard=True,
                            random_state=42,
                            training_length=20,
                            input_chunk_length=24,
                            force_reset=True,
                            save_checkpoints=True)

        exog_model = RNNModel(model="LSTM",
                            hidden_dim=20,
                            dropout=0,
                            batch_size=16,
                            n_epochs=150,
                            optimizer_kwargs={"lr": 1e-3},
                            model_name="LSTM",
                            log_tensorboard=True,
                            random_state=42,
                            training_length=20,
                            input_chunk_length=24,
                            force_reset=True,
                            save_checkpoints=True)

        transformer = Scaler()
        train_transformed = transformer.fit_transform(train_ts)
        no_exog_model.fit(
            train_transformed,
            verbose=True,
        )
        exog_model.fit(
            train_transformed,
            verbose=True,
            future_covariates = exog_df
        )

        no_exog_forecasts = no_exog_model.predict(n=6)
        no_exog_forecasts = transformer.inverse_transform(no_exog_forecasts)
        no_exog_rmse = darts.metrics.rmse(actual_series=val_ts, pred_series=no_exog_forecasts)
        no_exog_forecasts = TimeSeries.pd_series(no_exog_forecasts)

        exog_forecasts = exog_model.predict(n=6)
        exog_forecasts = transformer.inverse_transform(exog_forecasts)
        exog_rmse = darts.metrics.rmse(actual_series=val_ts, pred_series=exog_forecasts)
        exog_forecasts = TimeSeries.pd_series(exog_forecasts)

        forecasts = pd.DataFrame([])
        forecasts['No exog'] = no_exog_forecasts
        forecasts['Exog'] = exog_forecasts

        print(desc)
        print('Exog rsme: ' +str(exog_rmse))
        print('No exog rsme: ' + str(no_exog_rmse))



        fname = r'/Users/Josh/Desktop/RPI_Forecaster/Exogenous Variables/Food Forecasts/'+str(desc)+'.csv'
        forecasts.to_csv(fname)

    pass

def create_fwd_looking_exog():

    # metrics_df = pd.read_csv(r'/Users/Josh/Desktop/RPI_Forecaster/Model Selection/RMSE_results.csv', index_col=0)
    best_models = {}
    forecasts_df = pd.DataFrame([])

    for desc in descriptions[:43]:

        train = food_df[desc][:-6].pct_change().dropna()
        train_ts = TimeSeries.from_series(train)
        val = food_df[desc][-7:].pct_change()
        val_ts = TimeSeries.from_series(val[1:])
        series = food_df[desc].pct_change().dropna()
        series_ts = TimeSeries.from_series(series)


        exog_model = RNNModel(model="LSTM",
                            hidden_dim=20,
                            dropout=0,
                            batch_size=16,
                            n_epochs=115,
                            optimizer_kwargs={"lr": 1e-3},
                            model_name="LSTM",
                            log_tensorboard=True,
                            random_state=42,
                            training_length=20,
                            input_chunk_length=24,
                            force_reset=True,
                            save_checkpoints=True)

        transformer = Scaler()
        series_transformed = transformer.fit_transform(series_ts)

        exog_model.fit(
            series_transformed,
            verbose=True,
            future_covariates = exog_df
        )

        exog_forecasts = exog_model.predict(n=6)
        exog_forecasts = transformer.inverse_transform(exog_forecasts)
        exog_forecasts = TimeSeries.pd_series(exog_forecasts)
        forecasts_df[desc] = exog_forecasts

        fname = r'/Users/Josh/Desktop/RPI_Forecaster/Exogenous Variables/Forward Looking Forecasts/'+str(desc)+'.csv'
        exog_forecasts.to_csv(fname)

    wb = xw.Book(r'/Users/Josh/Desktop/RPI_Forecaster/RPI_Forecaster.xlsm')
    exog_sheet = wb.sheets['Explanatory Variables']
    exog_sheet.range('A30').value = forecasts_df

# full_df = pd.DataFrame([])
#
# for desc in descriptions[:30]:
#     df = pd.read_csv(r'/Users/Josh/Desktop/RPI_Forecaster/Exogenous Variables/Forward Looking Forecasts/'+str(desc)+'.csv', index_col=0)
#     full_df[desc] = df
# wb = xw.Book(r'/Users/Josh/Desktop/RPI_Forecaster/RPI_Forecaster.xlsm')
# exog_sheet = wb.sheets['Explanatory Variables']
# exog_sheet.range('A30').value = full_df
#
# print(full_df)

# create_with_without_exog()
create_fwd_looking_exog()