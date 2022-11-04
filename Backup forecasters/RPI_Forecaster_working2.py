import xlwings as xw
import pandas as pd
import requests
import numpy as np
import dateutil
from darts import TimeSeries
from darts.models import ExponentialSmoothing, RegressionModel, AutoARIMA, Prophet, Theta, NBEATSModel


class Subcomponent:

    def __init__(self, code, description, subgroup):
        self.short_code = code
        self.description = description
        self.subgroup = subgroup

    def retrieve(self, timeseries_id):

        api_endpoint = "https://api.ons.gov.uk/timeseries/"
        api_params = {'dataset':'MM23',
                      'time_series':timeseries_id}
        url = (api_endpoint + '/'+api_params['time_series']+'/dataset/'+api_params['dataset']+'/data')

        data = requests.get(url).json()
        data = pd.DataFrame(pd.json_normalize(data['months']))
        data['value'] = data['value'].astype(float)
        data['date'] = pd.to_datetime(data['date'])

        data['log_ret'] = np.log(data.value) - np.log(data.value.shift(1))
        data['Month Index'] = pd.DatetimeIndex(data['date']).month
        data['Year Index'] = pd.DatetimeIndex(data['date']).year
        data['pct change'] = data['value'].pct_change()

        data['Easter'] = data['Year Index'].apply(dateutil.easter.easter)
        data['Easter Month'] = pd.DatetimeIndex(data['Easter']).month
        data['Easter Day'] = pd.DatetimeIndex(data['Easter']).day
        data['Easter Regressor'] = data['Easter Month'] = data.apply(lambda row: int(row['Month Index']== row['Easter Month']), axis =1)

        return data

    def forecast(self, data):

            df = data
            series = TimeSeries.from_dataframe(df, 'date', 'value')
            train, val = series[:-1], series[-1:]
            models = [AutoARIMA()]

            prediction_dfs = []
            for model in models:

                model.fit(train)
                prediction = model.predict(26)
                # series.plot()
                #
                # prediction.plot(label='forecast', low_quantile=0.33, high_quantile=0.66)
                prediction_dfs.append(prediction)
                # plt.legend()
                # plt.title(self.description)
                # plt.show()

            #todo perform some model selection here (maybe use a bokeh GUI to perform this?)
            #prediction_df = lowest MAPE prediction
            #model = lowest MAPE model
            best_prediction = prediction_dfs[0].pd_dataframe()
            best_model = models[0]
            mom_predictions = best_prediction.pct_change()

            return best_prediction, best_model, mom_predictions

@xw.func
def main():

    wb = xw.Book.caller()

    subcomponents = r"C:\Users\joshb\Desktop\RPI Forecasting\RPI_Subcomponents.xlsx"
    weights = r"C:\Users\joshb\Desktop\RPI Forecasting\Weights.xlsx"
    model_overrides = r"C:\Users\joshb\Desktop\RPI Forecasting\Model_Overrides.xlsx"
    codes = pd.read_excel(subcomponents)['Code_TS']
    weights = pd.read_excel(weights)
    descriptions = pd.read_excel(subcomponents)['Description']
    latest_weights = pd.read_excel(subcomponents)['Weight']
    subgroups = pd.read_excel(subcomponents)['SubGroup']
    model_overrides = pd.read_excel(model_overrides)

    hist_data = []
    forecasts = []

    forecast_df = pd.DataFrame([], columns=descriptions)
    hist_df = pd.DataFrame([], columns=descriptions)

    # if want to get new data
    RETRIEVE = False

    if RETRIEVE == True:

        # get subcomponent level data and create respective forecasts
        for idx, code in enumerate(codes[:]):
            code = Subcomponent(codes.iloc[idx], descriptions.iloc[idx], subgroups.iloc[idx])
            print(code.description)
            data = code.retrieve(code.short_code)

            # check that we want to use time series forecasting method for this subcomponent
            if code.short_code not in model_overrides:
                prediction, model, mom_forecasts = code.forecast(data)
                # update the first forecast mom versus last available index value
                mom_forecasts.iloc[0] = prediction.iloc[0] / data['value'].iloc[-1] - 1
                prediction = Forecast(prediction, mom_forecasts, codes.iloc[idx],
                                      TimeSeries.from_dataframe(prediction).time_index[:1],
                                      TimeSeries.from_dataframe(prediction).time_index[-1:],
                                      latest_weights.iloc[idx],
                                      model)

                hist_data.append(data)
                forecasts.append(prediction)

                mom_forecasts.append(mom_forecasts)

            # otherwise pull correct forecasting model
            else:
                # todo generate forecasts for overriden models
                hist_data.append(data)
                forecasts.append(prediction)
                mom_forecasts.append(mom_forecasts)
                pass

        picklecreator(hist_data)
        # pd.DataFrame(hist_data).to_csv(r"C:\Users\joshb\Desktop\RPI Forecasting\HistData.csv")

    else:
        for idx, code in enumerate(codes[:]):
            code = Subcomponent(codes.iloc[idx], descriptions.iloc[idx], subgroups.iloc[idx])
            print(code.description)
            fname = r"C:\Users\joshb\Desktop\RPI Forecasting\Pickles\filename_{:02d}.pkl".format(idx + 1)
            data = pd.read_pickle(fname)
            hist_data.append(data)

            # check that we want to use time series forecasting method for this subcomponent
            if code.short_code not in model_overrides:
                prediction, model, mom_forecasts = code.forecast(data)
                # update the first forecast mom versus last available index value
                if idx == 0:
                    forecast_df[code.description] = mom_forecasts['value']
                    forecast_df.index = mom_forecasts.index
                    dates = data['date']
                    hist_df[code.description] = data['value']

                else:
                    forecast_df[code.description] = mom_forecasts['value']
                    hist_df[code.description] = data['value']

            else:
                # todo generate forecasts for overriden models
                hist_data.append(data)
                forecasts.append(prediction)
                mom_forecasts.append(mom_forecasts)
                pass

        #wb = xw.Book(r"C:\Users\joshb\Desktop\RPI Forecasting\Forecaster.xlsx")
        mom_sheet = wb.sheets['MoM Forecasts']
        mom_sheet.range('A1').value = forecast_df
        hist_df.index = dates
        hist_data_sheet = wb.sheets['Hist Data']
        print(hist_df)
        #hist_data_sheet.range('A1').value = hist_df

@xw.func
def hello(name):
    return f"Hello {name}!"

@xw.func
def load_weights():
    weights = r"C:\Users\joshb\Desktop\RPI Forecasting\Weights.xlsx"
    weights = pd.read_excel(weights)
    weights = weights.set_index('Description')
    weights = weights.transpose()
    wb = xw.Book.caller()
    weights_sheet = wb.sheets['Hist Weights']
    weights_sheet.range("A1").value = weights


if __name__ == "__main__":
    xw.Book("RPI_Forecaster.xlsm").set_mock_caller()
    main()
