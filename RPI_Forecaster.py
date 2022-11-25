import codecs

import xlwings as xw
import pandas as pd
import requests
import numpy as np
import dateutil
from darts import TimeSeries
from darts.models import ExponentialSmoothing, RegressionModel, AutoARIMA, Prophet, Theta
from darts.metrics import mape
import importlib
import sys
from petrol_api import get_petrol_prices
from ts_models import *
import ast
import io
import datetime

global workstation
workstation = "MAC"
if workstation == "PC":
    subcomponents = r"C:\Users\joshb\Desktop\RPI Forecasting\RPI_Subcomponents.xlsx"

if workstation == "MAC":
    subcomponents = r"/Users/Josh/Desktop/RPI_Forecaster/RPI_Subcomponents.xlsx"

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

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

    def forecast(self, data, idx):

            wb = xw.Book.caller()
            idx = idx
            df = data
            series = TimeSeries.from_dataframe(df, 'date', 'value')
            train, val = series[:-24], series[-24:]
            models = [AutoARIMA(), ExponentialSmoothing()]
            prediction_dfs = []
            selected_models = []

            BACKTESTING = False

            if BACKTESTING == True:
                backtests = [model.historical_forecasts(series=series, start=.75, forecast_horizon=12) for model in models]
                lowest_mape = 100

                for i, m in enumerate(models):
                    err = mape(backtests[i], series)
                    if err < lowest_mape:
                        lowest_mape = err
                        best_model_idx = i

                model = models[best_model_idx]
                selected_models.append(model)
            else:
                model_list = wb.sheets("Front").range('Ak1:AK92').value
                model = model_list[idx]
                if model == "AutoARIMA":
                    new_model = models[0]
                else:
                    new_model = models[1]

            # new_model = Prophet()

            new_model.fit(series=series)
            prediction = new_model.predict(27)
            prediction_dfs.append(prediction)

            #todo perform some model selection here (maybe use a bokeh GUI to perform this?)
            #prediction_df = lowest MAPE prediction
            #model = lowest MAPE model
            best_prediction = prediction_dfs[0].pd_dataframe()
            best_model = models[0]
            mom_predictions = best_prediction.pct_change()

            return best_prediction, best_model, mom_predictions, selected_models

def picklecreator(dflist):
    for i, b in enumerate(dflist):
        if workstation == "PC":
            b.to_pickle(r"C:\Users\joshb\Desktop\RPI Forecasting\Pickles\filename_{:02d}.pkl".format(i+1))
        if workstation == "MAC":
            b.to_pickle(r"/Users/Josh/Desktop/RPI_Forecaster/Pickles/filename_{:02d}.pkl".format(i + 1))

    return 1

@xw.func()
def pull_rpi_data():

    if workstation == "PC":
        subcomponents = r"C:\Users\joshb\Desktop\RPI Forecasting\RPI_Subcomponents.xlsx"

    if workstation == "MAC":
        subcomponents = r"/Users/Josh/Desktop/RPI_Forecaster/RPI_Subcomponents.xlsx"

    codes = pd.read_excel(subcomponents)['Code_TS']
    descriptions = pd.read_excel(subcomponents)['Description']
    subgroups = pd.read_excel(subcomponents)['SubGroup']

    hist_data = []
    hist_df = pd.DataFrame([], columns=descriptions)
    wb = xw.Book.caller()
    hist_data_sheet = wb.sheets['Hist Data']
    # if want to get new data
    RETRIEVE = True

    if RETRIEVE == True:

        for idx, code in enumerate(codes[:]):
            code = Subcomponent(codes.iloc[idx], descriptions.iloc[idx], subgroups.iloc[idx])
            print(code.description)
            data = code.retrieve(code.short_code)
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

        picklecreator(hist_data)


    else:
        for idx, code in enumerate(codes[:]):
            code = Subcomponent(codes.iloc[idx], descriptions.iloc[idx], subgroups.iloc[idx])
            print(code.description)
            if workstation == "PC":
                fname = r"C:\Users\joshb\Desktop\RPI Forecasting\Pickles\filename_{:02d}.pkl".format(idx + 1)
            if workstation == "MAC":
                fname = r"/Users/Josh/Desktop/RPI_Forecaster/Pickles/filename_{:02d}.pkl".format(idx + 1)

            data = pd.read_pickle(fname)
            hist_data.append(data)

            # check that we want to use time series forecasting method for this subcomponent

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

    hist_data_sheet.range('A1').value = hist_df

@xw.func()
def pull_subgroup_data():
    wb = xw.Book.caller()
    subgroups_sheet = wb.sheets['Subgroups']
    subgroup_codes = subgroups_sheet['B1:O1'].value
    subgroup_desc = subgroups_sheet['B2:O2'].value
    hist_data = []
    hist_df = pd.DataFrame([], columns=subgroup_desc)

    for idx, code in enumerate(subgroup_codes[:]):
        code = Subcomponent(subgroup_codes[idx], subgroup_desc[idx], subgroup_desc[idx])
        print(code.description)
        data = code.retrieve(code.short_code)
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

    subgroups_sheet.range('a2').value = hist_df[-21:]

@xw.func()
def pull_petrol_data():
    petrol_df, petrol_prices, hist_petrol_df, hist_diesel_df = get_petrol_prices()
    wb = xw.Book.caller()
    petrol_sheet = wb.sheets("Petrol")
    petrol_sheet.range('h30').value = petrol_df
    # hist_petrol_df = hist_petrol_df.sort_values(axis=0, ascending=False)
    petrol_sheet.range('A30').value = hist_petrol_df

@xw.func()
def pull_hpi_data():
    wb = xw.Book.caller()
    housing_sheet = wb.sheets("Housing")
    front_sheet = wb.sheets("Front")
    latestdate = str(front_sheet.range("L4").value)
    url = "http://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/Average-prices-" + latestdate + \
          ".csv?utm_medium=GOV.UK&utm_source=datadownload&utm_campaign=average_price&utm_term=9.30_14_09_22"
    df = pd.read_csv(url)
    df = df.loc[df['Region_Name'] == "United Kingdom"]
    df.set_index('Date', inplace=True)
    df = df['Average_Price']
    df = df.sort_values(axis=0, ascending=False)
    housing_sheet.range("A38").value = df

@xw.func()
def pull_ofgem_data():
    wb = xw.Book.caller()
    if workstation == "MAC":
        annex_2_wb = xw.Book(
            "/Users/Josh/Desktop/RPI_Forecaster/Energy/OFGEM/Annex_2_-_wholesale_cost_allowance_methodology_v1.12 (input_data_removed).xlsx")
        cap_calculator_wb = xw.Book(
            "/Users/Josh/Desktop/RPI_Forecaster/Energy/OFGEM/Default_tariff_cap_level_v1.12.xlsx")
    if workstation == "PC":
        pass

    elec_offset = int(annex_2_wb.sheets('3d(ii) Price data elec Q+n').range('h6').value)
    gas_offset = int(annex_2_wb.sheets('3e Price data gas').range('i6').value)

    elec_prices_used = (annex_2_wb.sheets("Josh Price elec Q+n").range('l1:t23').offset(elec_offset).options(
        pd.DataFrame).value.dropna())
    gas_prices_used = annex_2_wb.sheets("Josh Price gas").range('k1:s16').offset(gas_offset).options(
        pd.DataFrame).value.dropna()
    cap_mom = cap_calculator_wb.sheets("Cap Table").range('J5:L77').options(pd.DataFrame).value.dropna()

    wb.sheets('OFGEM').range('b2').value = cap_mom
    wb.sheets('OFGEM').range('h2').value = elec_prices_used
    wb.sheets('OFGEM').range('h24').value = gas_prices_used

@xw.func()
def update_xlsm_forecast_tab():

    wb = xw.Book(r'/Users/Josh/Desktop/RPI_Forecaster/RPI_Forecaster.xlsm')

    if workstation == "PC":
        subcomponents = r"C:\Users\joshb\Desktop\RPI Forecasting\RPI_Subcomponents.xlsx"

    if workstation == "MAC":
        subcomponents = r"/Users/Josh/Desktop/RPI_Forecaster/RPI_Subcomponents.xlsx"

    descriptions = pd.read_excel(subcomponents)['Description']
    metrics_df = pd.read_csv(r'/Users/Josh/Desktop/RPI_Forecaster/Model Selection/RMSE_results.csv', index_col=0)
    best_models = {}
    best_forecasts_df = pd.DataFrame(columns=descriptions)
    for desc in descriptions:
        i = metrics_df.loc[desc, 'RMSE']
        min_rmse = min(ast.literal_eval(i).values())
        forecast_df = pd.read_csv(r'/Users/Josh/Desktop/RPI_Forecaster/Forward Looking Forecasts/'+str(desc)+'.csv', index_col=0)
        # best_model = [key for key in ast.literal_eval(i) if ast.literal_eval(i)[key] == min_rmse]
        # best_models[desc] = best_model[0]
        best_forecasts_df[desc] = forecast_df

    best_forecasts_df.index = forecast_df.index
    print(best_forecasts_df)
    mom_sheet = wb.sheets['MoM Forecasts']
    mom_sheet.range('A1').value = best_forecasts_df

@xw.func()
def pull_boe_mips():
    wb = xw.Book.caller()
    url_endpoint = 'http://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp?csv.x=yes'
    date_to = datetime.date(datetime.datetime.today().year, datetime.datetime.today().month, 1)
    date_to_str = date_to.strftime('%d/%b/%Y')
    payload = {
        'Datefrom': '01/Jan/2000',
        'Dateto': date_to_str,
        'SeriesCodes': 'IUMZO27,IUMZICQ,CFMZ6IY',
        'CSVF': 'TN',
        'UsingCodes': 'Y',
        'VPD': 'Y',
        'VFD': 'N'
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/54.0.2840.90 '
                      'Safari/537.36'
    }
    response = requests.get(url_endpoint, params=payload, headers=headers)
    df = pd.read_csv(io.BytesIO(response.content))
    df.set_index('DATE', inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=False)
    df = df.rename(columns={'IUMZO27': '5y 60% LTV', 'IUMZICQ': '2y 60% LTV', 'CFMZ6IY': 'Base rate tracker'})
    wb.sheets('Housing').range('AG1').value = df

@xw.func
def main():

    xw.Book("RPI_Forecaster.xlsm").set_mock_caller()
    wb = xw.Book.caller()

    if workstation == "PC":
        subcomponents = r"C:\Users\joshb\Desktop\RPI Forecasting\RPI_Subcomponents.xlsx"

    if workstation == "MAC":
        subcomponents = r"/Users/Josh/Desktop/RPI_Forecaster/RPI_Subcomponents.xlsx"

    codes = pd.read_excel(subcomponents)['Code_TS']
    descriptions = pd.read_excel(subcomponents)['Description']
    subgroups = pd.read_excel(subcomponents)['SubGroup']
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
            #change this bit for updated forecast method

            prediction, model, mom_forecasts, selected_models = code.forecast(data, idx)


            # update the first forecast mom versus last available index value
            mom_forecasts.iloc[0] = prediction.iloc[0] / data['value'].iloc[-1] - 1


            hist_data.append(data)
            forecasts.append(prediction)
            mom_forecasts.append(mom_forecasts)

        picklecreator(hist_data)
        # pd.DataFrame(hist_data).to_csv(r"C:\Users\joshb\Desktop\RPI Forecasting\HistData.csv")

    else:
        for idx, code in enumerate(codes[:]):
            code = Subcomponent(codes.iloc[idx], descriptions.iloc[idx], subgroups.iloc[idx])
            print(code.description)
            if workstation == "PC":
                fname = r"C:\Users\joshb\Desktop\RPI Forecasting\Pickles\filename_{:02d}.pkl".format(idx + 1)
            if workstation == "MAC":
                fname = r"/Users/Josh/Desktop/RPI_Forecaster/Pickles/filename_{:02d}.pkl".format(idx + 1)

            data = pd.read_pickle(fname)
            hist_data.append(data)

            prediction, model, mom_forecasts, selected_models = code.forecast(data, idx)

            # update the first forecast mom versus last available index value
            if idx == 0:
                forecast_df[code.description] = mom_forecasts['value']
                forecast_df.index = mom_forecasts.index
                dates = data['date']
                hist_df[code.description] = data['value']

            else:
                forecast_df[code.description] = mom_forecasts['value']
                hist_df[code.description] = data['value']

        mom_sheet = wb.sheets['MoM Forecasts']
        mom_sheet.range('A1').value = forecast_df
        hist_df.index = dates
        hist_data_sheet = wb.sheets['Hist Data']
        print(hist_df)
        #hist_data_sheet.range('A1').value = hist_df
        front_sheet = wb.sheets['Front']
        front_sheet.range('A40').options(transpose=True).value = selected_models

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
    # xw.Book("RPI_Forecaster.xlsm").set_mock_caller()
    update_xlsm_forecast_tab()
    # main()
