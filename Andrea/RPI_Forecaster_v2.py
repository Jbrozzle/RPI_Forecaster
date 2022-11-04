#!/usr/bin/env python
# coding: utf-8

# # IMPORTS AND SETTINGS

# In[44]:


import codecs

import xlwings as xw
import pandas as pd
import requests
import numpy as np
import dateutil
from darts import TimeSeries
# from darts.models import ExponentialSmoothing, RegressionModel, AutoARIMA, Theta , NBEATSModel
# from darts.models import Prophet
# from darts.models import ExponentialSmoothing
from darts.metrics import mape
import os
#statsmodels
import statsmodels.api as sm

import matplotlib.pyplot as plt
# warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

import importlib
import sys
# from petrol_api import get_petrol_prices

idx = pd.IndexSlice

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 700)

from IPython.display import display, HTML

display(HTML(data="""
<style>
    div#notebook-container    { width: 100%; }
    div#menubar-container     { width: 100%; }
    div#maintoolbar-container { width: 100%; }
</style>
"""))


# PATHS

# In[5]:


project_path=os.sep.join(os.getcwd().split("\\")[:-1])
project_path


# In[6]:


data_path=os.path.join(project_path,"data")
data_path


# # Forecasting Models

# In[225]:

#
# code=Subcomponent(code="CHOO", description="Depreciation", subgroup="Housing")
# data=code.retrieve(code.short_code)
# data=data.set_index("date")["value"]
# train, val=data[:-24], data[-24:]
#
#
# # In[45]:
#
#
# decomposition = sm.tsa.seasonal_decompose(df.set_index("date")["value"], model='additive')
# fig = decomposition.plot()
# plt.show()
#
#
# # In[67]:
#
#
# data.diff().diff().plot()


# In[209]:


def SARIMAX(data):
    p_max=3
    d_max=2
    q_max=3

    IC=pd.DataFrame(columns=["BIC"])

    for d in range(1,d_max+1):
        for p in range(0,p_max+1):
            for q in range(0,q_max+1):

                mod = sm.tsa.statespace.SARIMAX(data,
                                                order=(p, d, q),
                                                seasonal_order=(1, 1, 0, 12),
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)


                results = mod.fit()
                IC.loc[str(d)+str(p)+str(q),"BIC"]=results.bic


    #re-estimate the best model            
    d_optimal=IC.sort_values(by="BIC").iloc[0].name[0]
    d_optimal=IC.sort_values(by="BIC").iloc[0].name[1]
    d_optimal=IC.sort_values(by="BIC").iloc[0].name[2]

    mod = sm.tsa.statespace.SARIMAX(data,
                        order=(p, d, q),
                        seasonal_order=(1, 1, 0, 12),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
    results = mod.fit()
    return results.forecast(24)-val

SARIMAX(train)


# In[ ]:


models={"SARIMAX":SARIMAX} #add models here

results=[] #list of dataframes

#loop on subcomponents
for code in codes:
    
        #data loading
        data = code.retrieve(code.short_code)
        data=df.set_index("date")["value"]
        train, val=data[:-24], data[-24:]
        
        #loop on the models
        for model in models:
            print(model)
            results_i=models[model](train)
            results_i=pd.DataFrame(results_i, columns=["errors"])
            results_i.insert(0,"model",model)
            results_i.insert(1,"subcomponent",code.short_code)
            results.append(results_i)
            
results=pd.concat(results, axis=0)
results


# # CLASSES and FUNCTIONS

# In[26]:


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

            global new_model
            wb = xw.Book.caller()
            idx = idx
            df = data
            # prepare the date for Darts models
            # To do prepare the date for a generic model:
            # split train and validation
            series = TimeSeries.from_dataframe(df, 'date', 'value')
            train, val = series[:-24], series[-24:]
            models = [AutoARIMA(), ExponentialSmoothing()]
            prediction_dfs = []
            selected_models = []

            BACKTESTING = False

            if BACKTESTING == True:
                #to do save into backtest the forecasting errors for every model
                backtests = [model.historical_forecasts(series=series, start=.75, forecast_horizon=12) for model in models]
                lowest_mape = 100

                #to do loop across forecasting metrics and think about a selection procedure
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
#         b.to_pickle(r"C:\Users\joshb\Desktop\RPI Forecasting\Pickles\filename_{:02d}.pkl".format(i+1))
        b.to_pickle(os.path.join(data,"Pickles\\filename_{:02d}.pkl".format(i+1)))
    return 1

@xw.func()
def pull_rpi_data():
#     subcomponents = r"C:\Users\joshb\Desktop\RPI Forecasting\RPI_Subcomponents.xlsx"
    subcomponents = os.path.join(data_path,"RPI_Subcomponents.xlsx")
    codes = pd.read_excel(subcomponents)['Code_TS']
    descriptions = pd.read_excel(subcomponents)['Description']
    subgroups = pd.read_excel(subcomponents)['SubGroup']

    hist_data = []
    hist_df = pd.DataFrame([], columns=descriptions)
    wb = xw.Book.caller()
    hist_data_sheet = wb.sheets['Hist Data']
    # if want to get new data
    RETRIEVE = False

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
#             fname = r"C:\Users\joshb\Desktop\RPI Forecasting\Pickles\filename_{:02d}.pkl".format(idx + 1)
            fname = os.path.join(data_path,"Pickles\\filename_{:02d}.pkl".format(idx + 1))
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
def pull_petrol_data():
    petrol_df, petrol_prices = get_petrol_prices()
    #xw.Book("RPI_Forecaster.xlsm").set_mock_caller()
    wb = xw.Book.caller()
    petrol_sheet = wb.sheets("Petrol")
    petrol_sheet.range('AA1').value = petrol_df


@xw.func
def main():

#     xw.Book("RPI_Forecaster.xlsm").set_mock_caller()
    xw.Book(os.path.join(data_path,"RPI_Forecaster.xlsm")).set_mock_caller()
    wb = xw.Book.caller()

#     subcomponents = r"C:\Users\joshb\Desktop\RPI Forecasting\RPI_Subcomponents.xlsx"
    subcomponents = os.path.join(data_path,"RPI_Subcomponents.xlsx")
#     weights = r"C:\Users\joshb\Desktop\RPI Forecasting\Weights.xlsx"
    weights = os.path.join(data_path,"Weights.xlsx")
#     model_overrides = r"C:\Users\joshb\Desktop\RPI Forecasting\Model_Overrides.xlsx"
    model_overrides = os.path.join(data_path,"Model_Overrides.xlsx")
    
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
                prediction, model, mom_forecasts, selected_models = code.forecast(data, idx)
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
#             fname = r"C:\Users\joshb\Desktop\RPI Forecasting\Pickles\filename_{:02d}.pkl".format(idx + 1)
            fname = os.path.join(data_path,"Pickles\\filename_{:02d}.pkl".format(idx + 1))
            data = pd.read_pickle(fname)
            hist_data.append(data)

            # check that we want to use time series forecasting method for this subcomponent
            if code.short_code not in model_overrides:
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
        front_sheet = wb.sheets['Front']
        front_sheet.range('A40').options(transpose=True).value = selected_models

@xw.func
def hello(name):
    return f"Hello {name}!"

@xw.func
def load_weights():
#     weights = r"C:\Users\joshb\Desktop\RPI Forecasting\Weights.xlsx"
    os.path.join(data_path,"Weights.xlsx")
    weights = pd.read_excel(weights)
    weights = weights.set_index('Description')
    weights = weights.transpose()
    wb = xw.Book.caller()
    weights_sheet = wb.sheets['Hist Weights']
    weights_sheet.range("A1").value = weights

if __name__ == "__main__":
    xw.Book(os.path.join(data_path,"RPI_Forecaster.xlsm")).set_mock_caller()
    main()

