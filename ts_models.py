
import xlwings as xw
import pandas as pd
import requests
import numpy as np
import dateutil
from darts import TimeSeries
from darts.metrics import mape
import os
#statsmodels
import statsmodels.api as sm
from darts.utils.statistics import check_seasonality




def SARIMAX(data):

    p_max = 3
    d_max = 2
    q_max = 3

    IC = pd.DataFrame(columns=["BIC"])

    for d in range(1,d_max+1):
        for p in range(0,p_max+1):
            for q in range(0,q_max+1):

                mod = sm.tsa.statespace.SARIMAX(data,
                                                order=(p, d, q),
                                                seasonal_order=(1, 1, 0, 12),
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit(disp=False)
                IC.loc[str(d)+str(p)+str(q),"BIC"] = results.bic

    #re-estimate the best model
    d_optimal = IC.sort_values(by="BIC").iloc[0].name[0]
    p_optimal = IC.sort_values(by="BIC").iloc[0].name[1]
    q_optimal = IC.sort_values(by="BIC").iloc[0].name[2]

    mod = sm.tsa.statespace.SARIMAX(data, order=(int(p_optimal), int(d_optimal), int(q_optimal)),
                                    seasonal_order=(1, 1, 0, 12), enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()
    params = {'p': int(p_optimal), 'd': int(d_optimal), 'q': int(q_optimal)}

    return results.forecast(24), params

def SARIMAX_known(data, p, d, q):

    mod = sm.tsa.statespace.SARIMAX(data,
                                    order=(p, d, q),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit(disp=False)
    return results.forecast(25)
