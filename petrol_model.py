import pandas as pd
from petrol_api import *
import xlwings as xw
import statsmodels.api as sm

# url2 = "https://docs.google.com/spreadsheets/d/1NdWKr1TiDbuU0lbZT8EaWX0eGidxKbj_zfMGmnrSOKg/export?format=csv&id=1NdWKr1TiDbuU0lbZT8EaWX0eGidxKbj_zfMGmnrSOKg"
# df2 = pd.read_csv(url2)
def check_rca():
    df, dict1, df2, df3 = get_petrol_prices()


    df2 = df2['Unleaded pump (inc VAT)']
    print(df2)
    df2.index = pd.to_datetime(df2.index)
    wb = r"/Users/Josh/Desktop/RPI_Forecaster/RPI_Forecaster.xlsm"

    hist_df = pd.read_excel(wb, sheet_name='Hist Data')
    #print(hist_df[' Petrol and oil'])

    comp_df = hist_df[' Petrol and oil']
    comp_df.index = hist_df['date']
    comp_df.index = comp_df.index.to_period('M')
    df2 = df2.groupby([df2.index.strftime('%Y-%m')]).last().reset_index()
    print(df2)
    df2.to_csv(r"/Users/Josh/Desktop/RPI_Forecaster/Petrol_model.csv")


def resize_data():
    df = pd.read_csv(r'/Users/Josh/Downloads/Gasoline RBOB Futures Historical Data (2).csv')
    #df = pd.read_csv(r'/Users/Josh/Downloads/Crude Oil WTI Futures Historical Data (2).csv')
    #df = pd.read_csv(r'/Users/Josh/Downloads/GBP_USD Historical Data.csv')
    df = df[['Price', 'Date']]
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index, yearfirst=True, format="%d/%m/%Y")
    print(df)
    df = df.groupby([df.index.strftime('%Y-%m')]).last().reset_index()
    df.set_index('Date', inplace=True)
    print(df.tail(12))
    df.to_csv(r"/Users/Josh/Desktop/RPI_Forecaster/Gasoline.csv")

#resize_data()

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_excel(r'/Users/Josh/Desktop/RPI_Forecaster/Petrol_model.xls', sheet_name = "Regressions", index_col='Date')
print(df)

target = np.array(df['Unleaded after fuel duty']).reshape(-1,1)
independents = np.array(df['Sterling gasoline']).reshape(-1,1)
#independents = np.array(independents).reshape(1, -1)
X_train, X_test, y_train, y_test = train_test_split(independents, target, test_size=0.2)

lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()
print('Score:', model.score(X_test, y_test))