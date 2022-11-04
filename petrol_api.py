import pandas as pd
import requests
import html
from bs4 import BeautifulSoup


def get_petrol_prices():

    url = "https://docs.google.com/spreadsheets/d/1EmzU-rNhTj1nBjPs0QAREeoQ-2tNryVHjkDkKSCcZ-8/export?format=csv&id=1EmzU-rNhTj1nBjPs0QAREeoQ-2tNryVHjkDkKSCcZ-8"
    url2 = "https://docs.google.com/spreadsheets/d/1NdWKr1TiDbuU0lbZT8EaWX0eGidxKbj_zfMGmnrSOKg/export?format=csv&id=1NdWKr1TiDbuU0lbZT8EaWX0eGidxKbj_zfMGmnrSOKg"
    url3 ="https://docs.google.com/spreadsheets/d/1LhjnjbFbj9fWRfyNNtdfL18fHQeGJd8x0tgxZwtz_1Q/export?format=csv&id=1LhjnjbFbj9fWRfyNNtdfL18fHQeGJd8x0tgxZwtz_1Q"
    df = pd.read_csv(url)
    df2 = pd.read_csv(url2)
    df3 = pd.read_csv(url3)

    print(df)
    unleaded = df['Unleaded'][0]
    super_unleaded = df['Super unleaded'][0]
    diesel = df['Diesel'][0]
    lpg = df['LPG'][0]
    df2.set_index('Date', inplace=True)
    df3.set_index('Date', inplace=True)


    return df, {'Unleaded':unleaded, 'Super Unleaded':super_unleaded, 'Diesel':diesel, 'LPG':lpg}, df2, df3

