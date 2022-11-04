import pandas as pd
import xlwings as xw

def pull_house_prices():
      latestdate = '2022-08'
      url = "http://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/Average-prices-"+latestdate+\
            ".csv?utm_medium=GOV.UK&utm_source=datadownload&utm_campaign=average_price&utm_term=9.30_14_09_22"

      url = 'http://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/Indices-'+latestdate+\
            '.csv?utm_medium=GOV.UK&utm_source=datadownload&utm_campaign=index&utm_term=9.30_14_09_22'
      df = pd.read_csv(url)
      df = df.loc[df['Region_Name']=="United Kingdom"]
      df.set_index('Date', inplace=True)
      print(df.columns)
      # df = df['Index']



      print(df.tail(20))

# url = "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/personalandhouseholdfinances/expenditure/" \
#       "datasets/familyspendingworkbook5expenditureonhousing/fye2021/workbook5housing.xlsx"
#
# df = pd.read_csv(url)
# print(df)
pull_house_prices()