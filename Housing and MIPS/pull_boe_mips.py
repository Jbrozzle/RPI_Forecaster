import pandas as pd
import requests
import io
import datetime

url_endpoint = 'http://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp?csv.x=yes'
date_to = datetime.date(datetime.datetime.today().year, datetime.datetime.today().month,1)
date_to_str = date_to.strftime('%d/%b/%Y')
payload = {
    'Datefrom'   : '01/Jan/2000',
    'Dateto'     : date_to_str,
    'SeriesCodes': 'IUMZO27,IUMZICQ,CFMZ6IY',
    'CSVF'       : 'TN',
    'UsingCodes' : 'Y',
    'VPD'        : 'Y',
    'VFD'        : 'N'
}

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/54.0.2840.90 '
                  'Safari/537.36'
}
response = requests.get(url_endpoint, params=payload, headers=headers)
df = pd.read_csv(io.BytesIO(response.content))



print(df.tail())