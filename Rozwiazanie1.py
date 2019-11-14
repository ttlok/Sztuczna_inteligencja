#Zad1
import pandas as pd
import requests
import pandas as pd
import matplotlib.pyplot as plt
from numpy import corrcoef, array


#Pobieranie danych jako listy słowników
def get_data(start_date, end_date, currency):
    request_url = 'http://api.nbp.pl/api/exchangerates/rates/A/' + currency + '/' + start_date + '/' + end_date + '/'
    currency_req = requests.get(request_url)
    currency_data = currency_req.json()
    return currency_data['rates'] 

#Wykorzystanie powyższej funkcji do pobrania danych jako dataframe
def get_data_as_json(start_date, end_date, currency):
    currency_data = get_data(start_date, end_date, currency)
    return pd.DataFrame.from_dict(currency_data)

#Zad2
#kurs franka szwajcarskiego w pierwszym miesiącu tego roku
k_chf = get_data_as_json('2019-01-01', '2019-01-31', 'CHF')
#kurs złotego w tym samym okresie
k_uah = get_data_as_json('2019-01-01', '2019-01-31', 'UAH')

#Zad3


#dla kursu franka

k_chf.head() #podgląd zawartości dataframe
k_chf.dtypes #sprawdzanie typu pobranych danych
k_chf['effectiveDate'] = pd.to_datetime(k_chf['effectiveDate']) #zmiana typu danych kolumny z datą na typ datatime
k_chf['effectiveDate'].dtypes #sprawdzenie typu danych kolumny z datą po zmianie
k_chf = k_chf.set_index("effectiveDate").drop(columns='no') #zmiana indeksu na datę i usunięcie kolumny "no"
k_chf.head()

#dla kursu złotego

k_uah.head() #podgląd zawartości dataframe
k_uah.dtypes #sprawdzanie typu pobranych danych
k_uah['effectiveDate'] = pd.to_datetime(k_uah['effectiveDate']) #zmiana typu danych kolumny z datą na typ datatime
k_uah['effectiveDate'].dtypes #sprawdzenie typu danych kolumny z datą po zmianie
k_uah = k_uah.set_index("effectiveDate").drop(columns='no') #zmiana indeksu na datę i usunięcie kolumny "no"
k_uah.head()

#zad 4

#generowanie listy wartości waluty względem złotego 
k_chf2 = []
for tmp in k_chf:
    k_chf2.append(tmp['mid'])

k_uah2 = []
for tmp in k_uah:
    k_uah2.append(tmp['mid'])

#tablica korelacji
corrcoef(array(k_chf2), array(k_uah2))

#Zad5

chart_data_chf = k_chf.set_index(['effectiveDate'])['mid']
chart_data_pln = k_uah.set_index(['effectiveDate'])['mid']

fig, axs = plt.subplots(1,2, sharex=True, sharey=True) 
axs[0].plot(chart_data_chf) #dodanie danych k_chf do wykresu 1.
axs[1].plot(chart_data_pln) #dodanie danych k_uah do wykresu 2.

plt.show()