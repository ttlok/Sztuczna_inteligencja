from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
%matplotlib inline

# Wczytanie zbioru cech nieruchomości i ich cen
boston_nieruchomosci = load_boston()

print('Klucze dostępne w zbiorze danych: ', boston_nieruchomosci.keys())
print(boston_nieruchomosci.DESCR)

print('Przykładowe wartości cech:\n', boston_nieruchomosci.data[:3])
print('Przykładowe kwoty: ', boston_nieruchomosci.target[:3])

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
%matplotlib inline

import seaborn as sns

# konwersja na obiekt pandas.DataFrame
boston_df = pd.DataFrame(boston_nieruchomosci['data'], columns=boston_nieruchomosci['feature_names'])

# doklejenie informacji o cenie do reszty dataframe
boston_df['target'] = np.array(list(boston_nieruchomosci['target']))

# wykres
sns.pairplot(boston_df)
plt.show()

# Dużo zmiennych... na początek spróbujmy z jedną
l_pokoi = boston_nieruchomosci['data'][:, np.newaxis, 5]
plt.scatter(l_pokoi, boston_nieruchomosci['target'])
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Stworzenie regresora liniowego
linreg = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(l_pokoi, boston_nieruchomosci['target'], test_size = 0.3)

linreg.fit(X_train, y_train)

# przewidywanie ceny
y_pred = linreg.predict(X_test)

# domyślna metryka
print('Metryka domyślna: ', linreg.score(X_test, y_test))

# wskaźnik (metryka) r^2
# dokumentacja: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
print('Metryka r2: ', r2_score(y_test, y_pred))

# współczynniki regresji
print('Współczynniki regresji:\n', linreg.coef_)

# Wykres regresji
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=2)
plt.show()


# dokumentacja: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
from sklearn.model_selection import cross_val_score

cv_score_r2 = cross_val_score(linreg, l_pokoi, boston_nieruchomosci.target, cv=5, scoring='r2')
print(cv_score_r2)

# używamy innej metryki 
# dokumentacja:
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
cv_score_ev = cross_val_score(linreg, l_pokoi, boston_nieruchomosci.target, cv=5, scoring='explained_variance')
print(cv_score_ev)

# ...i jeszcze innej
cv_score_mse = cross_val_score(linreg, l_pokoi, boston_nieruchomosci.target, cv=5, scoring='neg_mean_squared_error')
print(cv_score_mse)



# Zadanie 1
# Wybierz inną cechę i spróbuj przewidzieć ceny mieszkań. Użyj walidacji krzyżowej.
# Zadanie bonus: zaimportuj dane nt cukrzycy przy pomocy funkcji load_diabetes
# Następnie przeanalizuj dane i zaproponuj regresor liniowy. Sprawdź jakość modelu za pomocą walidacji krzyżowej.


# Zadanie 1
# Wybierz inną cechę i spróbuj przewidzieć ceny mieszkań. Użyj walidacji krzyżowej.
# Zadanie bonus: zaimportuj dane nt cukrzycy przy pomocy funkcji load_diabetes
# Następnie przeanalizuj dane i zaproponuj regresor liniowy. Sprawdź jakość modelu za pomocą walidacji krzyżowej.


from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
%matplotlib inline



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
%matplotlib inline

import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict



from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


# Wczytanie zbioru cech nieruchomości i ich cen
boston_nieruchomosci = load_boston()

# wybralem zmienna dotyczaca dostepu do drog
idx_drog = boston_nieruchomosci['data'][:, np.newaxis, 5]

# Stworzenie regresora liniowego
linreg = LinearRegression()

#wybor zrbioru uczacego i testowego
X_train, X_test, y_train, y_test = train_test_split(idx_drog, boston_nieruchomosci['target'], test_size = 0.3)

#nauka regresora liniowego
linreg.fit(X_train, y_train)

# przewidywanie ceny
y_pred = linreg.predict(X_test)

# Wykres regresji
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=2)
plt.show()

# domyślna metryka
print('Metryka domyślna: ', linreg.score(X_test, y_test))

# teraz dla walidacji krzyzowej

#metryki dla walidacji krzyzowej
cv_score = cross_val_score(linreg, idx_drog, boston_nieruchomosci.target, cv=10, scoring='r2')
print(cv_score)

# prewidujemy wartosci po kolei 
y_predict = cross_val_predict(linreg, idx_drog, boston_nieruchomosci.target, cv=5)
plt.scatter(idx_drog, y_predict)


plt.plot(idx_drog, y_predict, color='blue', linewidth=2)
plt.show()
