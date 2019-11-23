
from sklearn.decomposition import PCA
from sklearn import datasets

# Wczytaj przykładowy zbiór danych - dane dotyczące trzech gatunków Irysów
iris = datasets.load_iris()

# Podzielmy zbiór na cechy oraz etykiety
# Zostawiamy tym razem wszystkie cechy - będziemy próbować odgadnąć które cechy są najważniejsze
X = iris.data
y = iris.target

# Inicjalizacja. Można od razu wypełnić n_components, na razie wykorzystujemy wszystkie cechy 
# pca = PCA(n_components=3)

pca = PCA()
pca.fit(X)

print(pca.n_components_)
print(pca.explained_variance_ratio_)


# Jedna cecha tłumaczy prawie wszystko? Sprawdźmy!

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
%matplotlib inline

# wykresy będą tworzone przy pomocy pakietu seaborn
import seaborn as sns

# konwersja na obiekt pandas.DataFrame
iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

# funkcja która nam zamieni wartości 0, 1, 2 na pełny opis tekstowy dla gatunku
targets = map(lambda x: iris['target_names'][x], iris['target'] )

# doklejenie informacji o gatunku do reszty dataframe
iris_df['species'] = np.array(list(targets))

# wykres
sns.pairplot(iris_df, hue='species')
plt.show()



pca_limit = PCA(n_components = 1)

X_new = pca_limit.fit_transform(X)

print(pca_limit.n_components_)
print(pca_limit.components_)
print(pca_limit.explained_variance_ratio_)


# Po użyciu funkcji transform (lub fit_transform) dekompozycja pozostawiła nam tylko liczbę cech, którą skonfigurowaliśmy
# Dodatkowo została od nich odjęta średnia, więc dane zawierają tylko wariancję

X_new[:5]


plt.scatter(X_new, y)
plt.show()

# Zadanie 1:
# Wyjaśnij w kilku zdaniach która cecha została wybrana przez PCA i dlaczego według Ciebie właśnie ta
#powierzchnia płatka
#obydwie skladowe cechy dla nizszych wartosci daja ta sama klase, dla sredniej ta sama i dla najwyzszej ta sama klase
#do sztucznej cechy zostala wzieta dlugosc platka, ktora ma najwieksza rozpietosc wartosci

# Zadanie bonus: przeprowadź dekompozycję PCA na wybranym przez siebie innym zbiorze danych