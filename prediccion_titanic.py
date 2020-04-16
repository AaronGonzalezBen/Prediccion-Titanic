"""
EJERCICIO DE PREDICCION DE MUERTES DEL TITANIC
"""

# ANALIZANDO EL DATASET

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Importamos los datos
dir_train = "train.csv"
dir_test = "test.csv"

df_train = pd.read_csv(dir_train)
df_test = pd.read_csv(dir_test)

print(df_train.head())
print(df_test.head())

# Verifico la cantidad de datos del dataset
print("Cantidad de datos:")
print(df_train.shape)
print(df_test.shape)    # El test no posee el target, ya que con estos datos es que se implementa la prueba de la prediccion

# Verifico los tipos de datos de ambos datasets
print("Tipos de datos: ")
print(df_train.info())
print(df_test.info())

# Verificamos datos faltantes
print("Datos faltantes: ")
print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())

# Revisamos las estadisticas del dataset
print("Estadisticas generales:")
print(df_train.describe())
print(df_test.describe())

# PREPROCESAMIENTO DE LOS DATOS

# 1. Cambio de tipo de datos columna Sexo a numerico
df_train['Sex'].replace(['female','male'],[0,1], inplace = True)
df_test['Sex'].replace(['female','male'],[0,1], inplace = True)

# 2. Cambio de tipo de datos columna Embarque a numerico
df_train['Embarked'].replace(['Q','S','C'],[0,1,2],inplace=True)
df_test['Embarked'].replace(['Q','S','C'],[0,1,2],inplace=True)

# 3. Reemplazo los datos faltantes de Edad por la media
mean_train = round(df_train['Age'].mean())
mean_test = round(df_test['Age'].mean())
df_train['Age'] = df_train['Age'].replace(np.nan, mean_train)
df_test['Age'] = df_test['Age'].replace(np.nan, mean_test)

# 4. Creacion de grupos a partir de la Edad
# Bandas: 0-8, 9-15, 16-18, 19-25, 26-40, 41-60, 61-100
bins = [0,8,15,18,25,40,60,100]
names = ['1','2','3','4','5','6','7']
df_train['Age'] = pd.cut(df_train['Age'], bins, labels = names)
df_test['Age'] = pd.cut(df_test['Age'], bins, labels = names)

# 5. Se elimina la columna "Cabina" debido a la cantidad de datos perdidos que tiene
df_train.drop(['Cabin'], axis = 1, inplace = True)
df_test.drop(['Cabin'], axis = 1, inplace = True)

# 6. Se eliminan las columnas no necesarias para el analisis
df_train = df_train.drop(['PassengerId','Name','Ticket'], axis = 1)
df_test = df_test.drop(['Name','Ticket'], axis = 1)     # Se utilizara mas adelante la columna PassengerId

# 7. Se eliminan las filas con los datos perdidos
df_train.dropna(axis = 0, how = 'any', inplace = True)
df_test.dropna(axis = 0, how = 'any', inplace = True)

# 8. Verificamos los datos
# Tambien funciona df_train.isnull().sum()

print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())

print(df_train.shape)
print(df_test.shape)

# Implementacion del algoritmo de ML

# 1. Separo la columna con la informacion de los sobrevivientes (target)
X = np.array(df_train.drop(['Survived'],1))
y = np.array(df_train['Survived'])

# 2. Separo los datos de train y test para el algoritmo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# 3. Aplico un modelo de Regresion Logistica
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
print()
print('Precisipn de la Regresion Logistica: ')
print(logreg.score(X_train, y_train))

# 4. Aplico un modelo de SVM
svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
print()
print('Precision de algoritmo SVM:')
print(svc.score(X_train, y_train))

# 5. Aplico un modelo de K vecinos mas cercanos (KNN)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
print()
print('Precision del algoritmo KNN:')
print(knn.score(X_train, y_train))

# APLICAMOS UNA PREDICCION CON LOS DATOS DE PRUEBA

ids = df_test['PassengerId']

# 1. Regresion logistica
prediccion_logreg = logreg.predict(df_test.drop('PassengerId', axis = 1))
out_logreg = pd.DataFrame({'PassengerId' : ids, 'Survived' : prediccion_logreg})
print('Prediccion Regresion Logistica:')
print(out_logreg.head())

# 2. SVM
prediccion_svc = svc.predict(df_test.drop('PassengerId', axis = 1))
out_svc = pd.DataFrame({'PassengerId' : ids, 'Survived' : prediccion_svc})
print('Prediccion SVC:')
print(out_svc.head())

# 3. KNN
prediccion_knn = knn.predict(df_test.drop('PassengerId', axis = 1))
out_knn = pd.DataFrame({'PassengerId' : ids, 'Survived' : prediccion_knn})
print('Prediccion KNN:')
print(out_knn.head())
