# -*- coding: utf-8 -*-
"""Uso de framework.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Cqr5J-OHZq-H1ldiU3GHEVDlVdAkY6Yt
"""

#importar librerias necesarias
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score,  classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense


#leer la data
df=pd.read_csv("titanic.csv")
#mostrar la forma de los datos originales
print("Data sin limpieza:", df.shape)
#eliminar columnas que no son importantes para el analisis
df.drop(["PassengerId", "Name","Ticket","Cabin"], axis=1, inplace=True)
#encontrar los valores vacios
print("Valores vacios en la data:\n",df.isnull().sum())
#rellenar los valores vacios en la columna Age con la mediana de los datos
mage = df['Age'].median()
df['Age'].fillna(mage, inplace=True)
#eliminar el resto de valores vacios pues son pocos
df.dropna(inplace=True)
print("Valores vacios en la data despues de la limpeza:\n",df.isnull().sum())
#utilizar labelencoder para reemplazar los valores categoricos por numericos asignado valores apartir de 1 dependiendo de los valores unicos presentes
# en cada calumna
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
#dado que el modelo que deseamos uitlizar se ve altamente afectado por las grandes diferencias numericas que existen en nuestros datos
#utilizaremos MinMaxscaler para realizar una transformacion en los datos
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])
print("normalizacion de la data")
#finalmente reacomodaremos los datos
df=df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
       'Embarked','Survived']]

#con ayuda de las funciones de sklearn hacemos la division de la data en 3 conjuntos, uno de entrenmiento con el 60% de los datos,
#uno de validacion y test con el 20% de los datos respectivamente
X = df.drop('Survived', axis=1)
y = df['Survived']
# Dividir los datos en tres conjuntos: entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#generación del modelo
model = Sequential()
#agregamos 3 capas ocultas la primera de 64 nueronas, la segunda de 32 y la ultima de 16,
#emulando la arquitectura de una red Neuronal Feedforward pues podemos jugar con su configuracion
#facimente mas adelante, ademas por su alto rendimiento sera efectiva para nuestra gran cantidad de datos
#entodas utilizamos la funcion de activacion relu pues es la mas convencional

model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
#capa de salida (1 neurona para la clasificación binaria, es por ello que la funcion es sigmoide)
model.add(Dense(1, activation='sigmoid'))
#compilación del modelo, se utilizo binary_crossentropy pues es la mas adecuada en problemas de clasificacion
#binaria, un optimizador de adam pues es de los mas comunes y eficientes y de metrica accuracy pues se pueden
#obtener las demas con la funcion reporte
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#entrenamiento del modelo con 50 epocas para tener una cantidad considerable y revisar si eran suficientes o
#demaciadas, un batch size de 32 dada la forma del data set pues es una catidad considerable de datos
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
#evaluación del modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')


#creamos una lista que guardara las matrices de confusion de cada iteracion
list_cm=[]
#realizar predicciones en el conjunto de prueba
y_test_pred = model.predict(X_test)
y_test_pred_binary = (y_test_pred > 0.5)  # Convertir a etiquetas binarias (0 o 1)
#obtener el informe de clasificación
class_report_test = classification_report(y_test, y_test_pred_binary)
#obtener la matriz de confusión
list_cm.append(confusion_matrix(y_test, y_test_pred_binary))
#mostrar el informe de clasificación
print("\nInforme de Clasificación para el conjunto de prueba:")
print(class_report_test)

#realizar predicciones en el conjunto de validacion
y_val_pred = model.predict(X_val)
y_val_pred_binary = (y_val_pred > 0.5)  # Convertir a etiquetas binarias (0 o 1)
#obtener el informe de clasificación
class_report_test = classification_report(y_val, y_val_pred_binary)
#obtener la matriz de confusión
list_cm.append(confusion_matrix(y_val, y_val_pred_binary))
#mostrar el informe de clasificación
print("Informe de Clasificación para el conjunto de prueba:")
print(class_report_test)

# Creamos una figura con 2 subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Mostramos las matrices de confusión en subplots con ayuda de la lista generada anteriormente
for i in range(2):
    sns.heatmap(list_cm[i], annot=True, cmap="Blues", fmt="d", ax=axes[i],xticklabels=["No sobrevive", "Sobrevive"], yticklabels=["No sobrevive", "Sobrevive"])
    axes[i].set_title('Matriz de Confusión ' + str(i + 1))
#se agrega espacio entre los subplots y se despliega la grafica
plt.tight_layout()
plt.show()