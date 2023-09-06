#importar librerias necesarias
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#leer la data
df=pd.read_csv("C:/Users/Admin/Downloads/titanic.csv")
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

# Crear un modelo de árbol de decisión
clf = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo en el conjunto de entrenamiento
clf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de validación
y_val_pred = clf.predict(X_val)

# Calcular la precisión en el conjunto de validación
accuracy_val = accuracy_score(y_val, y_val_pred)
print("Precisión en el conjunto de validación:", accuracy_val)

# Realizar predicciones en el conjunto de prueba
y_test_pred = clf.predict(X_test)

# Calcular la precisión en el conjunto de prueba
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Precisión en el conjunto de prueba:", accuracy_test)

# Mostrar un informe de clasificación para el conjunto de prueba
class_report_test = classification_report(y_test, y_test_pred)
print("Informe de clasificación para el conjunto de prueba:\n", class_report_test)