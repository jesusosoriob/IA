# Predicción de Series Temporales utilizando LSTM con Precios de Acciones

En este notebook, realizaremos una predicción de series temporales utilizando un modelo LSTM (Long Short-Term Memory)
para predecir el precio de cierre de una acción en función de datos históricos.
Usaremos la API de Yahoo Finance para descargar los datos y luego normalizaremos los valores antes de entrenar el modelo.

## 1. Importar las Bibliotecas y Descargar Datos

```python
import yfinance as yf

# Descargamos datos históricos de una acción específica (ejemplo: Apple)
data = yf.download('AAPL', start='2010-01-01', end='2023-01-01')

# Mostramos las primeras filas del dataset
#print(data.head())
```
# Descripción:

Utilizamos yfinance para descargar datos históricos de la acción de Apple (#AAPL) desde el 1 de enero de 2010 hasta el 1 de enero de 2023.
El print(data.head()) está comentado, pero mostraría las primeras filas del conjunto de datos descargado.

## 2. Normalización de los Datos

```python
from sklearn.preprocessing import MinMaxScaler

# Normalizamos la columna 'Close'
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

print(scaled_data[:5])  # Visualizamos los primeros 5 valores escalados

#dividir los datos en conjunto de entrenamiento y prueba 
from sklearn.model_selection import train_test_split
```
# Descripción:

Utilizamos MinMaxScaler de sklearn.preprocessing para normalizar los precios de cierre (Close), ajustando los valores entre 0 y 1.
Esto es importante en modelos como LSTM, ya que ayuda a mejorar la eficiencia del entrenamiento y la convergencia del modelo.

## 3. Creación del Conjunto de Datos para Series Temporales

```python
# crear el conjunto de datos para series temporales

import numpy as np

def create_dataset(data, time_step=1):
    X, y = []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 10  # Por ejemplo, si usas ventanas de 10 días
X, y = create_dataset(scaled_data, time_step)
```

# Descripción:

La función create_dataset transforma los datos de series temporales en un formato adecuado para el modelo LSTM. Aquí, time_step=10 significa que estamos utilizando ventanas deslizantes de 10 días para predecir el día siguiente.
X contiene los valores de entrada (historia de precios) y y contiene el valor objetivo (precio siguiente).

## 4. División del Conjunto de Datos en Entrenamiento y Prueba

```python
from sklearn.model_selection import train_test_split

# Dividimos los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

# Descripción:

Usamos train_test_split para dividir los datos en conjuntos de entrenamiento (80%) y prueba (20%).
Esto es esencial para evaluar el rendimiento del modelo y evitar el sobreajuste.

## 5. Preparación de los Datos para el Modelo LSTM

```python
# reshape para el modelo de deep learning

# Reshape para LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
```

# Descripción:

Reajustamos la forma de X_train y X_test a un formato compatible con las capas LSTM, donde cada muestra debe tener 3 dimensiones: (muestras, pasos temporales, características).

## 6. Construcción del Modelo LSTM

```python
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

input_layer = Input(shape=(time_step, 1))
x = LSTM(50, return_sequences=True)(input_layer)
x = LSTM(50, return_sequences=False)(x)
output_layer = Dense(1)(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mean_squared_error')
```

# Descripción:

Utilizamos la API funcional de Keras para construir un modelo LSTM.
Capa de entrada (Input): toma entradas de forma (time_step, 1), donde time_step es el número de días de historia.
Capa LSTM: Dos capas LSTM secuenciales, la primera con return_sequences=True para devolver toda la secuencia, y la segunda con return_sequences=False para devolver solo el último valor de la secuencia.
Capa de salida (Dense): Produce un solo valor como predicción.
Compilamos el modelo con el optimizador adam y la función de pérdida mean_squared_error, adecuada para regresión.

## 7. Entrenamiento del Modelo

```python
# Entrenamos el modelo con el conjunto de datos
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)
```

# Descripción:

Entrenamos el modelo con los datos de entrenamiento usando 20 épocas y un tamaño de lote (batch_size) de 32.
Usamos el 10% de los datos de entrenamiento para validación (validation_split=0.1) durante el entrenamiento.

## 8. Evaluación del Modelo

```python
# Evaluación del rendimiento del modelo
loss = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')
```

## Descripción:

Evaluamos el rendimiento del modelo con el conjunto de prueba (X_test y y_test) y mostramos la pérdida (loss), que indica el error del modelo en la predicción de precios de las acciones.

## 9. Predicciones y Visualización de Resultados

```python
# Hacer predicciones
predictions = model.predict(X_test)

# Visualizar resultados
import matplotlib.pyplot as plt

plt.plot(y_test, label='Real')
plt.plot(predictions, label='Predicción')
plt.legend()
plt.show()
```

# Descripción:

Utilizamos el modelo entrenado para predecir los precios en el conjunto de prueba.
Visualizamos los resultados utilizando matplotlib. Se grafica la serie de precios reales versus las predicciones del modelo para evaluar visualmente su desempeño.

## Conclusión:

Este notebook demuestra cómo construir y entrenar un modelo LSTM para la predicción de series temporales utilizando datos históricos de precios de acciones. El uso de LSTM es beneficioso para captar dependencias a largo plazo en los datos, como las series temporales financieras.
