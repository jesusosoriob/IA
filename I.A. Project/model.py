import yfinance as yf

# Descargamos datos históricos de una acción específica (ejemplo: Apple)
data = yf.download('AAPL', start='2010-01-01', end='2023-01-01')


# NORMALIZACION DE DATOS (entre 0 - 1)

from sklearn.preprocessing import MinMaxScaler

# Normalizamos la columna 'Close'
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

print(scaled_data[:5])  # Visualizamos los primeros 5 valores escalados

# dividir los datos en conjunto de entrenamiento y prueba 

from sklearn.model_selection import train_test_split


# crear el conjunto de datos para series temporales

import numpy as np

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 10  # Por ejemplo, si usas ventanas de 10 días
X, y = create_dataset(scaled_data, time_step)

# dividimos los datos en conjunto de entrenamiento y prueba 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Reshape para el modelo deep learning
X = X.reshape((X.shape[0], X.shape[1], 1))

# construir el modelo LSTM

from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

input_layer = Input(shape=(time_step, 1))
x = LSTM(50, return_sequences=True)(input_layer)
x = LSTM(50, return_sequences=False)(x)
output_layer = Dense(1)(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mean_squared_error')


# entrenamos el modelo con el conjunto de datos 

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# evalua el rendimiento del modelo 

loss = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')

# hacer predicciones 

predictions = model.predict(X_test)

# visualizar resultados 

import matplotlib.pyplot as plt

plt.plot(y_test, label='Real')
plt.plot(predictions, label='Predicción')
plt.legend()
plt.show()
