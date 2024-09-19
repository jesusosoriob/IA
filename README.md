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
Descripción:

Utilizamos #yfinance para descargar datos históricos de la acción de Apple (#AAPL) desde el 1 de enero de 2010 hasta el 1 de enero de 2023.
El ##print(data.head()) está comentado, pero mostraría las primeras filas del conjunto de datos descargado.
