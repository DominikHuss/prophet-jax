from jax_prophet.logistic_prophet import fit_logistic_prophet, infer_logistic_prophet
from jax_prophet.utils import plot, apply_transformations
import pandas as pd
import numpy as np


url = 'https://raw.githubusercontent.com/facebook/prophet/master/python/prophet/tests/data.csv'
df = pd.read_csv(url, error_bad_lines=False)
df['ds'] = pd.to_datetime(df['ds'])
df['ds'] = df.ds.values.astype(np.int64) // 10 ** 9

x_data = np.array(df["ds"].values)[:,np.newaxis]
x_data = (x_data - np.min(x_data))
x_data = x_data / np.max(x_data)
y_data = np.array(df["y"].values)[:,np.newaxis]

print(x_data.shape)
train_split = 500
Y = y_data[:train_split,:]
changepoints, A, X = apply_transformations(x_data[:train_split,:], trend="logistic", seasonality=[{"period": 0.5, "fourier_order": 10}, {"period": 0.2, "fourier_order": 10}, {"period": 0.01, "fourier_order": 10}])
w = fit_logistic_prophet(X, Y, 0.005, 1000, changepoints, A)
y_pred = infer_logistic_prophet(X, w, changepoints, A)
plot(x_data[:train_split,:], y_pred, y_data[:train_split,:])

changepoints, A, X = apply_transformations(x_data, trend="logistic", seasonality=[{"period": 0.5, "fourier_order": 10}, {"period": 0.2, "fourier_order": 10}, {"period": 0.01, "fourier_order": 10}])
y_pred = infer_logistic_prophet(X, w, changepoints, A)
plot(x_data, y_pred, y_data, f_name="b.png")