from jax_prophet.linear_prophet import fit_linear_prophet, infer_linear_prophet
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


Y = y_data
X = apply_transformations(x_data, trend="linear", seasonality=[{"period": 0.5, "fourier_order": 10}, {"period": 0.2, "fourier_order": 10}, {"period": 0.01, "fourier_order": 10}])
w = fit_linear_prophet(X, y_data, 0.1, 10000)
y_pred = infer_linear_prophet(X, w)
plot(x_data, y_pred, y_true=y_data)
