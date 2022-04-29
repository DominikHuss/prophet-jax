import os
from typing import List, Dict, Any
from functools import partial
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax

PI = np.arccos(np.zeros(1)).item() * 2

def _generate_changepoint_matrix(T: np.ndarray,
                                 n_changepoints: int,
                                 changepoint_range: float):
    T_shape = T.shape[0]
    A = np.ones((T_shape, n_changepoints))
    changepoints = []
    period = int((T_shape * changepoint_range) / n_changepoints)
    for i in range(n_changepoints):
        A[i * period : (i + 1) * period] = np.concatenate(
            (
                np.ones((period, i + 1)),
                np.zeros((period, n_changepoints - i - 1)),
            ),
            1,
        )
        changepoints.append([T[i * period][0]])
    changepoints = np.array(changepoints)
    _changepoint_matrix = A
    return changepoints, _changepoint_matrix

def _add_seasonality(T: np.ndarray,
                    period: float,
                    fourier_order: int,
                    *,
                    X: np.ndarray = None):
    arg = np.array([(2 * PI * o) / period for o in range(1, fourier_order + 1)])
    arg = np.tile(arg, (T.shape[0], 1))
    _T = np.tile(T, (1, fourier_order)) * arg
    sin, cos = np.sin(_T), np.cos(_T)
    A = np.concatenate((sin, cos), -1)
    if X is not None:
        A = np.concatenate((X, A), -1)
    return A

def apply_transformations(T: np.ndarray,
                          *,
                          trend: str = "linear",
                          seasonality: List[Dict[str, Any]] = None,
                          holidays: List[Dict[str, Any]] = None) -> np.ndarray:
    X = None
    if seasonality is not None:
        for s in seasonality:
            X = _add_seasonality(T, s["period"], s["fourier_order"], X=X)

    changepoints, A = _generate_changepoint_matrix(T, n_changepoints=25, changepoint_range=1.)
    if trend == "linear":
        AT = A*T
        AS = -A*changepoints.T
        if X is None:
            X = np.concatenate((AT+AS, np.ones_like(T)), -1)
        else:
            X = np.concatenate((X, AT+AS, np.ones_like(T)), -1)
        return X
    elif trend == "logistic":
        if X is None:
            X = np.concatenate((np.ones_like(T), T), -1)
        else:
            X = np.concatenate((X, np.ones_like(T), T), -1)
        return changepoints, A, X

def plot(X, y_pred, y_true=None, f_name="a.png"):
    plt.figure()
    plt.plot(X, y_pred)
    if y_true is not None:
        plt.plot(X, y_true)
    plt.savefig(os.path.join("output", f_name))