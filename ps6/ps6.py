# RUN THIS CELL FIRST
import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from numpy import allclose, isclose

# Task 1.2
def polyfit(x, y, loss_fn, n, lr, n_iter):
    """
    Parameters
    ----------
        x : A tensor of shape (1, n)
        y : A tensor of shape (1, n)
        loss_fn : Function to measure loss
        n : The nth-degree polynomial
        lr : Learning rate
        n_iter : The number of iterations of gradient descent
        
    Returns
    -------
        Near-optimal coefficients of the nth-degree polynomial as a tensor of shape (1, n+1) after `n_iter` epochs.
    """
    
    


x = torch.linspace(-math.pi, math.pi, 1000)

# Original true values
y = torch.sin(x)
plt.plot(x, y, linestyle='solid', label='sin(x)')

# MSE
mse = lambda y_true, y_pred: torch.mean(torch.square(y_pred - y_true))
a, b, c, d = polyfit(x, y, mse, 3, 1e-3, 5000)
y_pred_mse = a + b * x + c * x ** 2 + d * x ** 3
plt.plot(x, y_pred_mse.detach().numpy(), linestyle='dashed', label=f'mse')

# MAE
mae = lambda y_true, y_pred: torch.mean(torch.abs(y_pred - y_true))
a, b, c, d = polyfit(x, y, mae, 3, 1e-3, 5000)
y_pred_mae = a + b * x + c * x ** 2 + d * x ** 3
plt.plot(x, y_pred_mae.detach().numpy(), linestyle='dashed', label=f'mae')

plt.axis('equal')
plt.title('Comparison of different fits')
plt.legend()
plt.show()

if __name__ == "__main__":
    x = torch.linspace(-math.pi, math.pi, 10)
    y = torch.sin(x)

    def mse(y_true, y_pred):
        assert y_true.shape == y_pred.shape, f"Your ground truth and predicted values need to have the same shape {y_true.shape} vs {y_pred.shape}"
        return torch.mean(torch.square(y_pred - y_true))
    def mae(y_true, y_pred):
        assert y_true.shape == y_pred.shape, f"Your ground truth and predicted values need to have the same shape {y_true.shape} vs {y_pred.shape}"
        return torch.mean(torch.abs(y_pred - y_true))

    test1 = polyfit(x, x, mse, 1, 1e-1, 100).tolist()
    test2 = polyfit(x, x**2, mse, 2, 1e-2, 2000).tolist()
    test3 = polyfit(x, y, mse, 3, 1e-3, 5000).tolist()
    test4 = polyfit(x, y, mae, 3, 1e-3, 5000).tolist()

    assert allclose(test1, [0.0, 1.0], atol=1e-6)
    assert allclose(test2, [0.0, 0.0, 1.0], atol=1e-5)
    assert allclose(test3, [0.0, 0.81909, 0.0, -0.08469], atol=1e-3)
    assert allclose(test4, [0.0, 0.83506, 0.0, -0.08974], atol=1e-3)