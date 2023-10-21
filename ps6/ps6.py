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
    X = torch.vander(x, n + 1, increasing=True)
    w = torch.zeros(n + 1, requires_grad=True)

    for i in range(n_iter):
        yHat = torch.mv(X, w)
        loss = loss_fn(y, yHat)

        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            w.grad.zero_()

    return w

# Task 1.3
x = torch.linspace(-math.pi, math.pi, 1000)

# Original true values
y = torch.sin(x)
plt.plot(x, y, linestyle='solid', label='sin(x)')

# MSE - Graph 1
mse = lambda y_true, y_pred: torch.mean(torch.square(y_pred - y_true))
a, b, c, d = polyfit(x, y, mse, 3, 1e-6, 5000)
y_pred_mse1 = a + b * x + c * x ** 2 + d * x ** 3
plt.plot(x, y_pred_mse1.detach().numpy(), linestyle='dashed', label=f'mse-1')

# MSE - Graph 2
a, b, c, d = polyfit(x, y, mse, 3, 1e6, 5000)
y_pred_mse2 = a + b * x + c * x ** 2 + d * x ** 3
plt.plot(x, y_pred_mse2.detach().numpy(), linestyle='dashed', label=f'mse-2')

# MSE - Graph 3
a, b = polyfit(x, y, mse, 1, 1e-3, 5000)
y_pred_mse3 = a + b * x
plt.plot(x, y_pred_mse3.detach().numpy(), linestyle='dashed', label=f'mse-3')

# MSE - Graph 4
a, b, c, d, e, f, g = polyfit(x, y, mse, 6, 1e-3, 5000)
y_pred_mse4 = a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4 + f * x ** 5 + g * x ** 6
plt.plot(x, y_pred_mse4.detach().numpy(), linestyle='dashed', label=f'mse-4')

plt.axis('equal')
plt.title('Comparison of different fits')
plt.legend()
plt.show()

if __name__ == "__main__":
    print("plotting")