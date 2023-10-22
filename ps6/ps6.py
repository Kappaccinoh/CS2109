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
# x = torch.linspace(-math.pi, math.pi, 1000)

# # Original true values
# y = torch.sin(x)
# plt.plot(x, y, linestyle='solid', label='sin(x)')

# # MSE Function
# mse = lambda y_true, y_pred: torch.mean(torch.square(y_pred - y_true))

# # MSE - Graph 1
# a, b, c, d = polyfit(x, y, mse, 3, 1e-6, 5000)
# y_pred_mse1 = a + b * x + c * x ** 2 + d * x ** 3
# plt.plot(x, y_pred_mse1.detach().numpy(), linestyle='dashed', label=f'mse-1')

# # MSE - Graph 2
# a, b, c, d = polyfit(x, y, mse, 3, 1e6, 5000)
# y_pred_mse2 = a + b * x + c * x ** 2 + d * x ** 3
# plt.plot(x, y_pred_mse2.detach().numpy(), linestyle='dashed', label=f'mse-2')

# # MSE - Graph 3
# a, b = polyfit(x, y, mse, 1, 1e-3, 5000)
# y_pred_mse3 = a + b * x
# plt.plot(x, y_pred_mse3.detach().numpy(), linestyle='dashed', label=f'mse-3')

# # MSE - Graph 4
# a, b, c, d, e, f, g = polyfit(x, y, mse, 6, 1e-8, 5000)
# y_pred_mse4 = a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4 + f * x ** 5 + g * x ** 6
# plt.plot(x, y_pred_mse4.detach().numpy(), linestyle='dashed', label=f'mse-4')

# plt.axis('equal')
# plt.title('Comparison of different fits')
# plt.legend()
# plt.show()


# Task 2.1
x = torch.linspace(-10, 10, 1000).reshape(-1, 1)
y = torch.abs(x-1)

def forward_pass(x, w0, w1, activation_fn):
    '''
    x - (n, 1) -> add bias to make (n , 2)
    w0 - (2,2)
    w1 - (3,1) -> add bias to make (3,2)
    '''
    x = torch.cat((x, torch.ones((len(x), 1))), dim=1)
    hiddenOutput0 = activation_fn(
        torch.matmul(x, w0[0])
    ).reshape(-1,1) # (n,1)

    hiddenOutput1 = activation_fn(
        torch.matmul(x, w0[1])
    ).reshape(-1,1) # (n,1)

    hiddenOutput = torch.cat((hiddenOutput0, hiddenOutput1), axis=1) # (n,2)
    ones = torch.ones((len(x), 1))
    hiddenOutput = torch.cat((ones, hiddenOutput), axis=1) # (n,3)

    finalOut = torch.matmul(hiddenOutput, w1) # (n,1)

    return finalOut


# # Exact weights
# w0 = torch.tensor([[-1., 1.], [1., -1.]], requires_grad=True)
# w1 = torch.tensor([[0.], [1.], [1.]], requires_grad=True)

# # Performing a forward pass on exact solution for weights will give us the correct y values
# x_sample = torch.linspace(-2, 2, 5).reshape(-1, 1)
# forward_pass(x_sample, w0, w1, torch.relu) # tensor([[3.], [2.], [1.], [0.], [1.]])

# Task 2.2
torch.manual_seed(1) # Set seed to some fixed value

w0 = torch.randn(2, 2, requires_grad=True)
w1 = torch.randn(3, 1, requires_grad=True)

learning_rate = 1e-3
print('iter', 'loss', '\n----', '----', sep='\t')
for t in range(1, 10001):
    # Forward pass: compute predicted y
    y_pred = forward_pass(x, w0, w1, torch.relu)

    loss = torch.mean(torch.square(y - y_pred))
    loss.backward()

    if t % 1000 == 0:
        print(t, loss.item(), sep='\t')

    with torch.no_grad():
        w0 -= learning_rate * w0.grad
        w1 -= learning_rate * w1.grad
        w0.grad.zero_()
        w1.grad.zero_()
        

print("--- w0 ---", w0, sep='\n')
print("--- w1 ---", w1, sep='\n')
y_pred = forward_pass(x, w0, w1, torch.relu)
plt.plot(x, y, linestyle='solid', label='|x-1|')
plt.plot(x, y_pred.detach().numpy(), linestyle='dashed', label='perceptron')
plt.axis('equal')
plt.title('Fit NN on abs function')
plt.legend()
plt.show()

# Task 5: Submit the values of `w0`, `w1`, and `loss` values after fitting
# Note: An acceptable loss value should be less than 1.0
#       You should try adjusting the random seed, learning rate, or 
#       number of iterations to improve your model.

w0   = [[1.2394, -1.2318], [0.3287,  2.6828]] # to be computed
w1   = [[9.1564], [1.6083], [-3.0357]]     # to be computed
loss = 0.106                       # to be computed

# # Task 2.3


# Task 3


if __name__ == "__main__":
    # print("main")
    w0 = torch.tensor([[-1., 1.], [1., -1.]], requires_grad=True)
    w1 = torch.tensor([[0.], [1.], [1.]], requires_grad=True)

    output0 = forward_pass(torch.linspace(0,1,50).reshape(-1, 1), w0, w1, torch.relu)
    x_sample = torch.linspace(-2, 2, 5).reshape(-1, 1)
    test1 = forward_pass(x_sample, w0, w1, torch.relu).tolist()
    output1 = [[3.], [2.], [1.], [0.], [1.]]

    assert output0.shape == torch.Size([50, 1])
    assert test1 == output1