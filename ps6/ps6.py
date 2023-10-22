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

# # Task 2.2
# torch.manual_seed(1) # Set seed to some fixed value

# w0 = torch.randn(2, 2, requires_grad=True)
# w1 = torch.randn(3, 1, requires_grad=True)

# learning_rate = 1e-3
# print('iter', 'loss', '\n----', '----', sep='\t')
# for t in range(1, 100000):
#     # Forward pass: compute predicted y
#     y_pred = forward_pass(x, w0, w1, torch.relu)

#     loss = torch.mean(torch.square(y - y_pred))
#     loss.backward()

#     if t % 1000 == 0:
#         print(t, loss.item(), sep='\t')

#     with torch.no_grad():
#         w0 -= learning_rate * w0.grad
#         w1 -= learning_rate * w1.grad
#         w0.grad.zero_()
#         w1.grad.zero_()
        

# print("--- w0 ---", w0, sep='\n')
# print("--- w1 ---", w1, sep='\n')
# y_pred = forward_pass(x, w0, w1, torch.relu)
# plt.plot(x, y, linestyle='solid', label='|x-1|')
# plt.plot(x, y_pred.detach().numpy(), linestyle='dashed', label='perceptron')
# plt.axis('equal')
# plt.title('Fit NN on abs function')
# plt.legend()
# plt.show()

# # Task 5: Submit the values of `w0`, `w1`, and `loss` values after fitting
# # Note: An acceptable loss value should be less than 1.0
# #       You should try adjusting the random seed, learning rate, or 
# #       number of iterations to improve your model.

# w0   = [[1.2394, -1.2318], [0.3287,  2.6828]] # to be computed
# w1   = [[9.1564], [1.6083], [-3.0357]]     # to be computed
# loss = 0.106                       # to be computed

# Task 2.3
'''
torch.manual_seed(1) # Set seed to some fixed value

# Pass 1

w0 = torch.randn(2, 2, requires_grad=True)
w1 = torch.randn(3, 1, requires_grad=True)

learning_rate = 1e-3
print('iter', 'loss', '\n----', '----', sep='\t')
for t in range(1, 10000):
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
plt.plot(x, y_pred.detach().numpy(), linestyle='dashed', label='perceptron-1')

# Pass 2

w0 = torch.randn(2, 2, requires_grad=True)
w1 = torch.randn(3, 1, requires_grad=True)

learning_rate = 1e-3
print('iter', 'loss', '\n----', '----', sep='\t')
for t in range(1, 10000):
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
plt.plot(x, y_pred.detach().numpy(), linestyle='dashed', label='perceptron-2')

# Pass 3

w0 = torch.randn(2, 2, requires_grad=True)
w1 = torch.randn(3, 1, requires_grad=True)

learning_rate = 1e-3
print('iter', 'loss', '\n----', '----', sep='\t')
for t in range(1, 10000):
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
plt.plot(x, y_pred.detach().numpy(), linestyle='dashed', label='perceptron-3')

# Plotting Final
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
'''

# Task 3.1
class MyFirstNeuralNet(nn.Module):
    def __init__(self): # set the arguments you'd need
        super().__init__()
        self.l1 = nn.Linear(1, 2) # bias included by default
        self.l2 = nn.Linear(2, 1) # bias included by default
        self.relu = nn.ReLU()
 
    # Task 3.1: Forward pass
    def forward(self, x):
        '''
        Forward pass to process input through two linear layers and ReLU activation function.

        Parameters
        ----------
        x : A tensor of of shape (n, 1) where n is the number of training instances

        Returns
        -------
            Tensor of shape (n, 1)
        '''
        x = self.relu(self.l1(x))
        x = self.l2(x)
        return x

# Task 3.2
'''
torch.manual_seed(6) # Set seed to some fixed value

epochs = 10000

model = MyFirstNeuralNet()
# the optimizer controls the learning rate
optimiser = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0)
loss_fn = nn.MSELoss()

x = torch.linspace(-10, 10, 1000).reshape(-1, 1)
y = torch.abs(x-1)

print('Epoch', 'Loss', '\n-----', '----', sep='\t')
for i in range(1, epochs+1):
    # reset gradients to 0
    optimiser.zero_grad()
    # get predictions
    y_pred = model(x)
    # compute loss
    loss = loss_fn(y_pred, y)
    # backpropagate
    loss.backward()
    # update the model weights
    optimiser.step()

    if i % 1000 == 0:
        print (f"{i:5d}", loss.item(), sep='\t')

y_pred = model(x)
plt.plot(x, y, linestyle='solid', label='|x-1|')
plt.plot(x, y_pred.detach().numpy(), linestyle='dashed', label='perceptron')
plt.axis('equal')
plt.title('Fit NN on y=|x-1| function')
plt.legend()
plt.show()

# To submit this output
print("--- Submit the OrderedDict below ---")
print(model.state_dict())
'''

# Task 3.3
# DO NOT REMOVE THIS CELL – THIS DOWNLOADS THE MNIST DATASET
# RUN THIS CELL BEFORE YOU RUN THE REST OF THE CELLS BELOW
from torchvision import datasets

# This downloads the MNIST datasets ~63MB
mnist_train = datasets.MNIST("./", train=True, download=True)
mnist_test  = datasets.MNIST("./", train=False, download=True)

x_train = mnist_train.data.reshape(-1, 784) / 255
y_train = mnist_train.targets
    
x_test = mnist_test.data.reshape(-1, 784) / 255
y_test = mnist_test.targets

class DigitNet(nn.Module):
    def __init__(self, input_dimensions, num_classes): # set the arguments you'd need
        super().__init__()
        """
        YOUR CODE HERE
        - DO NOT hardcode the input_dimensions, use the parameter in the function
        - Your network should work for any input and output size 
        - Create the 3 layers (and a ReLU layer) using the torch.nn layers API
        """
        self.l1 = nn.Linear(input_dimensions,512)
        self.l2 = nn.Linear(512,128)
        self.l3 = nn.Linear(128,num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Performs the forward pass for the network.
        
        Parameters
        ----------
        x : Input tensor (batch size is the entire dataset)

        Returns
        -------
            The output of the entire 3-layer model.
        """
        
        """
        YOUR CODE
        
        - Pass the inputs through the sequence of layers
        - Run the final output through the Softmax function on the right dimension!
        """
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = torch.softmax(self.l3(x), dim=1)
        return x

# Task 3.4
def train_model(x_train, y_train, epochs=20):
    """
    Trains the model for 20 epochs/iterations
    
    Parameters
    ----------
        x_train : A tensor of training features of shape (60000, 784)
        y_train : A tensor of training labels of shape (60000, 1)
        epochs  : Number of epochs, default of 20
        
    Returns
    -------
        The final model 
    """
    model = DigitNet(784, 10)
    optimiser = torch.optim.Adam(model.parameters()) # use Adam
    loss_fn = nn.CrossEntropyLoss()   # use CrossEntropyLoss

    for i in range(epochs):
        optimiser.zero_grad()
        output = model(x_train)
        loss = loss_fn(output, y_train.squeeze())
        loss.backward()
        optimiser.step()

    return model
                
digit_model = train_model(x_train, y_train)

# Task 3.5
def get_accuracy(scores, labels):
    """
    Helper function that returns accuracy of model
    
    Parameters
    ----------
        scores : The raw softmax scores of the network
        labels : The ground truth labels
        
    Returns
    -------
        Accuracy of the model. Return a number in range [0, 1].
        0 means 0% accuracy while 1 means 100% accuracy
    """
    _, predictions = torch.max(scores, 1)
    correct = (predictions == labels.squeeze()).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

scores = digit_model(x_test) # n x 10 tensor
get_accuracy(scores, y_test)

if __name__ == "__main__":
    torch.manual_seed(0)
    for n in torch.randint(50, 100, (5,)):
        y_true = torch.randint(0, 9, (n,))
        scores = torch.rand(n, 10)
        _, y_pred = torch.max(scores, 1)
        acc_true = (y_pred == y_true).float().mean().item()
        assert get_accuracy(scores, y_true) == acc_true