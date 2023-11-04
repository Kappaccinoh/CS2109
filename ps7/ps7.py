# do not remove this cell
# run this cell before moving on

# DL libraries
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import datasets, transforms

# Computational libraries
import math
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

def conv2d(img, kernel):
    """
    PARAMS
        img: the 2-dim image with a specific height and width
        kernel: a 2-dim kernel (smaller than image dimensions) that convolves the given image
    
    RETURNS
        the convolved 2-dim image
    """
    img_height, img_width = img.shape
    kernel_height, kernel_width = kernel.shape
    output_height = img_height - kernel_height + 1
    output_width = img_width - kernel_width + 1
    output = torch.zeros(output_height, output_width)

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = torch.sum(img[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output

def maxpool2d(img, size):
    """
    PARAMS
        img: the 2-dim image with a specific height and width
        size: an integer corresponding to the window size for Max Pooling
    
    RETURNS
        the 2-dim output after Max Pooling
    """
    img_height, img_width = img.shape
    output_height = img_height - size + 1
    output_width = img_width - size + 1
    output = torch.zeros(output_height, output_width)

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = torch.max(img[i:i+size, j:j+size])

    return output

# do not remove this cell
# run this before moving on

T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

"""
Note: You can update the path to point to the directory containing `MNIST` 
directory to avoid downloading the MNIST data again.
"""
mnist_train = datasets.MNIST("./", train=True, download=True, transform=T)
mnist_test = datasets.MNIST("./", train=False, download=True, transform=T)

"""
if you feel your computer can't handle too much data, you can reduce the batch
size to 64 or 32 accordingly, but it will make training slower. 

We recommend sticking to 128 but do choose an appropriate batch size that your
computer can manage. The training phase tends to require quite a bit of memory.
"""
train_loader = torch.utils.data.DataLoader(mnist_train, shuffle=True, batch_size=256)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=10000)

def get_accuracy(scores, labels):
    ''' accuracy metric '''
    _, predicted = torch.max(scores.data, 1)
    correct = (predicted == labels).sum().item()   
    return correct / scores.size(0)


if __name__ == "__main__":
    print()