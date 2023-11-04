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

'''
# no need to code
# run this before moving on

train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()
print(f"Label: {label}")
'''

class RawCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        """
        classes: integer that corresponds to the number of classes for MNIST
        """
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.mp1 = nn.MaxPool2d(2, 2)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.mp2 = nn.MaxPool2d(2, 2)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(256, 128)
        self.lrelu4 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(128, classes)
        
    def forward(self, x):
        # YOUR CODE HERE     
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.lrelu2(x)
        x = x.view(-1, 64*5*5) # Flattening – do not remove this line

        # YOUR CODE HERE
        x = self.fc1(x)
        x = self.lrelu3(x)
        x = self.fc2(x)
        x = self.lrelu4(x)
        x = self.fc3(x)
        return x

'''
# Test your network's forward pass
num_samples, num_channels, width, height = 20, 1, 28, 28
x = torch.rand(num_samples, num_channels, width, height)
net = RawCNN(10)
y = net(x)
print(y.shape) # torch.Size([20, 10])
'''


class DropoutCNN(nn.Module):
    def __init__(self, classes, drop_prob=0.5):
        super().__init__()
        """
        classes: integer that corresponds to the number of classes for MNIST
        drop_prob: probability of dropping a node in the neural network
        """
        
        # YOUR CODE HERE
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.mp1 = nn.MaxPool2d(2, 2)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.do1 = nn.Dropout(drop_prob)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.mp2 = nn.MaxPool2d(2, 2)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.do2 = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.do3 = nn.Dropout(drop_prob)
        self.fc2 = nn.Linear(256, 128)
        self.lrelu4 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(128, classes)
        
    def forward(self, x):
        # YOUR CODE HERE
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.lrelu1(x)
        x = self.do1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.lrelu2(x)
        x = self.do2(x)

        x = x.view(-1, 64*5*5) # Flattening – do not remove

        # YOUR CODE HERE
        x = self.fc1(x)
        x = self.lrelu3(x)
        x = self.do3(x)
        x = self.fc2(x)
        x = self.lrelu4(x)
        x = self.fc3(x)
        return x

# Test your network's forward pass
num_samples, num_channels, width, height = 20, 1, 28, 28
x = torch.rand(num_samples, num_channels, width, height)
net = DropoutCNN(10)
y = net(x)
print(y.shape) # torch.Size([20, 10])

# %%time 
# do not remove the above line

def train_model(loader, model):
    """
    PARAMS
    loader: the data loader used to generate training batches
    model: the model to train
  
    RETURNS
        the final trained model 
    """

    """
    YOUR CODE HERE
    
    - create the loss and optimizer
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    epoch_losses = []
    for i in range(10):
        epoch_loss = 0.0
        
        for idx, data in enumerate(loader):
            x, y = data

            """
            YOUR CODE HERE
            
            - reset the optimizer
            - perform forward pass
            - compute loss
            - perform backward pass
            """

            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # COMPUTE STATS
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(loader)
        epoch_losses.append(epoch_loss)
        print ("Epoch: {}, Loss: {}".format(i, epoch_loss))
        

    # YOUR CODE HERE
    return model, epoch_losses

'''
vanilla_model, losses = train_model(train_loader, RawCNN(10))
do_model, losses = train_model(train_loader, DropoutCNN(10))

# do not remove – nothing to code here
# run this cell before moving on

with torch.no_grad():
    vanilla_model.eval()
    for i, data in enumerate(test_loader):
        x, y = data
        pred_vanilla = vanilla_model(x)
        acc = get_accuracy(pred_vanilla, y)
        print(f"vanilla acc: {acc}")
        
    do_model.eval()
    for i, data in enumerate(test_loader):
        x, y = data
        pred_do = do_model(x)
        acc = get_accuracy(pred_do, y)
        print(f"drop-out (0.5) acc: {acc}")
        
"""
The network with Dropout might under- or outperform the network without
Dropout. However, in terms of generalisation, we are assured that the Dropout
network will not overfit – that's the guarantee of Dropout.

A very nifty trick indeed!
"""


# %%time 
# do not remove – nothing to code here
# run this before moving on

do10_model, do10_losses = train_model(train_loader, DropoutCNN(10, 0.10))
do95_model, do95_losses = train_model(train_loader, DropoutCNN(10, 0.95))

# do not remove – nothing to code here
# run this cell before moving on

with torch.no_grad():
    do10_model.eval()
    for i, data in enumerate(test_loader):
        x, y = data
        pred_do = do10_model(x)
        acc = get_accuracy(pred_do, y)
        print(acc)

    do95_model.eval()
    for i, data in enumerate(test_loader):
        x, y = data
        pred_do = do95_model(x)
        acc = get_accuracy(pred_do, y)
        print(acc)

from sklearn.metrics import confusion_matrix

with torch.no_grad():
    vanilla_model.eval()
    for i, data in enumerate(test_loader):
        x, y = data
        pred_vanilla = vanilla_model(x)
        acc = get_accuracy(pred_vanilla, y)
        print(f"vanilla acc: {acc}")
        
    do_model.eval()
    for i, data in enumerate(test_loader):
        x, y = data
        pred_do = do_model(x)
        acc = get_accuracy(pred_do, y)
        print(f"drop-out (0.5) acc: {acc}")

cm = confusion_matrix(mnist_test.targets, pred_vanilla.argmax(dim=1))
plt.figure(figsize=(10,7))
plt.title('Confusion Matrix for vanilla_model')
# np.fill_diagonal(cm, 0) # you can zero-out the diagonal to highlight the errors better
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
# print(cm) # if seaborn does not work, you can always print out the array
    
cm = confusion_matrix(mnist_test.targets, pred_do.argmax(dim=1))
plt.figure(figsize=(10,7))
plt.title('Confusion Matrix for do_model')
# np.fill_diagonal(cm, 0) # you can zero-out the diagonal to highlight the errors better
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
# print(cm) # if seaborn does not work, you can always print out the array

plt.show()
'''

if __name__ == "__main__":
    vanilla_model, losses = train_model(train_loader, RawCNN(10))
    vanilla_model_scripted = torch.jit.script(vanilla_model)
    vanilla_model_scripted.save('vanilla_model')

    do_model, losses = train_model(train_loader, DropoutCNN(10))
    do_model_scripted = torch.jit.script(do_model)
    do_model_scripted.save('do_model')

    do10_model, do10_losses = train_model(train_loader, DropoutCNN(10, 0.10))
    do10_model_scripted = torch.jit.script(do10_model)
    do10_model_scripted.save('do10_model')

    do95_model, do95_losses = train_model(train_loader, DropoutCNN(10, 0.95))
    do95_model_scripted = torch.jit.script(do95_model)
    do95_model_scripted.save('do95_model')
