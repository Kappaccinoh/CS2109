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


# vanilla_model, losses = train_model(train_loader, RawCNN(10))
# do_model, losses = train_model(train_loader, DropoutCNN(10))
vanilla_model = torch.jit.load('vanilla_model')

do_model = torch.jit.load('do_model')

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

'''
# %%time 
# do not remove – nothing to code here
# run this before moving on

# do10_model, do10_losses = train_model(train_loader, DropoutCNN(10, 0.10))
# do95_model, do95_losses = train_model(train_loader, DropoutCNN(10, 0.95))
do10_model = torch.jit.load('do10_model')
do95_model = torch.jit.load('do95_model')

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

from sklearn.metrics import f1_score, precision_score, recall_score

'''
print(f1_score(pred_do.argmax(dim=1), mnist_test.targets, average='macro'))
print(precision_score(pred_do.argmax(dim=1), mnist_test.targets, average='macro'))
print(recall_score(pred_do.argmax(dim=1), mnist_test.targets, average='macro'))
'''

'''
cifar_train = datasets.CIFAR10("./", train=True, download=True, transform=transforms.ToTensor())
cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=128, shuffle=True)

train_features, train_labels = next(iter(cifar_train_loader))
img = train_features[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,7))
transform = transforms.Compose([transforms.RandomHorizontalFlip()
                                # YOUR CODE HERE
                                ]) # add in your own transformations to test
tensor_img = transform(img)
ax1.imshow(img.permute(1,2,0))
ax1.axis("off")
ax1.set_title("Before Transformation")
ax2.imshow(tensor_img.permute(1, 2, 0))
ax2.axis("off")
ax2.set_title("After Transformation")
plt.show()
'''

# pick your data augmentations here
def get_augmentations():
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ])
    
    return T

# Rationale Behind Chosen Augmentaions

# RandomHorizontal/VerticalFlip - In this case I want to generalise the orientation of the object since it doesnt affect the label outcome. This will create more variations in the dataset and make it more generalisable.
# RandomRotation - Again makes the model become more robust since as the images are rotated by a random angle.
# ColourJitter - Makes the model more generalisable to changes in lighting conditions, which I vary using brightness/contrast/saturation/hue

# I avoided randomcropping or resizing in this case since the images are already quite small (32px x 32px), so any additional cropping or resizing may lead to information loss.

# do not remove this cell
# run this before moving on

T = get_augmentations()

cifar_train = datasets.CIFAR10("./", train=True, download=True, transform=T)
cifar_test = datasets.CIFAR10("./", train=False, download=True, transform=T)

"""
if you feel your computer can't handle too much data, you can reduce the batch
size to 64 or 32 accordingly, but it will make training slower. 

We recommend sticking to 128 but dochoose an appropriate batch size that your
computer can manage. The training phase tends to require quite a bit of memory.

CIFAR-10 images have dimensions 3x32x32, while MNIST is 1x28x28
"""
cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=128, shuffle=True)
cifar_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=10000)

class CIFARCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        """
        classes: integer that corresponds to the number of classes for CIFAR-10
        """
        self.conv = nn.Sequential(
                        nn.Conv2d(3, 32, 3),
                        nn.MaxPool2d(2),
                        nn.LeakyReLU(0.1),
                        nn.Conv2d(32, 64, 3),
                        nn.MaxPool2d(2),
                        nn.LeakyReLU(0.1)
                    )

        self.fc = nn.Sequential(
                        nn.Linear(64, 256),
                        nn.LeakyReLU(0.1),
                        nn.Linear(256, 128),
                        nn.LeakyReLU(0.1),
                        nn.Linear(128, classes)
                    )
        
    def forward(self, x):
        # YOUR CODE HERE
        x = self.conv(x)

        x = x.view(x.shape[0], 64, 6*6).mean(2) # GAP – do not remove this line
        
        # YOUR CODE HERE
        x = self.fc(x)
        return x

# %%time
# do not remove – nothing to code here
# run this cell before moving on

# cifar10_model, losses = train_model(cifar_train_loader, CIFARCNN(10))
# cifar10_model_scripted = torch.jit.script(cifar10_model)
# cifar10_model_scripted.save("cifar10_model.pt")
cifar10_model = torch.jit.load("cifar10_model.pt")

# do not remove – nothing to code here
# run this cell before moving on

with torch.no_grad():
    cifar10_model.eval()
    for i, data in enumerate(cifar_test_loader):
        x, y = data
        pred = cifar10_model(x)
        acc = get_accuracy(pred, y)
        print(f"cifar accuracy: {acc}")
        
# don't worry if the CIFAR-10 accuracy is low, it's a tough dataset to crack.
# as long as you get something shy of 50%, you should be alright!

def get_CAM(feature_map, weight, class_idx):
    """
    PARAMS
        feature_map: the output of the final pre-GAP layer in the ConvNet
        weight: the parameters of the first linear layer post-GAP
        class_idx: the final prediction label of the ConvNet
    
    RETURNS
        a CAM heatmap of the areas the ConvNet is focusing on more
    """
    
    # do not remove these lines
    size_upsample = (32, 32)
    bz, nc, h, w = feature_map.shape

    before_dot = feature_map.reshape((nc, h*w))
    cam = weight[class_idx].unsqueeze(0) @ before_dot
    
    """
    YOUR CODE HERE - perform the steps listed above
    """
    
    cam = torch.squeeze(cam) ## remove the first dimension of cam using torch.squeeze(...)
    cam = cam.view(h, w) ## reshape cam to h x w
    cam = cam - torch.min(cam) ## get the difference of cam and the minimum elements of cam
    cam = cam / torch.max(cam) ## divide cam by the maximum elements of cam
    cam = torch.clip(cam, 0, 1) ## clip the values of cam so they are within the [0, 1] range
    
    # here, `cam` is the final processed heatmap
    # we upsample/resize the heatmap to the original image's dimensions
    # do not remove these lines
    img = transforms.Resize(size_upsample)(cam.unsqueeze(0))
    
    return img.detach().numpy(), cam

# do not remove this cell
# run this cell before moving on

cifar10_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

def plot_cam(img, cam):
    ''' Visualization function '''
    img = img.permute(1, 2, 0)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,7))
    ax1.imshow(img)
    ax1.set_title(f"Input image\nLabel: {cifar10_classes[y]}")

    ax2.imshow(cam.reshape(32, 32), cmap="jet")
    ax2.set_title("Raw CAM.")

    ax3.imshow(img)
    ax3.imshow(cam.reshape(32, 32), cmap="jet", alpha=0.2)
    ax3.set_title(f"Overlayed CAM.\nPrediction: {cifar10_classes[idx[0]]}")
    plt.show()

# do not remove this cell
# run this cell before moving on

rand_idx = torch.randint(0, 10000, size=[1]) # pick a random index from the test set

x = cifar_test[rand_idx][0] # test image
y = cifar_test[rand_idx][1] # associated test label

cifar10_model.eval()
scores = cifar10_model(x.unsqueeze(0)) # get the raw scores
probs = scores.data.squeeze()
probs, idx = probs.sort(0, True)

print('true class: ', cifar10_classes[y])
print('predicated class: ', cifar10_classes[idx[0]])

assert y == idx[0], "We want to visualize what the model is focusing on for a correct prediction, run again for another random sample!"

# if the printed prediction and label are different, it means the model misclassified it. 
# Rerun this cell until you get the same class printed for both. It will help for the visualisation later.

# Get the first Linear layer's weights and final Feature Map
params = list(cifar10_model.fc.parameters()) # access the model layers
weight = params[0].data # grab the first layer's weights

feature_maps = cifar10_model.conv(x.unsqueeze(0))

# Creating the heatmap
heatmap, _ = get_CAM(feature_maps, weight, idx[0])
    
plot_cam(x, heatmap)
# Red "hot" areas represent where the model is focusing on more
# if the shading isn't that great, rerun the cell to get another random sample

if __name__ == "__main__":
    print()
