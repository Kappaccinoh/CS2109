import pandas as pd
import os
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
# from torchvision.models.detection import focal_loss

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 4)  # Adjust output size based on your number of classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def step1(images, labels):
    data = images
    flattened_images = data.reshape((data.shape[0], -1))
    columns = [f'pixel_{i}' for i in range(flattened_images.shape[1])]
    images = pd.DataFrame(flattened_images, columns=columns)
    labels = pd.DataFrame(labels)
    
    return images, labels

def step2(images, labels):
    # Images
    images = images.apply(pd.to_numeric, errors='coerce')
    for col in images.columns:
        images[col].interpolate(method='nearest', limit_direction='both', inplace=True)

    images.fillna(images.median(), inplace=True)
    
    # Labels
    labels = labels.fillna(3.0)
    
    return images, labels

def step3(images, labels):
    images = images.clip(lower=0, upper=256)
    
    return images, labels

def step4(images, labels, num_entries):
    images_nparray = images.values
    images_reshaped = images_nparray.reshape((num_entries, 3, 16, 16))

    return images_reshaped, labels

def prepareData(X):
    data = X
    flattened_images = data.reshape((data.shape[0], -1))
    columns = [f'pixel_{i}' for i in range(flattened_images.shape[1])]
    images = pd.DataFrame(flattened_images, columns=columns)
    
    # Step 2
    images = images.apply(pd.to_numeric, errors='coerce')
    for col in images.columns:
        images[col].interpolate(method='nearest', limit_direction='both', inplace=True)

    images.fillna(images.median(), inplace=True)
    
    # Step 3
    images = images.clip(lower=0, upper=256)
    
    images_nparray = images.values
    images_reshaped = images_nparray.reshape((X.shape[0], 3, 16, 16))

    return images_reshaped


class Model:  
    """
    This class represents an AI model.
    """
    
    def __init__(self):
        """
        Constructor for Model class.
  
        Parameters
        ----------
        self : object
            The instance of the object passed by Python.
        """
        learning_rate = 0.001
        num_epochs = 100
        
        self.model = SimpleCNN()
        
        total_samples = 2619
        class_weights = [
            total_samples / (0.8217 * total_samples), # 0.0
            total_samples / (0.0697 * total_samples), # 1.0
            total_samples / (0.0086 * total_samples), # 2.0
            total_samples / (0.1000 * total_samples) # 3.0
        ]
        
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
#         self.criterion = focal_loss.FocalLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
    
    def fit(self, X, y):
        """
        Train the model using the input data.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, channel, height, width)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Returns an instance of the trained model.
        """
        # Format to Pandas DataFrame
        X, y = step1(X, y)
        
        # Handle NANs
        X, y = step2(X, y)
        
        # Handle Outliers
        X, y = step3(X, y)
        
        num_entries =  X.shape[0]
        
        # Reshape data to original format
        X, y = step4(X, y, num_entries)
        
        images_numpy = X
        labels_numpy = y.values
        
        images_tensor = torch.tensor(images_numpy, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_numpy, dtype=torch.long)
        
        for epoch in range(self.num_epochs):
            # Forward pass
            outputs = self.model(images_tensor)
            loss = self.criterion(outputs, labels_tensor.squeeze())

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')

        return self
            
    def predict(self, X):
        """
        Use the trained model to make predictions.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, channel, height, width)
            Input data.
            
        Returns
        -------
        ndarray of shape (n_samples,)
        Predicted target values per element in X.
           
        """
        X_test = prepareData(X)
        
        X_test = torch.tensor(X_test, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(X_test)
            _, predicted_class = torch.max(val_outputs, 1)

        predictions = predicted_class.numpy()

        return predictions
        
