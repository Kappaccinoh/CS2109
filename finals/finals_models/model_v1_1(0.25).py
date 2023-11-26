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

class SimpleCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * (num_features // 4), 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension for Conv1d
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
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

def step4(images, labels):
    merged_data = pd.merge(images, labels, left_index=True, right_index=True, how='inner')
    
    return merged_data

def step5(merged_data):
    '''
    # Map numeric labels to string representations
    label_mapping = {0.0: 'A', 1.0: 'B', 2.0: 'C', 3.0: 'D'}
    merged_data.iloc[:, -1] = merged_data.iloc[:, -1].map(label_mapping)
    '''
    merged_data_categorical = merged_data

    # Rename Column for sanity
    current_columns = merged_data_categorical.columns.tolist()
    new_column_name = 'target'
    merged_data_categorical.rename(columns={current_columns[-1]: new_column_name}, inplace=True)
    
    return merged_data_categorical

def step6(merged_data):
    return merged_data

def step7(merged_data):
    return merged_data

def step8(df_combined):
    # Separate features (X) and target (y)
    X = df_combined.iloc[:, :-1].values
    y = df_combined['target'].values

    # Standardize the features (important for PCA)
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Apply PCA with the desired number of components
    n_components = 10  # Adjust as needed
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_standardized)

    # Create a DataFrame with the PCA components and target column
    columns_pca = [f'PCA_{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(data=X_pca, columns=columns_pca)
    df_pca['target'] = y

    return df_pca

def step9(df_pca):
    '''
    # Extract the principal components and target column
    X_pca = df_pca.iloc[:, :-1].values
    y = df_pca['target'].values

    # Apply PolynomialFeatures to create higher-order terms
    degree = 2  # Adjust as needed
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X_pca)

    # Create a DataFrame with the new features
    df_poly = pd.DataFrame(X_poly)
    df_poly['target'] = y
    
    # Splitting the Dataset into Train and Test at 0.2 ratio split
    X = df_poly.drop(['target'], axis=1)
    y = df_poly['target']
    '''
    X = df_pca.drop(['target'], axis=1)
    y = df_pca['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def step10(X_train, X_test, y_train, y_test):
    # Convert NumPy arrays to PyTorch tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_test = torch.tensor(y_test.values, dtype=torch.long)
    
    return X_train, X_test, y_train, y_test
    
def prepareData(X):
    # Data Preprocessing
    # Step 1
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
    
    X = images
    
    # Apply PCA 
    X_pca = X.iloc[:, :-1].values
    
    # Standardize the features (important for PCA)
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Apply PCA with the desired number of components
    n_components = 10  # Adjust as needed
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_standardized)

    # Create a DataFrame with the PCA components and target column
    columns_pca = [f'PCA_{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(data=X_pca, columns=columns_pca)
    
    '''
    # Apply PolynomialFeatures to create higher-order terms
    X_pca = df_pca
    degree = 2  # Adjust as needed
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X_pca)

    # Create a DataFrame with the new features
    df_poly = pd.DataFrame(X_poly)
    '''
    df_poly = df_pca
    
    # Convert DataFrame to Tensor
    df_poly = torch.tensor(df_poly.values, dtype=torch.float32)
    
    return df_poly
    

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
        num_features = 10
        num_classes = 4
        learning_rate = 0.001
        num_epochs = 100
        batch_size = 32
        
        self.model = SimpleCNN(num_features=num_features, num_classes=num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
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
        
        # Merge Image and Labels into one Set
        merged = step4(X, y)
        
        # Convert Labels data into Categorical (A,B,C,D)
        merged = step5(merged)
        
        # Nil
        merged = step6(merged)
        
        # Nil
        merged = step7(merged)
        
        # Feature Selection
        merged = step8(merged)
        
        # Feature Engineering + Split Data
        X_train, X_test, y_train, y_test = step9(merged)
        
        # Preparing Data to right format
        X_train, X_test, y_train, y_test = step10(X_train, X_test, y_train, y_test)
        
        # Training the Model
        for epoch in range(self.num_epochs):
            self.model.train()
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i+self.batch_size]
                batch_y = y_train[i:i+self.batch_size]

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item():.4f}')
        
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
        
        # Change to eval mode
        self.model.eval()
        
        with torch.no_grad():
            val_outputs = self.model(X_test)
            _, predicted_class = torch.max(val_outputs, 1)
        
        predictions = predicted_class.numpy()
        
        return predictions