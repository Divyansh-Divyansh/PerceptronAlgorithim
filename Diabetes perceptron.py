#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd

diabetes = pd.read_csv("C:\\Users\\Jitendra Sharma\\Downloads\\archive\\diabetes.csv")


# In[9]:


diabetes


# In[22]:


import numpy as np 

class Perceptron: 
    def __init__(self, learning_rate = 0.01, n_iters = 1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None
        
    def fit(self, X, y): # Number of rows - samples, columns - features.
        n_samples , n_features = X.shape
        
        #init Weights
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # convert the value to 0 or 1 for perceptron if not already
        y_ = np.array([1 if i > 0 else 0 for i in y])
        
        #
        for _ in range(self.n_iters):
        # second loop
            for idx, x_i in enumerate(X):
                linear_operator = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_operator)
                
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update 
    
    def predict(self, X):
        linear_operator = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_operator)
        return y_predicted
        
        
    def _unit_step_func(self, x):
        return np.where(x>=0, 1, 0)


# In[3]:


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# In[66]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import matplotlib.pyplot as plt

# Train datasets
X = diabetes.iloc[:,:7]
y = diabetes["Outcome"]
#print(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.4, random_state=42)


# In[62]:


# Initialize class.
p = Perceptron(learning_rate=0.01, n_iters=5000)
p.fit(X_train ,y_train)
predictions = p.predict(X_test)

print("Perceptron classification accuracy", round(accuracy(y_test, predictions)*100,2))


# In[ ]:




