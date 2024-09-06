#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Load dataset
data = pd.read_csv('/content/data.csv')

# train-test split function with random suffle, y_train,y_test is one-hot encoded , X_train,X_test is normalize
def train_test_split(dataset, train_ratio, test_ratio):
    x = dataset.sample(frac=1, random_state=19).reset_index(drop=True)
    X_train = (x.iloc[:int(train_ratio * x.shape[0]), 1:] / 255).values
    y_train = pd.get_dummies(x.iloc[:int(train_ratio * x.shape[0]), 0]).values
    X_test = (x.iloc[int(train_ratio * x.shape[0]):, 1:] / 255).values
    y_test = pd.get_dummies(x.iloc[int(train_ratio * x.shape[0]):, 0]).values
    return X_train, y_train, X_test, y_test

# Initial parameters initialization using xavier weight initialization method , random seed = 4
def initial_parameters():
    np.random.seed(4)
    w1 = np.random.randn(128, 784) * np.sqrt(1 / 784)
    b1 = np.ones((128, 1))
    w2 = np.random.randn(64, 128) * np.sqrt(1 / 128)
    b2 = np.ones((64, 1))
    w3 = np.random.randn(32, 64) * np.sqrt(1 / 64)
    b3 = np.ones((32, 1))
    w4 = np.random.randn(10, 32) * np.sqrt(1 / 32)
    b4 = np.ones((10, 1))
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2, 'w3': w3, 'b3': b3, 'w4': w4, 'b4': b4}
    return parameters

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / exp_z.sum(axis=0, keepdims=True)

#Forward propogation
def forward(X, parameters):
    w1, b1, w2, b2, w3, b3, w4, b4 = [parameters[key] for key in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4')]
    # a's are preactivation and h's are output of activation functions
    # 1st hidden layer
    a1 = w1 @ X.T + b1
    h1 = sigmoid(a1)
    #2nd hidden layer
    a2 = w2 @ h1 + b2
    h2 = sigmoid(a2)
    # 3rd hidden layer
    a3 = w3 @ h2 + b3
    h3 = sigmoid(a3)
    # 4th hidden layer
    a4 = w4 @ h3 + b4
    y_hat = softmax(a4)
    forward_cache = {'a1': a1, 'h1': h1, 'a2': a2, 'h2': h2, 'a3': a3, 'h3': h3, 'a4': a4, 'y_hat': y_hat}
    return forward_cache

# Cross - Entropy cost function
def cost_function(y_hat, y):
    m = y.shape[0]
    cost = np.sum(-np.log(y_hat) * y.T) / m
    return cost

# Backward propogation
def backward(X, y, parameters, forward_cache):
    w1, b1, w2, b2, w3, b3, w4, b4 = [parameters[key] for key in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4')]
    h1, h2, h3, y_hat = [forward_cache[key] for key in ('h1', 'h2', 'h3', 'y_hat')]
    m = X.shape[0]
    # At output layer
    da4 = y_hat - y.T
    dw4 = (da4 @ h3.T) / m
    db4 = np.sum(da4, axis=1, keepdims=True) / m
    # 3rd layer
    da3 = (w4.T @ da4) * h3 * (1 - h3)
    dw3 = (da3 @ h2.T) / m
    db3 = np.sum(da3, axis=1, keepdims=True) / m
   #2nd layer
    da2 = (w3.T @ da3) * h2 * (1 - h2)
    dw2 = (da2 @ h1.T) / m
    db2 = np.sum(da2, axis=1, keepdims=True) / m
   # 1st layer
    da1 = (w2.T @ da2) * h1 * (1 - h1)
    dw1 = (da1 @ X) / m
    db1 = np.sum(da1, axis=1, keepdims=True) / m
    gradients = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2, 'dw3': dw3, 'db3': db3, 'dw4': dw4, 'db4': db4}
    return gradients

# Parameters update
def update_parameters(parameters, gradients, lr):
    w1, b1, w2, b2, w3, b3, w4, b4 = [parameters[key] for key in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4')]
    dw1, db1, dw2, db2, dw3, db3, dw4, db4 = [gradients[key] for key in ('dw1', 'db1', 'dw2', 'db2', 'dw3', 'db3', 'dw4', 'db4')]

    w4 = w4 - lr * dw4
    b4 = b4 - lr * db4
    w3 = w3 - lr * dw3
    b3 = b3 - lr * db3
    w2 = w2 - lr * dw2
    b2 = b2 - lr * db2
    w1 = w1 - lr * dw1
    b1 = b1 - lr * db1
    updated_parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2, 'w3': w3, 'b3': b3, 'w4': w4, 'b4': b4}
    return updated_parameters

#Accuracy calculation
def calculate_accuracy(X, y, parameters):
    forward_cache = forward(X, parameters)
    predicted_labels = np.argmax(forward_cache['y_hat'], axis=0)
    true_labels = np.argmax(y, axis=1).flatten()
    accuracy = np.mean(predicted_labels == true_labels)
    return accuracy

#model train
def model_train(X_train, y_train, lr, epochs, batch_size):
    parameters = initial_parameters()
    costs = []
    accuracies = []

    for i in range(epochs):
        for j in range(0, X_train.shape[0], batch_size):
            X = X_train[j:j+batch_size]
            y = y_train[j:j+batch_size]
            forward_cache = forward(X, parameters)
            gradients = backward(X, y, parameters, forward_cache)
            parameters = update_parameters(parameters, gradients, lr)

        # Calculate loss and accuracy
        forward_cache = forward(X_train, parameters)
        loss = cost_function(forward_cache['y_hat'], y_train)
        costs.append(loss)
        accuracy = calculate_accuracy(X_train, y_train, parameters)
        accuracies.append(accuracy)
        print('Epoch: {}, Loss: {:.4f}, Accuracy: {:.2%}'.format(i+1, costs[-1], accuracy))

    # Plot loss per epoch
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(costs, label='Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.grid()
    plt.legend()

    # Plot accuracy per epochs
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    return parameters, costs

#confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 12))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Oranges', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel('y_predicted')
    plt.ylabel('y_true')
    plt.title('Confusion Matrix')
    plt.show()


#model test and plot confusion matrix
def model_test(X_test, y_test, trained_parameters):
    forward_cache = forward(X_test, trained_parameters)
    y_pred = np.argmax(forward_cache['y_hat'], axis=0)
    y_true = np.argmax(y_test, axis=1).flatten()
    accuracy = calculate_accuracy(X_test, y_test, trained_parameters)
    print('Test Accuracy:{:.2%}'.format(accuracy))
    plot_confusion_matrix(y_true, y_pred)

# train-test ratio 70:30
X_train, y_train, X_test, y_test = train_test_split(dataset=data, train_ratio=0.7,test_ratio=0.3)
trained_parameters, training_costs = model_train(X_train, y_train, lr=0.02, epochs=25, batch_size=23)
model_test(X_test, y_test, trained_parameters)

# train-test ratio 80:20
X_train, y_train, X_test, y_test = train_test_split(dataset=data, train_ratio=0.8,test_ratio=0.2)
trained_parameters, training_costs = model_train(X_train, y_train, lr=0.02, epochs=25, batch_size=23)
model_test(X_test, y_test, trained_parameters)

# train-test ratio 90:10
X_train, y_train, X_test, y_test = train_test_split(dataset=data, train_ratio=0.9,test_ratio=0.1)
trained_parameters, training_costs = model_train(X_train, y_train, lr=0.02, epochs=25, batch_size=23)
model_test(X_test, y_test, trained_parameters)

# Calculation of trainable and non trainable parameters
total_trainable_params= np.sum([np.product(trained_parameters[key].shape) for key in trained_parameters if 'w' in key or 'b' in key])
total_nontrainable_params = np.product(trained_parameters['w1'].shape[1] * trained_parameters['w4'].shape[0])
print("Number of Trainable Parameters:", total_trainable_params)
print("Number of Non-Trainable Parameters:", total_nontrainable_params)

