import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import mlflow
from vanilla_snn import StochasticNN
import pandas as pd
import os
from torch import optim 

EPOCHS = 100
LR_RATE = 0.06

INPUT_DIM = 4     
OUTPUT_DIM = 3

def import_data():
    """
    This function is used to import the training data.
    :return: X: X1 and X2 features of the dataset which becomes our input variable for the model.
             Y: Y is our target variable.
    """
    path = os.getcwd() + '/snn_testing/data/iris_train.dat'
    data = pd.read_csv(path, header=None, names=['X1', 'X2', 'X3', 'X4', 'Y'])
    X = data[['X1', 'X2', 'X3', 'X4']].to_numpy()
    Y = np.array(data['Y'])

    return torch.Tensor(X), torch.Tensor(Y)

def one_hot_encode(Y):
    """
    performing one hot encoding over target variable Y
    :param Y:
    :return: one hot encoded Y
    """
    num_classes_k = len(np.unique(Y))
    new_y = np.zeros((Y.shape[0], num_classes_k))
    for i in range(len(Y)):
        if Y[i] == 0:
            new_y[i][0] = 1
        else:
            new_y[i][1] = 1

    return new_y


def train(snn, x, y):
    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()  # since this is a classification problem
    optimizer = optim.Adam(snn.parameters(), lr=LR_RATE)

    snn.train()  # put the model in training mode

    for epoch in range(EPOCHS):
        optimizer.zero_grad()  # zero the gradients from the last iteration

        # Reparameterize weights and biases
        output = snn(x.T)  # forward pass

        # Compute loss
        loss = loss_fn(output.T, y)
        
        # Backpropagation
        loss.backward()

        # Optimize weights
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    print("Training complete.")


def test():
    pass


def main():
    snn = StochasticNN(input_dim= INPUT_DIM, z_dim=OUTPUT_DIM)

    X, Y = import_data()
    # ground_truth_Y = Y.copy()
    Y = torch.Tensor(one_hot_encode(Y))

    train(snn, X, Y)

    

if __name__ == "__main__":
    main()

