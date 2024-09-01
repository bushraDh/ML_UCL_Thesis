import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(LinearRegressionModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim, output_dim)  # 1 * 512 * 512 = 262144

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)


def LinearReg(input_dim=256 * 256 ,output_dim= 2):
    # Get the input and output dimensions
    # input_dim = 512 * 512  # Flattened image size
    # output_dim = 2  # Number of output labels
    # Initialize the model
    model = LinearRegressionModel(input_dim=input_dim, output_dim=output_dim)
    return model
