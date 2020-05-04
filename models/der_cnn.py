# The following code makes up the Derived CNN (der_CNN) model. It is derived from the example CNN model from the COMPSYS
# 302 labs (found in conv.py).

import torch.nn as nn
import torch.nn.functional as func
import torch

# This function represents the Sigmoid Linear Unit (SiLU) activation function.
def silu(input):
    return input * torch.sigmoid(input)

class der_CNN(nn.Module):
    def __init__(self):
        super(der_CNN, self).__init__()
        # 1: input channels 32: output channels, 3: kernel size, 1: stride
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # "Filter out" some of the input based on their probability of being zeroed.
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.25)
        # Fully connected layer: input size, output size
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    # This function links all of the layers of the model together and passes the input data through them.
    # Activation function: ReLU
    # Loss function: Softmax
    def forward(self, input):
        input = self.conv1(input)
        input = silu(input)
        input = self.conv2(input)
        input = silu(input)
        input = func.max_pool2d(input, 2)
        input = self.dropout1(input)
        input = torch.flatten(input, 1)
        input = self.fc1(input)
        input = silu(input)
        input = self.dropout2(input)
        input = self.fc2(input)
        output = func.log_softmax(input, dim=1)
        return output