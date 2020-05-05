# The following code makes up the Derived RNN (der_RNN) model. It is derived from the example RNN model from the COMPSYS
# 302 labs (found in rnn_conv.py).

import torch.nn as nn
import torch


class der_RNN(nn.Module):
    def __init__(self, device):
        super(der_RNN, self).__init__()
        # RNN layer specifications
        self.device = device
        self.neurons = 256  # Hidden size
        self.batch_size = 64
        self.steps = 28
        self.inputs = 28
        self.outputs = 10
        self.dropout = 0    # Dropout layers
        self.bidirectional = False
        self.layers = 1     # Recurrent layers
        self.activ_func = "relu"
        # RNN layer
        self.rnn_layer = nn.RNN(self.inputs, self.neurons, self.layers, nonlinearity=self.activ_func,
                                dropout=self.dropout, bidirectional=self.bidirectional)
        # Fully connected layer
        self.fc = nn.Linear(self.neurons, self.outputs)

    # This function initialises a hidden layer
    def init_hidden(self, ):
        return (torch.zeros(self.layers, self.batch_size, self.neurons)).to(self.device)

    # This function links all of the layers of the model together and passes the input data through them.
    def forward(self, input):
        # Transform input to dimensions 28x64x28 (steps x batch_size x inputs)
        input = input.permute(1, 0, 2)
        self.batch_size = input.size(1)
        self.hidden = self.init_hidden()    # Create the hidden layer
        lstm_out, self.hidden = self.rnn_layer(input, self.hidden)
        out = self.fc(self.hidden)
        # Output from fully connected layer directly
        return out.view(-1, self.outputs)  # batch_size x outputs
