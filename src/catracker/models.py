from torch import nn
import torch

"""models.py: A module for defining the neural network models used in the project.

This module provides classes for the ConvLSTM neural network model used for tracking neurons.
"""

class Encoder(nn.Module):
    """Encoder model for extracting latent features from images
    
    Applies two convolutional layers to the input image and then flattens the output to a linear layer
    
    Attributes:
        capacity: number of kernels to apply in the first layer of the encoder
        latent_dims: final number of latent dimensions to extract from the image
    """
    def __init__(self, capacity, latent_dims):
        """Initializes the Encoder model
        
        Args:
            capacity (int): number of channels in the first layer of the encoder
            latent_dims (int): number of latent dimensions to extract from the image
        """
        super(Encoder, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=3,
                               stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2,
                               kernel_size=3, stride=2, padding=0)
        self.fc = nn.Linear(in_features=c*2*4*4, out_features=latent_dims)

    def forward(self, x):
        """Forward pass of the Encoder model
        
        Args:
            x (Tensor): input image tensor
            
        Returns:
            Tensor: latent features extracted from the input image
        """
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class ConvLSTM(nn.Module):
    """ConvLSTM model for tracking neurons
    
    Combines the Encoder model with an LSTM model to track neurons in a sequence of images
    
    Attributes:
        input_size_lstm: size of the input to the LSTM model
        conv_latent_dims: number of latent dimensions extracted by the Encoder model
        hidden_size_lstm: size of the hidden state in the LSTM model
        hidden_size_linear: size of the linear layer after the LSTM model
        num_layers: number of layers in the LSTM model
        batch_first: whether the input is batch-first or not
        dropout1: dropout rate for the input to the LSTM model
        dropout2: dropout rate for the output of the LSTM model
    """

    def __init__(self, input_size_lstm, capacity=16, conv_latent_dims=100, hidden_size_lstm=16, hidden_size_linear=64,num_layers=2, batch_first=True, dropout1=0.8, dropout2=0.8):
        """Initializes the ConvLSTM model"""
        super(ConvLSTM, self).__init__()
        self.conv = Encoder(capacity=capacity, latent_dims=conv_latent_dims)
        self.lstm = nn.LSTM(input_size=input_size_lstm,
                            hidden_size=hidden_size_lstm,
                            bidirectional=False,
                            num_layers=num_layers,
                            batch_first=batch_first)
        self.dropout_input = nn.Dropout(dropout1)
        self.linear = nn.Linear(hidden_size_lstm+2, hidden_size_linear)
        self.dropout_output = nn.Dropout(dropout2)

    def forward(self, x):
        num_sequences, sequence_length, _ = x.shape
        conv = self.conv(torch.reshape(x[:, :, 2:], (-1, 1, 20, 20)))
        conv = torch.reshape(conv, (num_sequences, sequence_length, -1))
        position_conv = torch.cat((x[:, :, :2], conv), dim=2)
        output, _status = self.lstm(position_conv)
        output = self.dropout_input(output)  # Apply dropout to input
        output = torch.cat((x[:, :, :2], output), dim=2)
        output = self.linear(output)
        output = self.dropout_output(output)  # Apply dropout to output
        return output