import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear,  GELU, Conv2d, BatchNorm2d, Flatten, Sigmoid, Tanh

import numpy as np

import config

def get_params(model):
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    return total_param

def get_size(model):
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    return size_model, (size_model / 8e6)

class RLModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        
        super().__init__()

        # the input to the board is (H, W, C) : (8, 8, 19) 
        # the output shpae is (4672, 1) : (policy_head, value_head)

        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = config.ACTIVATION
        self.norm = config.NORMALIZATION

        self.n_conv_layers = config.AMOUNT_OF_RESIDUAL_BLOCKS
        self.conv_filters = config.CONVOLUTION_FILTERS

        self.initial_convolution = nn.Sequential(Conv2d(self.input_dim[2], self.conv_filters, kernel_size=(3, 3), stride=(1, 1), padding='same'),
                                           BatchNorm2d(self.conv_filters),
                                           GELU())

        self.convolutions = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(self.n_conv_layers):
            self.convolutions.append(nn.Sequential(Conv2d(self.conv_filters, self.conv_filters, kernel_size=(3, 3), stride=(1, 1), padding='same'),
                                           BatchNorm2d(self.conv_filters),
                                           GELU(),
                                           Conv2d(self.conv_filters, self.conv_filters, kernel_size=(3, 3), stride=(1, 1), padding='same'),
                                           BatchNorm2d(self.conv_filters)))
            self.activations.append(GELU()) 

        
        self.policy_head = nn.Sequential(Conv2d(self.conv_filters, 2, kernel_size=(1,1), stride=1, padding='same'),
                                          BatchNorm2d(2),
                                          GELU(),
                                          Flatten(),  ## the output of flatten is 2*self.input_dim[0]* self.input_dim[1]
                                          Linear(2*self.input_dim[0]* self.input_dim[1], self.output_dim[0]),  ## the output of ploicy head is 4672
                                          Sigmoid())
                
        self.value_head = nn.Sequential(Conv2d(self.conv_filters, 1, kernel_size=(1,1), stride=1, padding='same'),
                                          BatchNorm2d(1),
                                          GELU(),
                                          Flatten(),  ## the output of flatten is 1*self.input_dim[0]* self.input_dim[1])
                                          Linear(1*self.input_dim[0]* self.input_dim[1], 256),
                                          GELU(),
                                          Linear(256, self.output_dim[1]),  # the outsput size of value head is (1)
                                          Tanh())


    def reset_parameters(self):
        pass

    def forward(self, x):
        
        # x is of shape (B, H, W, C)

        x = x.view(-1, self.input_dim[2], self.input_dim[1], self.input_dim[0])

        x = self.initial_convolution(x)

        for conv, act in zip(self.convolutions, self.activations):
            x = x + conv(x)
            x = act(x)

        policy = self.policy_head(x)
        
        value = self.value_head(x)

        return (policy, value)


if __name__ == "__main__":

    model = RLModel(config.INPUT_SHAPE, config.OUTPUT_SHAPE)

    ## testing with random input
    print(f"the model has {get_params(model)} parameters and {get_size(model)[1]} MB.")
    
    x = torch.randn(1, config.amount_of_input_planes, config.n, config.n)
    
    print(x.shape)

    model.train()

    policy, value = model(x)
    print(policy.shape)
    print(value.shape)