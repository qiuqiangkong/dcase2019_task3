import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_utils import interpolate


def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2)):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=pool_size)
        
        return x
    
    
class Cnn_9layers(nn.Module):
    def __init__(self, classes_num):
        
        super(Cnn_9layers, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=4, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.event_fc = nn.Linear(512, classes_num, bias=True)
        self.elevation_fc = nn.Linear(512, classes_num, bias=True)
        self.azimuth_fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.event_fc)
        init_layer(self.elevation_fc)
        init_layer(self.azimuth_fc)

    def forward(self, input):
        '''
        Input: (channels_num, batch_size, times_steps, freq_bins)'''
        
        interpolate_ratio = 8
        
        x = input.transpose(0, 1)
        '''(batch_size, channels_num, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2))
        x = self.conv_block2(x, pool_size=(2, 2))
        x = self.conv_block3(x, pool_size=(2, 2))
        x = self.conv_block4(x, pool_size=(1, 1))

        (x, _) = torch.max(x, dim=3)    # (batch_size, feature_maps, time_steps)
        x = x.transpose(1, 2)   # (batch_size, time_steps, feature_maps)
        
        event_output = torch.sigmoid(self.event_fc(x))  # (batch_size, time_steps, classes_num)
        elevation_output = self.elevation_fc(x)     # (batch_size, time_steps, classes_num)
        azimuth_output = self.azimuth_fc(x)     # (batch_size, time_steps, classes_num)
        
        # Interpolate
        event_output = interpolate(event_output, interpolate_ratio)
        elevation_output = interpolate(elevation_output, interpolate_ratio)
        azimuth_output = interpolate(azimuth_output, interpolate_ratio)

        output_dict = {
            'event': event_output, 
            'elevation': elevation_output, 
            'azimuth': azimuth_output}

        return output_dict