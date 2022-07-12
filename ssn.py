# SSN definition script
# This script just needed to build the feature extractor
# Pytorch implementation from: https://github.com/andrewsonga/ssn_pytorch
# Original paper is: https://varunjampani.github.io/ssn/

import torch
import torch.nn as nn
import torch.nn.functional as F

class crop(nn.Module):
    # all dimensions up to but excluding 'axis' are preserved
    # while the dimensions including and trailing 'axis' are cropped
    # (since the standard dimensions are N, C, H, W,  the default is a spatial crop)

    def __init__(self, axis = 2, offset = 0):
        super(crop, self).__init__()
        self.axis = axis
        self.offset = offset
        
    def forward(self, x, ref):
        for axis in range(self.axis, x.dim()):
            ref_size = ref.size(axis)
            indices = torch.arange(self.offset, self.offset + ref_size)
            indices = x.data.new().resize_(indices.size()).copy_(indices)
            x = x.index_select(axis, indices.long())
        return x

######################
#  Define the model  #
######################

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_pixel_features):
        super(CNN, self).__init__()
        
        ##############################################
        ########## 1st convolutional layer ###########
        self.conv1_bn_relu_layer = nn.Sequential()
        self.conv1_bn_relu_layer.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        self.conv1_bn_relu_layer.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        self.conv1_bn_relu_layer.add_module("relu", nn.ReLU())

        ##############################################
        ###### 2nd/4th/6th convolutional layers ######
        self.conv2_bn_relu_layer = nn.Sequential()
        self.conv2_bn_relu_layer.add_module("conv", nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        self.conv2_bn_relu_layer.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        self.conv2_bn_relu_layer.add_module("relu", nn.ReLU())

        self.conv4_bn_relu_layer = nn.Sequential()
        self.conv4_bn_relu_layer.add_module("conv", nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        self.conv4_bn_relu_layer.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        self.conv4_bn_relu_layer.add_module("relu", nn.ReLU())

        self.conv6_bn_relu_layer = nn.Sequential()
        self.conv6_bn_relu_layer.add_module("conv", nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        self.conv6_bn_relu_layer.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        self.conv6_bn_relu_layer.add_module("relu", nn.ReLU())
        
        ##############################################
        ######## 3rd/5th convolutional layers ########
        self.pool_conv3_bn_relu_layer = nn.Sequential()
        self.pool_conv3_bn_relu_layer.add_module("maxpool", nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        self.pool_conv3_bn_relu_layer.add_module("conv", nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        self.pool_conv3_bn_relu_layer.add_module("batchnorm", nn.BatchNorm2d(out_channels)) # the gamma and betas are trainable parameters of Batchnorm
        self.pool_conv3_bn_relu_layer.add_module("relu", nn.ReLU())

        self.pool_conv5_bn_relu_layer = nn.Sequential()
        self.pool_conv5_bn_relu_layer.add_module("maxpool", nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        self.pool_conv5_bn_relu_layer.add_module("conv", nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        self.pool_conv5_bn_relu_layer.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        self.pool_conv5_bn_relu_layer.add_module("relu", nn.ReLU())

        ##############################################
        ####### 7th (Last) convolutional layer #######
        self.conv7_relu_layer = nn.Sequential()
        self.conv7_relu_layer.add_module("conv", nn.Conv2d(3 * out_channels + in_channels, num_pixel_features - in_channels, kernel_size = 3, stride = 1, padding = 1))
        self.conv7_relu_layer.add_module("relu", nn.ReLU())

        ##############################################
        ################### crop #####################
        self.crop = crop()

    def forward(self, x):

        conv1 = self.conv1_bn_relu_layer(x)
        conv2 = self.conv2_bn_relu_layer(conv1)
        conv3 = self.pool_conv3_bn_relu_layer(conv2)
        conv4 = self.conv4_bn_relu_layer(conv3)
        conv5 = self.pool_conv5_bn_relu_layer(conv4)
        conv6 = self.conv6_bn_relu_layer(conv5)

        # the input data is assumed to be of the form minibatch x channels x [Optinal depth] x [optional height] x width
        # hence, for spatial inputs, we expect a 4D Tensor
        # one can EITHER give a "scale_factor" or a the target output "size" to calculate thje output size (cannot give both, as it's ambiguous)
        conv4_upsample_crop = self.crop(F.interpolate(conv4, scale_factor = 2, mode = 'bilinear'), conv2)
        conv6_upsample_crop = self.crop(F.interpolate(conv6, scale_factor = 4, mode = 'bilinear'), conv2)

        conv7_input = torch.cat((x, conv2, conv4_upsample_crop, conv6_upsample_crop), dim = 1)
        conv7 = self.conv7_relu_layer(conv7_input)

        return conv7
