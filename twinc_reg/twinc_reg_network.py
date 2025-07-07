"""
twinc_reg_network.py
Author: Anupama Jha <anupamaj@uw.edu>
TwinC regression model architecture.
"""

import torch
import numpy as np
from twinc_reg_utils import count_pos_neg, decode_chrome_order_dict, decode_list
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve


def encoder_linear_block(layer1_in,
                         layer1_out,
                         layer2_in,
                         layer2_out,
                         kernel1,
                         kernel2,
                         padding1,
                         padding2):
    """
    This function creates a sequential container using
    linear 1D convolutional layers followed by batch
    normalization.
    :param layer1_in: int, input size for the first conv1D layer.
    :param layer1_out: int, output size for the first conv1D layer.
    :param layer2_in: int, input size for the second conv1D layer.
    :param layer2_out: int, output size for the second conv1D layer.
    :param kernel1: int, filter size for the first conv1D layer.
    :param kernel2: int, filter size for the second conv1D layer.
    :param padding1: int, padding to add to the first layer.
    :param padding2: int, padding to add to the second layer.
    :return: torch.nn.Sequential container
    """
    first_conv = torch.nn.Conv1d(layer1_in,
                                 layer1_out,
                                 kernel_size=kernel1,
                                 padding=padding1)
    first_norm = torch.nn.BatchNorm1d(layer1_out)
    second_conv = torch.nn.Conv1d(layer2_in,
                                  layer2_out,
                                  kernel_size=kernel2,
                                  padding=padding2)
    second_norm = torch.nn.BatchNorm1d(layer2_out)

    linear_conv = torch.nn.Sequential(first_conv,
                                      first_norm,
                                      second_conv,
                                      second_norm)
    return linear_conv


def encoder_linear_block_maxpool(layer1_in,
                                 layer1_out,
                                 layer2_in,
                                 layer2_out,
                                 pool_kernel,
                                 pool_stride,
                                 kernel1,
                                 kernel2,
                                 padding1,
                                 padding2):
    """
    This function creates a sequential container using
    max pooling followed by linear 1D convolutional layers
    and batch normalization.
    :param layer1_in: int, input size for the first conv1D layer.
    :param layer1_out: int, output size for the first conv1D layer.
    :param layer2_in: int, input size for the second conv1D layer.
    :param layer2_out: int, output size for the second conv1D layer.
    :param pool_kernel: int, filter size for the max pooling layer.
    :param pool_stride: int, stride size for the max pooling layer.
    :param kernel1: int, filter size for the first conv1D layer.
    :param kernel2: int, filter size for the second conv1D layer.
    :param padding1: int, padding to add to the first layer.
    :param padding2: int, padding to add to the second layer.
    :return: torch.nn.Sequential container
    """
    maxpool = torch.nn.MaxPool1d(kernel_size=pool_kernel,
                                 stride=pool_stride)
    first_conv = torch.nn.Conv1d(layer1_in,
                                 layer1_out,
                                 kernel_size=kernel1,
                                 padding=padding1)
    first_norm = torch.nn.BatchNorm1d(layer1_out)
    second_conv = torch.nn.Conv1d(layer2_in,
                                  layer2_out,
                                  kernel_size=kernel2,
                                  padding=padding2)
    second_norm = torch.nn.BatchNorm1d(layer2_out)

    pool_linear_conv = torch.nn.Sequential(maxpool,
                                           first_conv,
                                           first_norm,
                                           second_conv,
                                           second_norm)
    return pool_linear_conv


def encoder_nonlinear_block(layer1_in,
                            layer1_out,
                            layer2_in,
                            layer2_out,
                            kernel1,
                            kernel2,
                            padding1,
                            padding2):
    """
    This function creates a sequential container using
    linear 1D convolutional layers followed by batch
    normalization and ReLU nonlinearity.
    :param layer1_in: int, input size for the first conv1D layer.
    :param layer1_out: int, output size for the first conv1D layer.
    :param layer2_in: int, input size for the second conv1D layer.
    :param layer2_out: int, output size for the second conv1D layer.
    :param kernel1: int, filter size for the first conv1D layer.
    :param kernel2: int, filter size for the second conv1D layer.
    :param padding1: int, padding to add to the first layer.
    :param padding2: int, padding to add to the second layer.
    :return: torch.nn.Sequential container
    """
    first_conv = torch.nn.Conv1d(layer1_in,
                                 layer1_out,
                                 kernel_size=kernel1,
                                 padding=padding1)
    first_norm = torch.nn.BatchNorm1d(layer1_out)

    second_conv = torch.nn.Conv1d(layer2_in,
                                  layer2_out,
                                  kernel_size=kernel2,
                                  padding=padding2)
    second_norm = torch.nn.BatchNorm1d(layer2_out)

    relu = torch.nn.ReLU(inplace=True)

    nonlinear_conv = torch.nn.Sequential(first_conv,
                                         first_norm,
                                         relu,
                                         second_conv,
                                         second_norm,
                                         relu)
    return nonlinear_conv


def decoder_linear_block(layer1_in,
                         layer1_out,
                         layer2_in,
                         layer2_out,
                         kernel1,
                         kernel2,
                         padding1,
                         padding2,
                         dilation1,
                         dilation2):
    """
    This function creates a sequential container using
    linear 2D convolutional layers followed by batch
    normalization.
    :param layer1_in: int, input size for the first conv2D layer.
    :param layer1_out: int, output size for the first conv2D layer.
    :param layer2_in: int, input size for the second conv2D layer.
    :param layer2_out: int, output size for the second conv2D layer.
    :param kernel1: (int, int), filter size for the first conv2D layer.
    :param kernel2: (int, int), filter size for the second conv2D layer.
    :param padding1: int, padding to add to the first layer.
    :param padding2: int, padding to add to the second layer.
    :param dilation1: int, dilation to add to the first layer.
    :param dilation2: int, dilation to add to the second layer.
    :return: torch.nn.Sequential container
    """
    first_conv = torch.nn.Conv2d(layer1_in,
                                 layer1_out,
                                 kernel_size=kernel1,
                                 padding=padding1,
                                 dilation=dilation1)
    first_norm = torch.nn.BatchNorm2d(layer1_out)
    second_conv = torch.nn.Conv2d(layer2_in,
                                  layer2_out,
                                  kernel_size=kernel2,
                                  padding=padding2,
                                  dilation=dilation2)
    second_norm = torch.nn.BatchNorm2d(layer2_out)

    linear_conv = torch.nn.Sequential(first_conv,
                                      first_norm,
                                      second_conv,
                                      second_norm)
    return linear_conv


def decoder_linear_block_drop(layer1_in,
                              layer1_out,
                              layer2_in,
                              layer2_out,
                              kernel1,
                              kernel2,
                              padding1,
                              padding2,
                              dilation1,
                              dilation2,
                              drop_prob):
    """
    This function creates a sequential container using
    dropout followed by linear 2D convolutional layers and batch
    normalization.
    :param layer1_in: int, input size for the first conv2D layer.
    :param layer1_out: int, output size for the first conv2D layer.
    :param layer2_in: int, input size for the second conv2D layer.
    :param layer2_out: int, output size for the second conv2D layer.
    :param kernel1: (int, int), filter size for the first conv2D layer.
    :param kernel2: (int, int), filter size for the second conv2D layer.
    :param padding1: int, padding to add to the first layer.
    :param padding2: int, padding to add to the second layer.
    :param dilation1: int, dilation to add to the first layer.
    :param dilation2: int, dilation to add to the second layer.
    :param drop_prob: float, dropout probability
    :return: torch.nn.Sequential container
    """
    dropout = torch.nn.Dropout(p=drop_prob)
    first_conv = torch.nn.Conv2d(layer1_in,
                                 layer1_out,
                                 kernel_size=kernel1,
                                 padding=padding1,
                                 dilation=dilation1)
    first_norm = torch.nn.BatchNorm2d(layer1_out)
    second_conv = torch.nn.Conv2d(layer2_in,
                                  layer2_out,
                                  kernel_size=kernel2,
                                  padding=padding2,
                                  dilation=dilation2)
    second_norm = torch.nn.BatchNorm2d(layer2_out)

    drop_linear_conv = torch.nn.Sequential(dropout,
                                           first_conv,
                                           first_norm,
                                           second_conv,
                                           second_norm)
    return drop_linear_conv


def decoder_linear_block_maxpool(layer1_in,
                                 layer1_out,
                                 layer2_in,
                                 layer2_out,
                                 pool_kernel,
                                 pool_stride,
                                 kernel1,
                                 kernel2,
                                 padding1,
                                 padding2,
                                 dilation1,
                                 dilation2):
    """
    This function creates a sequential container using
    max pooling followed by linear 1D convolutional layers
    and batch normalization.
    :param layer1_in: int, input size for the first conv1D layer.
    :param layer1_out: int, output size for the first conv1D layer.
    :param layer2_in: int, input size for the second conv1D layer.
    :param layer2_out: int, output size for the second conv1D layer.
    :param pool_kernel: int, filter size for the max pooling layer.
    :param pool_stride: int, stride size for the max pooling layer.
    :param kernel1: int, filter size for the first conv1D layer.
    :param kernel2: int, filter size for the second conv1D layer.
    :param padding1: int, padding to add to the first layer.
    :param padding2: int, padding to add to the second layer.
    :param dilation1: int, dilation to add to the first layer.
    :param dilation2: int, dilation to add to the second layer.
    :return: torch.nn.Sequential container
    """
    maxpool = torch.nn.MaxPool2d(kernel_size=pool_kernel,
                                 stride=pool_stride)
    first_conv = torch.nn.Conv2d(layer1_in,
                                 layer1_out,
                                 kernel_size=kernel1,
                                 padding=padding1,
                                 dilation=dilation1)
    first_norm = torch.nn.BatchNorm2d(layer1_out)
    second_conv = torch.nn.Conv2d(layer2_in,
                                  layer2_out,
                                  kernel_size=kernel2,
                                  padding=padding2,
                                  dilation=dilation2)
    second_norm = torch.nn.BatchNorm2d(layer2_out)

    pool_linear_conv = torch.nn.Sequential(maxpool,
                                           first_conv,
                                           first_norm,
                                           second_conv,
                                           second_norm)
    return pool_linear_conv


def decoder_nonlinear_block(layer1_in,
                            layer1_out,
                            layer2_in,
                            layer2_out,
                            kernel1,
                            kernel2,
                            padding1,
                            padding2,
                            dilation1,
                            dilation2):
    """
    This function creates a sequential container using
    linear 2D convolutional layers, ReLU activation
    and batch normalization.
    :param layer1_in: int, input size for the first conv2D layer.
    :param layer1_out: int, output size for the first conv2D layer.
    :param layer2_in: int, input size for the second conv2D layer.
    :param layer2_out: int, output size for the second conv2D layer.
    :param kernel1: (int, int), filter size for the first conv2D layer.
    :param kernel2: (int, int), filter size for the second conv2D layer.
    :param padding1: int, padding to add to the first layer.
    :param padding2: int, padding to add to the second layer.
    :param dilation1: int, dilation to add to the first layer.
    :param dilation2: int, dilation to add to the second layer.
    :return: torch.nn.Sequential container
    """
    first_conv = torch.nn.Conv2d(layer1_in,
                                 layer1_out,
                                 kernel_size=kernel1,
                                 padding=padding1,
                                 dilation=dilation1)
    first_norm = torch.nn.BatchNorm2d(layer1_out)

    second_conv = torch.nn.Conv2d(layer2_in,
                                  layer2_out,
                                  kernel_size=kernel2,
                                  padding=padding2,
                                  dilation=dilation2)
    second_norm = torch.nn.BatchNorm2d(layer2_out)

    relu = torch.nn.ReLU(inplace=True)

    nonlinear_conv = torch.nn.Sequential(first_conv,
                                         first_norm,
                                         relu,
                                         second_conv,
                                         second_norm,
                                         relu)
    return nonlinear_conv


def decoder_final_block(layer1_in,
                        layer1_out,
                        layer2_in,
                        layer2_out,
                        kernel1,
                        kernel2,
                        padding1,
                        padding2):
    """
    This function creates a sequential container using
    linear 2D convolutional layers, batch normalization
    and ReLU activation.
    :param layer1_in: int, input size for the first conv2D layer.
    :param layer1_out: int, output size for the first conv2D layer.
    :param layer2_in: int, input size for the second conv2D layer.
    :param layer2_out: int, output size for the second conv2D layer.
    :param kernel1: (int, int), filter size for the first conv2D layer.
    :param kernel2: (int, int), filter size for the second conv2D layer.
    :param padding1: int, padding to add to the first layer.
    :param padding2: int, padding to add to the second layer.
    :return: torch.nn.Sequential container
    """
    first_conv = torch.nn.Conv2d(layer1_in,
                                 layer1_out,
                                 kernel_size=kernel1,
                                 padding=padding1)
    first_norm = torch.nn.BatchNorm2d(layer1_out)

    relu = torch.nn.ReLU(inplace=True)

    second_conv = torch.nn.Conv2d(layer2_in,
                                  layer2_out,
                                  kernel_size=kernel2,
                                  padding=padding2)

    final_conv = torch.nn.Sequential(first_conv,
                                     first_norm,
                                     relu,
                                     second_conv)
    return final_conv


class TwinCRegNet(torch.nn.Module):
    def __init__(self):
        """
        Constructor for the TwinCNet
        """
        super(TwinCRegNet, self).__init__()

        # Encoder linear layers
        self.lconv1 = encoder_linear_block(layer1_in=4,
                                           layer1_out=64,
                                           layer2_in=64,
                                           layer2_out=64,
                                           kernel1=9,
                                           kernel2=9,
                                           padding1=4,
                                           padding2=4)

        self.lconv2 = encoder_linear_block_maxpool(layer1_in=64,
                                                   layer1_out=96,
                                                   layer2_in=96,
                                                   layer2_out=96,
                                                   pool_kernel=4,
                                                   pool_stride=4,
                                                   kernel1=9,
                                                   kernel2=9,
                                                   padding1=4,
                                                   padding2=4)

        self.lconv3 = encoder_linear_block_maxpool(layer1_in=96,
                                                   layer1_out=128,
                                                   layer2_in=128,
                                                   layer2_out=128,
                                                   pool_kernel=4,
                                                   pool_stride=4,
                                                   kernel1=9,
                                                   kernel2=9,
                                                   padding1=4,
                                                   padding2=4)

        self.lconv4 = encoder_linear_block_maxpool(layer1_in=128,
                                                   layer1_out=128,
                                                   layer2_in=128,
                                                   layer2_out=128,
                                                   pool_kernel=4,
                                                   pool_stride=4,
                                                   kernel1=9,
                                                   kernel2=9,
                                                   padding1=4,
                                                   padding2=4)

        self.lconv5 = encoder_linear_block_maxpool(layer1_in=128,
                                                   layer1_out=128,
                                                   layer2_in=128,
                                                   layer2_out=128,
                                                   pool_kernel=5,
                                                   pool_stride=5,
                                                   kernel1=9,
                                                   kernel2=9,
                                                   padding1=4,
                                                   padding2=4)

        self.lconv6 = encoder_linear_block_maxpool(layer1_in=128,
                                                   layer1_out=128,
                                                   layer2_in=128,
                                                   layer2_out=128,
                                                   pool_kernel=5,
                                                   pool_stride=5,
                                                   kernel1=9,
                                                   kernel2=9,
                                                   padding1=4,
                                                   padding2=4)

        self.lconv7 = encoder_linear_block_maxpool(layer1_in=128,
                                                   layer1_out=128,
                                                   layer2_in=128,
                                                   layer2_out=128,
                                                   pool_kernel=2,
                                                   pool_stride=2,
                                                   kernel1=9,
                                                   kernel2=9,
                                                   padding1=4,
                                                   padding2=4)

        self.lconv8 = encoder_linear_block_maxpool(layer1_in=128,
                                                   layer1_out=128,
                                                   layer2_in=128,
                                                   layer2_out=128,
                                                   pool_kernel=2,
                                                   pool_stride=2,
                                                   kernel1=9,
                                                   kernel2=9,
                                                   padding1=4,
                                                   padding2=4)

        self.lconv9 = encoder_linear_block_maxpool(layer1_in=128,
                                                   layer1_out=128,
                                                   layer2_in=128,
                                                   layer2_out=128,
                                                   pool_kernel=2,
                                                   pool_stride=2,
                                                   kernel1=9,
                                                   kernel2=9,
                                                   padding1=4,
                                                   padding2=4)

        self.lconv10 = encoder_linear_block_maxpool(layer1_in=128,
                                                    layer1_out=128,
                                                    layer2_in=128,
                                                    layer2_out=128,
                                                    pool_kernel=5,
                                                    pool_stride=5,
                                                    kernel1=9,
                                                    kernel2=9,
                                                    padding1=4,
                                                    padding2=4)
        # Encoder linear module
        lconvs = torch.nn.ModuleList(
            [self.lconv1,
             self.lconv2,
             self.lconv3,
             self.lconv4,
             self.lconv5,
             self.lconv6,
             self.lconv7,
             self.lconv8,
             self.lconv9,
             self.lconv10,
             ]
        )

        # Encoder nonlinear layers
        self.conv1 = encoder_nonlinear_block(layer1_in=64,
                                             layer1_out=64,
                                             layer2_in=64,
                                             layer2_out=64,
                                             kernel1=9,
                                             kernel2=9,
                                             padding1=4,
                                             padding2=4)

        self.conv2 = encoder_nonlinear_block(layer1_in=96,
                                             layer1_out=96,
                                             layer2_in=96,
                                             layer2_out=96,
                                             kernel1=9,
                                             kernel2=9,
                                             padding1=4,
                                             padding2=4)

        self.conv3 = encoder_nonlinear_block(layer1_in=128,
                                             layer1_out=128,
                                             layer2_in=128,
                                             layer2_out=128,
                                             kernel1=9,
                                             kernel2=9,
                                             padding1=4,
                                             padding2=4)

        self.conv4 = encoder_nonlinear_block(layer1_in=128,
                                             layer1_out=128,
                                             layer2_in=128,
                                             layer2_out=128,
                                             kernel1=9,
                                             kernel2=9,
                                             padding1=4,
                                             padding2=4)

        self.conv5 = encoder_nonlinear_block(layer1_in=128,
                                             layer1_out=128,
                                             layer2_in=128,
                                             layer2_out=128,
                                             kernel1=9,
                                             kernel2=9,
                                             padding1=4,
                                             padding2=4)

        self.conv6 = encoder_nonlinear_block(layer1_in=128,
                                             layer1_out=128,
                                             layer2_in=128,
                                             layer2_out=128,
                                             kernel1=9,
                                             kernel2=9,
                                             padding1=4,
                                             padding2=4)

        self.conv7 = encoder_nonlinear_block(layer1_in=128,
                                             layer1_out=128,
                                             layer2_in=128,
                                             layer2_out=128,
                                             kernel1=9,
                                             kernel2=9,
                                             padding1=4,
                                             padding2=4)

        self.conv8 = encoder_nonlinear_block(layer1_in=128,
                                             layer1_out=128,
                                             layer2_in=128,
                                             layer2_out=128,
                                             kernel1=9,
                                             kernel2=9,
                                             padding1=4,
                                             padding2=4)

        self.conv9 = encoder_nonlinear_block(layer1_in=128,
                                             layer1_out=128,
                                             layer2_in=128,
                                             layer2_out=128,
                                             kernel1=9,
                                             kernel2=9,
                                             padding1=4,
                                             padding2=4)

        self.conv10 = encoder_nonlinear_block(layer1_in=128,
                                              layer1_out=128,
                                              layer2_in=128,
                                              layer2_out=128,
                                              kernel1=9,
                                              kernel2=9,
                                              padding1=4,
                                              padding2=4)

        # Encoder nonlinear module
        convs = torch.nn.ModuleList(
            [self.conv1,
             self.conv2,
             self.conv3,
             self.conv4,
             self.conv5,
             self.conv6,
             self.conv7,
             self.conv8,
             self.conv9,
             self.conv10,
             ]
        )

        # Decoder linear layers
        lconvtwo1 = decoder_linear_block_drop(layer1_in=128,
                                              layer1_out=32,
                                              layer2_in=32,
                                              layer2_out=64,
                                              kernel1=(3, 3),
                                              kernel2=(3, 3),
                                              padding1=1,
                                              padding2=1,
                                              dilation1=1,
                                              dilation2=1,
                                              drop_prob=0.1)

        lconvtwo2_maxpool = decoder_linear_block_maxpool(layer1_in=64,
                                                         layer1_out=32,
                                                         layer2_in=32,
                                                         layer2_out=64,
                                                         pool_kernel=5,
                                                         pool_stride=5,
                                                         kernel1=(3, 3),
                                                         kernel2=(3, 3),
                                                         padding1=2,
                                                         padding2=2,
                                                         dilation1=2,
                                                         dilation2=2)

        lconvtwo2 = decoder_linear_block(layer1_in=64,
                                         layer1_out=32,
                                         layer2_in=32,
                                         layer2_out=64,
                                         kernel1=(3, 3),
                                         kernel2=(3, 3),
                                         padding1=2,
                                         padding2=2,
                                         dilation1=2,
                                         dilation2=2)

        lconvtwo3 = decoder_linear_block(layer1_in=64,
                                         layer1_out=32,
                                         layer2_in=32,
                                         layer2_out=64,
                                         kernel1=(3, 3),
                                         kernel2=(3, 3),
                                         padding1=4,
                                         padding2=4,
                                         dilation1=4,
                                         dilation2=4)

        lconvtwo4_maxpool = decoder_linear_block_maxpool(layer1_in=64,
                                                         layer1_out=32,
                                                         layer2_in=32,
                                                         layer2_out=64,
                                                         pool_kernel=2,
                                                         pool_stride=2,
                                                         kernel1=(3, 3),
                                                         kernel2=(3, 3),
                                                         padding1=8,
                                                         padding2=8,
                                                         dilation1=8,
                                                         dilation2=8)

        lconvtwo4 = decoder_linear_block(layer1_in=64,
                                         layer1_out=32,
                                         layer2_in=32,
                                         layer2_out=64,
                                         kernel1=(3, 3),
                                         kernel2=(3, 3),
                                         padding1=8,
                                         padding2=8,
                                         dilation1=8,
                                         dilation2=8)

        lconvtwo5 = decoder_linear_block(layer1_in=64,
                                         layer1_out=32,
                                         layer2_in=32,
                                         layer2_out=64,
                                         kernel1=(3, 3),
                                         kernel2=(3, 3),
                                         padding1=16,
                                         padding2=16,
                                         dilation1=16,
                                         dilation2=16)

        lconvtwo6 = decoder_linear_block(layer1_in=64,
                                         layer1_out=32,
                                         layer2_in=32,
                                         layer2_out=64,
                                         kernel1=(3, 3),
                                         kernel2=(3, 3),
                                         padding1=32,
                                         padding2=32,
                                         dilation1=32,
                                         dilation2=32)

        lconvtwo7 = decoder_linear_block(layer1_in=64,
                                         layer1_out=32,
                                         layer2_in=32,
                                         layer2_out=64,
                                         kernel1=(3, 3),
                                         kernel2=(3, 3),
                                         padding1=64,
                                         padding2=64,
                                         dilation1=64,
                                         dilation2=64)

        # Decoder nonlinear layers
        convtwo1 = decoder_nonlinear_block(layer1_in=64,
                                           layer1_out=32,
                                           layer2_in=32,
                                           layer2_out=64,
                                           kernel1=(3, 3),
                                           kernel2=(3, 3),
                                           padding1=1,
                                           padding2=1,
                                           dilation1=1,
                                           dilation2=1)

        convtwo2 = decoder_nonlinear_block(layer1_in=64,
                                           layer1_out=32,
                                           layer2_in=32,
                                           layer2_out=64,
                                           kernel1=(3, 3),
                                           kernel2=(3, 3),
                                           padding1=2,
                                           padding2=2,
                                           dilation1=2,
                                           dilation2=2)

        convtwo3 = decoder_nonlinear_block(layer1_in=64,
                                           layer1_out=32,
                                           layer2_in=32,
                                           layer2_out=64,
                                           kernel1=(3, 3),
                                           kernel2=(3, 3),
                                           padding1=4,
                                           padding2=4,
                                           dilation1=4,
                                           dilation2=4)

        convtwo4 = decoder_nonlinear_block(layer1_in=64,
                                           layer1_out=32,
                                           layer2_in=32,
                                           layer2_out=64,
                                           kernel1=(3, 3),
                                           kernel2=(3, 3),
                                           padding1=8,
                                           padding2=8,
                                           dilation1=8,
                                           dilation2=8)

        convtwo5 = decoder_nonlinear_block(layer1_in=64,
                                           layer1_out=32,
                                           layer2_in=32,
                                           layer2_out=64,
                                           kernel1=(3, 3),
                                           kernel2=(3, 3),
                                           padding1=16,
                                           padding2=16,
                                           dilation1=16,
                                           dilation2=16)

        convtwo6 = decoder_nonlinear_block(layer1_in=64,
                                           layer1_out=32,
                                           layer2_in=32,
                                           layer2_out=64,
                                           kernel1=(3, 3),
                                           kernel2=(3, 3),
                                           padding1=32,
                                           padding2=32,
                                           dilation1=32,
                                           dilation2=32)

        convtwo7 = decoder_nonlinear_block(layer1_in=64,
                                           layer1_out=32,
                                           layer2_in=32,
                                           layer2_out=64,
                                           kernel1=(3, 3),
                                           kernel2=(3, 3),
                                           padding1=64,
                                           padding2=64,
                                           dilation1=64,
                                           dilation2=64)

        # Decoder linear module
        self.lconvtwos = torch.nn.ModuleList(
            [lconvtwo1,
             lconvtwo2_maxpool,
             lconvtwo3,
             lconvtwo4_maxpool,
             lconvtwo5,
             lconvtwo6,
             lconvtwo7,
             lconvtwo2,
             lconvtwo3,
             lconvtwo4,
             lconvtwo5,
             lconvtwo6,
             lconvtwo7,
             lconvtwo2,
             lconvtwo3,
             lconvtwo4,
             lconvtwo5,
             lconvtwo6,
             lconvtwo7
             ]
        )

        # Decoder nonlinear module
        self.convtwos = torch.nn.ModuleList(
            [convtwo1,
             convtwo2,
             convtwo3,
             convtwo4,
             convtwo5,
             convtwo6,
             convtwo7,
             convtwo2,
             convtwo3,
             convtwo4,
             convtwo5,
             convtwo6,
             convtwo7,
             convtwo2,
             convtwo3,
             convtwo4,
             convtwo5,
             convtwo6,
             convtwo7,
             ]
        )

        # Final layer before output
        self.final = decoder_final_block(layer1_in=64,
                                         layer1_out=5,
                                         layer2_in=5,
                                         layer2_out=1,
                                         kernel1=(1, 1),
                                         kernel2=(1, 1),
                                         padding1=0,
                                         padding2=0)

        # Loss functions
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.cross_entropy_loss = torch.nn.BCELoss(reduction="mean")

    def forward_twin_encoder(self, x):
        """
        Both sequences pass through the same ENCODER to generate
        the sequence embedding.
        :param x: tensor, one-hot-encoded sequence
        :return: tensor, sequence embedding
        """
        lx = self.lconv1(x)
        x = self.conv1(lx)
        print(f"x1: {x.shape}")
        lx = self.lconv2(x + lx)
        x = self.conv2(lx)
        print(f"x2: {x.shape}")
        lx = self.lconv3(x + lx)
        x = self.conv3(lx)
        print(f"x3: {x.shape}")
        lx = self.lconv4(x + lx)
        x = self.conv4(lx)
        print(f"x4: {x.shape}")
        lx = self.lconv5(x + lx)
        x = self.conv5(lx)
        print(f"x5: {x.shape}")
        lx = self.lconv6(x + lx)
        x = self.conv6(lx)
        print(f"x6: {x.shape}")
        lx = self.lconv7(x + lx)
        x = self.conv7(lx)
        print(f"x7: {x.shape}")
        lx = self.lconv8(x + lx)
        x = self.conv8(lx)
        print(f"x8: {x.shape}")
        lx = self.lconv9(x + lx)
        x = self.conv9(lx)
        print(f"x9: {x.shape}")
        return x

    def forward_decoder(self, x1, x2):
        """
        Takes two sequence embeddings and generates output trans Hi-C contact
        :param x1: tensor, embedding for the first sequence
        :param x2: tensor, embedding for the second sequence
        :return: tensor, output trans Hi-C contact
        """
        embed = x1[:, :, :, None] + x2[:, :, None, :]

        first = True
        for lconvtwo, convtwo in zip(self.lconvtwos, self.convtwos):
            if first:
                embed = lconvtwo(embed)
                embed = convtwo(embed) + embed
                first = False
                print(f"embed: {embed.shape}")
            else:
                lembed = lconvtwo(embed)
                if lembed.size() == embed.size():
                    embed = lembed + embed
                else:
                    embed = lembed
                embed = convtwo(embed) + embed
                print(f"embed: {embed.shape}")
        embed = self.final(embed)
        embed = torch.squeeze(embed)
        print(f"embed: {embed.shape}")
        if len(embed.shape) == 2:
            embed = embed[None, :, :]
        # embed = self.flatten(embed)
        return embed

    def forward(self, X1, X2):
        """A forward pass of the model.
        This method takes in two nucleotide sequences X1 and X2
        and makes predictions for the trans-Hi-C contacts between
        them.
        Parameters
        ----------
        X1: torch.tensor, shape=(batch_size, 4, sequence_length)
            The one-hot encoded batch of sequences.
        X2: torch.tensor, shape=(batch_size, 4, sequence_length)
            The one-hot encoded batch of sequences.
        Returns
        -------
        y: torch.tensor, shape=(batch_size, out_length)
            The trans-Hi-C predictions.
        """
        X1 = self.forward_twin_encoder(X1)
        X2 = self.forward_twin_encoder(X2)
        y = self.forward_decoder(X1, X2)
        # y = self.softmax(self.label(X))
        return y

    def fit_supervised(
            self,
            training_data,
            model_optimizer,
            validation_data,
            max_epochs=10,
            validation_iter=1000,
            device="cpu",
            best_save_model="",
            final_save_model=""
    ):
        """
        Training procedure for the supervised version
        of m6A CNN.
        :param training_data: torch.DataLoader,
                              training data generator
        :param model_optimizer: torch.Optimizer,
                                An optimizer to
                                training our model
        :param X_valid: numpy array, validation features
        :param y_valid: numpy array, validation labels
        :param max_epochs: int, maximum epochs to run
                                the model for
        :param validation_iter: int,After how many
                                    iterations should
                                    we compute validation
                                    stats.
        :param device: str, GPU versus CPU, defaults to CPU
        :param best_save_model: str, path to save best model
        :param final_save_model: str, path to save final model
        :return: None
        """

        best_loss = np.inf
        best_corr = -np.inf
        for epoch in range(max_epochs):
            # to log cross-entropy loss to
            # average over batches
            avg_train_loss = 0
            avg_train_iter = 0
            iteration = 0
            for data in training_data:
                # Get features and label batch
                X1, X2, y = data
                # Convert them to float
                X1, X2, y = X1.float(), X2.float(), y.float()
                X1, X2, y = X1.to(device), X2.to(device), y.to(device)

                # Clear the optimizer and
                # set the model to training mode
                model_optimizer.zero_grad()
                self.train()

                # Run forward pass
                train_pred = self.forward(X1, X2)

                mse_loss = ((train_pred[~torch.isnan(y)] - y[~torch.isnan(y)]) ** 2).mean()

                mse_loss.backward()
                # Extract the cross entropy loss for logging
                mse_loss_item = mse_loss.detach().cpu().numpy()
                model_optimizer.step()

                # log loss to average over training batches
                avg_train_loss += mse_loss_item
                avg_train_iter += 1
                train_loss = avg_train_loss / avg_train_iter

                print(f"Epoch {epoch}, iteration {iteration},"
                      f" train loss: {train_loss:.4f},", flush=True
                      )

                # If current iteration is a validation iteration compute validation stats.
                if iteration % validation_iter == 0 and iteration > 0:
                    with torch.no_grad():
                        # Set the model to
                        # evaluation mode
                        self.eval()
                        y_valid = torch.empty((0, 5, 5))
                        valid_preds = torch.empty((0, 5, 5)).to(device)
                        cnt = 0
                        for data in validation_data:
                            # Get features and label batch
                            X1, X2, y = data
                            y_valid = torch.cat((y_valid, y))
                            # Convert them to float
                            X1, X2, y = X1.float(), X2.float(), y.float()
                            X1, X2, y = X1.to(device), X2.to(device), y.to(device)

                            # Run forward pass
                            val_pred = self.forward(X1, X2)
                            valid_preds = torch.cat((valid_preds, val_pred))
                            valid_preds = valid_preds.to(device)
                            cnt += 1
                            if cnt > 1000:
                                break
                        valid_preds, y_valid = valid_preds.to(device), y_valid.to(device)

                        valid_corr = spearmanr(valid_preds[~torch.isnan(y_valid)].detach().cpu().numpy(),
                                               y_valid[~torch.isnan(y_valid)].detach().cpu().numpy())[0]

                        mse_loss = ((valid_preds[~torch.isnan(y_valid)] - y_valid[~torch.isnan(y_valid)]) ** 2).mean()

                        # Extract the validation loss
                        valid_loss = mse_loss.detach().cpu().numpy()
                        train_loss = avg_train_loss / avg_train_iter

                        print(
                            f"Epoch {epoch}, iteration {iteration},"
                            f" train loss: {train_loss:4.4f},"
                            f" validation loss: {valid_loss:4.4f}",
                            f" validation spearman corr: {valid_corr:4.4f}",
                            flush=True
                        )

                        if valid_loss < best_loss:
                            torch.save(self.state_dict(), best_save_model)
                            best_loss = valid_loss
                        if valid_corr > best_corr:
                            torch.save(self.state_dict(), final_save_model)
                            best_corr = valid_corr

                        avg_train_loss = 0
                        avg_train_iter = 0

                iteration += 1

        torch.save(self.state_dict(), final_save_model)


if __name__ == '__main__':
    best_save_model = "../models/TwinC_reg_H1ESC.torch"
    trans_hic_model = TwinCRegNet()
    trans_hic_model.load_state_dict(torch.load(best_save_model))
    input_tensor1 = torch.randn(1, 4, 640000)
    input_tensor2 = torch.randn(1, 4, 640000)

    output = trans_hic_model(input_tensor1, input_tensor2)
    print(f"output: {output.shape}")
