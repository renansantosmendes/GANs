import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import utils


class Generator(nn.Module):
    def __init__(self, z_dim: int, generator_hidden_layer_size: int, image_channels: int) -> None:
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            nn.ConvTranspose2d(z_dim,
                               generator_hidden_layer_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(generator_hidden_layer_size * 8),
            nn.ReLU(True),
            # 2nd layer
            nn.ConvTranspose2d(generator_hidden_layer_size * 8,
                               generator_hidden_layer_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_hidden_layer_size * 4),
            nn.ReLU(True),
            # 3rd layer
            nn.ConvTranspose2d(generator_hidden_layer_size * 4,
                               generator_hidden_layer_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_hidden_layer_size * 2),
            nn.ReLU(True),
            # 4th layer
            nn.ConvTranspose2d(generator_hidden_layer_size * 2,
                               generator_hidden_layer_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_hidden_layer_size),
            nn.ReLU(True),
            # output layer
            nn.ConvTranspose2d(generator_hidden_layer_size,
                               image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            )

    def forward(self, input_data):
        return self.main(input_data)


def weights_init(layers):
    classname = layers.__class__.__name__
    if classname.find('Conv') != -1:
        layers.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        layers.weight.data.normal_(1.0, 0.02)
        layers.bias.data.fill_(0)


class Discriminator(nn.Module):
    def __init__(self, image_channels, discriminator_hidden_layer_size):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(image_channels,
                      discriminator_hidden_layer_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(discriminator_hidden_layer_size,
                      discriminator_hidden_layer_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_hidden_layer_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(discriminator_hidden_layer_size * 2,
                      discriminator_hidden_layer_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_hidden_layer_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(discriminator_hidden_layer_size * 4,
                      discriminator_hidden_layer_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_hidden_layer_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Conv2d(discriminator_hidden_layer_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)
