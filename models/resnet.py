'''VGG11/13/16/19 in Pytorch.'''
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW, lr_scheduler
from utils.utils import evaluate
import time

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, act='relu', norm='bn', bias=False):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=bias)
        if norm == 'bn':
            self.bn1 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = nn.GroupNorm(1, planes)
        self.conv2 = conv3x3(planes, planes, bias=bias)
        if norm == 'bn':
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.bn2 = nn.GroupNorm(1, planes)

        if act == 'relu':
            self.act = torch.nn.ReLU(inplace=True)
        else:
            self.act = torch.nn.Mish(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if norm == 'bn':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias),
                    nn.GroupNorm(1, self.expansion*planes)
                )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.act(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, act='relu', norm='bn', bias=False):
        super().__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64, bias=bias)
        if norm == 'bn':
            self.bn1 = nn.BatchNorm2d(64)
        elif norm == 'gn':
            self.bn1 = nn.GroupNorm(1, 64)

        if act == 'relu':
            self.act = torch.nn.ReLU(inplace=True)
        else:
            self.act = torch.nn.Mish(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, act=act, norm=norm, bias=bias)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, act=act, norm=norm, bias=bias)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, act=act, norm=norm, bias=bias)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, act=act, norm=norm, bias=bias)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, act, norm, bias=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, act, norm, bias=bias))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def forward_layers(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out1.reshape(out.size(0), -1), out2.reshape(out.size(0), -1), out3.reshape(out.size(0), -1), out4.reshape(out.size(0), -1)

def ResNet18(model_name, act="relu", norm="bn", num_classes=10, bias=False):
    return ResNet(BasicBlock, [2,2,2,2], num_classes, act=act, norm=norm, bias=bias)


def get_blocks(net):
    return nn.Sequential(nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool),
                         *net.layer1, *net.layer2, *net.layer3, *net.layer4)

