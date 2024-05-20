import pdb
from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models
from functools import partial
from torchvision.ops.misc import Permute
import torch.nn.functional as F
import math


class BCNN(nn.Module):
    def __init__(self, thresh=1e-8, is_vec=True, input_dim=512):
        super(BCNN, self).__init__()
        self.thresh = thresh
        self.is_vec = is_vec
        self.output_dim = input_dim * input_dim

    def _bilinearpool(self, x):
        batchSize, dim, h, w = x.data.shape
        x = x.reshape(batchSize, dim, h * w)
        x = 1. / (h * w) * x.bmm(x.transpose(1, 2))
        return x

    def _signed_sqrt(self, x):
        x = torch.mul(x.sign(), torch.sqrt(x.abs() + self.thresh))
        return x

    def _l2norm(self, x):
        x = nn.functional.normalize(x)
        return x

    def forward(self, x):
        x = self._bilinearpool(x)
        x = self._signed_sqrt(x)
        if self.is_vec:
            x = x.view(x.size(0), -1)
        x = self._l2norm(x)
        return x


class BaseCNN(nn.Module):
    def __init__(self, config):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        self.config = config





        if self.config.backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
        elif self.config.backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
        elif self.config.backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
        elif self.config.backbone == 'SwinB':
            self.backbone = models.swin_b(weights='Swin_B_Weights.DEFAULT')
            self.backbone.head = Identity()
        if config.std_modeling:
            outdim = 2
        else:
            outdim = 1
        if config.representation == 'BCNN':
            assert ((self.config.backbone == 'resnet18') | (self.config.backbone == 'resnet34')), "The backbone network must be resnet18 or resnet34"
            self.representation = BCNN()
            self.fc = nn.Linear(512 * 512, outdim)
        elif self.config.backbone == 'SwinB':
            self.fc = nn.Linear(1024, outdim)
        else:
            # self.fc = nn.Linear(512, outdim)
            self.fc = nn.Linear(2048, outdim)

        if self.config.fc:
            # Freeze all previous layers.
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Initialize the fc layers.
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)


    def forward(self, x):
        """Forward pass of the network.
        """
        if self.config.backbone == 'SwinB':
            x = self.backbone(x)
        else:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)

        if self.config.backbone != 'SwinB':
            if self.config.representation == 'BCNN':
                x = self.representation(x)
            else:
                x = self.backbone.avgpool(x)
                x = x.view(x.size()[0], x.size()[1])

        x = self.fc(x)

        if self.config.std_modeling:
            mean = x[:, 0]
            t = x[:, 1]
            var = nn.functional.softplus(t)
            return mean, var
        else:
            return x




class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x