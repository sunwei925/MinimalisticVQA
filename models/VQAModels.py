import torch
import torch.nn as nn
import torchvision.models as models
import timm

import math
from .pretrained_IQA_model import BaseCNN

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


# Model (I): ResNet-50 pre-trained on ImageNet-1K as the spatial quality analyzer 
# with no temporal quality analyzer (as the baseline model)
class Model_I(nn.Module):
    def __init__(self, pretrained = True):
        super(Model_I, self).__init__()

        model = models.resnet50(weights='DEFAULT')

        # spatial quality analyzer
        self.feature_extraction = nn.Sequential(*list(model.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # quality regressor
        self.quality = self.quality_regression(2048, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x):

        # x size: batch x frames x 3 x height x width
        x_size = x.shape
        
        # x size: (batch * frames) x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
            
        x = self.feature_extraction(x)

        # x size: (batch * frames) x channel x 1 x 1
        x = self.avgpool(x)

        # x size: (batch * frames) x channel
        x = torch.flatten(x, 1)

        # x size: (batch * frames)
        x = self.quality(x)
        
        # x size: batch x frames
        x = x.view(x_size[0], x_size[1])

        # x size: batch
        x = torch.mean(x, dim = 1)
            
        return x




# Model (II): pre-training Model (I) on the combination of four IQA datasets, the model is loaded from https://github.com/zwx8981/TCSVT-2022-BVQA
class Model_II(torch.nn.Module):
    def __init__(self, config):
        super(Model_II, self).__init__()
        # define the model (i.e., ResNet-50)
        model = BaseCNN(config)
        model = torch.nn.DataParallel(model).cuda()
        # load the pre-trained model
        ckpt = 'ckpts/DataParallel-00008.pt'
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint['state_dict'])
        resnet = nn.Sequential(*list(model.module.backbone.children())[:-2])
        resnet.head = Identity()

        # spatial quality analyzer
        self.feature_extraction = resnet
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # quality regressor
        self.quality = self.quality_regression(2048, 128, 1)

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

        return regression_block

    def forward(self, x):

        # x size: batch x frames x 3 x height x width
        x_size = x.shape
        
        # x size: (batch * frames) x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
            
        x = self.feature_extraction(x)

        # x size: (batch * frames) x channel x 1 x 1
        x = self.avgpool(x)

        # x size: (batch * frames) x channel
        x = torch.flatten(x, 1)

        # x size: (batch * frames)
        x = self.quality(x)
        
        # x size: batch x frames
        x = x.view(x_size[0], x_size[1])

        # x size: batch
        x = torch.mean(x, dim = 1)
            
        return x




# Model (III): pre-training Model (I) on the training set of LSVQ (load the weights when during the training)
class Model_III(nn.Module):
    def __init__(self):
        super(Model_III, self).__init__()

        model = models.resnet50()

        # spatial quality analyzer
        self.feature_extraction = nn.Sequential(*list(model.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # quality regressor
        self.quality = self.quality_regression(2048, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x):

        # x size: batch x frames x 3 x height x width
        x_size = x.shape
        
        # x size: (batch * frames) x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
            
        x = self.feature_extraction(x)

        # x size: (batch * frames) x channel x 1 x 1
        x = self.avgpool(x)

        # x size: (batch * frames) x channel
        x = torch.flatten(x, 1)

        # x size: (batch * frames)
        x = self.quality(x)
        
        # x size: batch x frames
        x = x.view(x_size[0], x_size[1])

        # x size: batch
        x = torch.mean(x, dim = 1)
            
        return x


# Model (IV): adding the temporal analyzer to Model (I)
class Model_IV(nn.Module):
    def __init__(self, pretrained = True):
        super(Model_IV, self).__init__()


        model = models.resnet50(weights='DEFAULT')

        # spatial quality analyzer
        self.feature_extraction = nn.Sequential(*list(model.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # quality regressor
        self.quality = self.quality_regression(2048+256, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x, x_temporal_featurs):

        # x size: batch x frames x 3 x height x width
        x_size = x.shape

        # x_temporal_featurs size: batch x frames x 2048
        x_temporal_featurs_size = x_temporal_featurs.shape

        # x size: (batch * frames) x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        # x_temporal_featurs size: (batch * frames) x 256
        x_temporal_featurs = x_temporal_featurs.view(-1, x_temporal_featurs_size[2])
        
        
        
        x = self.feature_extraction(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        # x: (batch * frames) x (channel + 256)
        x = torch.cat((x, x_temporal_featurs), dim = 1)

        x = self.quality(x)
        
        # x: batch x frames
        x = x.view(x_size[0], x_size[1])

        # x: batch
        x = torch.mean(x, dim = 1)

            
        return x


# Model (V): adding the temporal analyzer to Model (II)
class Model_V(torch.nn.Module):
    def __init__(self):
        super(Model_V, self).__init__()
        # define the model (i.e., ResNet-50)
        model = BaseCNN()
        model = torch.nn.DataParallel(model).cuda()
        # load the pre-trained model
        ckpt = '/home/sunwei/code/VQA/MinimalisticVQA/ckpts/DataParallel-00008.pt'
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint['state_dict'])
        resnet = nn.Sequential(*list(model.module.backbone.children())[:-2])
        resnet.head = Identity()

        # spatial quality analyzer
        self.feature_extraction = resnet
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # quality regressor
        self.quality = self.quality_regression(2048 + 256, 128, 1)

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

        return regression_block

    def forward(self, x, x_temporal_featurs):

        # x size: batch x frames x 3 x height x width
        x_size = x.shape

        # x_temporal_featurs size: batch x frames x 2048
        x_temporal_featurs_size = x_temporal_featurs.shape

        # x size: (batch * frames) x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        # x_temporal_featurs size: (batch * frames) x 256
        x_temporal_featurs = x_temporal_featurs.view(-1, x_temporal_featurs_size[2])
        
        
        
        x = self.feature_extraction(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        # x: (batch * frames) x (channel + 256)
        x = torch.cat((x, x_temporal_featurs), dim = 1)

        x = self.quality(x)
        
        # x: batch x frames
        x = x.view(x_size[0], x_size[1])

        # x: batch
        x = torch.mean(x, dim = 1)

            
        return x

# Model (VI): adding the temporal analyzer to Model (III) (load the weights when during the training)
class Model_VI(nn.Module):
    def __init__(self, pretrained = True):
        super(Model_VI, self).__init__()

        model = models.resnet50()

        # spatial quality analyzer
        self.feature_extraction = nn.Sequential(*list(model.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # quality regressor
        self.quality = self.quality_regression(2048+256, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x, x_temporal_featurs):

        # x size: batch x frames x 3 x height x width
        x_size = x.shape

        # x_temporal_featurs size: batch x frames x 2048
        x_temporal_featurs_size = x_temporal_featurs.shape

        # x size: (batch * frames) x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        # x_temporal_featurs size: (batch * frames) x 256
        x_temporal_featurs = x_temporal_featurs.view(-1, x_temporal_featurs_size[2])
        
        
        
        x = self.feature_extraction(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        # x: (batch * frames) x (channel + 256)
        x = torch.cat((x, x_temporal_featurs), dim = 1)

        x = self.quality(x)
        
        # x: batch x frames
        x = x.view(x_size[0], x_size[1])

        # x: batch
        x = torch.mean(x, dim = 1)
            
        return x
    


# Model (VII): Swin-B pre-trained on ImageNet-1K as the spatial quality analyzer 
# with no temporal quality analyzer (as the baseline model)
class Model_VII(nn.Module):
    def __init__(self):
        super(Model_VII, self).__init__()

        model = models.swin_b(weights='Swin_B_Weights.DEFAULT')
        model.head = Identity()

        # spatial quality analyzer
        self.feature_extraction = model

        # quality regressor
        self.quality = self.quality_regression(1024, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x):

        # x size: batch x frames x 3 x height x width
        x_size = x.shape
        
        # x size: (batch * frames) x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
            
        x = self.feature_extraction(x)

        # x size: (batch * frames) x channel
        x = torch.flatten(x, 1)

        # x size: (batch * frames)
        x = self.quality(x)
        
        # x size: batch x frames
        x = x.view(x_size[0], x_size[1])

        # x size: batch
        x = torch.mean(x, dim = 1)
            
        return x


# Model (VIII): pre-training Model (VII) on the training set of LSVQ (load the weights when during the training)
class Model_VIII(nn.Module):
    def __init__(self):
        super(Model_VIII, self).__init__()

        model = models.swin_b()
        model.head = Identity()

        # spatial quality analyzer
        self.feature_extraction = model


        # quality regressor
        self.quality = self.quality_regression(1024, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x):

        # x size: batch x frames x 3 x height x width
        x_size = x.shape
        
        # x size: (batch * frames) x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
            
        x = self.feature_extraction(x)


        # x size: (batch * frames) x channel
        x = torch.flatten(x, 1)

        # x size: (batch * frames)
        x = self.quality(x)
        
        # x size: batch x frames
        x = x.view(x_size[0], x_size[1])

        # x size: batch
        x = torch.mean(x, dim = 1)
            
        return x

# Model (IX): adding the temporal analyzer to Model (VII)
class Model_IX(nn.Module):
    def __init__(self):
        super(Model_IX, self).__init__()

        model = models.swin_b(weights='Swin_B_Weights.DEFAULT')
        model.head = Identity()

        # spatial quality analyzer
        self.feature_extraction = model


        # quality regressor
        self.quality = self.quality_regression(1024+256, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x, x_temporal_featurs):

        # x size: batch x frames x 3 x height x width
        x_size = x.shape

        # x_temporal_featurs size: batch x frames x 2048
        x_temporal_featurs_size = x_temporal_featurs.shape

        # x size: (batch * frames) x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        # x_temporal_featurs size: (batch * frames) x 256
        x_temporal_featurs = x_temporal_featurs.view(-1, x_temporal_featurs_size[2])
        
        
        
        x = self.feature_extraction(x)


        x = torch.flatten(x, 1)

        # x: (batch * frames) x (channel + 256)
        x = torch.cat((x, x_temporal_featurs), dim = 1)

        x = self.quality(x)
        
        # x: batch x frames
        x = x.view(x_size[0], x_size[1])

        # x: batch
        x = torch.mean(x, dim = 1)
            
        return x


# Model (X): adding the temporal analyzer to Model (VIII) (load the weights when during the training)
class Model_X(nn.Module):
    def __init__(self):
        super(Model_X, self).__init__()

        model = models.swin_b()
        model.head = Identity()

        # spatial quality analyzer
        self.feature_extraction = model

        # quality regressor
        self.quality = self.quality_regression(1024+256, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x, x_temporal_featurs):

        # x size: batch x frames x 3 x height x width
        x_size = x.shape

        # x_temporal_featurs size: batch x frames x 2048
        x_temporal_featurs_size = x_temporal_featurs.shape

        # x size: (batch * frames) x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        # x_temporal_featurs size: (batch * frames) x 256
        x_temporal_featurs = x_temporal_featurs.view(-1, x_temporal_featurs_size[2])
        
        
        
        x = self.feature_extraction(x)

        x = torch.flatten(x, 1)

        # x: (batch * frames) x (channel + 256)
        x = torch.cat((x, x_temporal_featurs), dim = 1)

        x = self.quality(x)
        
        # x: batch x frames
        x = x.view(x_size[0], x_size[1])

        # x: batch
        x = torch.mean(x, dim = 1)
            
        return x


class Model_Swinb_in22k_SlowFast(nn.Module):
    def __init__(self):
        super(Model_Swinb_in22k_SlowFast, self).__init__()

        model = timm.create_model('swin_base_patch4_window12_384_in22k', pretrained=True)
        model.head = Identity()

        # spatial quality analyzer
        self.feature_extraction = model
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # quality regressor
        self.quality = self.quality_regression(1024+256, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x, x_temporal_featurs):

        # x size: batch x frames x 3 x height x width
        x_size = x.shape

        # x_temporal_featurs size: batch x frames x 2048
        x_temporal_featurs_size = x_temporal_featurs.shape

        # x size: (batch * frames) x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        # x_temporal_featurs size: (batch * frames) x 256
        x_temporal_featurs = x_temporal_featurs.view(-1, x_temporal_featurs_size[2])

        x = self.feature_extraction(x)
        x = x.permute(0,3,1,2)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        # x: (batch * frames) x (channel + 256)
        x = torch.cat((x, x_temporal_featurs), dim = 1)

        x = self.quality(x)
        
        # x: batch x frames
        x = x.view(x_size[0], x_size[1])

        # x: batch
        x = torch.mean(x, dim = 1)
            
        return x

if __name__ == '__main__':
    import argparse

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model =  Model_II(pretrained = True)


    from thop import profile
    from thop import clever_format

    parser = argparse.ArgumentParser()
    config = parser.parse_args()

    config.backbone = 'resnet50'
    config.fc = False
    config.representation = 'NOTBCNN'
    config.std_modeling = 'True'

    print(config)

    model =  Model_V(config).to(device)
    # model = Model_X().to(device)

    input = torch.randn(8,8,3,384,384).to(device)
    input2 = torch.randn(8,8,256).to(device)

    flops, params = profile(model, inputs=(input,input2,))
    # flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")

    print(flops)
    print(params)