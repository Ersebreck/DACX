import torch.nn as nn
import torchvision.models as models
import torch
from functools import partial


class Resnext50(nn.Module):

    def __init__(self, n_classes):

        super().__init__()

        resnet = models.resnext50_32x4d(pretrained=True)

        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes))

        self.base_model = resnet

        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))
        
class eff(nn.Module):

    def __init__(self, n_classes):

        super().__init__()
        
        eff = models.efficientnet_b0(pretrained=True)
        
        eff.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=eff.classifier[1].in_features, out_features=n_classes),
        )


        self.base_model = eff

        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))
        
class convext(nn.Module):

    def __init__(self, n_classes):

        super().__init__()
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        convext = models.convnext_base(pretrained=True)
        
        convext.classifier = nn.Sequential(
            norm_layer(convext.classifier[2].in_features), 
            nn.Flatten(1), 
            nn.Linear(in_features=convext.classifier[2].in_features, out_features=n_classes)
        )


        self.base_model = convext

        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))
        
class vit_b_16(nn.Module):

    def __init__(self, n_classes):

        super().__init__()
        
        vit = models.vit_b_16(pretrained= True)
        
        vit.heads = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=vit.heads[0].in_features, out_features=n_classes),
        )


        self.base_model = vit

        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))