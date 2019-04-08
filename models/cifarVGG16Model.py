import torch.nn as nn
from torchvision import models
# VGG16 for cifar10 and cifar100
class cifarVGG16Model(nn.Module):
    def __init__(self,pretrained=True,num_classes=100):
        super(cifarVGG16Model,self).__init__()
        model=models.vgg16(pretrained=pretrained)
        self.features=model.features

        self.classifer=nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifer(x)
        return x

def vgg16_cifar(pretrained=True, num_classes=100):
    model = cifarVGG16Model(pretrained,num_classes)
    return model