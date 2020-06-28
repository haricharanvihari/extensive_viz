import torch.nn as nn
import torch.nn.functional as F

class ResnetNew_Cifar10DNN(nn.Module):
    def __init__(self):
        super(ResnetNew_Cifar10DNN, self).__init__()

        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.resnet_block_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.resnet_block_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer4_pool = nn.MaxPool2d(4, 1) 
 
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=512, out_features=10),
        )

    def forward(self, x):
        x1 = self.prep_layer(x)
        
        x2 = self.layer1(x1)
        x3 = x2 + self.resnet_block_layer1(x2)
        
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = x5 + self.resnet_block_layer3(x5)

        x7 = self.layer4_pool(x6)
     
        x8 = x7.view(-1, 512)
        x9 = self.fc_layer(x8)

        return F.log_softmax(x9, dim=-1)