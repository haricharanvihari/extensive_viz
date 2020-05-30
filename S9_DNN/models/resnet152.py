from models.resnetbase import BasicBlock, ResNet

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])