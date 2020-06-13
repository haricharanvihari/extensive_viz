from models.resnetbase import BasicBlock, ResNet

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])