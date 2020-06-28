from models.resnetbase import BasicBlock, ResNet

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)