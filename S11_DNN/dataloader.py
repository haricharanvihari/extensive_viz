import torch
import torchvision

from transformation import TransformationFactory

class ImageData(object):

  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
      'frog', 'horse', 'ship', 'truck')

  def __init__(self):
    super(ImageData, self).__init__()
    self.trainloader = None
    self.testloader = None

  def load(self, transformation_type="pytorch"):
   # Choose from "albumentations" or "pytorch". Default is "pytorch"
   t = TransformationFactory(transformation_type)
   train_transform = t.load(is_train=True)
   test_transform = t.load(is_train=False)

   SEED = 1

   # CUDA?
   cuda = torch.cuda.is_available()
   print("CUDA Available?", cuda)

   # For reproducibility
   torch.manual_seed(SEED)

   if cuda:
     torch.cuda.manual_seed(SEED)

   # dataloader arguments - something you'll fetch these from cmdprmt
   dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

   trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)

   testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)

   self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4)
   self.testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=4)