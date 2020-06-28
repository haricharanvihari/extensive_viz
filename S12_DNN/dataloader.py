from os import path

import torch
import torchvision

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from transformation import TransformationFactory
from tinyimagenetloader import TinyImagenetLoader, load_data

class ImageData(object):

    classes = None

    def __init__(self):
        super(ImageData, self).__init__()
        self.trainloader = None
        self.testloader = None
        self.trainset = None
        self.testset = None
        self.cuda = None

    def load(self, transformation_type="pytorch"):
        # Choose from "albumentations" or "pytorch". Default is "pytorch"
        t = TransformationFactory(transformation_type)
        train_transform = t.load(is_train=True)
        test_transform = t.load(is_train=False)

        SEED = 1

        # CUDA?
        self.cuda = torch.cuda.is_available()
        print("CUDA Available?", self.cuda)

        # For reproducibility
        torch.manual_seed(SEED)

        if self.cuda:
            torch.cuda.manual_seed(SEED)

        return train_transform, test_transform

    def load_CIFAR(self, transformation_type="pytorch"):
        train_transform, test_transform = self.load(transformation_type)

        # dataloader arguments - something you'll fetch these from cmdprmt
        dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if self.cuda else dict(shuffle=True, batch_size=64)

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=test_transform)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=512, shuffle=True, num_workers=4)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=512, shuffle=False, num_workers=4)

    def load_TINY_IMAGENET(self, image_path, transformation_type="pytorch"):
        train_transform, test_transform = self.load(transformation_type)

        full_dataset, class_names = load_data(image_path)

        self.classes = class_names

        train_size = int(0.7 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        temp_train_dataset, temp_test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

        self.trainset = TinyImagenetLoader(temp_train_dataset, image_path, transform=train_transform)
        self.testset = TinyImagenetLoader(temp_test_dataset, image_path, transform=test_transform)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=256, shuffle=True, num_workers=4)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=256, shuffle=False, num_workers=4)

    def get_class_distribution_loaders(self, dataset_obj):
        idx2class = {v: k for k, v in dataset_obj.class_to_idx.items()}
        count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}

        for element in dataset_obj:
            y_lbl = element[1]
            y_lbl = idx2class[y_lbl]
            count_dict[y_lbl] += 1
            
        return count_dict

    def plot_class_distribution(self, dataset_path):
        natural_img_dataset = torchvision.datasets.ImageFolder(
                              root = dataset_path
                       )

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,7))
        sns.barplot(data = pd.DataFrame.from_dict([self.get_class_distribution_loaders(self.trainset)]).melt(), 
                    x = "variable", y="value", hue="variable",  ax=axes[0]).set_title('Train Set')
        sns.barplot(data = pd.DataFrame.from_dict([self.get_class_distribution_loaders(self.testset)]).melt(), 
                    x = "variable", y="value", hue="variable",  ax=axes[1]).set_title('Test Set')

        # del natural_img_dataset