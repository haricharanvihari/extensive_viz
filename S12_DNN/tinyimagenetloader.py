import torch
import numpy as np

from skimage import io
from torch.utils.data import Dataset
import pathlib
import random

class TinyImagenetLoader(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images, self.labels = zip(*data)
        self.__init()

    def __init(self):
        self.jpg_files = [f for f in pathlib.Path(self.root_dir).glob('**/*.jpg')
                          if '.ipynb_checkpoints' not in f.parts]
        self.class_to_idx = {clazz: i for i, clazz in enumerate(self.labels)}
        print(self.class_to_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.images[idx], as_gray=False, pilmode="RGB")

        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx] #label

def load_data(image_path):
    class_ids = [line.strip() for line in open(image_path + 'wnids.txt', 'r')]
    id_dict = {x:i for i, x in enumerate(class_ids)}
    all_classes = {line.split('\t')[0] : line.split('\t')[1].strip() for line in open(image_path + 'words.txt', 'r')}
    class_names = [all_classes[x] for x in class_ids]

    images = []
    labels = []

    # train data
    for value, key in enumerate(class_ids):
        images += [f'{image_path}train/{key}/images/{key}_{i}.JPEG' for i in range(500)]
        labels += [value for i in range(500)]

    # validation data
    for line in open( image_path + 'val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        images.append(f'{image_path}val/images/{img_name}')
        labels.append(id_dict[class_id])

    dataset = list(zip(images, labels))
    random.shuffle(dataset)

    return dataset, class_names