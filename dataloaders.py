import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
import os
import json
from PIL import Image
import random


class SpecDataset(Dataset):
    def __init__(self, img_path, label_path, train=False):
        self.img_path = img_path
        self.label_path = label_path
        self.images = []
        self.train = train
        # self.path_images = os.listdir(path)
        for subdir in os.listdir(self.img_path):
            if os.path.isdir(os.path.join(self.img_path, subdir)):
                for file in os.listdir(os.path.join(self.img_path, subdir)):
                    if file != "Thumbs.db":
                        self.images.append(file)

        self.length_images = len(self.images)
        if self.train:
            self.transform = transforms.Compose(
                [
                    # transforms.RandomAffine(
                    #     degrees=5,
                    #     # translate=(random.randint(-5, 5), random.randint(-5, 5)),
                    #     scale=(0.9, 1.1),
                    # ),
                    transforms.Resize([400, 400]),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.341, 0.312, 0.321), (0.275, 0.264, 0.270)),
                    # transforms.RandomRotation(5),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize([400, 400]),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.341, 0.312, 0.321), (0.275, 0.264, 0.270)),
                ]
            )

    def __len__(self):
        return self.length_images

    def __getitem__(self, index):

        with open(self.label_path, "r") as file:
            labels_dict = json.load(file)
        # print(self.images)
        # print(labels_dict)
        # print(index)
        # print(self.images[index])
        label = int(labels_dict[str(self.images[index])])

        # print(type(label))

        for subdir in os.listdir(self.img_path):
            if os.path.isdir(os.path.join(self.img_path, subdir)):
                for file in os.listdir(os.path.join(self.img_path, subdir)):
                    # print(file)
                    if file == self.images[index]:

                        image = Image.open(os.path.join(self.img_path, subdir, file))
                        image = self.transform(image)
                        # print(type(image))
                        # print(type(label))
                        return image, torch.tensor(label)
