import torch
from torch import nn

# import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np


class ClassifierCNN(nn.Module):
    def __init__(self, num_class):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same"),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding="same"),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )

        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )

        self.fc1 = nn.LazyLinear(128)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)
        # self.fc2 = nn.Linear(256, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=-1)
        return x


# test the the above model:
def test():

    x = torch.randn((8, 3, 48, 48))

    model = ClassifierCNN(2, 48)
    preds = model(x)
    # print("outputs:", preds)
    print(preds.shape)


if __name__ == "__main__":
    test()
