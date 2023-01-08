import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
from tqdm import tqdm
from tabulate import tabulate

# import pytorch_lightning as pl
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from net1 import ClassifierCNN
from dataloaders import *
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
import seaborn as sn
import pandas as pd
import io
from cls_desc import *
from config import *

####################################################################################################################################################
test_data = SpecDataset(test_img_path, test_label_path, train=False)
test_loader = DataLoader(test_data, batch_size=4, num_workers=4)


eval_model = ClassifierCNN(2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# eval_model.load_state_dict(torch.load(model_location, map_location=device))
eval_model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
eval_model.to(device)
eval_model.eval()


# instantiate Confusion amtrix from torch metrics
cm = ConfusionMatrix(num_classes=2)
# Perform inference:

pred_labels = []
true_labels = []
correct = 0
with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)
        outputs = eval_model(images)

        # calculate accuracy:
        # accuracy
        max_index = outputs.max(dim=1)[1]
        correct += (max_index == labels).sum()

        pred_labels.append(max_index.cpu().numpy())
        true_labels.append(labels.cpu().numpy())

        # Plot confusion matrix and calculate per class accuracy
    pred_labels_all = np.concatenate(pred_labels)
    true_labels_all = np.concatenate(true_labels)

    cm.update(torch.tensor(pred_labels_all), torch.tensor(true_labels_all))
    conf_mat = cm.compute().detach().cpu().numpy().astype(np.int)
    print("type of confusion matrix:", type(conf_mat))
    print("shape of confusion matrix:", conf_mat.shape)
    df_cm = pd.DataFrame(conf_mat, index=np.arange(2), columns=np.arange(2))
    fig1 = plt.figure(figsize=(15, 15))
    sn.set(font_scale=1.1)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 5}, fmt="d")
    buf = io.BytesIO()
    plt.savefig(cm_path)

    # write class wise accuracy to a text file and save it
    table_all = []
    count = 0
    for i in conf_mat:
        table_cls = []
        table_cls.append(count)
        table_cls.append(cls_des[count])
        table_cls.append(sum(i))
        table_cls.append(i[count])
        table_cls.append((i[count] / sum(i)) * 100)
        count += 1
        table_all.append(table_cls)
    headers = [
        "class ID",
        "Class description",
        "total samples",
        "correct predictions",
        "Class accuracy(%)",
    ]
    with open(
        accu_path,
        "w",
    ) as outputfile:
        outputfile.write(
            tabulate(table_all, headers, tablefmt="grid", numalign="center")
        )
