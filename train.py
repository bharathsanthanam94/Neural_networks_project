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
from config import *
import io
from cls_desc import *

# from pytorch_lightning import Trainer

# input_size = 3
# # hidden_size = 20
# num_classes = 2
# num_epochs = 12
# batch_size = 4
# learning_Rate = 0.001


#################################################################################################################################################

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train_data = SpecDataset(img_path, label_path, train=True)
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=4
)
val_data = SpecDataset(val_img_path, val_label_path)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=8)

# instantiate the model:
torch.manual_seed(45)
model = ClassifierCNN(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    model.to(device)
    print("Models moved to GPU.")
else:
    print("Only CPU available.")


# define optimizers
weight_d = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_Rate, weight_decay=weight_d)
train_loss_epoch = []
val_loss_epoch = []
train_acc_epoch = []
val_acc_epoch = []
pred_labels_epoch = []
true_labels_epoch = []

# print moddel params
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# instantiate confusion matrix:
cm = ConfusionMatrix(num_classes=2)
# training loop
for epoch in tqdm(
    range(1, num_epochs + 1), desc="Epochs", total=num_epochs, disable=False
):
    # for epoch in range(num_epochs + 1):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Epoch:", epoch)

    step_train_loss = []
    step_val_loss = []

    model.train()
    train_batch = 0
    val_batch = 0
    correct = 0
    correct_val = 0
    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        # loss_criterion = nn.NLLLoss(weight=weights_tensor)
        loss_criterion = nn.NLLLoss()
        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        step_train_loss.append(loss.item())
        train_batch += 1

        # accuracy
        max_index = outputs.max(dim=1)[1]
        correct += (max_index == labels).sum()

    train_loss_epoch.append(np.mean(step_train_loss))
    train_acc_epoch.append((correct / len(train_loader.dataset)).cpu().numpy())

    # initialize local variables to zero:
    correct = 0

    model.eval()
    with torch.no_grad():
        for val_images, val_labels in val_loader:

            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)

            # val_images = val_images.to(device)
            # val_images = val_labels.to(device)

            val_outputs = model(val_images)

            # loss_criterion = nn.NLLLoss(weight=weights_tensor)
            loss_criterion = nn.NLLLoss()
            val_loss = loss_criterion(val_outputs, val_labels)
            step_val_loss.append(val_loss.item())
            val_batch += 1

            # accuracy
            max_index_val = val_outputs.max(dim=1)[1]
            correct_val += (max_index_val == val_labels).sum()

            if epoch == num_epochs:

                # For confusion matrix
                pred_labels_epoch.append(max_index_val.cpu().numpy())
                true_labels_epoch.append(val_labels.cpu().numpy())

    val_acc_epoch.append((correct_val / len(val_loader.dataset)).cpu().numpy())
    val_loss_epoch.append(np.mean(step_val_loss))

    if epoch == num_epochs:
        # join values of all batches to find confusion matrix
        pred_labels_all = np.concatenate(pred_labels_epoch)
        true_labels_all = np.concatenate(true_labels_epoch)
        pred_labels_epoch = []
        true_labels_epoch = []
        # cm.update(
        #     torch.from_numpy(true_labels_all).cuda(),
        #     torch.from_numpy(pred_labels_all).cuda(),
        # )
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
        with open("filename.txt", "w") as outputfile:
            outputfile.write(
                tabulate(table_all, headers, tablefmt="grid", numalign="center")
            )

        # save the model:
        checkpoint_dir = checkpoint_dir
        model_path = os.path.join(checkpoint_dir, "classifier_model.pkl")
        torch.save(model.state_dict(), model_path)
    # initialize local variables to zero:
    correct_val = 0

    # print("Training loss:", np.mean(step_train_loss))
    # print("validation loss:", np.mean(step_val_loss))
## plot train and validation epochs

print("-------------------------------------------")
# print("training losses all epochs:", train_loss_epoch)
# print("validation losses all epochs:", val_loss_epoch)
print("training accuracies all epochs:", train_acc_epoch)
print("validation accuracies all epochs", val_acc_epoch)
x = [i for i in range(num_epochs)]
# print("x axis values:", x)
fig1 = plt.figure(figsize=(15, 15))
plt.plot(x, train_loss_epoch)
plt.plot(x, val_loss_epoch)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["train_loss", "val_loss"])
plt.savefig(loss_plot_path)


fig2 = plt.figure(figsize=(15, 5))
plt.plot(x, train_acc_epoch)
plt.plot(x, val_acc_epoch)
plt.xlabel("epochs")
plt.ylabel("acc")
plt.legend(["train_acc", "val_acc"])
plt.savefig(accuracy_plot_path)
