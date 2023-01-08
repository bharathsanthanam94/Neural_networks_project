input_size = 3  # no of channels
num_classes = 2
num_epochs = 8
batch_size = 4
learning_Rate = 0.001

# specify paths for training
##choose data paths
img_path = (
    "/home/bharath/Desktop/NN_project/Source/images_CNN_channels/train/all_channels"
)
label_path = "/home/bharath/Desktop/NN_project/Source/images_CNN_channels/label/10mm/train_label/train_label_10mm.json"
# label_path = "/home/bharath/Desktop/NN_project/Source/images_CNN_channnnels/label/all_labels_combined/train_all/train_5_10.json"


# Path for validation dataset
val_img_path = (
    "/home/bharath/Desktop/NN_project/Source/images_CNN_channels/val/all_channels"
)
val_label_path = "//home/bharath/Desktop/NN_project/Source/images_CNN_channels/label/10mm/val_label/val_label_10mm.json"
# val_label_path = "/home/bharath/Desktop/NN_project/Source/images_CNN_channels/label/all_labels_combined/val_all/val_5_10.json"


# location to save trained model
checkpoint_dir = (
    "/home/bharath/Desktop/NN_project/Neural_networks/results/pretrained_model"
)
loss_plot_path = (
    "/home/bharath/Desktop/NN_project/Neural_networks/results/loss_plots.PNG"
)
accuracy_plot_path = (
    "/home/bharath/Desktop/NN_project/Neural_networks/results/accuracy_plots.PNG"
)
cm_path = "/home/bharath/Desktop/NN_project/Neural_networks/results/CM_train.PNG"


# specify path for testing

# load test data loaders:
# Location of Test images:
test_img_path = "/home/bharath/Desktop/NN_project/Source/images_CNN_channels/test"
# Location of test labels:
test_label_path = "/home/bharath/Desktop/NN_project/Source/images_CNN_channels/label/10mm/test_label/test_label_10mm.json"
# test_label_path = "/home/bharath/Desktop/NN_project/Source/images_CNN_channels/label/all_labels_combined/test_all/test_5_10.json"
model_location = "/home/bharath/Desktop/NN_project/Neural_networks/results/pretrained_model/classifier_model.pkl"
cm_path = "/home/bharath/Desktop/NN_project/Neural_networks/results/CM_test.PNG"
accu_path = (
    "/home/bharath/Desktop/NN_project/Neural_networks/results/test_accuracies.txt"
)
#################################################################################################################################################
