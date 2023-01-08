input_size = 3  # no of channels
num_classes = 2
num_epochs = 15
batch_size = 4
learning_Rate = 0.001

# specify paths for training
##choose data paths
# choose the path of "train_all_channels/all_channels from the extracted directory. "All channels should be final directory"
img_path = "./train_all_channels/all_channels"
##choose the path of train_label_10mm.json
label_path = "../train_label_10mm.json"


# Path for validation dataset
""
# choose the path of "val_all_channels/all_channels from the extracted directory. "all_channels" should be final directory
val_img_path = "../val_all_channels/all_channels"
##choose the path of val_label_10mm.json
val_label_path = "../val_label_10mm.json"


# location to save trained model
# choose the directory "trained_model"
checkpoint_dir = "/trained_model"
loss_plot_path = "loss_plots.PNG"
accuracy_plot_path = "accuracy_plots.PNG"
cm_path = "CM_train.PNG"


# specify path for testing
##Choose the path of "test_10mm_manual" from the extracted directory
test_img_path = "../test_10mm_manual"
# Location of test labels:
## Choose the path of test_label_10mm.json from the downloaded files
test_label_path = "../test_label_10mm.json"
model_location = "trained_model/classifier_model.pkl"
cm_path = "CM_test.PNG"
accu_path = "est_accuracies.txt"
#################################################################################################################################################
