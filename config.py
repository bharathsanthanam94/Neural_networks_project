import os


input_size = 3  # no of channels
num_classes = 2
num_epochs = 15
batch_size = 4
learning_Rate = 0.001

# specify paths for training
##choose data paths
# choose the path of "train_all_channels/all_channels from the extracted directory. "All channels should be final directory"
img_path = "/home/bharath/Desktop/NN_project/NN_dependency_trial/Neural_networks_project/Neural_networks_dataset-20230108T225542Z-001/Neural_networks_dataset/train_all_channels/all_channels"
##choose the path of train_label_10mm.json
label_path = "/home/bharath/Desktop/NN_project/NN_dependency_trial/Neural_networks_project/Neural_networks_dataset-20230108T225542Z-001/Neural_networks_dataset/train_label_10mm.json"


# Path for validation dataset
""
# choose the path of "val_all_channels" from the extracted directory. "val_all_channels" should be top directory
val_img_path = "/home/bharath/Desktop/NN_project/NN_dependency_trial/Neural_networks_project/Neural_networks_dataset-20230108T225542Z-001/Neural_networks_dataset/val_all_channels"
##choose the path of val_label_10mm.json
val_label_path = "/home/bharath/Desktop/NN_project/NN_dependency_trial/Neural_networks_project/Neural_networks_dataset-20230108T225542Z-001/Neural_networks_dataset/val_label_10mm.json"


# location to save trained model
# choose the directory "trained_model"
# specify any directory to store saved model
# To test the model, store the location of pretrained model "classifier_trained.pkl" available in the google drive
checkpoint_dir = (
    "/home/bharath/Desktop/NN_project/NN_dependency_trial/Neural_networks_project"
)
loss_plot_path = "loss_plots.PNG"
accuracy_plot_path = "accuracy_plots.PNG"
cm_path = "CM_train.PNG"


# specify path for testing
##Choose the path of "test_10mm_manual" from the extracted directory
test_img_path = "/home/bharath/Desktop/NN_project/NN_dependency_trial/Neural_networks_project/Neural_networks_dataset-20230108T225542Z-001/Neural_networks_dataset/test_10mm_manual"
# Location of test labels:
## Choose the path of test_label_10mm.json from the downloaded files
test_label_path = "/home/bharath/Desktop/NN_project/NN_dependency_trial/Neural_networks_project/Neural_networks_dataset-20230108T225542Z-001/Neural_networks_dataset/test_label_10mm.json"
model_location = os.path.join(checkpoint_dir, "classifier_model.pkl")
cm_path = "CM_test.PNG"
accu_path = "est_accuracies.txt"
#################################################################################################################################################
