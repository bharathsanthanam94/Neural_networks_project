# Neural_networks_project

This repository contains CNN based implementation of a model that classifies scaled and de-scaled pipes.


### Download training data
___

The data is not open sourced, hence it is not direcltly available. Contact for the dataset link.

The code for generating data is given in "CNN_Descaling_new_feature.ipynb"
### Installations and dependencies
___
- Install dependencies mentioned in "install_dependencies.txt"
- Change the file paths given in config.py to the local directory where the above data is extracted.

### Training steps:
___
- The training, validation and test images locations has to be altered based on the downloaded directory in "config.py" file.
- Model parameters can be adjusted in "config.py" file
- Run "train.py" to train the model

### Testing steps
___
- The pretrained model is also stored in the above mentioned google drive link
- Make sure the test images and label locations are correct in the "config.py" file
- Donwload the pretrained model and save the downloaded directory in config.py in the variable "checkpoint_dir"
