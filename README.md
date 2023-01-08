# Neural_networks_project

This repository contains CNN based implementation of a model that classifies scaled and de-scaled pipes.


### Download training data
___

To download the already processed data, follow the google drive link and extract in your local directory [google drive](https://drive.google.com/drive/folders/1dY1YWo-ZCTR9aCRMs4dcl-bn45RDva_Q?usp=sharing)

The code for generating data is given in "CNN_Descaling_new_feature.ipynb"
### Installations and dependencies
___
- Install dependencies mentioned in "dependencies.txt"
- Change the file paths given in config.py to the local directory where the above data is extracted.

### Training steps:
___
- The training, validation and test images locations has to be altered based on the downloaded directory in "config.py" file.
- Model parameters can be adjusted in "config.py" file
- Run "train.py" to train the model
