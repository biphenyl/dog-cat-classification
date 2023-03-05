# dog-cat-classification

A ResNet-50 model to classify dog and cat images. Dataset came from https://www.kaggle.com/competitions/dogs-vs-cats/data.

# Environment

The codes was test on Ubuntu 20.04.3 LTS with Python 3.8.10. 
Module requirements were listed in requirements.txt.

# How to run

Unzip the dataset to current diretory, or elsewhere and change setting in config.ini
```
image_location = <path to image diretory>
```

Install required modules:
```
pip install -r requirements.txt
```

Trained the model:
```
python3 src/main_train.py
```

Evaluate training results and output validation results:
```
python3 src/main_evaluate.py
```

Modify **config.ini** to change the setting of training and validation.

# Explanations

The codes would perform a k-fold cross validation on the ResNet-50 model, then calculate the prediction metrics, plot roc curves and draw confusion matrixes.

### main_train.py
The script would parse the dataset to eastablish the labels of each images, split the data into k-fold and save the split into a csv file. 
The number of folds could be change by set **n_splits** in the confifuration.
If **keep_old_split** is set true in the configuration and an old split file existed, this action would not be perform. 

Then, it would establish ResNet-50 models and train them by the data split above, then save the trained models to **model_save_location**.

### main_evaluation.py
The code would load the trained models and the data splits to perform k-fold validation. Each image would be validated by the model of its fold. 
The script would also generate a combined result of every fold, and save the results of each folds and combined one.

The prediction metrics(accuracy, recall, precision and AUROC) would be saved as a csv file. The roc curves and confusion matrixes would be saved into images.

### file_formatter.py
Classes for establish labels of each image and prefrom k-fold split, and load data from a certain fold.

### model_training.py
A class contained a ResNet-50 model and methods to bulid and train the model. Also included a generator class for models to load and preprocess images during training process.

### evalutaion_utilts.py
A class perform evaluation and process evaluation result into demaned forms (metrics, roc curves and confusion matrixes).

### custom_metrics.py
Additional custom metrics for monitoring training process.They were only used in training process for now.