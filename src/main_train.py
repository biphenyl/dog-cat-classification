import tensorflow as tf
import numpy as np
import configparser
import os

from file_formatter import FileFormatter, FormattedDataLoader
from model_training import ResNetModel, DogCatGenerator
import custom_metrics


if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('config.ini')

    # load dataset info
    image_location = config['FILE_PATHES']['image_location']
    label_dict = {'cat':0, 'dog':1}
    n_splits = int(config['K_FOLD_SPLIT_INFO']['n_splits'])
    shuffile = config['K_FOLD_SPLIT_INFO'].getboolean('shuffle')
    random_state = int(config['K_FOLD_SPLIT_INFO']['random_state'])
    split_csv_save_location = config['FILE_PATHES']['split_csv_save_location']
    keep_old_split = config.getboolean(config['FILE_PATHES']['keep_old_split'])

    # format dataset and split data for k-fold validation or use old split
    if (not os.path.isfile(split_csv_save_location)) or (not keep_old_split):
        my_file_formatter = FileFormatter(image_location)
        my_file_formatter.generate_label(label_dict)
        my_file_formatter.k_fold_split(n_splits=n_splits, shuffle=shuffile, random_state=random_state)
        my_file_formatter.output_csv(split_csv_save_location)
        print('dataset info created.')
    else:
        print('split csv existed and use_old_split==True, not creating new dataset info.')

    # load metadata from built csv
    formatted_data_loader = FormattedDataLoader(split_csv_save_location)

    # load training parameters
    log_dir = config['FILE_PATHES']['model_save_location']
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    batch_size = int(config['TRAINING_PARAMETERS']['batch_size'])
    pre_trained_weights = config['TRAINING_PARAMETERS']['pre_trained_weights']
    image_size = (int(config['IMAGE_INFO']['width']), int(config['IMAGE_INFO']['height']), int(config['IMAGE_INFO']['channel']))
    epoch = int(config['TRAINING_PARAMETERS']['epoch'])

    # define metrics for evaluation
    accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

    # k-fold training 
    for k in range(n_splits):

        # establish model
        cur_model = ResNetModel(log_dir, batch_size, image_size, epoch, pre_trained_weights)
        cur_model.compile_model()

        # get the training data and validation data of this fold
        ((train_img_path, train_label), (valid_img_path, valid_label)) = formatted_data_loader.get_fold_data(k)

        # create generators for the model to load and preprocess images during training
        train_generator = DogCatGenerator(train_img_path, train_label, batch_size, image_size)
        valid_generator = DogCatGenerator(valid_img_path, valid_label, batch_size, image_size)

        record_save_path = os.path.join(log_dir, str(k))
        if not os.path.isdir(record_save_path):
            os.mkdir(record_save_path)

        # set metrics to be evaluated during training
        cur_model.add_metrics([accuracy_metric, custom_metrics.tp, custom_metrics.tn, custom_metrics.fp, custom_metrics.fn])
        
        # train and record model after training completed
        cur_model.train_model(train_generator, valid_generator, record_save_path)
        cur_model.model.save(os.path.join((cur_model.log_dir), ('model' +  str(k) + '.h5')))




