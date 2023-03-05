import tensorflow as tf
import configparser
import os

from model_training import ResNetModel
from file_formatter import FormattedDataLoader
from evalutaion_utilts import ModelValidator


if __name__ == '__main__':
    
    # load configuration
    config = configparser.ConfigParser()
    config.read('config.ini')

    model_save_location = config['FILE_PATHES']['model_save_location']
    split_csv_save_location = config['FILE_PATHES']['split_csv_save_location']
    result_save_location = config['FILE_PATHES']['result_save_location']
    if not os.path.isdir(result_save_location):
        os.mkdir(result_save_location)

    result_csv_file_name = config['FILE_PATHES']['result_csv_file_name']
    confusion_matrix_file_name = config['FILE_PATHES']['confusion_matrix_file_name']
    roc_curve_file_name = config['FILE_PATHES']['roc_curve_file_name']

    batch_size = int(config['TRAINING_PARAMETERS']['batch_size'])
    image_size = (int(config['IMAGE_INFO']['width']), int(config['IMAGE_INFO']['height']), int(config['IMAGE_INFO']['channel']))
    n_splits = int(config['K_FOLD_SPLIT_INFO']['n_splits'])
    
    # the model for evaluation
    model_obj = ResNetModel(model_save_location,)

    # use same data split as training process
    formatted_data_loader = FormattedDataLoader(split_csv_save_location)

    # plot training graph first

    model_validator = ModelValidator(model_obj.model)

    # evaluate each fold by corresponding trained model
    for k in range(n_splits):
        print('evaluating fold', k, '...')

        # load validation data from formatted csv
        ((_, _), (valid_img_path, valid_label)) = formatted_data_loader.get_fold_data(k)
        
        # load weights of the trained model of certain fold into model_validator
        model_weights_file_path = os.path.join(model_save_location, str(k), 'best_model.h5')
        model_validator.load_model_weights(model_weights_file_path)
        print('Successfully load weights from', model_weights_file_path)
 
        # perform prediction on each image and print the metrics
        model_validator.perform_validation(valid_img_path, valid_label, batch_size, image_size)
        metrics_dict = model_validator.get_fold_metrics(k)
        print(metrics_dict)

    # get prediction metrics of all validation combined
    combined_metrics_dict = model_validator.get_combined_metrics()
    print('Combined Result:')
    print(combined_metrics_dict)

    # write prediction metrics of all fold and combined result to a csv file
    model_validator.output_metrics_csv(os.path.join(result_save_location, result_csv_file_name))

    # draw confusion matrix and roc curve images
    model_validator.draw_confusion_matrix(os.path.join(result_save_location, confusion_matrix_file_name))
    model_validator.plot_roc_curve(os.path.join(result_save_location, roc_curve_file_name))