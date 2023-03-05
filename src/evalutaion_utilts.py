import tensorflow as tf
import numpy as np
from PIL import Image
import sklearn.metrics
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from model_training import pad_and_resize

class ModelValidator():
    '''
    A class perform evaluation and process evaluation result into demaned forms.
    '''
    def __init__(self, model):
        self.model = model
        self.predicted_scores_list = []
        self.true_labels_list = []


    def load_model_weights(self, model_weights_file_path):
        self.model.load_weights(model_weights_file_path)


    def perform_validation(self, valid_img_path, valid_label, batch_size=64, img_size=(224,224)):
        '''
        Perform evaluation by given image pathes and labels, the result would be append to self.predicted_scores_list.
        The labels will be appended to self.true_labels_list for easy comparison.
        '''

        predict_score = np.zeros(len(valid_img_path))
        true_label = np.argmax(valid_label, axis=1)

        # load images, perform preprocess and prediction
        batch_images = []
        cur_batch_start = 0
        for i in range(len(valid_img_path)):

            cur_img = Image.open(valid_img_path[i])
            cur_img = pad_and_resize(cur_img, img_size)
            cur_img = np.array(cur_img)
            batch_images.append(cur_img)
            
            # perform one prediction when a image in a batch reach batch_size or run out of images
            if (len(batch_images) >= batch_size) or (i==len(valid_img_path)-1):
                batch_images = tf.keras.applications.densenet.preprocess_input(np.array(batch_images))
                predict_score[cur_batch_start:i+1] = self.model.predict(batch_images)[:, 1]

                cur_batch_start += batch_size
                batch_images = []

        
        print(f'Validation complete for {len(valid_img_path)} images.')
        self.predicted_scores_list.append(predict_score)
        self.true_labels_list.append(true_label)


    def get_fold_metrics(self, k):
        '''
        Get metrics of certain fold.
        '''
        predict_score_k = self.predicted_scores_list[k]
        true_label_k = self.true_labels_list[k]

        return self.get_metrics_dict(predict_score_k, true_label_k)
    
    
    def get_combined_metrics(self):
        '''
        Get metrics of combined validation result
        '''
        combined_predict_score = np.concatenate(self.predicted_scores_list)
        combined_true_label = np.concatenate(self.true_labels_list)

        return self.get_metrics_dict(combined_predict_score, combined_true_label)


    def get_metrics_dict(self, predict_score, true_label):
        '''
        Get dictionary of needed metrics, the metrics were fixed now.
        '''
        predict_label = np.rint(predict_score)

        metrics_dict = {}
        metrics_dict['accuracy'] = sklearn.metrics.accuracy_score(true_label, predict_label)
        metrics_dict['dog_recall'] = sklearn.metrics.recall_score(true_label, predict_label)
        metrics_dict['dog_precision'] = sklearn.metrics.precision_score(true_label, predict_label)
        metrics_dict['cat_recall'] = sklearn.metrics.recall_score(true_label, predict_label, pos_label=0)
        metrics_dict['car_precision'] = sklearn.metrics.precision_score(true_label, predict_label, pos_label=0)
        metrics_dict['auroc'] = sklearn.metrics.roc_auc_score(true_label, predict_score)

        return metrics_dict
    

    def output_metrics_csv(self, csv_save_path):
        '''
        Write the metrics of each fold that performed validation and their combined result into a csv file.
        '''

        csv_cols = []
        csv_indices = []

        # add metrics dictionary of each fold into list 
        all_result_list = []
        for k in range(len(self.predicted_scores_list)):
            csv_indices.append(str(k))

            metrics_k = self.get_fold_metrics(k)
            metrics_k_list = [v for k, v in metrics_k.items()]

            all_result_list.append(metrics_k_list)

        # also add combined metrics into list
        csv_indices.append(str('combined'))

        metrics_combined = self.get_combined_metrics()
        metrics_combined_list = [v for k, v in metrics_combined.items()]

        all_result_list.append(metrics_combined_list)
        csv_cols = list(metrics_combined.keys())

        df = pd.DataFrame(all_result_list, columns=csv_cols, index=csv_indices)

        try:
            df.to_csv(csv_save_path)
            print('evaluation metrics written to', csv_save_path)
        except OSError:
            print('failed to write metrics to', csv_save_path)


    def draw_confusion_matrix(self, cm_image_save_path):
        '''
        Draw confusion matrix of each fold and their combined result, and save them into a single image.
        '''

        # setup canvas
        subplot_height = int(np.ceil((len(self.predicted_scores_list)+1)/5))
        plt.figure(figsize = (20, subplot_height*4))
        sn.set(font_scale=2.0)

        # draw confusion matrix of each fold
        for k in range(len(self.predicted_scores_list)):
            predict_label = np.rint(self.predicted_scores_list[k])
            cm = sklearn.metrics.confusion_matrix(self.true_labels_list[k], predict_label)

            plt.subplot(subplot_height, 5, k+1)
            sn.heatmap(cm, annot=True, fmt='d', cbar=False)
            plt.ylabel('true label')
            plt.xlabel('prediction')
            plt.title('fold ' + str(k))

        # draw confusion matrix of combined_result
        combined_predict_score = np.concatenate(self.predicted_scores_list)
        combined_predict_label = np.rint(combined_predict_score)
        combined_true_label = np.concatenate(self.true_labels_list)
        cm = sklearn.metrics.confusion_matrix(combined_true_label, combined_predict_label)

        plt.subplot(subplot_height, 5, len(self.predicted_scores_list)+1)
        sn.heatmap(cm, annot=True, fmt='d', cbar=False)
        plt.ylabel('true label')
        plt.xlabel('prediction')
        plt.title('combined')

        plt.tight_layout()

        try:
            plt.savefig(cm_image_save_path)
            print('confusion matrix image saved to', cm_image_save_path)
        except OSError:
            print('failed to save confusion matrix image at', cm_image_save_path)

        sn.reset_defaults()


    def plot_roc_curve(self, roc_plot_save_path):
        '''
        Plot ROC curve of each folds on one plot and the curve of combined result on another one, save them into a single image.
        '''

        plt.figure(figsize = (8, 16))
        plt.subplot(2,1,1)
        plt.title('ROC - cross validation')
        for k in range(len(self.predicted_scores_list)):
            fpr, tpr, threshold = sklearn.metrics.roc_curve(self.true_labels_list[k], self.predicted_scores_list[k])
            plt.plot(fpr, tpr, label='fold '+str(k))
            plt.legend()
            plt.grid()

        combined_predict_score = np.concatenate(self.predicted_scores_list)
        combined_true_label = np.concatenate(self.true_labels_list)
        fpr, tpr, threshold = sklearn.metrics.roc_curve(combined_true_label, combined_predict_score)
        
        plt.subplot(2,1,2)
        plt.title('ROC - combined result')
        plt.plot(fpr, tpr, label='combined')
        plt.grid()

        try:
            plt.savefig(roc_plot_save_path)
            print('roc curve image saved to', roc_plot_save_path)
        except OSError:
            print('failed to save roc curve image at', roc_plot_save_path)
    