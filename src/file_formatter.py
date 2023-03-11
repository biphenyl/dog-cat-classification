import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

class FileFormatter():

    def __init__(self, img_dir):
        self.img_dir = img_dir

        # shuffle files orders to prevent cat or dog data stick together
        self.img_file_list = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir)]
        np.random.shuffle(self.img_file_list)

    def generate_label(self, label_type_dict):
        '''
        Generate label list by given label dictionary
        '''
        self.label_list = np.ones(len(self.img_file_list), dtype=np.int32) * -1
        for i in range(len(self.img_file_list)):
            for  k in label_type_dict:
                if k in self.img_file_list[i]:
                    self.label_list[i] = label_type_dict[k]
                    break


    def k_fold_split(self, n_splits, shuffle=True, random_state=42):
        '''
        Split data into different folds
        '''
        self.fold_index = np.zeros(len(self.img_file_list), dtype=np.int32)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for i, (train_index, test_index) in enumerate(skf.split(np.arange(len(self.img_file_list)), self.label_list)):
            self.fold_index[test_index] = i


    def output_csv(self, csv_path=''):
        '''
        Saved image path, label and fold to a csv file
        '''
        output_list = [[self.img_file_list[i], self.label_list[i], self.fold_index[i]] for i in range(len(self.img_file_list))]
        df = pd.DataFrame(output_list, columns=['image_name', 'label', 'fold'])
        df.to_csv(csv_path, index=False)


class FormattedDataLoader():
    def __init__(self, split_csv_save_location):
        '''
        Load image pathes, label and fold form csv file and perform one-hot encoding of the label.
        '''
        df = pd.read_csv(split_csv_save_location)

        self.image_path = np.array(df['image_name'])
        self.label = np.array(df['label'], dtype=np.int32)
        self.fold = np.array(df['fold'], dtype=np.int32)

        self.class_n = len(np.unique(self.label))

        self.one_hot_label = np.zeros((len(self.label), self.class_n), dtype=np.int32)
        for i in range(self.class_n):
            self.one_hot_label[self.label==i, i] = 1

    def get_fold_data(self, k):
        '''
        Return training and validation data/labels by given fold.
        '''

        train_img_path = self.image_path[self.fold!=k]
        train_label = self.one_hot_label[self.fold!=k]

        valid_img_path = self.image_path[self.fold==k]
        valid_label = self.one_hot_label[self.fold==k]

        return ((train_img_path, train_label), (valid_img_path, valid_label))
