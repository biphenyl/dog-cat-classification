import tensorflow as tf
import numpy as np
import os
import math
from PIL import Image, ImageOps

# class of CNN model and its method of establish and training
class ResNetModel():

    def __init__(self, log_dir, batch_size=64, image_size=(224,224,3), epoch=50, pre_trained_weights='imagenet'):
        self.batch_size = batch_size
        self.image_size = image_size
        self.epoch = epoch
        self.pre_trained_weights = pre_trained_weights
        self.log_dir = log_dir
        self.metrics = []

        self.create_model()


    def create_model(self):
        '''
        Create a resnet-50 model for two class classification
        '''
        
        input_layer = tf.keras.Input(self.image_size)

        base_model = tf.keras.applications.resnet50.ResNet50(weights=self.pre_trained_weights, 
                                                             input_shape=self.image_size, include_top=False)
        output_layer = base_model(input_layer, training=False)

        # add classification layers
        output_layer = tf.keras.layers.AveragePooling2D(pool_size=(7,7), name='top_avgpool')(output_layer)
        output_layer = tf.keras.layers.Flatten(name='top_flatten')(output_layer)
        output_layer = tf.keras.layers.Dense(2, activation='softmax', name='top_fc')(output_layer)
        
        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)


    def add_metrics(self, metrics):
        '''
        Add given metrics to be monitored during training
        '''
        if isinstance(metrics, list):
            self.metrics.extend(metrics)
        else:
            self.metrics.append(metrics)


    def compile_model(self):
        '''
        Define optimizer, loss and metrics and compiled the model for training.
        The parameters were fixed now.
        '''
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                        loss='categorical_crossentropy', metrics=self.metrics)


    def train_model(self, train_generator, valid_generator, record_save_path):
        '''
        Train the model by given training and validation generators, also record trained models to record_save_path.
        '''

        if not os.path.isdir(record_save_path):
            os.mkdir(record_save_path)

        # callbacks record training process and store the model with best accuracy
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=record_save_path, histogram_freq=0)
        best_checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(record_save_path, 'best_model.h5'), 
                                                    monitor='val_categorical_accuracy', verbose=0, save_best_only=True, 
                                                    save_weights_only=False, mode='max', save_freq='epoch')
        history = self.model.fit(x=train_generator, batch_size=self.batch_size, epochs=self.epoch,
            validation_data=valid_generator, shuffle=True,
            callbacks=[tensorboard_callback, best_checkpoint])

    def load_weights(self, model_weights_file_path):
         self.model.load_weights(model_weights_file_path)


class DogCatGenerator(tf.keras.utils.Sequence):
    '''
    A generator of the dataset, allow the model to load and preprocess images during training.
    '''
    def __init__(self, x_set, y_set, batch_size, img_size):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, index):
        batch_x_path = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x = [self.img_path_to_arr(path) for path in batch_x_path]
        batch_x = tf.keras.applications.densenet.preprocess_input(np.array(batch_x))

        batch_y = np.array(self.y[index * self.batch_size:(index + 1) * self.batch_size])

        return batch_x, batch_y
    

    def img_path_to_arr(self, path):
        '''
        load data from given path, preprocess it and cast it to an numpy array.
        '''
        img = tf.keras.preprocessing.image.load_img(path)
        img = pad_and_resize(img, (self.img_size))
        img_arr = tf.keras.preprocessing.image.img_to_array(img)

        return img_arr
    

def pad_and_resize(img: Image.Image, new_size):
    '''
    pad image into a square and resized it into given size 
    '''
    long_side = img.height if img.height>img.width else img.width
    img = ImageOps.pad(img, (long_side, long_side))
    img = img.resize((new_size[0], new_size[1]))

    return img