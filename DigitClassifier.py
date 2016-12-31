#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger
import pandas as pd
from scipy.misc import imsave
from scipy import misc
#from PIL import Image,ImageFilter

import os, errno

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# model serialization
import json
from pprint import pprint
from keras.models import model_from_json
from keras.models import model_from_yaml

def create_path_if_doesnt_exist(path_to_file):
    if not os.path.exists(os.path.dirname(path_to_file)):
        try:
            os.makedirs(os.path.dirname(path_to_file))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

class DigitClassifier:
    def __init__(self, model_name):
        print "DigitClassifier Keras Theano"
        self.architecture_yaml_string = ''
        self.model_json = ''
        self.weights_model = ''
        self.model_name = model_name
        self.model = None
        self.model_filename = 'models/'+self.model_name+'/'+self.model_name+'.h5'
        self.json_filename = 'models/'+self.model_name+'/'+self.model_name+'.json'

        # the data, shuffled and split between tran and test sets, (60000, 28, 28), (60000, 1) dimensions np.ndarray
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

        # if K.image_dim_ordering() == 'th':
        #     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        #     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        #     input_shape = (1, img_rows, img_cols)
        # else:
        #     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        #     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        #     input_shape = (img_rows, img_cols, 1)

        self.y_train = self.y_train.reshape(self.y_train.shape[0], 1)
        self.y_test = self.y_test.reshape(self.y_test.shape[0], 1)

        print("X_train, y_train original shape", self.X_train.shape, self.y_train.shape, type(self.X_train), type(self.y_train))
        print("X_test, y_test original shape", self.X_test.shape, self.y_test.shape)
        print(len(self.X_train), ' train sequences')
        print(len(self.X_test), ' test sequences')

        # Plotting one sample of each digit
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.imshow(self.X_train[i], cmap='gray', interpolation='none')
            plt.title("Class {}".format(self.y_train[i]))

        # re-shape data: flatten 28*28 images to a 784 vector for each image
        self.n_pixels = self.X_train.shape[1] * self.X_train.shape[2]
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.n_pixels).astype('float32')
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.n_pixels).astype('float32')


        # Format the data for training: normalization pre-processing:
        # divide by max and substract mean
        self.X_train = self.normalize(self.X_train)
        self.X_test = self.normalize(self.X_test)
        # total pixels: 784
        self.input_dim = self.X_train.shape[1]
        self.Y_train, self.Y_test = self.labels_to_categorical_vars(self.y_train, self.y_test)

        print("Training matrix X, Y reshaped", self.X_train.shape, self.Y_train.shape)
        print("Testing matrix X, Y reshaped", self.X_test.shape, self.Y_test.shape)
        print("Input_dim, n_pixels: ", self.input_dim, self.n_pixels)

    def normalize(self, x):
        scale = np.max(x)
        x /= scale  # 255

        mean = np.std(x)
        x -= mean
        return x

    def labels_to_categorical_vars(self, y_train, y_test):
        # convert list of labels to binary class matrix via One hot encoding of output labels
        self.Y_train = np_utils.to_categorical(y_train)
        self.Y_test = np_utils.to_categorical(y_test)
        self.n_classes = self.Y_test.shape[1] # 10
        print "labels_to_categorical_vars- classes: ", self.n_classes
        return self.Y_train, self.Y_test

    def download_imgs(self, train_file_input, test_file_input, train_folder_output, test_folder_output):
        data_train = pd.read_csv(train_file_input, header = 0) # header is first line of data (not of file)
        data_test = pd.read_csv(test_file_input, header = 0)
        create_path_if_doesnt_exist(train_folder_output)
        create_path_if_doesnt_exist(test_folder_output)

        # we resize each of the 784 rows into a 28*28 image
        print "train and test file shapes: ", data_train.shape, data_test.shape, " files in train and test folders: ",os.listdir(train_folder_output), " ",len(os.listdir(test_folder_output))
        print data_train.head(),'\n', data_test.head()
        img_index = 0
        n_images = len(data_train.values)
        print "train files' shape: ",data_train.shape,  n_images
        while img_index< n_images:
            train_img = data_train.values[img_index:img_index+1, 1:]  # first column is the label
            im = train_img.reshape((28,28))
            imsave((train_folder_output+str(img_index)+'.png'),im)
            img_index +=1
        img_index = 0
        n_images = len(data_test.values)
        print "test files' shape: ", len(data_test), n_images
        while img_index< n_images:
            test_img = data_test.values[img_index:img_index+1]
            im = test_img.reshape((28,28))
            imsave((test_folder_output+str(img_index)+'.png'),im)
            img_index +=1

    def compile_CNN(self):
        model = Sequential()
        model.add(Dense(input_dim=self.input_dim, output_dim=512)) #512, input_shape=(self.n_pixels,1)))
        model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
        # of the layer above. Here, with a "rectified linear unit", we clamp all values below 0 to 0.
        model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data

        model.add(Dense(input_dim=512, output_dim=512)) #512
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(self.n_classes))
        model.add(Activation('softmax')) # This special "softmax" activation among other things,
                                 # ensures the output is a valid probability distribution, that is
                                 # that its values are all non-negative and sum to 1.
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        return model
        # self.batch_size = 128
        # self.n_epoch = 12
        #
        # # input image dimensions
        # img_rows, img_cols = 28, 28
        # # number of convolutional filters to use
        # n_filters = 32
        # # size of pooling area for max pooling
        # pool_size = (2, 2)
        # # convolution kernel size
        # kernel_size = (3, 3)
        # model = Sequential()
        #
        # model.add(Convolution2D(n_filters, kernel_size[0], kernel_size[1],
        #                         border_mode='valid',
        #                         input_shape=input_shape))
        # model.add(Activation('relu'))
        # model.add(Convolution2D(n_filters, kernel_size[0], kernel_size[1]))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=pool_size))
        # model.add(Dropout(0.25))
        #
        # model.add(Flatten())
        # model.add(Dense(128))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(self.n_classes))
        # model.add(Activation('softmax'))
        #
        # model.compile(loss='categorical_crossentropy',
        #               optimizer='adadelta',
        #               metrics=['accuracy'])

        #model.fit(X_train, Y_train, batch_size=self.batch_size, self.nb_epoch= self.n_epoch,
                  #verbose=1, validation_data=(X_test, Y_test))


    def compile_MLP(self):
        model = Sequential()
        model.add(Dense(input_dim=self.input_dim, output_dim=512)) #512, input_shape=(self.n_pixels,1)))
        model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
        # of the layer above. Here, with a "rectified linear unit", we clamp all values below 0 to 0.
        model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data

        model.add(Dense(input_dim=512, output_dim=512)) #512
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(self.n_classes))
        model.add(Activation('softmax')) # This special "softmax" activation among other things,
                                 # ensures the output is a valid probability distribution, that is
                                 # that its values are all non-negative and sum to 1.
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        return model

    def compile_MLP2(self):
        # MLP
        model = Sequential()
        model.add(Dense(input_dim=self.input_dim, output_dim=128))
        model.add(Activation('relu'))
        model.add(Dropout(0.15))

        model.add(Dense(input_dim=128, output_dim=128))
        model.add(Activation('relu'))
        model.add(Dropout(0.15))

        model.add(Dense(self.n_classes))
        model.add(Activation('softmax'))

        # we'll use categorical xent for the loss, and RMSprop as the optimizer
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
        return model

    def train(self, model_name):
        self.model_name = model_name
        self.model_filename = 'models/'+self.model_name+'/'+self.model_name+'.h5'
        self.json_filename = 'models/'+self.model_name+'/'+self.model_name+'.json'
        #self.weights_filename = 'models/'+self.model_name+'/DigitClassifierWeights.h5'
        print "Compiling model..."
        if model_name =='MLP':
            model = self.compile_MLP()
            print "Fitting model..."
            model.fit(self.X_train, self.Y_train,
                      nb_epoch=10, batch_size=16,
                      validation_split=0.1, verbose=2)
            return model
        elif model_name== 'CNN':
            model = self.compile_CNN()
            print "Fitting model..."
            model.fit(self.X_train, self.Y_train,
              batch_size=128, nb_epoch=4,
              verbose=2, # one line per epoch
              validation_data=(self.X_test, self.Y_test))
        else:
            print "Invalid model name: use MLP or CNN"
            sys.exit(-1)
        self.save_model(model)
        return model

    def evaluate(self, model):#, X_test=self.X_test, y_test=self.y_test):
        """
        Returns the loss value and specified metric value on test data
        """
        print "Evaluating... "
        score = model.evaluate(self.X_test, self.Y_test, verbose=0)
        print "%s: %.2f%%" % (model.metrics_names[1], score[1]*100)
        print('Test score (loss):', score[0])
        print('Test accuracy:', score[1])
        print score
        return score

    def write_predictions(self, preds):
        filename='data/predicted/'+self.model_name+'/predicted.csv'
        create_path_if_doesnt_exist(filename)
        pd.DataFrame({"ImageID": list(range(1,len(preds)+1)), "Label": preds}).to_csv(filename, index=False, header=True)

    def write_prediction(self, image_filename, preds):
        filename='data/predicted_imgs/'+self.model_name+'/predicted_imgs.csv'
        create_path_if_doesnt_exist(filename)
        pd.DataFrame({"Image_filename": image_filename, "Label": preds}).to_csv(filename, index=False, header=True)

    def predict_image(self, img_file, model):
        im = misc.imread(img_file) # np.ndarray
        x = im.reshape((1, self.n_pixels)).astype('float32') # np.ndarray
        x = self.normalize(x)

        predicted_class = model.predict_classes(x, verbose=0)[0]
        print "predict_instance for image ",img_file, " (input is ",type(x), x.shape,") Prediction: ", predicted_class

        #self.write_prediction(img_file, predicted_class)
        return predicted_class

    def predict(self, model, output_dir='output/'):#, X_test = self.X_test, y_test = self.y_test, output_dir=self.output_dir):
        # The predict_classes function outputs the highest probability class
        # according to the trained classifier for each input example.
        print "Predicting... (input: ",type(self.X_test), self.X_test.shape,")\n"#, self.X_test[0]
        predicted_classes = model.predict_classes(self.X_test, verbose = 0)
        self.write_predictions(predicted_classes)
        # Check which items we got right / wrong
        correct_indices = np.nonzero(predicted_classes == self.y_test)[0]
        incorrect_indices = np.nonzero(predicted_classes != self.y_test)[0]

        plt.figure()
        if self.X_test.shape[0]==1:
             plt.subplot(1,1,1)
             plt.imshow(self.X_test[0].reshape(28,28), cmap='gray', interpolation='none')
             plt.title("Predicted {}, Class {}".format(predicted_classes[0], self.y_test[0]))
        else:
             for i, correct in enumerate(correct_indices[:9]):
                 plt.subplot(3,3,i+1)
                 plt.imshow(self.X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
                 plt.title("Predicted {}, Class {}".format(predicted_classes[correct], self.y_test[correct]))
             for i, incorrect in enumerate(incorrect_indices[:9]):
                 plt.subplot(3,3,i+1)
                 plt.imshow(self.X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
                 plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], self.y_test[incorrect]))
        print "Predicted labels: ",  predicted_classes
        return predicted_classes

    def save_model(self, model):
        # serialize model to JSON
        model_json = model.to_json()
        #create_path_if_doesnt_exist(self.json_filename)
        with open(self.json_filename,  "w") as json_file:
            pprint(model_json)
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(self.model_filename)
        print("Saving model and weights to disk in files: ", self.json_filename, " ",self.model_filename)

    def load_model(self):
        # load json and create model
        print "loading model from file: ",self.json_filename
        with open(self.json_filename, 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(self.model_filename)
        print("Loaded model json and instance from disk ")#, loaded_model_json,"\n", loaded_model)
        return loaded_model


if __name__ == '__main__':
    dc = DigitClassifier('CNN')
    #dc.download_imgs("data/train.csv", "data/test.csv", "./data/train/", "./data/test/")

    # Model 1: CNN: train and compile
    model = dc.train('CNN')

    dc.evaluate(model)
    predicted = dc.predict(model)

    # Load trained model
    loaded_model = dc.load_model()
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # evaluate loaded model on test data
    predicted_label = dc.predict_image('data/test/3808.png', loaded_model)

    # Model 2 MLP
    # model = dc.train('MLP')
    #
    # dc.evaluate(model)
