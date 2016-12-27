import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

import pandas as pd
from scipy.misc import imsave
import os

from keras import backend as K
K.set_image_dim_ordering('th')

class DigitClassifier:
    def __init__(self):
        print "DigitClassifier Keras Theano"
        # the data, shuffled and split between tran and test sets
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

        self.y_train = self.y_train.reshape(self.y_train.shape[0], 1)
        self.y_test = self.y_test.reshape(self.y_test.shape[0], 1)

        print("X_train, y_train original shape", self.X_train.shape, self.y_train.shape)
        print("X_test, y_test original shape", self.X_test.shape, self.y_test.shape)
        print(len(self.X_train), ' train sequences')
        print(len(self.X_test), ' test sequences')

        # Plotting one sample of each digit
        # for i in range(9):
        #     plt.subplot(3,3,i+1)
        #     plt.imshow(self.X_train[i], cmap='gray', interpolation='none')
        #     plt.title("Class {}".format(self.y_train[i]))

        # re-shape data: flatten 28*28 images to a 784 vector for each image
        self.n_pixels = self.X_train.shape[1] * self.X_train.shape[2]
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.n_pixels).astype('float32')
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.n_pixels).astype('float32')


        # Format the data for training: normalization pre-processing:
        # divide by max and substract mean
        self.X_train, self.y_train, self.X_test, self.y_test = self.normalize_data(self.X_train, self.y_train, self.X_test, self.y_test)
        # total pixels: 784
        self.input_dim = self.X_train.shape[1]
        self.Y_train, self.Y_test = self.labels_to_categorical_vars(self.y_train, self.y_test)

        print("Training matrix X, Y reshaped", self.X_train.shape, self.Y_train.shape)
        print("Testing matrix X, Y reshaped", self.X_test.shape, self.Y_test.shape)
        print("Input_dim, n_pixels: ", self.input_dim, self.n_pixels)

    def normalize_data(self, X_train, y_train, X_test, y_test):
        # pre-processing: divide by max and substract mean
        # normalize inputs from 0-255 to 0-1
        scale = np.max(X_train)
        X_train /= scale  # 255
        X_test /= scale

        mean = np.std(X_train)
        X_train -= mean
        X_test -= mean
        return X_train, y_train, X_test, y_test

    def labels_to_categorical_vars(self, y_train, y_test):
        # convert list of labels to binary class matrix via One hot encoding of output labels
        self.Y_train = np_utils.to_categorical(y_train)
        self.Y_test = np_utils.to_categorical(y_test)
        self.n_classes = y_test.shape[1] # 10
        print "labels_to_categorical_vars- classes: ", self.n_classes
        return self.Y_train, self.Y_test

    def download_imgs(self, train_file_input, test_file_input, train_folder_output, test_folder_output):
        data_train = pd.read_csv(train_file_input, header = 0) # header is first line of data (not of file)
        data_test = pd.read_csv(test_file_input, header = 0)
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
        #print len(os.listdir(test_folder_output))# ==0:
        img_index = 0
        n_images = len(data_test.values)
        print "test files' shape: ", len(data_test), n_images
        while img_index< n_images:
            test_img = data_test.values[img_index:img_index+1]
            im = test_img.reshape((28,28))
            imsave((test_folder_output+str(img_index)+'.png'),im)
            img_index +=1

    def compile_model(self):
        model = Sequential()
        model.add(Dense(512, input_shape=(self.n_pixels,1)))
        model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
                                      # of the layer above. Here, with a "rectified linear unit",
                                      # we clamp all values below 0 to 0.

        model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data

        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(self.n_classes))
        model.add(Activation('softmax')) # This special "softmax" activation among other things,
                                 # ensures the output is a valid probability distribution, that is
                                 # that its values are all non-negative and sum to 1.

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        return model

    def compile_DDMLP(self):
        # Deep Dumb MLP (DDMLP)
        model = Sequential()
        model.add(Dense(128, input_dim=self.input_dim))
        model.add(Activation('relu'))
        model.add(Dropout(0.15))

        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.15))

        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        # we'll use categorical xent for the loss, and RMSprop as the optimizer
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        return model

    def train_DDMLP(self):
        model = self.compile_DDMLP()
        model.fit(self.X_train, self.Y_train, nb_epoch=10,
                  batch_size=16,
                  validation_split=0.1,
                  show_accuracy=True, verbose=2)
        return model

    def train(self):# X_train=self.X_train, y_train=self.y_train, X_test=self.X_test, y_test=self.y_test):
        print "Training..."
        model = self.compile_model()
        model.fit(self.X_train, self.Y_train,
          batch_size=128, nb_epoch=4,
          verbose=2, # one line per epoch
          validation_data=(self.X_test, self.Y_test))
        return model

    def evaluate(self, model):#, X_test=self.X_test, y_test=self.y_test):
        """
        Returns the loss value and specified metric value in test mode
        """
        print "Predict: "
        score = model.evaluate(self.X_test, self.Y_test,
                       show_accuracy=True, verbose=0)
        print('Test score (loss):', score[0])
        print('Test accuracy:', score[1])
        print score
        return score

    def write_preds(self, preds, fname):
        pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

    def predict(self, model, output_dir='/output/'):#, X_test = self.X_test, y_test = self.y_test, output_dir=self.output_dir):
        # The predict_classes function outputs the highest probability class
        # according to the trained classifier for each input example.
        predicted_classes = model.predict_classes(self.X_test, verbose = 0)
        self.write_preds(predicted_classes, output_dir+"predicted.csv")
        # Check which items we got right / wrong
        correct_indices = np.nonzero(predicted_classes == self.y_test)[0]
        incorrect_indices = np.nonzero(predicted_classes != self.y_test)[0]

        plt.figure()
        if X_test.shape[0]==1:
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
        return predicted_classes, self.y_test

    def save_architecture_and_weights(self, model):
        #The generated JSON / YAML files are human-readable and can be manually edited if needed.
        # model = model_from_json(json_string)
        json_string = model.to_json()
        import json
        with open('/model/json_model.txt', 'w') as outfile:
            json.dumps(json_string, outfile) #json.dump(data, outfile) # data is a dict

        #return a representation of the model as a YAML string (does not include the weights, only the architecture).
        yaml_string = model.to_yaml()
        # saves the weights of the model as a HDF5 file.
        model.save_weights('DigitClassifier.h5')
        print "Json and Yaml strings for model: ",json_string, yaml_string

    def retrieve_architecture(self, json_string):
        from keras.models import model_from_json
        return model_from_json(json_string)

    def retrieve_architecture_yaml(self, yaml_string):
        # model reconstruction from YAML (with reinitialized weights) from the YAML string via
        from models import model_from_yaml
        return model_from_yaml(yaml_string)

    def retrieve_weights(self, model, filename):
        # Load the weights you saved into a model with the same architecture:
        return model.load_weights(filename, by_name=True)

if __name__ == '__main__':
    dc = DigitClassifier()
    #dc.download_imgs("data/train.csv", "data/test.csv", "./data/train/", "./data/test/")
    model = dc.train()
    #model = dc.train_DDMLP()

    #predicted = dc.predict(model)
    dc.evaluate(model)
    dc.save_architecture_and_weights(model)
