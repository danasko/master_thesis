import numpy as np
from keras.utils import Sequence
# import tensorflow as tf
import os
from sklearn.utils import shuffle
# from sklearn.preprocessing import MinMaxScaler
# from scipy.io import loadmat


class DataGenerator(Sequence):

    def __init__(self, path, input_size, numJoints, steps=None, batch_size=32, shuffle=False, mode='train'):
        """Constructor can be expanded,
           with batch size, dimentation etc.
        """
        self.batch_size = batch_size
        self.input_size = input_size
        self.numJoints = numJoints
        self.path = path
        self.list_IDs = [str(i+1).zfill(6) for i in range(len(os.listdir(path + 'depth/')))]
        self.shuffle = shuffle
        self.steps = steps
        self.indexes = None
        # poses_file =

        # self.labels = np.asarray([poses_file[i][0] for i in range(poses_file.shape[0])])
        self.on_epoch_end()

    def __len__(self):
        """Take all batches in each iteration"""

        if self.steps is not None:
            return self.steps
        else:
            return len(self.list_IDs)  # int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Get next batch"""
        # # Generate indexes of the batch
        # indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        #
        # # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #
        # # Generate data
        # X, labels = self.__data_generation(list_IDs_temp)
        X, labels = self.__data_generation(self.list_IDs[self.indexes[index]])

        return X, labels

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        # X = np.empty((self.batch_size, self.input_size, self.input_size, 1))
        # y = np.empty((self.batch_size, self.numJoints * 3))
        #
        # # Generate data
        # for i, ID in enumerate(list_IDs_temp):
        #     # Store sample
        #     X[i,] = np.reshape(np.load(self.path + 'scaledpclglobal/' + ID + '.npy'), newshape=(self.numPoints, 1, 3))
        #     # Store labels
        #     p = np.load(self.path + 'posesglobalseparate/' + ID + '.npy', allow_pickle=True)
        #     y[i,] = np.reshape(p, newshape=(self.numJoints * 3))

        X = np.load(self.path + 'depth/' + list_IDs_temp + '.npy')
        y = np.load(self.path + 'pose/' + list_IDs_temp + '.npy')

        X, y = shuffle(X, y)

        return X, y
