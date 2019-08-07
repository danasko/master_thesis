import numpy as np
from keras.utils import Sequence
import os
from visualizer import *
from sklearn.preprocessing import MinMaxScaler


class DataGenerator(Sequence):

    def __init__(self, path, numPoints, numJoints, numRegions, steps=None, batch_size=32, shuffle=False):
        """Constructor can be expanded,
           with batch size, dimentation etc.
        """
        self.batch_size = batch_size
        self.numPoints = numPoints
        self.numJoints = numJoints
        self.numRegions = numRegions
        self.path = path
        self.list_IDs = [str(i).zfill(5) for i in range(len(os.listdir(path + 'scaledpclglobal/')))]
        self.shuffle = shuffle
        self.steps = steps
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        """Take all batches in each iteration"""

        if self.steps is not None:
            return self.steps
        else:
            return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Get next batch"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, labels = self.__data_generation(list_IDs_temp)

        return X, labels

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        X = np.empty((self.batch_size, self.numPoints, 1, 3))
        y = np.empty((self.batch_size, self.numJoints * 3))
        y_regions = np.zeros((self.batch_size, self.numPoints, 1, self.numRegions), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.reshape(np.load(self.path + 'scaledpclglobal/' + ID + '.npy'), newshape=(self.numPoints, 1, 3))
            # Store labels
            y[i,] = np.reshape(np.load(self.path + 'posesglobalseparate/' + ID + '.npy', allow_pickle=True),
                               newshape=(self.numJoints * 3))
            regs = np.asarray(np.load(self.path + '/region/' + ID + '.npy').flatten(), dtype=int)
            y_regions[i, np.arange(self.numPoints), 0, regs] = 1  # one-hot encoding
            # visualize_3D(X[i], regions=regs, pose=y[i], array=True, numJoints=self.numJoints)

        return X, {'output1': y, 'output2': y_regions}
