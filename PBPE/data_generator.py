import numpy as np
from keras.utils import Sequence
import tensorflow as tf
import os
from visualizer import *
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler


def shuffle_along_axis(x, regs, axis):
    idx = np.random.rand(*x.shape[:2]).argsort(axis=axis)
    # print(idx.shape)
    idx = np.expand_dims(idx, axis=-1)
    idx = np.expand_dims(idx, axis=-1)
    return np.take_along_axis(x, idx, axis=axis), np.take_along_axis(regs, idx, axis=axis)


class DataGenerator(Sequence):

    def __init__(self, path, numPoints, numJoints, numRegions, steps=None, batch_size=32, shuffle=False, fill=5,
                 loadBatches=False, singleview=False, test=False, elevensubs=False, segnet=False, four_channels=False,
                 predicted_regs=False):
        """
            path - directory of input data
            numPoints - num of points in point cloud
            numJoints - num of joints in pose

        """
        self.batch_size = batch_size
        self.numPoints = numPoints
        self.numJoints = numJoints
        self.numRegions = numRegions
        self.path = path
        self.loadBatches = loadBatches
        self.singleview = singleview
        self.elevensubs = elevensubs
        self.segnet = segnet
        self.four_channels = four_channels
        self.predicted_regs = predicted_regs
        if predicted_regs and test:
            self.regions = np.load(self.path + 'predicted_regs.npy')
        if loadBatches:
            if singleview:
                self.list_IDs = [str(i + 1).zfill(fill) for i in
                                 range(len(os.listdir(path + 'scaledpclglobalSWbatches/')))]
            elif numJoints == 35:
                self.list_IDs = [str(i + 1).zfill(fill) for i in
                                 range(len(os.listdir(path + 'scaledpclglobal35jbatches/')))]
            elif elevensubs:
                self.list_IDs = [str(i + 1).zfill(fill) for i in
                                 range(len(os.listdir(path + 'scaledpclglobal_11subsbatches/')))]
            else:
                self.list_IDs = [str(i + 1).zfill(fill) for i in
                                 range(len(os.listdir(path + 'scaledpclglobalbatches/')))]
        else:
            if singleview:
                self.list_IDs = [str(i).zfill(fill) for i in
                                 range(len(os.listdir(path + 'scaledpclglobalSW/')))]
            elif elevensubs:
                self.list_IDs = [str(i + 1).zfill(fill) for i in
                                 range(len(os.listdir(path + 'scaledpclglobal_11subs/')))]
            else:
                # assert not singleview
                self.list_IDs = [str(i).zfill(fill) for i in
                                 range(len(os.listdir(path + 'scaledpclglobal/')))]
        self.shuffle = shuffle
        self.steps = steps
        self.indexes = None
        self.test = test
        self.fill = fill
        self.on_epoch_end()

    def __len__(self):
        """Take all batches in each iteration"""

        if self.steps is not None:
            return self.steps
        elif self.loadBatches:
            return len(self.list_IDs)
        else:
            return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Get next batch"""
        # Generate indexes of the batch
        if self.loadBatches:
            X, labels = self.__data_generation(self.list_IDs[self.indexes[index]])
        else:
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

        if self.loadBatches:
            if self.singleview:
                X = np.load(self.path + 'scaledpclglobalSWbatches/' + list_IDs_temp + '.npy')
                if not self.segnet:
                    y = np.load(self.path + 'posesglobalseparateSWbatches/' + list_IDs_temp + '.npy')
                if not self.test or self.four_channels:
                    if self.predicted_regs:
                        regs = np.load(self.path + 'regionSW_predicted_batches/' + list_IDs_temp + '.npy').reshape(
                            (self.batch_size, self.numPoints))
                    else:
                        regs = np.load(self.path + 'regionSWbatches/' + list_IDs_temp + '.npy').reshape(
                            (self.batch_size, self.numPoints))
            else:
                # X = np.load(self.path + 'scaledpclglobalbatches/' + list_IDs_temp + '.npy')

                if self.numJoints == 35:
                    X = np.load(self.path + 'scaledpclglobal35jbatches/' + list_IDs_temp + '.npy')
                    if not self.segnet:
                        y = np.load(self.path + 'posesglobalseparate35jbatches/' + list_IDs_temp + '.npy')
                    if not self.test or self.four_channels:
                        if self.predicted_regs:
                            regs = np.load(self.path + 'region35j_predicted_batches/' + list_IDs_temp + '.npy').reshape(
                                (self.batch_size, self.numPoints))
                        else:
                            regs = np.load(self.path + 'region35jbatches/' + list_IDs_temp + '.npy').reshape(
                                (self.batch_size, self.numPoints))
                elif self.elevensubs:
                    X = np.load(self.path + 'scaledpclglobal_11subsbatches/' + list_IDs_temp + '.npy')
                    if not self.segnet:
                        y = np.load(self.path + 'posesglobalseparate_11subsbatches/' + list_IDs_temp + '.npy')
                    if not self.test or self.four_channels:
                        if self.predicted_regs:
                            regs = np.load(self.path + 'region_11subs_predicted_batches/' + list_IDs_temp + '.npy').reshape(
                                (self.batch_size, self.numPoints))
                        else:
                            regs = np.load(self.path + 'region_11subsbatches/' + list_IDs_temp + '.npy').reshape(
                                (self.batch_size, self.numPoints))
                else:
                    X = np.load(self.path + 'scaledpclglobalbatches/' + list_IDs_temp + '.npy')
                    if not self.segnet:
                        y = np.load(self.path + 'posesglobalseparatebatches/' + list_IDs_temp + '.npy')
                    if not self.test or self.four_channels:
                        if self.predicted_regs:
                            regs = np.load(self.path + 'region_predicted_batches/' + list_IDs_temp + '.npy').reshape(
                                (self.batch_size, self.numPoints))
                        else:
                            regs = np.load(self.path + 'regionbatches/' + list_IDs_temp + '.npy').reshape(
                                (self.batch_size, self.numPoints))
            if self.four_channels:
                regs = np.expand_dims(np.expand_dims(regs, -1), -1)
                X = np.concatenate([X, regs], axis=-1)
            if not self.test or self.segnet:
                y_regions = np.eye(self.numRegions)[regs]
                y_regions = y_regions.reshape((y_regions.shape[0], self.numPoints, 1, self.numRegions))
                # X, y_regions = shuffle_along_axis(X, y_regions, axis=1)  # TODO shuffle pts in pcl
                if not self.segnet:
                    X, y, y_regions = shuffle(X, y, y_regions)
                else:
                    X, y_regions = shuffle(X, y_regions)
            else:
                X, y = shuffle(X, y)

            if self.segnet:
                return X, y_regions
            elif self.test or self.four_channels:
                return X, y
            else:
                return X, {'output1': y, 'output2': y_regions}
        else:  # no batches saved ==> valid or test set
            if self.numJoints == 35:
                name = '35j'
            elif self.elevensubs:
                name = '_11subs'
            elif self.singleview:
                name = 'SW'
            else:
                name = ''
            X = np.empty((self.batch_size, self.numPoints, 1, 3))
            if self.four_channels:
                Xexp = np.empty((self.batch_size, self.numPoints, 1, 4))
            if not self.segnet:
                y = np.empty((self.batch_size, self.numJoints * 3))
            if not self.four_channels and (not self.test or self.segnet):
                y_regions = np.zeros((self.batch_size, self.numPoints, 1, self.numRegions), dtype=int)

            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                X[i,] = np.reshape(np.load(self.path + 'scaledpclglobal' + name + '/' + ID + '.npy'),
                                   newshape=(self.numPoints, 1, 3))
                if not self.segnet:
                    # Store labels
                    p = np.load(self.path + 'posesglobalseparate' + name + '/' + ID + '.npy', allow_pickle=True)
                    y[i,] = np.reshape(p, newshape=(self.numJoints * 3))
                if self.predicted_regs:
                    regs = self.regions[int(ID)]
                else:
                    regs = np.asarray(
                        np.load(self.path + '/region' + name + '/' + ID + '.npy', allow_pickle=True).flatten(),
                        dtype=int)
                if self.four_channels:
                    if not self.predicted_regs:
                        regs = np.expand_dims(np.expand_dims(regs, -1), -1)
                    Xexp[i,] = np.concatenate([X[i,], regs], axis=-1)
                if not self.test or self.segnet:
                    regs = np.asarray(
                        np.load(self.path + '/region' + name + '/' + ID + '.npy', allow_pickle=True).flatten(),
                        dtype=int)
                    y_regions[i, np.arange(self.numPoints), 0, regs] = 1  # one-hot encoding

            if self.segnet:
                return X, y_regions
            elif self.four_channels:
                return Xexp, y
            elif self.test:
                return X, y
            else:
                return X, {'output1': y, 'output2': y_regions}
