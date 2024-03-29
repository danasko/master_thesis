import numpy as np
from keras.utils import Sequence
import tensorflow as tf
import os
# from visualizer import *
from sklearn.utils import shuffle
from config import leaveout
from sklearn.preprocessing import MinMaxScaler


def shuffle_along_axis(x, regs, axis):
    idx = np.random.rand(*x.shape[:2]).argsort(axis=axis)
    # print(idx.shape)
    idx = np.expand_dims(idx, axis=-1)
    idx = np.expand_dims(idx, axis=-1)
    return np.take_along_axis(x, idx, axis=axis), np.take_along_axis(regs, idx, axis=axis)


class DataGenerator(Sequence):

    def __init__(self, path, numPoints, numJoints, numRegions, steps=None, batch_size=32, shuffle=False, fill=5,
                 loadbatch=False, singleview=False, test=False, elevensubs=False, sgpe_seg=False, four_channels=False,
                 predicted_regs=False, split=0):
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
        self.loadbatch = loadbatch
        self.singleview = singleview
        self.elevensubs = elevensubs
        self.sgpe_seg = sgpe_seg
        self.four_channels = four_channels
        self.predicted_regs = predicted_regs
        if predicted_regs and test:
            self.regions = np.load(self.path + 'predicted_regs' + (
                'SV' if singleview else '') + ('35j' if numJoints == 35 else '') + (
                                       '_11subs' + str(leaveout) if elevensubs else '') + '.npy')

        if singleview:
            if elevensubs:
                dir = 'scaledpclsSV_11subs' + str(leaveout)
            else:
                dir = 'scaledpclsSV'
        elif numJoints == 35:
            if elevensubs:
                dir = 'scaledpcls_11subs' + str(leaveout)
            elif loadbatch:
                dir = 'scaledpcls35j'
            else:
                dir = 'scaledpcls'
        elif elevensubs:
            dir = 'scaledpcls_11subs' + str(leaveout)
        else:
            dir = 'scaledpcls'

        if loadbatch:
            dir += 'batch'
            self.list_IDs = [str(i + 1).zfill(fill) for i in range(len(os.listdir(path + dir + '/')))]
        else:
            self.list_IDs = [str(i).zfill(fill) for i in range(len(os.listdir(path + dir + '/')))]
            self.list_IDs = self.list_IDs[:-(len(self.list_IDs) % self.batch_size)]

        self.split = split
        splitidx = ((len(self.list_IDs) // 2) // batch_size) * batch_size
        if split == 1:
            self.list_IDs = self.list_IDs[:splitidx]
        elif split == 2:
            self.list_IDs = self.list_IDs[splitidx:]

        self.shuffle = shuffle
        self.steps = steps
        self.indexes = None
        self.test = test
        self.fill = fill
        self.on_epoch_end()

    def __len__(self):
        """Take all batch in each iteration"""

        if self.steps is not None:
            return self.steps
        elif self.loadbatch:
            return len(self.list_IDs)
        else:
            return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Get next batch"""
        # Generate indexes of the batch
        if self.loadbatch:
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

        if self.loadbatch:
            if self.singleview:
                if self.elevensubs:
                    dir = 'SV_11subs' + str(leaveout)
                    if self.numJoints == 35:
                        posedir = 'SV35j_11subs' + str(leaveout)
                    else:
                        posedir = dir
                else:
                    dir = 'SV'
                    posedir = dir
                X = np.load(self.path + 'scaledpcls' + dir + 'batch/' + list_IDs_temp + '.npy')
                if not self.sgpe_seg:
                    y = np.load(self.path + 'scaledposes' + posedir + 'batch/' + list_IDs_temp + '.npy')
                if not self.test or self.four_channels:
                    if self.predicted_regs:
                        regs = np.load(
                            self.path + 'region' + posedir + '_predicted_batch/' + list_IDs_temp + '.npy').reshape(
                            (self.batch_size, self.numPoints))
                    else:
                        regs = np.load(self.path + 'region' + posedir + 'batch/' + list_IDs_temp + '.npy').reshape(
                            (self.batch_size, self.numPoints))
            elif self.numJoints == 35:
                if self.elevensubs:
                    dir = '_11subs' + str(leaveout)
                    pcldir = '_11subs' + str(leaveout)
                else:
                    dir = ''
                    pcldir = '35j'
                X = np.load(self.path + 'scaledpcls' + pcldir + 'batch/' + list_IDs_temp + '.npy')
                if not self.sgpe_seg:
                    y = np.load(self.path + 'scaledposes35j' + dir + 'batch/' + list_IDs_temp + '.npy')
                if not self.test or self.four_channels:
                    if self.predicted_regs:
                        regs = np.load(
                            self.path + 'region35j' + dir + '_predicted_batch/' + list_IDs_temp + '.npy').reshape(
                            (self.batch_size, self.numPoints))
                    else:
                        regs = np.load(self.path + 'region35j' + dir + 'batch/' + list_IDs_temp + '.npy').reshape(
                            (self.batch_size, self.numPoints))
            elif self.elevensubs:
                X = np.load(self.path + 'scaledpcls_11subs' + str(
                    leaveout) + 'batch/' + list_IDs_temp + '.npy')
                if not self.sgpe_seg:
                    y = np.load(self.path + 'scaledposes_11subs' + str(
                        leaveout) + 'batch/' + list_IDs_temp + '.npy')
                if not self.test or self.four_channels:
                    if self.predicted_regs:
                        regs = np.load(
                            self.path + 'region_11subs' + str(
                                leaveout) + '_predicted_batch/' + list_IDs_temp + '.npy').reshape(
                            (self.batch_size, self.numPoints))
                    else:
                        regs = np.load(self.path + 'region_11subs' + str(
                            leaveout) + 'batch/' + list_IDs_temp + '.npy').reshape(
                            (self.batch_size, self.numPoints))
            else:
                X = np.load(self.path + 'scaledpclsbatch/' + list_IDs_temp + '.npy')
                if not self.sgpe_seg:
                    y = np.load(self.path + 'scaledposesbatch/' + list_IDs_temp + '.npy')
                if not self.test or self.four_channels:
                    if self.predicted_regs:
                        regs = np.load(self.path + 'region_predicted_batch/' + list_IDs_temp + '.npy').reshape(
                            (self.batch_size, self.numPoints))
                    else:
                        regs = np.load(self.path + 'regionbatch/' + list_IDs_temp + '.npy').reshape(
                            (self.batch_size, self.numPoints))
            if self.four_channels:
                regs = np.expand_dims(np.expand_dims(regs, -1), -1)
                X = np.concatenate([X, regs], axis=-1)
            if not self.test or self.sgpe_seg:
                y_regions = np.eye(self.numRegions)[regs]
                y_regions = y_regions.reshape((y_regions.shape[0], self.numPoints, 1, self.numRegions))
                # X, y_regions = shuffle_along_axis(X, y_regions, axis=1)
                if not self.sgpe_seg:
                    X, y, y_regions = shuffle(X, y, y_regions)
                else:
                    X, y_regions = shuffle(X, y_regions)
            else:
                X, y = shuffle(X, y)

            if self.sgpe_seg:
                return X, y_regions
            elif self.test or self.four_channels:
                return X, y
            else:
                return X, {'output1': y, 'output2': y_regions}
        else:  # no batch saved ==> valid or test set
            if self.numJoints == 35:
                if self.elevensubs:
                    if self.singleview:
                        name = 'SV35j_11subs' + str(leaveout)
                        pclname = 'SV_11subs' + str(leaveout)
                    else:
                        name = '35j_11subs' + str(leaveout)
                        pclname = '_11subs' + str(leaveout)
                else:
                    name = '35j'
                    pclname = ''
            elif self.elevensubs:
                if self.singleview:
                    name = 'SV_11subs' + str(leaveout)
                    pclname = 'SV_11subs' + str(leaveout)
                else:
                    name = '_11subs' + str(leaveout)
                    pclname = '_11subs' + str(leaveout)
            elif self.singleview:
                name = 'SV'
                pclname = 'SV'
            else:
                name = ''
                pclname = ''
            X = np.empty((self.batch_size, self.numPoints, 1, 3))
            if self.four_channels:
                Xexp = np.empty((self.batch_size, self.numPoints, 1, 4))
            if not self.sgpe_seg:
                y = np.empty((self.batch_size, self.numJoints * 3))
            if not self.four_channels and (not self.test or self.sgpe_seg):
                y_regions = np.zeros((self.batch_size, self.numPoints, 1, self.numRegions), dtype=int)

            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                X[i,] = np.reshape(np.load(self.path + 'scaledpcls' + pclname + '/' + ID + '.npy'),
                                   newshape=(self.numPoints, 1, 3))
                if not self.sgpe_seg:
                    # Store labels
                    p = np.load(self.path + 'scaledposes' + name + '/' + ID + '.npy', allow_pickle=True)
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
                if not self.test or self.sgpe_seg:
                    regs = np.asarray(
                        np.load(self.path + '/region' + name + '/' + ID + '.npy', allow_pickle=True).flatten(),
                        dtype=int)
                    y_regions[i, np.arange(self.numPoints), 0, regs] = 1  # one-hot encoding

            if self.sgpe_seg:
                return X, y_regions
            elif self.four_channels:
                return Xexp, y
            elif self.test:
                return X, y
            else:
                return X, {'output1': y, 'output2': y_regions}
