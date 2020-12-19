import numpy as np
from keras.utils import Sequence
import tensorflow as tf
import os
# from visualizer import *
from sklearn.utils import shuffle
from config import seq_length, numPoints


class DataGeneratorSeq(Sequence):

    def __init__(self, path, numJoints, seq_idx, batch_size=32, pcls=False, gt_sequence=False):
        """
            path - directory of input data
            numJoints - num of joints in pose
            pcls - whether to infer from point clouds (else: refine predicted 3D poses)

        """
        self.batch_size = batch_size
        self.numJoints = numJoints
        self.path = path
        self.seq_idx = seq_idx
        self.pcls = pcls
        self.gt_sequence = gt_sequence
        if pcls:
            self.data = np.load(path + 'scaled_pcls_lzeromean_' + str(numPoints) + 'pts.npy',
                                allow_pickle=True)
        else:
            self.data = np.load(path + 'predictions.npy', allow_pickle=True)  # shape = (~88k, numJoints* 3)
        # rescale inputs and targets
        # self.data -= self.data.mean(axis=0)  # TODO try centering
        # [poses_min, poses_max] = np.load(path + 'poses_minmax.npy')
        # self.data = self.data.reshape((self.data.shape[0], numJoints, 3))
        # self.data = (self.data + 1) * (poses_max - poses_min) / 2 + poses_min
        # poses_min_new, poses_max_new = np.min(self.data, axis=(0, 1)), np.max(self.data, axis=(0, 1))
        # self.data = 2 * (self.data - poses_min_new) / (poses_max_new - poses_min_new) - 1
        # self.data = self.data.reshape((self.data.shape[0], numJoints * 3))

        self.gt_data = np.load(path + 'scaled_poses_lzeromean.npy', allow_pickle=True)  # shape = (~88k, numJoints, 3)
        # self.gt_data -= self.gt_data.mean(axis=0)  # TODO
        # self.gt_data = (self.gt_data + 1) * (poses_max - poses_min) / 2 + poses_min
        # self.gt_data = 2 * (self.gt_data - poses_min_new) / (poses_max_new - poses_min_new) - 1
        if pcls:
            self.data = self.data.reshape((self.data.shape[0], numPoints * 3))
        else:
            self.data = self.data.reshape((self.data.shape[0], numJoints * 3))
        self.on_epoch_end()

    def __len__(self):
        """Take all batch in each iteration"""

        # if self.steps is not None:
        #     return self.steps
        # else:
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """Get next batch"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, labels = self.__data_generation(indexes)

        return X, labels

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.data))
        self.curr_seq = 0

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        if self.pcls:
            x = np.zeros((self.batch_size, seq_length, numPoints * 3))
        else:
            x = np.zeros((self.batch_size, seq_length, self.numJoints * 3), dtype=np.float64)
        if self.gt_sequence:
            y = np.zeros((self.batch_size, seq_length, self.numJoints * 3), dtype=np.float64)
        else:
            y = np.zeros((self.batch_size, 1, self.numJoints * 3))

        for bidx, i in enumerate(list_IDs_temp):  # 0:32; 32:64 ...
            if i == 0:
                x[bidx, -1] = self.data[i]
                if self.gt_sequence:
                    y[bidx, -1] = self.gt_data[i].flatten()
            elif i in self.seq_idx and i != self.seq_idx[-1]:  # last id in seq_idx is last frame of the set
                x[bidx, -1] = self.data[i]
                if self.gt_sequence:
                    y[bidx, -1] = self.gt_data[i].flatten()
                self.curr_seq += 1
            elif i < seq_length:
                x[bidx, -i - 1:] = self.data[:i + 1]
                if self.gt_sequence:
                    y[bidx, -i - 1:] = self.gt_data[:i + 1].reshape((i + 1, -1))
            elif self.curr_seq > 0 and i < self.seq_idx[
                self.curr_seq - 1] + seq_length:  # beginning of a new sequence -> reset frames
                x[bidx, -(i - self.seq_idx[self.curr_seq - 1] + 1):] = self.data[self.seq_idx[self.curr_seq - 1]:i + 1]
                if self.gt_sequence:
                    y[bidx, -(i - self.seq_idx[self.curr_seq - 1] + 1):] = self.gt_data[self.seq_idx[
                                                                                            self.curr_seq - 1]:i + 1].reshape(
                        (-1, self.numJoints * 3))
            else:
                x[bidx] = self.data[i + 1 - seq_length:i + 1]
                if self.gt_sequence:
                    y[bidx] = self.gt_data[i + 1 - seq_length:i + 1].reshape((-1, self.numJoints * 3))
            if np.all(x[bidx, 0]) == 0:  # pad start frames by repeating first pose
                first_pose_idx = (x[bidx] != 0).argmax(axis=0)[0]
                x[bidx, :first_pose_idx] = x[bidx, first_pose_idx]

            if not self.gt_sequence:
                y[bidx, 0] = self.gt_data[i].flatten()
        return x, y
