import numpy as np
from keras.utils import Sequence
import tensorflow as tf
import os
# from visualizer import *
from sklearn.utils import shuffle
from config import seq_length, numPoints


class DataGeneratorLSTM(Sequence):

    def __init__(self, path, numJoints, seq_idx, batch_size=32, pcls=True, encoder=False, decoder=False):
        """
            path - directory of input data
            numJoints - num of joints in pose
            seq_idx - start indices of all motion sequences
            pcls - whether to infer from point clouds (else: refine predicted 3D poses)

        """
        self.batch_size = batch_size
        self.numJoints = numJoints
        self.path = path
        self.seq_idx = seq_idx
        self.pcls = pcls
        self.encoder = encoder
        self.decoder = decoder
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
        if not self.decoder:
            if self.pcls:
                x = np.zeros((self.batch_size, seq_length, numPoints * 3))
            else:
                x = np.zeros((self.batch_size, seq_length, self.numJoints * 3), dtype=np.float64)
        if not self.encoder:
            dec_input = np.zeros((self.batch_size, seq_length, self.numJoints * 3), dtype=np.float64)
        y = np.zeros((self.batch_size, seq_length, self.numJoints * 3), dtype=np.float64)

        # dec_input[:, -1, :] = self.start_symbol # leave zeros as start symbol # TODO try ones (as in paper) for start symbol
        for bidx, i in enumerate(list_IDs_temp):  # 0:32; 32:64 ...
            if i not in self.seq_idx and (i + seq_length - 1) not in self.seq_idx and \
                    (i + seq_length - 1) < self.seq_idx[self.curr_seq]:
                if not self.decoder:
                    x[bidx] = self.data[i:i + seq_length]
                if not self.encoder:
                    dec_input[bidx, 0] = np.zeros((self.numJoints * 3))
                    dec_input[bidx, 1:] = self.gt_data[i:i + seq_length - 1].reshape((seq_length - 1, -1))
                y[bidx] = self.gt_data[i:i + seq_length].reshape((seq_length, -1))
            elif i in self.seq_idx and i != self.seq_idx[-1]:  # last id in seq_idx is last frame of the set
                if not self.decoder:
                    x[bidx, :] = self.data[i:i + seq_length]
                if not self.encoder:
                    dec_input[bidx, 0] = np.zeros((self.numJoints * 3))
                    dec_input[bidx, 1:] = self.gt_data[i:i + seq_length - 1].reshape((seq_length - 1, -1))
                y[bidx, :] = self.gt_data[i:i + seq_length].reshape((seq_length, -1))
                self.curr_seq += 1
            elif (i + seq_length - 1) >= self.seq_idx[self.curr_seq] > i:
                # end of current sequence
                if not self.decoder:
                    x[bidx, :self.seq_idx[self.curr_seq] - i] = self.data[i:self.seq_idx[self.curr_seq]]
                if not self.encoder:
                    dec_input[bidx, 1:self.seq_idx[self.curr_seq] - i] = self.gt_data[
                                                                         i:self.seq_idx[self.curr_seq] - 1].reshape(
                        (-1, self.numJoints * 3))
                y[bidx, :self.seq_idx[self.curr_seq] - i] = self.gt_data[i:self.seq_idx[self.curr_seq]].reshape(
                    (-1, self.numJoints * 3))
            else:
                eod = min(i + seq_length, len(self.gt_data))  # end of data
                if not self.decoder:
                    x[bidx, :eod - i] = self.data[i:eod]
                if not self.encoder:
                    dec_input[bidx, 1:eod - i] = self.gt_data[i:eod - 1].reshape((-1, self.numJoints * 3))
                y[bidx, :eod - i] = self.gt_data[i:eod].reshape((-1, self.numJoints * 3))

            # if np.all(x[bidx, 0]) == 0:  # pad start frames by repeating first pose
            #     first_pose_idx = (x[bidx] != 0).argmax(axis=0)[0]
            #     x[bidx, :first_pose_idx] = x[bidx, first_pose_idx]
        if self.encoder:
            return x, y
        elif self.decoder:
            return dec_input, y
        else:
            return [x, dec_input], y
