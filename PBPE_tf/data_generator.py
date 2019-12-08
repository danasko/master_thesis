import numpy as np
import tensorflow as tf
import os
# from visualizer import *


# class DataGenerator():

# def __init__(self, path, numPoints, numJoints, numRegions, steps=None, batch_size=32, shuffle=False, fill=5):
#     """Constructor can be expanded,
#        with batch size, dimentation etc.
#     """
#     self.batch_size = batch_size
#     self.numPoints = numPoints
#     self.numJoints = numJoints
#     self.numRegions = numRegions
#     self.path = path
#     self.list_IDs = [str(i).zfill(fill) for i in range(len(os.listdir(path + 'scaledpclglobal/')))]
#     self.shuffle = shuffle
#     self.steps = steps
#     self.indexes = None
#     self.fill = fill
#     self.on_epoch_end()

# def __len__(self):
#     """Take all batches in each iteration"""
#
#     if self.steps is not None:
#         return self.steps
#     else:
#         return int(np.floor(len(self.list_IDs) / self.batch_size))

# def __getitem__(self, index):
#     """Get next batch"""
#     # Generate indexes of the batch
#     indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
#
#     # Find list of IDs
#     list_IDs_temp = [self.list_IDs[k] for k in indexes]
#
#     # Generate data
#     X, labels = self.__data_generation(list_IDs_temp)
#
#     return X, labels


