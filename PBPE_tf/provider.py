import os
import sys
import numpy as np

# import h5py

dataset = 'MHAD'

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.dirname(BASE_DIR))
npy_data_dir = os.path.join('D:/PycharmProjects/PBPE/data/' + dataset + '/train')
if dataset == 'UBC':
    fill = 5
else:
    fill = 6


def loadDataFile_with_seg(dir, cur_train_filename):
    pcl = np.load(dir + '/scaledpclglobal/' + cur_train_filename)
    label = np.load(dir + '/posesglobalseparate/' + cur_train_filename)
    label = label.flatten()
    region = np.load(dir + '/region/' + cur_train_filename)
    region = region.flatten()

    return [pcl, label, region]


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def data_generation(indexes, batch_size, numPoints, numJoints, numRegions):
    """Generates data containing batch_size samples"""

    # list_IDs = [str(i).zfill(fill) for i in range(len(os.listdir(npy_data_dir + '/scaledpclglobal/')))]
    # indexes = np.arange(len(list_IDs))
    for i in range(len(indexes) // batch_size):
        batch = indexes[i * batch_size:(i + 1) * batch_size]
        # list_IDs_temp = [indexes[k] for k in batch]

        X = np.empty((batch_size, numPoints, 3))
        y = np.empty((batch_size, numJoints * 3))
        y_regions = np.zeros((batch_size, numPoints, numRegions), dtype=int)

        # Generate data
        for idx in batch:
            # Store sample
            X[i,] = np.load(npy_data_dir + '/scaledpclglobal/' + str(idx).zfill(fill) + '.npy')
            # Store labels
            p = np.load(npy_data_dir + '/posesglobalseparate/' + str(idx).zfill(fill) + '.npy')
            y[i,] = np.reshape(p, newshape=(numJoints * 3))

            regs = np.asarray(np.load(npy_data_dir + '/region/' + str(idx).zfill(fill) + '.npy').flatten(),
                              dtype=np.int32)
            y_regions[i, np.arange(numPoints), regs] = 1  # one-hot encoding

        yield X, y, y_regions
