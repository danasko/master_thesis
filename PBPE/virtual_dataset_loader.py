import numpy as np
import os
import re
from preprocess import subsample
from config import numPoints, numJoints

path = 'G:/Datasets/Scans/'


def load_dataset(path):
    pcls_min = [1000000, 1000000, 1000000]
    pcls_max = [-1000000, -1000000, -1000000]
    no_scans = 4
    pcls = []
    poses = []

    # for f in os.listdir(path):
    for subdir, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.dat'):  # load pcls from .dat files
                with open(subdir + os.sep + f, 'rb') as file:
                    [no_scans, x_res, y_res] = np.fromfile(file, dtype=np.uint32, count=3)
                    data = np.fromfile(file, dtype=np.float32)
                    data = data.reshape((no_scans, x_res * y_res, 3))
                    for scan in data:
                        scan = subsample(scan, numPoints)
                        pcls.append(scan)

            elif f.endswith('.obj'):  # load poses from .obj files
                # TODO
                with open(subdir + os.sep + f, 'r') as file:
                    file.readline()
                    data = file.read()
                    data = data.replace('v ', '').strip()
                    data = re.split('[ \n]', data)
                    data = np.asarray(data, dtype=np.float32).reshape((no_scans, numJoints, 3))
                    for i in data:
                        poses.append(i)

    # TODO split test set - 80:20 random?


    # find min, max values in TRAINING scans
    pcls_min, pcls_max = [np.min(pcls, axis=(0, 1)), np.max(pcls, axis=(0, 1))]
    np.save('data/AMASS/train/pcls_minmax.npy', [pcls_min, pcls_max])
    assert len(pcls)==len(poses
                          )
    # zero centering
    for i in range(len(pcls)):
        poses[i] -= pcls[i].mean(axis=0)
        pcls[i] -= pcls[i].mean(axis=0)

    scale_dataset(pcls, poses, pcls_min, pcls_max)


def scale_dataset(pcls, poses, pcls_min, pcls_max):
    pcls = 2 * (pcls - pcls_min) / (pcls_max - pcls_min) - 1
    poses = 2 * (poses - pcls_min) / (pcls_max - pcls_min) - 1

    # TODO save in batches or individual .npy files


if __name__ == '__main__':
    load_dataset(path)
