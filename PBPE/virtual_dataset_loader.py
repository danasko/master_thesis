import numpy as np
import os
import re
from preprocess import subsample, remove_floor, automatic_annotation
from config import numPoints, numJoints, singleview, temp_convs, batch_size
from visualizer import visualize_3D, visualize_3D_pose
from sklearn.utils import shuffle

path = 'G:/Datasets/Scans/'


def load_dataset(path):
    pcls_min = [1000000, 1000000, 1000000]
    pcls_max = [-1000000, -1000000, -1000000]
    no_scans = 4
    pcls = []
    poses = []
    seq_starts = []
    curr_seq = -1

    # for subdir, dirs, files in os.walk(path):
    #     print(subdir)
    #         for f in files:
    #             # print(f)
    #             if f.endswith('.dat'):  # load pcls from .dat files
    #                 seq_num = f[3:f.find('-')]
    #                 if seq_num != curr_seq:
    #                     curr_seq = seq_num
    #                     seq_starts.append(len(pcls))  # mark start of a new sequence
    #                 with open(subdir + os.sep + f, 'rb') as file:
    #                     [no_scans, x_res, y_res] = np.fromfile(file, dtype=np.uint32, count=3)
    #                     data = np.fromfile(file, dtype=np.float32)
    #
    #                     if singleview:
    #                         data = data.reshape((no_scans, x_res * y_res, 3))
    #                         scan[:, :, 1:3] = scan[:, :, 2:0:-1]  # flip y,z axes (Y to be vertical axis)
    #                         for scan in data:
    #                             scan = remove_floor(scan)
    #                             scan = subsample(scan, numPoints)
    #                             # visualize_3D(scan)
    #                             pcls.append(scan)
    #                     else:  # multiview - merge into unified pcl
    #                         scan = data.reshape((no_scans * x_res * y_res, 3))
    #                         scan[:, 1:3] = scan[:, 2:0:-1]  # flip y,z axes (Y to be vertical axis)
    #                         scan = remove_floor(scan)
    #                         scan = subsample(scan, numPoints)
    #                         # visualize_3D(scan, pause=False)
    #                         pcls.append(scan)
    #
    #             elif f.endswith('.obj'):  # load poses from .obj files
    #                 with open(subdir + os.sep + f, 'r') as file:
    #                     file.readline()
    #                     data = file.read()
    #                     data = data.replace('v ', '').strip()
    #                     data = re.split('[ \n]', data)
    #                     data = np.asarray(data, dtype=np.float32).reshape((len(data) // 3, 3))
    #                     # crop out unnecessary finger joints
    #                     data = data[:22]
    #                     data[:, 1:3] = data[:, 2:0:-1]  # flip y,z axes (Y to be vertical axis)
    #                     # visualize_3D_pose(data)
    #                     if singleview:
    #                         for i in range(no_scans):
    #                             poses.append(data)
    #                     else:
    #                         poses.append(data)
    #
    #         np.save('data/AMASS/pcls_' + str(idx).zfill(2) + '.npy', pcls)
    #         np.save('data/AMASS/poses_' + str(idx).zfill(2) + '.npy', poses)
    #
    # np.save('data/AMASS/seq_idx.npy', seq_starts)
    # print(len(poses), len(pcls))
    pcls = np.load('data/AMASS/pcls.npy')
    poses = np.load('data/AMASS/poses.npy')
    print(pcls.shape, poses.shape)

    # TODO split test set - 80:20 random?
    if not temp_convs:  # random order for PE
        pcls, poses = shuffle(pcls, poses)
        split = int(len(poses) / 100 * 80)
        train_pcls, test_pcls = pcls[:split], pcls[split:]
        train_poses, test_poses = poses[:split], poses[split:]

    else:  # sequences for tracking # TODO
        train_pcls, test_pcls = None, None
        train_poses, test_poses = None, None

    # # find min, max values in TRAINING scans

    pcls_min, pcls_max = [np.min(train_pcls, axis=(0, 1)), np.max(train_pcls, axis=(0, 1))]
    np.save('data/AMASS/train/pcls_minmax.npy', [pcls_min, pcls_max])
    # assert len(pcls) == len(poses)
    train_regions = np.empty((train_poses.shape[0], numPoints, 1), dtype=np.int)
    test_regions = np.empty((test_poses.shape[0], numPoints, 1), dtype=np.int)

    # zero centering, region gt segmentation
    for i in range(train_pcls.shape[0]):
        train_regions[i] = automatic_annotation(train_poses[i], train_pcls[i])
        train_poses[i] -= train_pcls[i].mean(axis=0)
        train_pcls[i] -= train_pcls[i].mean(axis=0)

    for i in range(test_pcls.shape[0]):
        test_regions[i] = automatic_annotation(test_poses[i], test_pcls[i])
        test_poses[i] -= test_pcls[i].mean(axis=0)
        test_pcls[i] -= test_pcls[i].mean(axis=0)

    scaled_train_pcls, scaled_train_poses = scale_dataset(train_pcls, train_poses, pcls_min, pcls_max)
    scaled_test_pcls, scaled_test_poses = scale_dataset(test_pcls, test_poses, pcls_min, pcls_max)

    # TODO save in baches / individual npy files

    save_batches(scaled_train_pcls, scaled_train_poses, train_regions, mode='train')
    save_batches(scaled_test_pcls, scaled_test_poses, test_regions, mode='test')


def scale_dataset(pcls, poses, pcls_min, pcls_max):
    pcls = 2 * (pcls - pcls_min) / (pcls_max - pcls_min) - 1
    poses = 2 * (poses - pcls_min) / (pcls_max - pcls_min) - 1

    return pcls, poses


def save_batches(pcls, poses, regions, mode):
    bpcl = np.empty(shape=(batch_size, numPoints, 1, 3))
    bpose = np.empty(shape=(batch_size, numJoints * 3))
    breg = np.empty(shape=(batch_size, numPoints, 1), dtype=np.int)
    s = 0

    if singleview:
        name = 'SV'
    else:
        name = ''

    pcls = np.expand_dims(pcls, axis=2)
    poses = poses.reshape((poses.shape[0], numJoints * 3))

    for i in range(poses.shape[0]):
        pcl = pcls[i]
        pose = poses[i]
        region = regions[i]

        idx = s % batch_size
        if not idx and s > 0:
            np.save(
                'data/AMASS/' + mode + '/scaledpcls' + name + 'batch/' + str(s // batch_size).zfill(
                    6) + '.npy',
                bpcl)
            np.save('data/AMASS/' + mode + '/scaledposes' + name + 'batch/' + str(
                s // batch_size).zfill(
                6) + '.npy',
                    bpose)
            np.save('data/AMASS/' + mode + '/region' + name + 'batch/' + str(s // batch_size).zfill(
                6) + '.npy',
                    breg)
        bpcl[idx] = pcl
        bpose[idx] = pose
        breg[idx] = region
        s += 1


if __name__ == '__main__':
    load_dataset(path)
