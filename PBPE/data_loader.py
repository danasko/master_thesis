from scipy.io import loadmat  # from Matlab
from preprocess import *
import os
from config import *
from ITOP_data_loader import DataLoader
from sklearn.utils import shuffle

CMU_split_indices = []


# CMU_numsamples = 153378‬ (pose1,2,3)


def CMU_to_npy(seq_name):
    path = 'G:\\skola\\master\\datasets\\CMU-Panoptic\\panoptic-toolbox\\' + seq_name

    pcl_path = path + '\\kinoptic_ptclouds\\'
    pose_path = path + '\\kinoptic_poses\\'

    arr = []

    for file in os.listdir(pose_path):
        x = loadmat(pcl_path + 'ptcloud' + file[4:])['pclData'][0]  # shape (1000, 2048, 3)
        y = loadmat(pose_path + file)['poseData']  # shape (1000, [frameIdx, joints15]) (15, 3)
        print(file)
        for i, pcl in enumerate(x):
            if len(y[0][i][0][0]) > 1:
                arr.append((pcl, y[0][i][0][0][1]))  # pose (15, 3)

    arr = np.asarray(arr)
    print(arr.shape)
    np.save('data/CMU/test/' + seq_name + '.npy', arr)
    # np.save('data/CMU/train/pcls_poses/' + seq_name + '.npy', arr)


def UBC_convert_pcl_files(index=0, start=1, end=60, mode='train', random_subsampling=False):
    global pcls_min, pcls_max
    for j in range(start, end):
        if j != 6 or mode == 'valid':
            x = loadmat('D:/skola/master/datasets/UBC3V/ubc3v-master/exported_clouds_hard_'
                        + mode + str(j) + '.mat')['exported_clouds']
            # y_regions = \
            #     loadmat('D:/skola/master/datasets/UBC3V/ubc3v-master/exported_regions_hard_valid' + str(j) + '.mat')[
            #         'exported_regions']
            # print(x[0, 0][0])

            # subsampling input pointclouds to numPoints
            for i in range(x.shape[0]):
                # visualize_3D(x[i, 0][0])
                if random_subsampling:
                    t = subsample_random(x[i, 0][0], numPoints)
                else:
                    t = subsample(x[i, 0][0], numPoints)

                # pose = np.load('data/' + mode + '/pose/' + str(index).zfill(5) + '.npy')
                # t, pose = subsample(x[i, 0][0], pose, numPoints)
                if not i % 100:
                    print(i, ' pcls processed')
                np.save('data/' + mode + '/notscaledpcl/' + str(index).zfill(5) + '.npy', t)
                # np.save('data/' + mode + '/pose/' + str(index).zfill(5) + '.npy', pose)
                # np.save('data/' + mode + '/region_initial/'+ str(index).zfill(5) + '.npy', regions)
                index += 1
                # np.save('data/' + mode + '/pcls_minmax.npy', [pcls_min, pcls_max])


def UBC_convert_region_files(index=0, start=1, end=61, mode='train'):
    global pcls_min, pcls_max
    for j in range(start, end):
        if j != 6 or mode == 'valid':
            train_y_regions = \
                loadmat('G:/skola/master/datasets/UBC3V/exported_clouds_mat/hard-pose/train/regions/'
                        'exported_regions_hard_'
                        + mode + str(j) + '.mat')['exported_regions']
            # train_y_np = np.zeros(shape=(train_y_regions.shape[0], numPoints, 3))
            x = loadmat(
                'G:/skola/master/datasets/UBC3V/exported_clouds_mat/hard-pose/train/pcls/exported_clouds_hard_'
                + mode + str(j) + '.mat')['exported_clouds']

            # subsampling input pointclouds to numPoints
            for i in range(train_y_regions.shape[0]):  # train_y_regions.shape[0]
                # visualize_3D(train_x[i, 0][0])
                # pcl = np.load(
                #     'data/train/notscaledpcl/' + str(index).zfill(5) + '.npy')
                # train_y_np = train_y_np[indices]
                regs = np.asarray(train_y_regions[i, 0][0], dtype=np.int)
                regs = region_mapping(regs)
                # print(t)
                pcl, pcls_min, pcls_max, regs = subsample(x[i, 0][0], numPoints, pcls_min, pcls_max, regions=regs)
                if not i % 100:
                    print(i, ' region files processed')
                # visualize_3D(pcl, regions=t)

                np.save('data/UBC/' + mode + '/regions44/' + str(index).zfill(5) + '.npy', regs)
                np.save('data/UBC/' + mode + '/notscaledpcl/' + str(index).zfill(5) + '.npy', pcl)
                np.save('data/UBC/' + mode + '/pcls_minmax.npy', [pcls_min, pcls_max])
                # print(pcl.shape, regs.shape)
                index += 1


def MHAD_loadpcls(index=0, start=1, end=13, mode='train', singleview=False):
    if mode == 'train':
        pass
        # pcls_min = [1000000, 1000000, 1000000]
        # pcls_max = [-1000000, -1000000, -1000000]
    else:
        if test_method == '11subjects':
            [pcls_min, pcls_max] = np.load('data/MHAD/train/pcls_minmax_11subs.npy')
        elif singleview:
            [pcls_min, pcls_max] = np.load('data/MHAD/train/pcls_minmaxSW.npy')
        else:
            [pcls_min, pcls_max] = np.load('data/MHAD/train/pcls_minmax.npy')

    # [pcls_min, pcls_max] = np.load('data/MHAD/train/pcls_minmax.npy') # TODO

    # allp = np.empty((2410, numPoints, 3))
    # for i in range(0, 2410):
    #     s = np.load('data/MHAD/train/notscaledpcl/' + str(i).zfill(6) + '.npy')
    #     allp[i] = s

    if singleview:
        dir = 'pcl_singleview'
        name = 'SW/'
    elif test_method == '11subjects':
        dir = 'pcl'
        name = '_11subs/'
    else:
        dir = 'pcl'
        name = '/'

    for r in range(1, 6):
        for j in range(start, end):
            print('Subject ' + str(j) + ' Rep. ' + str(r))
            # if r == 1 and j == 1:
            #     file = h5py.File('G:/skola/master/datasets/MHAD/exported/pcl/S' + str(j).zfill(
            #         2) + 'R' + str(r).zfill(2) + '.mat', 'r')
            #     x = file.get('clouds').value
            # else:

            try:
                x = loadmat(
                    'G:/skola/master/datasets/MHAD/exported/' + dir + '/S' + str(j).zfill(
                        2) + 'R' + str(r).zfill(2) + '.mat')['clouds'][0]
            except KeyError:
                x = loadmat(
                    'G:/skola/master/datasets/MHAD/exported/' + dir + '/S' + str(j).zfill(
                        2) + 'R' + str(r).zfill(2) + '.mat')['s'][0]
            # xx = np.asarray([np.asarray(file[xi[0]]).T for xi in x])
            # print(xx.shape)
            # allp = np.concatenate([allp, xx], axis=0)

            for i in range(x.shape[0]):  # x.shape[0]
                # TODO scale to -1,1 - and save as scaledglobal
                # if j == 1:
                #     pcl = np.array(file[x[i][0]]).T  # x[i]
                # else:
                pcl = x[i]
                # print(pcl.shape)
                if mode == 'train':
                    np.save('data/MHAD/' + mode + '/notscaledpcl' + name + str(index).zfill(6) + '.npy', pcl)
                    # pcl = pcl - pcl.mean(axis=0)
                    # pcls_min = np.minimum(pcls_min, np.min(pcl, axis=0))
                    # pcls_max = np.maximum(pcls_max, np.max(pcl, axis=0))

                else:
                    # pcl = pcl - pcl.mean(axis=0)
                    # pcl = 2 * (pcl - pcls_min) / (pcls_max - pcls_min) - 1
                    # np.save('data/MHAD/test/scaledpclglobal' + name + str(index).zfill(6) + '.npy', pcl)
                    np.save('data/MHAD/test/notscaledpcl' + name + str(index).zfill(6) + '.npy', pcl)
                # visualize_3D(pcl)

                index += 1

            # if mode == 'train':
            #     np.save('data/MHAD/train/pcls_minmax.npy', [pcls_min, pcls_max])
    return index


def MHAD_load_poses(index=0, start=1, end=13, mode='train', sameaspcls=False, repetitions=(1, 6), singleview=False):
    if mode == 'train':
        poses_min = [1000000, 1000000, 1000000]
        poses_max = [-1000000, -1000000, -1000000]

    elif sameaspcls:
        [poses_min, poses_max] = np.load('data/MHAD/train/pcls_minmax.npy')
    elif test_method == '11subjects':
        [poses_min, poses_max] = np.load('data/MHAD/train/poses_minmax_11subs.npy')
    elif singleview:
        [poses_min, poses_max] = np.load('data/MHAD/train/poses_minmaxSW.npy')
    else:
        [poses_min, poses_max] = np.load('data/MHAD/train/poses_minmax.npy')

    if singleview:
        dir = 'pose_singleview'
        name = 'SW/'
    elif test_method == '11subjects':
        dir = 'pose'
        name = '_11subs/'
    else:
        dir = 'pose'
        name = '/'

    for r in range(repetitions[0], repetitions[1]):
        for i in range(start, end):
            y = loadmat(
                'G:/skola/master/datasets/MHAD/exported/' + dir + '/S' + str(i).zfill(
                    2) + '_R' + str(r).zfill(2) + '.mat')['final_poses'][0]
            # allp = []
            for a in y:  # 11 actions
                a = a.flatten()
                for pose in a:
                    # print(pose.shape)
                    if mode == 'test':
                        # pose = pose - pose.mean(axis=0)
                        # pose = 2 * (pose - poses_min) / (poses_max - poses_min) - 1
                        # np.save('data/MHAD/test/posesglobalseparate' + name + str(index).zfill(6) + '.npy', pose)
                        np.save('data/MHAD/test/notscaledpose' + name + str(index).zfill(6) + '.npy', pose)
                    else:

                        np.save('data/MHAD/' + mode + '/notscaledpose' + name + str(index).zfill(6) + '.npy', pose)
                        # pose = pose - pose.mean(axis=0)
                        # allp.append(pose)

                    index += 1

            # if mode == 'train':
            #     allp = np.asarray(allp)
            #     # print(allp.shape)
            #     poses_min = np.minimum(poses_min, np.min(allp, axis=(0, 1)))
            #     poses_max = np.maximum(poses_max, np.max(allp, axis=(0, 1)))
            #     np.save('data/MHAD/train/poses_minmax.npy', [poses_min, poses_max])
    return index


def MHAD_random_split(rate=0.25, folders=None, start=0, end=(numTrainSamples + numTestSamples)):
    # end = len(os.listdir('data/MHAD/train/' + folders[0] + '/'))  # TODO
    split = int(np.floor((end - start) * rate))
    arr = [str(i).zfill(6) for i in range(start, end)]  # list of file names
    sharr = shuffle(arr, random_state=128)
    # print(split)  # 28027
    testSamples = sharr[:split]
    idx = 0
    for file in testSamples:
        for dir in folders:  # ['scaledpclglobal', 'posesglobalseparate', 'region']:
            # move file + '.npy' from train directory to test
            os.rename('data/MHAD/train/' + dir + '/' + file + '.npy',
                      'data/MHAD/test/' + dir + '/' + str(idx).zfill(6) + '.npy')
        idx += 1

    trainSamples = os.listdir('data/MHAD/train/' + folders[0] + '/')[start:]
    idx = start
    for file in trainSamples:
        for dir in folders:  # ['scaledpclglobal', 'posesglobalseparate', 'region']:
            os.rename('data/MHAD/train/' + dir + '/' + file,
                      'data/MHAD/train/' + dir + '/' + str(idx).zfill(6) + '.npy')
        idx += 1


def make_batch_files(mode='train'):
    if mode == 'train':
        num = numTrainSamples
    else:
        num = numTestSamples
    indices = np.arange(num)
    indices = shuffle(indices, random_state=128)
    bpcl = np.empty(shape=(batch_size, numPoints, 1, 3))
    bpose = np.empty(shape=(batch_size, numJoints * 3))
    breg = np.empty(shape=(batch_size, numPoints, 1), dtype=np.int)
    s = 0
    if singleview:
        name = 'SW'
        namepose = 'SW'
    elif numJoints == 35:
        name = ''
        namepose = '35j'
    elif test_method == '11subjects':
        name = '_11subs'
        namepose = '_11subs'
    else:
        name = ''
        namepose = ''
    for i in indices:
        pcl = np.load(
            'data/' + dataset + '/' + mode + '/scaledpclglobal' + name + '/' + str(i).zfill(fill) + '.npy').reshape(
            (numPoints, 1, 3))
        pose = np.load(
            'data/' + dataset + '/' + mode + '/posesglobalseparate' + namepose + '/' + str(i).zfill(
                fill) + '.npy').reshape(
            numJoints * 3)
        reg = np.load('data/' + dataset + '/' + mode + '/region' + namepose + '/' + str(i).zfill(fill) + '.npy')

        idx = s % batch_size
        if not idx and s > 0:
            np.save(
                'data/' + dataset + '/' + mode + '/scaledpclglobal' + name + 'batches/' + str(s // batch_size).zfill(
                    fill) + '.npy',
                bpcl)
            np.save('data/' + dataset + '/' + mode + '/posesglobalseparate' + namepose + 'batches/' + str(
                s // batch_size).zfill(
                fill) + '.npy',
                    bpose)
            np.save('data/' + dataset + '/' + mode + '/region' + namepose + 'batches/' + str(s // batch_size).zfill(
                fill) + '.npy',
                    breg)
        bpcl[idx] = pcl
        bpose[idx] = pose
        breg[idx] = reg
        s += 1


def ITOP_load():
    train_dataset = DataLoader(
        'D:/skola/master/datasets/ITOP/pointclouds/ITOP_side_train_point_cloud.h5',
        'D:/skola/master/datasets/ITOP/labels/ITOP_side_train_labels.h5')
    test_dataset = DataLoader(
        'D:/skola/master/datasets/ITOP/pointclouds/ITOP_side_test_point_cloud.h5',
        'D:/skola/master/datasets/ITOP/labels/ITOP_side_test_labels.h5')

    # train_x = np.load('data/ITOP/train/train_data_notscaled_centered_x.npy', allow_pickle=True)
    # train_y = np.load('data/ITOP/train/train_data_notscaled_centered_y.npy', allow_pickle=True)
    # train_regs = np.load('data/ITOP/train/train_data_notscaled_centered_regs.npy', allow_pickle=True)
    [train_x, train_y, train_regs] = train_dataset.get_data(numPoints=numPoints, start=0, mode='train')  # 17889
    np.save('data/ITOP/train/train_data_notscaled_centered_x.npy', train_x)
    np.save('data/ITOP/train/train_data_notscaled_centered_y.npy', train_y)
    np.save('data/ITOP/train/train_data_regs.npy', train_regs)

    pcls_min = np.min(train_x, axis=(0, 1))
    pcls_max = np.max(train_x, axis=(0, 1))
    np.save('data/ITOP/train/pcls_minmax.npy', [pcls_min, pcls_max])
    print(pcls_min, pcls_max)

    poses_min = np.min(train_y, axis=(0, 1))
    poses_max = np.max(train_y, axis=(0, 1))
    np.save('data/ITOP/train/poses_minmax.npy', [poses_min, poses_max])
    print(poses_min, poses_max)

    train_x_scaled = 2 * (train_x - pcls_min) / (pcls_max - pcls_min) - 1
    train_y_scaled = 2 * (train_y - poses_min) / (poses_max - poses_min) - 1
    np.save('data/ITOP/train/train_data_x.npy', train_x_scaled)
    np.save('data/ITOP/train/train_data_y.npy', train_y_scaled)

    [test_x, test_y, test_regs] = test_dataset.get_data(numPoints=numPoints, mode='test')
    np.save('data/ITOP/test/test_data_notscaled_centered_x.npy', test_x)
    np.save('data/ITOP/test/test_data_notscaled_centered_y.npy', test_y)
    np.save('data/ITOP/test/test_data_regs.npy', test_regs)

    test_x_scaled = 2 * (test_x - pcls_min) / (pcls_max - pcls_min) - 1
    test_y_scaled = 2 * (test_y - poses_min) / (poses_max - poses_min) - 1
    np.save('data/ITOP/test/test_data_x.npy', test_x_scaled)
    np.save('data/ITOP/test/test_data_y.npy', test_y_scaled)

    print('ITOP dataset loaded.')


# def load_poses(index=0, mode='train'):
#     # pose_file = loadmat('D:/skola/master/datasets/UBC3V/ubc3v-master/exported_poses_hard_' + mode + '.mat')['poses'][0]
#     train_pose_file = loadmat('D:/skola/master/datasets/UBC3V/ubc3v-master/exported_poses_hard_train.mat')['poses'][0]
#     train_poses = np.asarray([train_pose_file[i][0] for i in range(train_pose_file.shape[0])])
#     # train_poses = np.reshape(train_poses, (train_poses.shape[0], numJoints * 3))
#
#     valid_pose_file = loadmat('D:/skola/master/datasets/UBC3V/ubc3v-master/exported_poses_hard_valid.mat')['poses'][0]
#     valid_poses = np.asarray([valid_pose_file[i][0] for i in range(valid_pose_file.shape[0])])
#     # valid_poses = np.reshape(valid_poses, (valid_poses.shape[0], numJoints * 3))
#
#     test_pose_file = loadmat('D:/skola/master/datasets/UBC3V/ubc3v-master/exported_poses_hard_test.mat')['poses'][0]
#     test_poses = np.asarray([test_pose_file[i][0] for i in range(test_pose_file.shape[0])])
#     # test_poses = np.reshape(test_poses, (test_poses.shape[0], numJoints * 3))
#
#     poses = np.concatenate([train_poses, valid_poses, test_poses], axis=0)
#     print(poses.shape)
#
#     for a in range(3):
#         # scale each axis separately
#         scaler = MinMaxScaler(feature_range=(-1, 1))
#         scaler.fit_transform(poses[:, :, a])
#
#         train_poses[:, :, a] = scaler.transform(train_poses[:, :, a])
#         valid_poses[:, :, a] = scaler.transform(valid_poses[:, :, a])
#         test_poses[:, :, a] = scaler.transform(test_poses[:, :, a])
#
#         np.save('data/pose_scaler' + str(a) + '.npy', np.asarray([scaler.min_, scaler.scale_]))
#
#     for j in range(index, train_poses.shape[0]):
#         np.save('data/train/scaledpose/' + str(j).zfill(5) + '.npy', train_poses[j])
#     for j in range(index, test_poses.shape[0]):
#         np.save('data/test/scaledpose/' + str(j).zfill(5) + '.npy', test_poses[j])
#     for j in range(index, valid_poses.shape[0]):
#         np.save('data/valid/scaledpose/' + str(j).zfill(5) + '.npy', valid_poses[j])


def scale_poses(mode='train', data='UBC'):
    # global poses_min, poses_max
    poses_min = [1000000, 1000000, 1000000]
    poses_max = [-1000000, -1000000, -1000000]

    if data == 'UBC':
        poses_file = \
            loadmat('D:/skola/master/datasets/UBC3V/ubc3v-master/exported_poses_hard_' + mode + '.mat')['poses'][0]
        poses = np.asarray([poses_file[i][0] for i in range(poses_file.shape[0])])

        if mode == 'train':
            poses_min = np.minimum(poses_min, np.min(poses, axis=(0, 1)))
            poses_max = np.maximum(poses_max, np.max(poses, axis=(0, 1)))
            np.save('data/' + data + '/train/poses_minmax.npy', [poses_min, poses_max])

        if mode == 'test' or mode == 'valid':
            [poses_min, poses_max] = np.load('data/' + data + '/train/poses_minmax.npy')

        for i, p in enumerate(poses):
            # if mode == 'test':
            #     np.save('data/' + data + '/test/notscaledpose/' + str(i).zfill(5) + '.npy', p)
            # p[:, 0] = 2 * (p[:, 0] - poses_min[0]) / (poses_max[0] - poses_min[0]) - 1
            # p[:, 1] = 2 * (p[:, 1] - poses_min[1]) / (poses_max[1] - poses_min[1]) - 1
            # p[:, 2] = 2 * (p[:, 2] - poses_min[2]) / (poses_max[2] - poses_min[2]) - 1
            p = 2 * (p - poses_min) / (poses_max - poses_min) - 1
            np.save('data/' + data + '/' + mode + '/posesglobalseparate/' + str(i).zfill(5) + '.npy', p)
    else:
        # if mode == 'train':
        #     num = numTrainSamples
        # else:
        #     num = numTestSamples
        if test_method == '11subjects':
            dir = '_11subs/'
        elif singleview:
            dir = 'SW/'
        else:
            dir = '/'
        num = len(os.listdir(('data/' + data + '/' + mode + '/notscaledpose' + dir)))
        # [poses_min_old, poses_max_old] = np.load('data/' + data + '/train/poses_minmax_old.npy')

        if test_method == '11subjects':
            [poses_min, poses_max] = np.load('data/' + data + '/train/poses_minmax_11subs.npy')
        elif singleview:
            [poses_min, poses_max] = np.load('data/' + data + '/train/poses_minmaxSW.npy')
        else:
            [poses_min, poses_max] = np.load('data/' + data + '/train/poses_minmax.npy')
        # global_mean = np.load('data/MHAD/train/poses_mean.npy')

        for i in range(num):  # scale
            p = np.load('data/' + data + '/' + mode + '/notscaledpose' + dir + str(i).zfill(6) + '.npy')
            p = p - p.mean(axis=0)
            # p = p - global_mean
            p = 2 * (p - poses_min) / (poses_max - poses_min) - 1
            np.save('data/' + data + '/' + mode + '/posesglobalseparate' + dir + str(i).zfill(6) + '.npy', p)

    # print(poses_min, poses_max)


def generate_regions(mode='train', data='UBC', start=None, end=None):
    # [pcls_min, pcls_max] = np.load('data/' + data + '/train/pcls_minmax.npy')
    # [poses_min, poses_max] = np.load('data/' + data + '/train/poses_minmax.npy')
    if data == 'CMU':
        # print('Generating regions for sequence ...')
        # pcls = np.load('data/CMU/train/pcls.npy')
        # poses = np.load('data/CMU/train/poses.npy')
        # regs = np.empty((pcls.shape[0], numPoints, 1), dtype=np.int)
        #
        # test_pcls = np.load('data/CMU/test/pcls.npy')
        # test_poses = np.load('data/CMU/test/poses.npy')
        # test_regs = np.empty((test_pcls.shape[0], numPoints, 1), dtype=np.int)
        #
        # for i in range(pcls.shape[0]):
        #     regs[i] = automatic_annotation(poses[i], pcls[i])
        # for i in range(test_pcls.shape[0]):
        #     test_regs[i] = automatic_annotation(test_poses[i], test_pcls[i])
        #
        # np.save('data/CMU/train/regions.npy', regs)
        # np.save('data/CMU/test/regions.npy', test_regs)

        for seq in ['171204_pose6.npy']:  # os.listdir('data/CMU/train/pcls_poses/'):
            # print(seq.split('.')[0])
            arr = np.load('data/CMU/test/' + seq, allow_pickle=True)
            regs = np.empty((arr.shape[0], numPoints, 1), dtype=np.int)
            for i in range(arr.shape[0]):
                regs[i] = automatic_annotation(pose=arr[i, 1], pcl=arr[i, 0])
            np.save('data/CMU/test/' + seq.split('.')[0] + '_regs.npy', regs)

    else:
        if data == 'UBC':
            fill = 5
        else:  # MHAD, ITOP, CMU
            fill = 6
        if mode == 'train':
            num = numTrainSamples
        elif mode == 'valid':
            num = numValSamples
        else:
            num = numTestSamples

        if start is not None and end is not None:
            r = range(start, end)
        else:
            r = range(num)

        for i in r:
            # if mode == 'test' and data == 'MHAD':
            #     pose = np.load('data/' + data + '/train/posesglobalseparate/' + str(i).zfill(fill) + '.npy')
            #     pose = (pose + 1) * (poses_max - poses_min) / 2 + poses_min
            #     pcl = np.load('data/' + data + '/train/scaledpclglobal/' + str(i).zfill(fill) + '.npy')
            #     pcl = (pcl + 1) * (pcls_max - pcls_min) / 2 + pcls_min
            # else:

            if mode == 'train':
                pose = np.load('data/' + data + '/' + mode + '/notscaledpose_11subs/' + str(i).zfill(fill) + '.npy')
                pcl = np.load('data/' + data + '/' + mode + '/notscaledpcl_11subs/' + str(i).zfill(fill) + '.npy')
            else:
                [poses_min, poses_max] = np.load('data/' + data + '/train/poses_minmax_11subs.npy')
                [pcls, pcls_max] = np.load('data/' + data + '/train/pcls_minmax_11subs.npy')
                pose = np.load(
                    'data/' + data + '/' + mode + '/posesglobalseparate_11subs/' + str(i).zfill(fill) + '.npy')
                pcl = np.load('data/' + data + '/' + mode + '/scaledpclglobal_11subs/' + str(i).zfill(fill) + '.npy')
                pose = (pose + 1) * (poses_max - poses_min) / 2 + poses_min
                pcl = (pcl + 1) * (pcls_max - pcls_min) / 2 + pcls_min
            regions = automatic_annotation(pose, pcl)
            # visualize_3D(pcl, regions=regions, pose=pose)
            if data == 'MHAD':
                np.save('data/' + data + '/' + mode + '/region_11subs/' + str(i).zfill(fill) + '.npy', regions)
            else:
                np.save('data/' + data + '/' + mode + '/region/' + str(i).zfill(fill) + '.npy', regions)


def validtotrain(index=59059):
    for i in range(numValSamples):
        p = np.load('data/UBC/valid/scaledpclglobal/' + str(i).zfill(5) + '.npy')
        s = np.load('data/UBC/valid/region/' + str(i).zfill(5) + '.npy')
        j = np.load('data/UBC/valid/posesglobalseparate/' + str(i).zfill(5) + '.npy')
        np.save('data/UBC/train_valid/scaledpclglobal/' + str(index).zfill(5) + '.npy', p)
        np.save('data/UBC/train_valid/region/' + str(index).zfill(5) + '.npy', s)
        np.save('data/UBC/train_valid/posesglobalseparate/' + str(index).zfill(5) + '.npy', j)
        index += 1


def scale(mode='train', data='UBC'):
    if test_method == '11subjects':
        [pcls_min, pcls_max] = np.load('data/' + data + '/train/pcls_minmax_11subs.npy')
    elif singleview:
        [pcls_min, pcls_max] = np.load('data/' + data + '/train/pcls_minmaxSW.npy')
    else:
        [pcls_min, pcls_max] = np.load('data/' + data + '/train/pcls_minmax.npy')
    # [pcls_min_old, pcls_max_old] = np.load('data/' + data + '/train/pcls_minmax_old.npy')
    if data == 'UBC':
        fill = 5
    else:  # MHAD
        fill = 6

    # if mode == 'train':
    #     num = numTrainSamples
    # elif mode == 'valid':
    #     num = numValSamples
    # else:
    #     num = numTestSamples
    if test_method == '11subjects':
        dir = '_11subs/'
    elif singleview:
        dir = 'SW/'
    else:
        dir = '/'

    num = len(os.listdir(('data/' + data + '/' + mode + '/notscaledpcl' + dir)))
    for i in range(num):
        pcl = np.load('data/' + data + '/' + mode + '/notscaledpcl' + dir + str(i).zfill(fill) + '.npy')
        # pose = np.load('data/' + mode + '/notscaledpose/' + str(i).zfill(fill) + '.npy')
        # pcl = (pcl + 1) * (pcls_max_old - pcls_min_old) / 2 + pcls_min_old  # unscale

        if data == 'MHAD':  # TODO also shift to zero mean
            pcl = pcl - pcl.mean(axis=0)

        pcl = 2 * (pcl - pcls_min) / (pcls_max - pcls_min) - 1  # scaled using new minmax values
        # pose = 2 * (pose - pcls_min) / (pcls_max - pcls_min) - 1

        np.save('data/' + data + '/' + mode + '/scaledpclglobal' + dir + str(i).zfill(fill) + '.npy', pcl)
        # np.save('data/' + mode + '/scaledposeglobal/' + str(i).zfill(fill) + '.npy', pose)


def find_minmax(data='MHAD', mode='train', pcls=True):
    pcls_min = [1000000, 1000000, 1000000]
    pcls_max = [-1000000, -1000000, -1000000]

    if data == 'CMU':
        pcls = np.load('data/CMU/train/pcls.npy', allow_pickle=True)
        poses = np.load('data/CMU/train/poses.npy', allow_pickle=True)

        test_pcls = np.load('data/CMU/test/pcls.npy', allow_pickle=True)
        test_poses = np.load('data/CMU/test/poses.npy', allow_pickle=True)

        print('pcls shape: ', pcls.shape)
        print('test pcls shape: ', test_pcls.shape)
        # TODO ? global zero mean --- same with test set
        # mean_pcl = pcls.mean(axis=(0,1))
        # pcls = pcls - mean_pcl
        # poses = poses - mean_pcl
        # test_pcls = test_pcls - mean_pcl
        # test_poses = test_poses - mean_pcl
        # TODO global/local zero mean
        for p in range(pcls.shape[0]):
            poses[p] -= pcls[p].mean(axis=0)
            pcls[p] -= pcls[p].mean(axis=0)
        for p in range(test_pcls.shape[0]):
            test_poses[p] -= test_pcls[p].mean(axis=0)
            test_pcls[p] -= test_pcls[p].mean(axis=0)

        pcls_min, pcls_max = [np.min(pcls, axis=(0, 1)), np.max(pcls, axis=(0, 1))]
        # poses_min, poses_max = [np.min(poses, axis=(0, 1)), np.max(poses, axis=(0, 1))]

        np.save('data/CMU/train/pcls_minmax_lzeromean.npy', [pcls_min, pcls_max])
        np.save('data/CMU/train/poses_minmax_lzeromean.npy', [pcls_min, pcls_max])  # [poses_min, poses_max]

        pcls = 2 * (pcls - pcls_min) / (pcls_max - pcls_min) - 1
        poses = 2 * (poses - pcls_min) / (pcls_max - pcls_min) - 1  # todo poses_min max

        np.save('data/CMU/train/scaled_pcls_lzeromean.npy', pcls)
        np.save('data/CMU/train/scaled_poses_lzeromean.npy', poses)

        test_pcls = 2 * (test_pcls - pcls_min) / (pcls_max - pcls_min) - 1
        test_poses = 2 * (test_poses - pcls_min) / (pcls_max - pcls_min) - 1  # todo poses_min max

        np.save('data/CMU/test/scaled_pcls_lzeromean.npy', test_pcls)
        np.save('data/CMU/test/scaled_poses_lzeromean.npy', test_poses)
    else:
        if data == 'UBC':
            fill = 5
        else:  # MHAD ...
            fill = 6

        # if mode == 'train':
        #     num = numTrainSamples
        # elif mode == 'valid':
        #     num = numValSamples
        # else:
        #     num = numTestSamples
        if test_method == '11subjects':
            num = len(os.listdir(('data/' + data + '/' + mode + '/notscaledpose_11subs/')))
        else:
            num = len(os.listdir(('data/' + data + '/' + mode + '/notscaledpose/')))

        if pcls:
            if test_method == '11subjects':
                dir = '/notscaledpcl_11subs/'
            else:
                dir = '/notscaledpcl/'
        else:
            if test_method == '11subjects':
                dir = '/notscaledpose_11subs/'
            else:
                dir = '/notscaledpose/'

        for i in range(num):
            p = np.load('data/' + data + '/' + mode + dir + str(i).zfill(fill) + '.npy')

            # pcl = (pcl + 1) * (pcls_max_old - pcls_min_old) / 2 + pcls_min_old  # unscale
            if data == 'MHAD':
                p = p - p.mean(axis=0)

            pcls_min = np.minimum(pcls_min, np.min(p, axis=0))
            pcls_max = np.maximum(pcls_max, np.max(p, axis=0))

        if pcls:
            if test_method == '11subjects':
                np.save('data/' + data + '/' + mode + '/pcls_minmax_11subs.npy', [pcls_min, pcls_max])
            elif singleview:
                np.save('data/' + data + '/' + mode + '/pcls_minmaxSW.npy', [pcls_min, pcls_max])
            else:
                np.save('data/' + data + '/' + mode + '/pcls_minmax.npy', [pcls_min, pcls_max])
        else:
            # allp = np.asarray(allp)
            # m = allp.mean(axis=(0, 1))
            # np.save('data/' + data + '/' + mode + '/poses_mean.npy', m)
            if test_method == '11subjects':
                np.save('data/' + data + '/' + mode + '/poses_minmax_11subs.npy', [pcls_min, pcls_max])
            elif singleview:
                np.save('data/' + data + '/' + mode + '/poses_minmaxSW.npy', [pcls_min, pcls_max])
            else:
                np.save('data/' + data + '/' + mode + '/poses_minmax.npy', [pcls_min, pcls_max])


def unscale_to_cm(pose, mode='train', data='UBC'):
    if test_method == '11subjects':
        [poses_min, poses_max] = np.load('data/' + data + '/' + mode + '/poses_minmax_11subs.npy')
    elif singleview:
        [poses_min, poses_max] = np.load('data/' + data + '/' + mode + '/poses_minmaxSW.npy')
    else:
        [poses_min, poses_max] = np.load('data/' + data + '/' + mode + '/poses_minmax.npy')
    # [poses_min, poses_max] = np.load('data/' + data + '/' + mode + '/pcls_minmax.npy')
    # pose2 = np.zeros_like(pose)
    # pose2[:, 0] = (pose[:, 0] + 1) * (poses_max[0] - poses_min[0]) / 2 + poses_min[0]
    # pose2[:, 1] = (pose[:, 1] + 1) * (poses_max[1] - poses_min[1]) / 2 + poses_min[1]
    # pose2[:, 2] = (pose[:, 2] + 1) * (poses_max[2] - poses_min[2]) / 2 + poses_min[2]
    pose2 = (pose + 1) * (poses_max - poses_min) / 2 + poses_min
    if data == 'MHAD':  # in mm
        pose2 *= 0.1
    elif data == 'ITOP':  # in m
        pose2 *= 100

    return pose2


def split_CMU(rate=0.2):
    all_pcls = np.empty((0, numPoints, 3))
    all_poses = np.empty((0, numJoints, 3))
    # all_regs = np.empty((0, numPoints, 1), dtype=np.int)

    for seq in os.listdir('data/CMU/train/pcls_poses/'):
        arr = np.load('data/CMU/train/pcls_poses/' + seq, allow_pickle=True)
        # regs = np.load('data/CMU/train/regions/' + seq + '_regs.npy', allow_pickle=True)
        pcls = arr[:, 0]
        poses = arr[:, 1]

        pcls = np.reshape(np.asarray([np.concatenate(i, axis=0) for i in pcls]), (pcls.shape[0], numPoints, 3))
        poses = np.reshape(np.asarray([np.concatenate(i, axis=0) for i in poses]), (poses.shape[0], numJoints, 3))

        all_pcls = np.concatenate([all_pcls, pcls])
        all_poses = np.concatenate([all_poses, poses])

    arange = np.arange(all_pcls.shape[0])
    arange = shuffle(arange)
    split = int(np.floor(arange.shape[0] * rate))
    test_indices = arange[:split]
    train_indices = arange[split:]
    print(train_indices.shape[0], ' train samples')
    print(test_indices.shape[0], ' test samples')

    train_set_x = all_pcls[train_indices]
    train_set_y = all_poses[train_indices]
    test_set_x = all_pcls[test_indices]
    test_set_y = all_poses[test_indices]

    np.save('data/CMU/train/pcls.npy', train_set_x)
    np.save('data/CMU/train/poses.npy', train_set_y)
    np.save('data/CMU/test/pcls.npy', test_set_x)
    np.save('data/CMU/test/poses.npy', test_set_y)


def scale_CMU(mode='train'):
    pcls_min, pcls_max = np.load('data/CMU/train/pcls_minmax.npy')
    poses_min, poses_max = np.load('data/CMU/train/poses_minmax.npy')
    # print(pcls_min, pcls_max)

    # all_pcls = np.empty((0, numPoints, 3))
    # all_poses = np.empty((0, numJoints, 3))
    # for seq in os.listdir('data/CMU/train/pcls_poses/'):
    #     arr = np.load('data/CMU/train/pcls_poses/' + seq, allow_pickle=True)
    #     pcls = arr[:, 0]
    #     poses = arr[:, 1]
    #
    #     pcls = np.reshape(np.asarray([np.concatenate(i, axis=0) for i in pcls]), (pcls.shape[0], numPoints, 3))
    #     poses = np.reshape(np.asarray([np.concatenate(i, axis=0) for i in poses]), (poses.shape[0], numJoints, 3))
    #
    #     # print(poses[0])

    pcls = np.load('data/CMU/' + mode + '/pcls.npy', allow_pickle=True)
    poses = np.load('data/CMU/' + mode + '/poses.npy', allow_pickle=True)

    pcls = 2 * (pcls - pcls_min) / (pcls_max - pcls_min) - 1
    poses = 2 * (poses - poses_min) / (poses_max - poses_min) - 1

    np.save('data/CMU/' + mode + '/scaled_pcls.npy', pcls)
    np.save('data/CMU/' + mode + '/scaled_poses.npy', poses)


if __name__ == "__main__":
    # CMU_to_npy('171204_pose5')
    # CMU_to_npy('171204_pose6')
    # CMU_to_npy('171026_pose1')
    # CMU_to_npy('171026_pose2')
    # CMU_to_npy('171026_pose3')

    # split_CMU(0.2) # wo pose6
    # generate_regions(data='CMU')
    # find_minmax(data='CMU')  # todo zero mean local
    # scale_CMU(mode='test')

    pass
