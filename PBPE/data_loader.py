from scipy.io import loadmat  # from Matlab
from preprocess import *
import os
from config import *
from ITOP_data_loader import DataLoader
from sklearn.utils import shuffle


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
    pcls_min, pcls_max = np.load('data/UBC/train/pcls_minmax.npy')
    for j in range(start, end):
        if j != 6 or mode == 'valid':
            x = loadmat('D:/skola/master/datasets/UBC3V/ubc3v-master/singleview/exported_clouds_hard_'
                        + mode + str(j) + '.mat')['exported_clouds']

            # subsampling input pointclouds to numPoints
            for i in range(x.shape[0]):
                # visualize_3D(x[i, 0][0])
                if random_subsampling:
                    t = subsample_random(x[i, 0][0], numPoints)
                else:
                    t = subsample(x[i, 0][0], numPoints)

                t = 2 * (t - pcls_min) / (pcls_max - pcls_min) - 1
                if not i % 100:
                    print(i, ' pcls processed')
                np.save('data/' + dataset + '/' + mode + '/scaledpclsSV/' + str(index).zfill(fill) + '.npy', t)

                index += 1
            print("end of " + str(j) + "th file: index " + str(index))


def UBC_convert_region_files(index=0, start=1, end=61, mode='train'):
    global pcls_min, pcls_max
    for j in range(start, end):
        if j != 6 or mode == 'valid':
            train_y_regions = \
                loadmat('G:/skola/master/datasets/UBC3V/exported_clouds_mat/hard-pose/train/regions/'
                        'exported_regions_hard_'
                        + mode + str(j) + '.mat')['exported_regions']
            x = loadmat(
                'G:/skola/master/datasets/UBC3V/exported_clouds_mat/hard-pose/train/pcls/exported_clouds_hard_'
                + mode + str(j) + '.mat')['exported_clouds']

            # subsampling input pointclouds to numPoints
            for i in range(train_y_regions.shape[0]):
                regs = np.asarray(train_y_regions[i, 0][0], dtype=np.int)
                regs = region_mapping(regs)

                pcl, pcls_min, pcls_max, regs = subsample(x[i, 0][0], numPoints, regions=regs)
                if not i % 100:
                    print(i, ' region files processed')

                np.save('data/UBC/' + mode + '/regions44/' + str(index).zfill(5) + '.npy', regs)
                # np.save('data/UBC/' + mode + '/notscaledpcl/' + str(index).zfill(5) + '.npy', pcl)
                # np.save('data/UBC/' + mode + '/pcls_minmax.npy', [pcls_min, pcls_max])

                index += 1


def MHAD_loadpcls(index=0, start=1, end=13, mode='train', singleview=False, leaveout=13):
    if singleview:
        if test_method == '11subjects':
            if numJoints == 35:
                dir = 'pcl_singleview'
                name = 'SV35j_11subs' + str(leaveout) + '/'
            else:
                dir = 'pcl_singleview'
                name = 'SV_11subs' + str(leaveout) + '/'
        else:
            dir = 'pcl_singleview'
            name = 'SV/'
    elif test_method == '11subjects':
        if numJoints == 35:
            dir = 'pcl'
            name = '35j_11subs' + str(leaveout) + '/'
        else:
            dir = 'pcl'
            name = '_11subs' + str(leaveout) + '/'
    else:
        dir = 'pcl'
        name = '/'

    for r in range(1, 6):  # 5 repetitions
        for j in range(start, end):  # subjects
            if mode == 'train':
                cond = j != leaveout
            else:
                cond = j == leaveout
            if cond:
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
                    pcl = x[i]
                    # if mode == 'train':
                    #     np.save('data/MHAD/' + mode + '/notscaledpcl' + name + str(index).zfill(6) + '.npy', pcl)
                    #     # pcl = pcl - pcl.mean(axis=0)
                    #     # pcls_min = np.minimum(pcls_min, np.min(pcl, axis=0))
                    #     # pcls_max = np.maximum(pcls_max, np.max(pcl, axis=0))
                    #
                    # else:
                    # pcl = pcl - pcl.mean(axis=0)
                    # pcl = 2 * (pcl - pcls_min) / (pcls_max - pcls_min) - 1
                    #     # np.save('data/MHAD/test/scaledpcls' + name + str(index).zfill(6) + '.npy', pcl)
                    np.save('data/MHAD/' + mode + '/notscaledpcl' + name + str(index).zfill(6) + '.npy', pcl)

                    index += 1

    return index


def MHAD_load_poses(index=0, start=1, end=13, mode='train', repetitions=(1, 6), singleview=False,
                    leaveout=13):
    if singleview:
        if test_method == '11subjects':
            if numJoints == 35:
                dir = 'pose_singleview_35j'
                name = 'SV35j_11subs' + str(leaveout) + '/'
            else:
                dir = 'pose_singleview'
                name = 'SV_11subs' + str(leaveout) + '/'
        else:
            dir = 'pose_singleview'
            name = 'SV/'
    elif test_method == '11subjects':
        if numJoints == 35:
            dir = 'pose_35j'
            name = '35j_11subs' + str(leaveout) + '/'
        else:
            dir = 'pose'
            name = '_11subs' + str(leaveout) + '/'
    else:
        dir = 'pose'
        name = '/'

    for r in range(repetitions[0], repetitions[1]):
        for i in range(start, end):
            if mode == 'train':
                cond = i != leaveout
            else:
                cond = i == leaveout
            if cond:
                y = loadmat(
                    'G:/skola/master/datasets/MHAD/exported/' + dir + '/S' + str(i).zfill(
                        2) + '_R' + str(r).zfill(2) + '.mat')['final_poses'][0]

                for a in y:  # 11 actions
                    a = a.flatten()
                    for pose in a:
                        # pose = pose - pose.mean(axis=0)
                        # pose = 2 * (pose - poses_min) / (poses_max - poses_min) - 1

                        np.save('data/MHAD/' + mode + '/notscaledpose' + name + str(index).zfill(6) + '.npy',
                                pose)
                        index += 1

    return index


def MHAD_random_split(rate=0.25, folders=None, start=0, end=(numTrainSamples + numTestSamples)):
    split = int(np.floor((end - start) * rate))
    arr = [str(i).zfill(6) for i in range(start, end)]  # list of file names
    sharr = shuffle(arr, random_state=128)
    testSamples = sharr[:split]
    idx = 0
    for file in testSamples:
        for dir in folders:  # ['scaledpcls', 'scaledposes', 'region']:
            # move file + '.npy' from train directory to test
            os.rename('data/MHAD/train/' + dir + '/' + file + '.npy',
                      'data/MHAD/test/' + dir + '/' + str(idx).zfill(6) + '.npy')
        idx += 1

    trainSamples = os.listdir('data/MHAD/train/' + folders[0] + '/')[start:]
    idx = start
    for file in trainSamples:
        for dir in folders:  # ['scaledpcls', 'scaledposes', 'region']:
            os.rename('data/MHAD/train/' + dir + '/' + file,
                      'data/MHAD/train/' + dir + '/' + str(idx).zfill(6) + '.npy')
        idx += 1


def make_batch_files(mode='train'):
    if test_method == '11subjects':
        if singleview:
            num = len(
                os.listdir('data/' + dataset + '/' + mode + '/scaledposesSV35j_11subs' + str(leaveout) + '/'))
        else:
            num = len(os.listdir('data/' + dataset + '/' + mode + '/scaledposes_11subs' + str(leaveout) + '/'))
    else:
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
        if test_method == '11subjects':
            if numJoints == 35:
                name = 'SV35j_11subs' + str(leaveout)
                namepose = 'SV35j_11subs' + str(leaveout)
            else:
                name = 'SV_11subs' + str(leaveout)
                namepose = 'SV_11subs' + str(leaveout)
        else:
            name = 'SV'
            namepose = 'SV'
    elif numJoints == 35:
        if test_method == '11subjects':
            name = '_11subs' + str(leaveout)
            namepose = '35j_11subs' + str(leaveout)
        else:
            name = ''
            namepose = '35j'
    elif test_method == '11subjects':
        name = '_11subs' + str(leaveout)
        namepose = '_11subs' + str(leaveout)
    else:
        name = ''
        namepose = ''
    for i in indices:
        pcl = np.load(
            'data/' + dataset + '/' + mode + '/scaledpcls' + name + '/' + str(i).zfill(fill) + '.npy').reshape(
            (numPoints, 1, 3))

        pose = np.load(
            'data/' + dataset + '/' + mode + '/scaledposes' + namepose + '/' + str(i).zfill(
                fill) + '.npy').reshape(
            numJoints * 3)  # scaling same as multiview

        reg = np.load('data/' + dataset + '/' + mode + '/region' + namepose + '/' + str(i).zfill(fill) + '.npy')

        idx = s % batch_size
        if not idx and s > 0:
            np.save(
                'data/' + dataset + '/' + mode + '/scaledpcls' + name + 'batch/' + str(s // batch_size).zfill(
                    fill) + '.npy',
                bpcl)
            np.save('data/' + dataset + '/' + mode + '/scaledposes' + namepose + 'batch/' + str(
                s // batch_size).zfill(
                fill) + '.npy',
                    bpose)
            np.save('data/' + dataset + '/' + mode + '/region' + namepose + 'batch/' + str(s // batch_size).zfill(
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


def load_poses(index=0, mode='train', start=1):
    [poses_min, poses_max] = np.load('data/UBC/train/poses_minmax.npy')

    if mode == 'train':
        numsections = 60
    else:
        numsections = 20
    for i in range(start, numsections):
        if i != 6 or mode == 'valid':
            train_pose_file = \
                loadmat('D:/skola/master/datasets/UBC3V/ubc3v-master/singleview/exported_poses_hard_' + mode + str(
                    i) + '.mat')['poses'][0]
            train_poses = np.asarray([train_pose_file[i][0] for i in range(train_pose_file.shape[0])])
            # train_poses = np.reshape(train_poses, (train_poses.shape[0], numJoints * 3))

            for j in range(train_poses.shape[0]):
                pose = 2 * (train_poses[j] - poses_min) / (poses_max - poses_min) - 1
                pose = pose.flatten()
                np.save('data/UBC/' + mode + '/scaledposesSV/' + str(index).zfill(fill) + '.npy', pose)
                index += 1


def scale_poses(mode='train', data='UBC'):
    [poses_min, poses_max] = np.load('data/' + data + '/train/poses_minmax.npy')

    if data == 'UBC':
        idx = 0
        poses_file = \
            loadmat(
                'D:/skola/master/datasets/UBC3V/ubc3v-master/singleview/exported_poses_hard_' + mode + '.mat')[
                'poses'][0]
        poses = np.asarray([poses_file[i][0] for i in range(poses_file.shape[0])])

        # if mode == 'train':
        #     poses_min = np.minimum(poses_min, np.min(poses, axis=(0, 1)))
        #     poses_max = np.maximum(poses_max, np.max(poses, axis=(0, 1)))
        #     np.save('data/' + data + '/train/poses_minmax.npy', [poses_min, poses_max])
        #
        # if mode == 'test' or mode == 'valid':
        #     [poses_min, poses_max] = np.load('data/' + data + '/train/poses_minmax.npy')

        for p in poses:
            # if mode == 'test':
            #     np.save('data/' + data + '/test/notscaledpose/' + str(i).zfill(5) + '.npy', p)
            p = 2 * (p - poses_min) / (poses_max - poses_min) - 1
            np.save('data/' + data + '/' + mode + '/scaledposesSV/' + str(idx).zfill(5) + '.npy', p)
            idx += 1
    else:
        # if mode == 'train':
        #     num = numTrainSamples
        # else:
        #     num = numTestSamples
        if test_method == '11subjects':
            if numJoints == 35:
                if singleview:
                    dir = 'SV35j_11subs' + str(leaveout) + '/'
                else:
                    dir = '35j_11subs' + str(leaveout) + '/'
            elif singleview:
                dir = 'SV_11subs' + str(leaveout) + '/'
            else:
                dir = '_11subs' + str(leaveout) + '/'
        elif singleview:
            dir = 'SV/'
        else:
            dir = '/'
        num = len(os.listdir(('data/' + data + '/' + mode + '/notscaledpose' + dir)))
        # [poses_min_old, poses_max_old] = np.load('data/' + data + '/train/poses_minmax_old.npy')

        if test_method == '11subjects':
            [poses_min, poses_max] = np.load('data/' + data + '/train/poses_minmax_11subs' + str(leaveout) + '.npy')
        elif singleview:
            [poses_min, poses_max] = np.load('data/' + data + '/train/poses_minmaxSV.npy')
        else:
            [poses_min, poses_max] = np.load('data/' + data + '/train/poses_minmax.npy')
        # global_mean = np.load('data/MHAD/train/poses_mean.npy')

        for i in range(num):  # scale
            p = np.load('data/' + data + '/' + mode + '/notscaledpose' + dir + str(i).zfill(6) + '.npy')
            p = p - p.mean(axis=0)
            # p = p - global_mean
            p = 2 * (p - poses_min) / (poses_max - poses_min) - 1
            np.save('data/' + data + '/' + mode + '/scaledposes' + dir + str(i).zfill(6) + '.npy', p)

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
            pcls_min, pcls_max = np.load('data/' + data + '/train/pcls_minmax.npy')
            poses_min, poses_max = np.load('data/' + data + '/train/poses_minmax.npy')
        else:  # MHAD, ITOP, CMU
            fill = 6
        # if mode == 'train':
        #     num = numTrainSamples
        # elif mode == 'valid':
        #     num = numValSamples
        # else:
        #     num = numTestSamples
        unscale_pcl = False
        if test_method == '11subjects':
            if singleview:
                if numJoints == 35:
                    dir = '/notscaledposeSV35j_11subs' + str(leaveout) + '/'
                    pcldir = '/notscaledpclSV35j_11subs' + str(leaveout) + '/'
                    regdir = '/regionSV35j_11subs' + str(leaveout) + '/'
                    num = len(
                        os.listdir(('data/' + data + '/' + mode + dir)))
                else:
                    dir = '/notscaledposeSV_11subs' + str(leaveout) + '/'
                    pcldir = '/notscaledpclSV_11subs' + str(leaveout) + '/'
                    regdir = '/regionSV_11subs' + str(leaveout) + '/'
                    num = len(
                        os.listdir(('data/' + data + '/' + mode + dir)))
            else:
                dir = '/notscaledpose_11subs' + str(leaveout) + '/'
                pcldir = '/notscaledpcl_11subs' + str(leaveout) + '/'
                regdir = '/region_11subs' + str(leaveout) + '/'
                num = len(os.listdir(('data/' + data + '/' + mode + dir)))
        else:
            dir = '/notscaledpose/'
            pcldir = '/notscaledpcl/'
            regdir = '/region/'
            num = len(os.listdir(('data/' + data + '/' + mode + dir)))

        if start is not None and end is not None:
            r = range(start, end)
        else:
            r = range(num)

        for i in r:
            pose = np.load('data/' + data + '/' + mode + dir + str(i).zfill(fill) + '.npy')
            pcl = np.load('data/' + data + '/' + mode + pcldir + str(i).zfill(fill) + '.npy')
            pose = pose.reshape((numJoints, 3))
            # unscale pcl and pose
            if unscale_pcl:
                pcl = (pcl + 1) * (pcls_max - pcls_min) / 2 + pcls_min
            # pose = (pose + 1) * (poses_max - poses_min) / 2 + poses_min
            # else:
            #     [poses_min, poses_max] = np.load('data/' + data + '/train/poses_minmax_11subs.npy')
            #     [pcls_min, pcls_max] = np.load('data/' + data + '/train/pcls_minmax_11subs.npy')
            #     pose = np.load(
            #         'data/' + data + '/' + mode + '/scaledposes_11subs/' + str(i).zfill(fill) + '.npy')
            #     pcl = np.load('data/' + data + '/' + mode + '/scaledpcls_11subs/' + str(i).zfill(fill) + '.npy')
            #     pose = (pose + 1) * (poses_max - poses_min) / 2 + poses_min
            #     pcl = (pcl + 1) * (pcls_max - pcls_min) / 2 + pcls_min

            regions = automatic_annotation(pose, pcl)

            np.save('data/' + data + '/' + mode + regdir + str(i).zfill(fill) + '.npy', regions)


def validtotrain(index=59059):
    for i in range(numValSamples):
        p = np.load('data/UBC/valid/scaledpcls/' + str(i).zfill(5) + '.npy')
        s = np.load('data/UBC/valid/region/' + str(i).zfill(5) + '.npy')
        j = np.load('data/UBC/valid/scaledposes/' + str(i).zfill(5) + '.npy')
        np.save('data/UBC/train_valid/scaledpcls/' + str(index).zfill(5) + '.npy', p)
        np.save('data/UBC/train_valid/region/' + str(index).zfill(5) + '.npy', s)
        np.save('data/UBC/train_valid/scaledposes/' + str(index).zfill(5) + '.npy', j)
        index += 1


def scale(mode='train', data='UBC'):
    if test_method == '11subjects':
        if singleview:
            [pcls_min, pcls_max] = np.load('data/' + data + '/train/pcls_minmaxSV_11subs' + str(leaveout) + '.npy')
        else:
            [pcls_min, pcls_max] = np.load('data/' + data + '/train/pcls_minmax_11subs' + str(leaveout) + '.npy')
    elif singleview:
        [pcls_min, pcls_max] = np.load('data/' + data + '/train/pcls_minmaxSV.npy')
    else:
        [pcls_min, pcls_max] = np.load('data/' + data + '/train/pcls_minmax.npy')
    # [pcls_min_old, pcls_max_old] = np.load('data/' + data + '/train/pcls_minmax_old.npy')
    if data == 'UBC':
        fill = 5
    else:  # MHAD
        fill = 6

    if test_method == '11subjects':
        if singleview:
            if numJoints == 35:
                dir = 'SV35j_11subs' + str(leaveout) + '/'
            else:
                dir = 'SV_11subs' + str(leaveout) + '/'
        else:
            dir = '_11subs' + str(leaveout) + '/'
    elif singleview:
        dir = 'SV/'
    else:
        dir = '/'

    num = len(os.listdir(('data/' + data + '/' + mode + '/notscaledpcl' + dir)))
    for i in range(num):
        pcl = np.load('data/' + data + '/' + mode + '/notscaledpcl' + dir + str(i).zfill(fill) + '.npy')

        if data == 'MHAD':  # shift to zero mean
            pcl = pcl - pcl.mean(axis=0)

        pcl = 2 * (pcl - pcls_min) / (pcls_max - pcls_min) - 1

        np.save('data/' + data + '/' + mode + '/scaledpcls' + dir + str(i).zfill(fill) + '.npy', pcl)


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
        poses = 2 * (poses - pcls_min) / (pcls_max - pcls_min) - 1

        np.save('data/CMU/train/scaled_pcls_lzeromean.npy', pcls)
        np.save('data/CMU/train/scaled_poses_lzeromean.npy', poses)

        test_pcls = 2 * (test_pcls - pcls_min) / (pcls_max - pcls_min) - 1
        test_poses = 2 * (test_poses - pcls_min) / (pcls_max - pcls_min) - 1

        np.save('data/CMU/test/scaled_pcls_lzeromean.npy', test_pcls)
        np.save('data/CMU/test/scaled_poses_lzeromean.npy', test_poses)
    else:
        if data == 'UBC':
            fill = 5
        else:  # MHAD ...
            fill = 6

        if pcls:
            if test_method == '11subjects':
                if singleview:
                    dir = '/notscaledpclSV_11subs' + str(leaveout) + '/'
                else:
                    dir = '/notscaledpcl_11subs' + str(leaveout) + '/'
            else:
                dir = '/notscaledpcl/'
        else:
            if test_method == '11subjects':
                if numJoints == 35:
                    if singleview:
                        dir = '/notscaledposeSV35j_11subs' + str(leaveout) + '/'
                    else:
                        dir = '/notscaledpose35j_11subs' + str(leaveout) + '/'
                elif singleview:
                    dir = '/notscaledposeSV_11subs' + str(leaveout) + '/'
                else:
                    dir = '/notscaledpose_11subs' + str(leaveout) + '/'
            else:
                dir = '/notscaledpose/'

        num = len(os.listdir('data/' + data + '/' + mode + dir))

        for i in range(num):
            p = np.load('data/' + data + '/' + mode + dir + str(i).zfill(fill) + '.npy')

            if data == 'MHAD':
                p = p - p.mean(axis=0)

            pcls_min = np.minimum(pcls_min, np.min(p, axis=0))
            pcls_max = np.maximum(pcls_max, np.max(p, axis=0))

        if pcls:
            if test_method == '11subjects':
                if singleview:
                    np.save('data/' + data + '/' + mode + '/pcls_minmaxSV_11subs' + str(leaveout) + '.npy',
                            [pcls_min, pcls_max])
                else:
                    np.save('data/' + data + '/' + mode + '/pcls_minmax_11subs' + str(leaveout) + '.npy',
                            [pcls_min, pcls_max])
            elif singleview:
                np.save('data/' + data + '/' + mode + '/pcls_minmaxSV.npy', [pcls_min, pcls_max])
            else:
                np.save('data/' + data + '/' + mode + '/pcls_minmax.npy', [pcls_min, pcls_max])
        else:
            if test_method == '11subjects':
                if singleview:
                    np.save('data/' + data + '/' + mode + '/poses_minmaxSV_11subs' + str(leaveout) + '.npy',
                            [pcls_min, pcls_max])
                else:
                    np.save('data/' + data + '/' + mode + '/poses_minmax_11subs' + str(leaveout) + '.npy',
                            [pcls_min, pcls_max])
            elif singleview:
                np.save('data/' + data + '/' + mode + '/poses_minmaxSV.npy', [pcls_min, pcls_max])
            else:
                np.save('data/' + data + '/' + mode + '/poses_minmax.npy', [pcls_min, pcls_max])


def unscale_to_cm(pose):
    if test_method == '11subjects':
        [poses_min, poses_max] = np.load('data/' + dataset + '/train/poses_minmax_11subs' + str(leaveout) + '.npy')
    elif singleview:
        [poses_min, poses_max] = np.load('data/' + dataset + '/train/poses_minmaxSV.npy')
    elif ordered:
        [poses_min, poses_max] = np.load('data/' + dataset + '/train/ordered/poses_minmax.npy')
    else:
        [poses_min, poses_max] = np.load('data/' + dataset + '/train/poses_minmax.npy')

    pose2 = (pose + 1) * (poses_max - poses_min) / 2 + poses_min
    if dataset == 'MHAD':  # in mm
        pose2 *= 0.1
    elif dataset == 'ITOP':  # in m
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

    # all_pcls = np.empty((0, numPoints, 3))
    # all_poses = np.empty((0, numJoints, 3))
    # for seq in os.listdir('data/CMU/train/pcls_poses/'):
    #     arr = np.load('data/CMU/train/pcls_poses/' + seq, allow_pickle=True)
    #     pcls = arr[:, 0]
    #     poses = arr[:, 1]
    #
    #     pcls = np.reshape(np.asarray([np.concatenate(i, axis=0) for i in pcls]), (pcls.shape[0], numPoints, 3))
    #     poses = np.reshape(np.asarray([np.concatenate(i, axis=0) for i in poses]), (poses.shape[0], numJoints, 3))

    pcls = np.load('data/CMU/' + mode + '/pcls.npy', allow_pickle=True)
    poses = np.load('data/CMU/' + mode + '/poses.npy', allow_pickle=True)

    pcls = 2 * (pcls - pcls_min) / (pcls_max - pcls_min) - 1
    poses = 2 * (poses - poses_min) / (poses_max - poses_min) - 1

    np.save('data/CMU/' + mode + '/scaled_pcls.npy', pcls)
    np.save('data/CMU/' + mode + '/scaled_poses.npy', poses)


# temporal convs

def order_dataset():
    '''order dataset to form original sequences (predict on SGPE model to obtain train data for temporal convs)'''
    if dataset == 'CMU':
        train_pcls = np.empty((0, numPoints, 3))
        train_poses = np.empty((0, numJoints, 3))
        test_pcls = np.empty((0, numPoints, 3))
        test_poses = np.empty((0, numJoints, 3))

        for i, seq in enumerate(os.listdir('data/CMU/train/pcls_poses/')):
            arr = np.load('data/CMU/train/pcls_poses/' + seq, allow_pickle=True)
            pcls = arr[:, 0]
            poses = arr[:, 1]

            pcls = np.reshape(np.asarray([np.concatenate(i, axis=0) for i in pcls]), (pcls.shape[0], numPoints, 3))
            poses = np.reshape(np.asarray([np.concatenate(i, axis=0) for i in poses]), (poses.shape[0], numJoints, 3))
            print(i, seq, poses.shape)

            if i < 6:  # train data
                train_pcls = np.concatenate([train_pcls, pcls])
                train_poses = np.concatenate([train_poses, poses])
            else:  # test data
                test_pcls = np.concatenate([test_pcls, pcls])
                test_poses = np.concatenate([test_poses, poses])

        # for p in range(train_poses.shape[0]):
        #     train_poses[p] -= train_pcls[p].mean(axis=0)
        #     train_pcls[p] -= train_pcls[p].mean(axis=0)
        #
        # for p in range(test_poses.shape[0]):
        #     test_poses[p] -= test_pcls[p].mean(axis=0)
        #     test_pcls[p] -= test_pcls[p].mean(axis=0)
        #
        # pcls_min, pcls_max = [np.min(train_pcls, axis=(0, 1)), np.max(train_pcls, axis=(0, 1))]
        # # poses_min, poses_max = [np.min(poses, axis=(0, 1)), np.max(poses, axis=(0, 1))]
        #
        # np.save('data/CMU/train/ordered/pcls_minmax.npy', [pcls_min, pcls_max])
        # np.save('data/CMU/train/ordered/poses_minmax.npy', [pcls_min, pcls_max])
        #
        # train_pcls = 2 * (train_pcls - pcls_min) / (pcls_max - pcls_min) - 1
        # train_poses = 2 * (train_poses - pcls_min) / (pcls_max - pcls_min) - 1
        #
        # np.save('data/CMU/train/ordered/scaled_pcls_lzeromean.npy', train_pcls)
        # np.save('data/CMU/train/ordered/scaled_poses_lzeromean.npy', train_poses)
        #
        # test_pcls = 2 * (test_pcls - pcls_min) / (pcls_max - pcls_min) - 1
        # test_poses = 2 * (test_poses - pcls_min) / (pcls_max - pcls_min) - 1
        #
        # np.save('data/CMU/test/ordered/scaled_pcls_lzeromean.npy', test_pcls)
        # np.save('data/CMU/test/ordered/scaled_poses_lzeromean.npy', test_poses)


def generate_sequences(save=True):
    if dataset == 'CMU':
        # save GT sequences for temp conv model
        poses = np.load('data/CMU/train/ordered/scaled_poses_lzeromean.npy', allow_pickle=True)

        num_sequences = poses.shape[0] // seq_length
        poses = poses[:num_sequences * seq_length]

        poses = poses.reshape((num_sequences, seq_length, numJoints * 3))
        if save:
            np.save('data/CMU/train/ordered/scaled_poses_lzeromean_seq.npy', poses)

        # save predicted sequences
        preds = np.load('data/CMU/train/ordered/predictions.npy', allow_pickle=True)

        num_sequences = preds.shape[0] // seq_length
        preds = preds[:num_sequences * seq_length]

        preds = preds.reshape((num_sequences, seq_length, numJoints * 3))
        if save:
            np.save('data/CMU/train/ordered/preds_seq.npy', preds)


def subsample_data():
    if dataset == 'CMU':
        train_pcls = np.load('data/CMU/train/ordered/scaled_pcls_lzeromean_2048pts.npy', allow_pickle=True)
        test_pcls = np.load('data/CMU/test/ordered/scaled_pcls_lzeromean_2048pts.npy', allow_pickle=True)
        # train_regions = np.load('data/CMU/train/regions_2048pts.npy', allow_pickle=True)
        # test_regions = np.load('data/CMU/test/regions_2048pts.npy', allow_pickle=True)

        train_pcls_new = np.empty((train_pcls.shape[0], numPoints, 3))
        test_pcls_new = np.empty((test_pcls.shape[0], numPoints, 3))
        # train_regions_new = np.empty((train_pcls.shape[0], numPoints, 1))
        # test_regions_new = np.empty((test_pcls.shape[0], numPoints, 1))

        for i, pcl in enumerate(train_pcls):
            # train_pcls_new[i], train_regions_new[i] = subsample(pcl, numPoints, train_regions[i])
            train_pcls_new[i] = subsample(pcl, numPoints)
        np.save('data/CMU/train/ordered/scaled_pcls_lzeromean_' + str(numPoints) + 'pts.npy', train_pcls_new)
        # np.save('data/CMU/train/regions_' + str(numPoints) + 'pts.npy', train_regions_new)

        for i, pcl in enumerate(test_pcls):
            # test_pcls_new[i], test_regions_new[i] = subsample(pcl, numPoints, test_regions[i])
            test_pcls_new[i] = subsample(pcl, numPoints)
        np.save('data/CMU/test/ordered/scaled_pcls_lzeromean_' + str(numPoints) + 'pts.npy', test_pcls_new)
        # np.save('data/CMU/test/regions_' + str(numPoints) + 'pts.npy', test_regions_new)


if __name__ == "__main__":
    # CMU_to_npy('171204_pose5')
    # CMU_to_npy('171204_pose6')
    # CMU_to_npy('171026_pose1')
    # CMU_to_npy('171026_pose2')
    # CMU_to_npy('171026_pose3')
    # make_batch_files(mode='test')

    # MHAD_loadpcls(index=0, mode='train', singleview=singleview, leaveout=leaveout)
    # MHAD_load_poses(index=0, mode='train', singleview=singleview, leaveout=leaveout)
    # MHAD_loadpcls(index=0, mode='test', singleview=singleview, leaveout=leaveout)
    # MHAD_load_poses(index=0, mode='test', singleview=singleview, leaveout=leaveout)
    # generate_regions(mode='train', data=dataset)
    # generate_regions(mode='test', data=dataset)
    # find_minmax(data=dataset, mode='train', pcls=True)
    # find_minmax(data=dataset, mode='train', pcls=False)
    # scale_poses(mode='train', data=dataset)
    # scale_poses(mode='test', data=dataset)
    # scale(mode='train', data=dataset)
    # scale(mode='test', data=dataset)
    # make_batch_files(mode='train')

    # order_dataset()
    subsample_data()
    # pass
