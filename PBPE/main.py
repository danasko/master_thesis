from keras.models import load_model, Model
from keras.layers.core import Flatten, Dense, Dropout, Reshape
from keras.layers import Input, Convolution2D, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import dot, BatchNormalization, concatenate, PReLU, LeakyReLU
from keras.optimizers import SGD, Adagrad, Adam, rmsprop
from keras.callbacks import LearningRateScheduler
from keras.backend.tensorflow_backend import set_session
from keras import regularizers
from keras.utils import plot_model, HDF5Matrix, CustomObjectScope
from keras.utils.generic_utils import get_custom_objects
import keras.callbacks
import tensorflow as tf
import numpy as np
import multiprocessing as mp
import keras.backend as Kb
from scipy.io import loadmat  # from Matlab
from tensorflow import set_random_seed
# from Adam_lr_mult import *
# from keras_contrib.callbacks import DeadReluDetector
# from visualize_weights import visualize_layer
from sklearn.utils import shuffle
import h5py

from preprocess import *
from visualizer import *
from data_generator import *

batch_size = 32
numPoints = 2048  # number of points in each pcl
numJoints = 18  # 29
numRegions = 18  # 18  # 45, 29

numTrainSamples = 59059
numValSamples = 19019
numTestSamples = 19019

# scaler_minX, scaler_minY, scaler_minZ = None, None, None
# scaler_scaleX, scaler_scaleY, scaler_scaleZ = None, None, None

pcls_min = [1000000, 1000000, 1000000]
pcls_max = [-1000000, -1000000, -1000000]
# pcls_min = None
# pcls_max = None

poses_min = [1000000, 1000000, 1000000]
poses_max = [-1000000, -1000000, -1000000]


# train_x = None  # input pointclouds
# train_y_joints = None  # x,y,z coordinates for each joint
# train_y_regions = None  # rgb color for each point


def tile(global_feature, numPoints):
    return Kb.repeat_elements(global_feature, numPoints, 1)


def PBPE():
    input_points = Input(shape=(numPoints, 1, 3))
    local_feature1 = Conv2D(filters=512, kernel_size=(1, 1), input_shape=(numPoints, 1, 3),
                            kernel_initializer='glorot_normal',
                            dim_ordering='tf', activation='relu')(input_points)  # glorot_normal
    # local_feature1 = LeakyReLU()(local_feature1)
    # local_feature1 = BatchNormalization(momentum=0.9)(x)  # momentum=0.9  # TODO batchnorm
    local_feature2 = Conv2D(filters=2048, kernel_size=(1, 1),
                            activation='relu', kernel_initializer='glorot_normal', dim_ordering='tf')(local_feature1)

    # local_feature2 = BatchNormalization(momentum=0.9)(x)  # shape = (b, numPoints=2048, 1, 2048)  # TODO batchnorm

    global_feature = MaxPooling2D(pool_size=(numPoints, 1))(local_feature2)  # strides  # shape= (b, 1, 1, 2048)
    global_feature_exp = Lambda(tile, arguments={'numPoints': numPoints})(
        global_feature)  # shape= (b, numPoints=2048, 1, 2048)

    f = Flatten()(global_feature)
    f = Dense(256, kernel_initializer='glorot_normal')(f)  # todo dense with activation
    # f = BatchNormalization(momentum=0.9)(f)  # TODO batchnorm

    f = Dense(256, kernel_initializer='glorot_normal')(f)  # todo dense with activation
    #  f = BatchNormalization(momentum=0.9)(f)  # TODO batchnorm

    # f = Dropout(0.3)(f)  # keep_prob = 0.7 => rate = 0.3

    output1 = Dense(3 * numJoints, name='output1')(f)

    # output1 = PReLU(name='output1')(f)

    # Auxiliary part-segmentation network - only for training - removed at test time

    c = concatenate([global_feature_exp, local_feature1, local_feature2], axis=-1)

    c = Conv2D(filters=256, kernel_size=(1, 1),
               activation='relu', kernel_initializer='glorot_normal', dim_ordering='tf')(c)

    c = BatchNormalization(momentum=0.9)(c)  # momentum=0.9

    c = Conv2D(filters=256, kernel_size=(1, 1),
               activation='relu', kernel_initializer='glorot_normal', dim_ordering='tf')(c)

    c = BatchNormalization(momentum=0.9)(c)

    # TODO dropout 0.2

    # c = Dropout(0.2)(c)

    c = Conv2D(filters=128, kernel_size=(1, 1),
               activation='relu', kernel_initializer='glorot_normal', dim_ordering='tf')(c)

    c = BatchNormalization(momentum=0.9)(c)

    # TODO dropout 0.2

    # c = Dropout(0.2)(c)

    output2 = Conv2D(numRegions, (1, 1), activation='softmax', kernel_initializer='glorot_normal', name='output2',
                     dim_ordering='tf')(c)

    model = Model(inputs=input_points, outputs=[output1, output2])  # , output2
    return model


def UBC_convert_pcl_files(index=0, start=1, end=60, mode='train'):
    global pcls_min, pcls_max
    for j in range(start, end):
        if j != 6:
            x = loadmat('D:/skola/master/datasets/UBC3V/ubc3v-master/exported_clouds_hard_'
                        + mode + str(j) + '.mat')['exported_clouds']
            # y_regions = \
            #     loadmat('D:/skola/master/datasets/UBC3V/ubc3v-master/exported_regions_hard_valid' + str(j) + '.mat')[
            #         'exported_regions']
            # print(x[0, 0][0])

            # subsampling input pointclouds to numPoints
            for i in range(x.shape[0]):
                # visualize_3D(x[i, 0][0])
                t, pcls_min, pcls_max = subsample(x[i, 0][0], numPoints, pcls_min,
                                                  pcls_max)

                # pose = np.load('data/' + mode + '/pose/' + str(index).zfill(5) + '.npy')
                # t, pose = subsample(x[i, 0][0], pose, numPoints)
                if not i % 100:
                    print(i, ' pcls processed')
                np.save('data/' + mode + '/notscaledpcl/' + str(index).zfill(5) + '.npy', t)
                # np.save('data/' + mode + '/pose/' + str(index).zfill(5) + '.npy', pose)
                # np.save('data/' + mode + '/region_initial/'+ str(index).zfill(5) + '.npy', regions)
                index += 1
                np.save('data/' + mode + '/pcls_minmax.npy', [pcls_min, pcls_max])


def UBC_convert_region_files(index=0, start=1, end=61, mode='train'):
    global pcls_min, pcls_max
    for j in range(start, end):
        if j != 6:
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


def MHAD_loadpcls(index=0, start=1, end=12):  # TODO last Subject as test set
    # allp = np.empty((2410, numPoints, 3))
    # for i in range(0, 2410):
    #     s = np.load('data/MHAD/train/notscaledpcl/' + str(i).zfill(6) + '.npy')
    #     allp[i] = s

    for r in range(1, 4):
        for j in range(start, end):
            print(j)
            # file = h5py.File('G:/skola/master/datasets/MHAD/exported/pcl/S' + str(j).zfill(
            #     2) + '.mat', 'r')
            # x = file.get('clouds').value
            # print(x.shape)
            x = loadmat(
                'G:/skola/master/datasets/MHAD/exported/pcl/S' + str(j).zfill(
                    2) + '_R' + str(r).zfill(2) + '.mat')['clouds'][0]

            xx = np.asarray([np.asarray(xi) for xi in x])
            allp = np.concatenate([allp, xx], axis=0)

            for i in range(x.shape[0]):  # x.shape[0]
                # TODO scale to -1,1 - and save as scaledglobal

                # pcl = np.array(file[x[0][0]]).T  # x[i]
                pcl = x[i]
                # [pcl, _, _] = subsample(pcl, numPoints, 0, 0)
                # visualize_3D(pcl)
                np.save('data/MHAD/train/notscaledpcl/' + str(index).zfill(6) + '.npy', pcl)
                index += 1

            # TODO min and max values for scaling
            minn = np.min(allp, axis=(0, 1))
            maxx = np.max(allp, axis=(0, 1))
            print(minn.shape)
            m1 = np.minimum(minn, pcls_min)
            m2 = np.maximum(maxx, pcls_max)
            np.save('data/MHAD/train/pcls_minmax.npy', [m1, m2])
    # print(minn, maxx)


def MHAD_random_split(rate=0.25):
    pass
    # newPath = shutil.move('sample1.txt', 'test')


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
    global poses_min, poses_max
    poses_file = loadmat('D:/skola/master/datasets/UBC3V/ubc3v-master/exported_poses_hard_' + mode + '.mat')['poses'][0]
    poses = np.asarray([poses_file[i][0] for i in range(poses_file.shape[0])])

    if mode == 'train':
        poses_min = np.minimum(poses_min, np.min(poses, axis=(0, 1)))
        poses_max = np.maximum(poses_max, np.max(poses, axis=(0, 1)))
        np.save('data/' + data + '/train/poses_minmax.npy', [poses_min, poses_max])

    # print(poses_min, poses_max)
    for i, p in enumerate(poses):
        if mode == 'test':
            np.save('data/' + data + '/test/notscaledpose/' + str(i).zfill(5) + '.npy', p)
        # p[:, 0] = 2 * (p[:, 0] - poses_min[0]) / (poses_max[0] - poses_min[0]) - 1
        # p[:, 1] = 2 * (p[:, 1] - poses_min[1]) / (poses_max[1] - poses_min[1]) - 1
        # p[:, 2] = 2 * (p[:, 2] - poses_min[2]) / (poses_max[2] - poses_min[2]) - 1
        p = 2 * (p - poses_min) / (poses_max - poses_min) - 1
        np.save('data/' + data + '/' + mode + '/posesglobalseparate/' + str(i).zfill(5) + '.npy', p)


def generate_regions_all(mode='train', data='UBC'):
    if mode == 'train':
        num = numTrainSamples
    elif mode == 'valid':
        num = numValSamples
    else:
        num = numTestSamples
    for i in range(num):
        pose = np.load('data/' + data + '/' + mode + '/notscaledpose/' + str(i).zfill(5) + '.npy')
        pcl = np.load('data/' + data + '/' + mode + '/notscaledpcl/' + str(i).zfill(5) + '.npy')
        regions = automatic_annotation(pose, pcl)
        # visualize_3D(pcl, regions=regions, pose=pose)
        np.save('data/' + data + '/' + mode + '/region/' + str(i).zfill(5) + '.npy', regions)


def scale(mode='train', data='UBC'):
    [pcls_min, pcls_max] = np.load('data/' + data + '/train/pcls_minmax.npy')  # TODO try also with valid min and max

    if mode == 'train':
        num = numTrainSamples
    elif mode == 'valid':
        num = numValSamples
    else:
        num = numTestSamples

    for i in range(num):
        pcl = np.load('data/' + data + '/' + mode + '/notscaledpcl/' + str(i).zfill(5) + '.npy')
        # pose = np.load('data/' + mode + '/notscaledpose/' + str(i).zfill(5) + '.npy')

        pcl = 2 * (pcl - pcls_min) / (pcls_max - pcls_min) - 1
        if data == 'MHAD':  # TODO also shift to zero mean
            pcl = pcl - pcl.mean(axis=0)
        # pose = 2 * (pose - pcls_min) / (pcls_max - pcls_min) - 1

        np.save('data/' + data + '/' + mode + '/scaledpclglobal/' + str(i).zfill(5) + '.npy', pcl)
        # np.save('data/' + mode + '/scaledposeglobal/' + str(i).zfill(5) + '.npy', pose)


def unscale_to_cm(pose, mode='train', data='UBC'):
    [poses_min, poses_max] = np.load('data/' + data + '/' + mode + '/poses_minmax.npy')
    # pose2 = np.zeros_like(pose)
    # pose2[:, 0] = (pose[:, 0] + 1) * (poses_max[0] - poses_min[0]) / 2 + poses_min[0]
    # pose2[:, 1] = (pose[:, 1] + 1) * (poses_max[1] - poses_min[1]) / 2 + poses_min[1]
    # pose2[:, 2] = (pose[:, 2] + 1) * (poses_max[2] - poses_min[2]) / 2 + poses_min[2]

    pose2 = (pose + 1) * (poses_max - poses_min) / 2 + poses_min

    return pose2


#
# def unscale_axis_to_cm(p, axis, mode='train'):  # shape = (batch, numJoints)
#     [poses_min, poses_max] = np.load('data/' + mode + '/poses_minmax.npy')
#     p2 = (p + 1) * (poses_max[axis] - poses_min[axis]) / 2 + poses_min[axis]
#     return p2

def avg_error(y_true, y_pred):  # shape=(batch, 3 * numJoints)
    y_pred = Reshape((numJoints, 3))(y_pred)
    y_true = Reshape((numJoints, 3))(y_true)  # shape=(batch, numJoints, 3)

    y_pred = unscale_to_cm(y_pred)
    y_true = unscale_to_cm(y_true)

    # y_predX = y_pred[:, :, 0]
    # y_predY = y_pred[:, :, 1]
    # y_predZ = y_pred[:, :, 2]
    #
    # y_trueX = y_true[:, :, 0]
    # y_trueY = y_true[:, :, 1]
    # y_trueZ = y_true[:, :, 2]
    #
    # # unscale back to cm
    # y_predX = unscale_axis_to_cm(y_predX, axis=0, mode='train')
    # y_predY = unscale_axis_to_cm(y_predY, axis=1, mode='train')
    # y_predZ = unscale_axis_to_cm(y_predZ, axis=2, mode='train')
    #
    # y_trueX = unscale_axis_to_cm(y_trueX, axis=0, mode='train')
    # y_trueY = unscale_axis_to_cm(y_trueY, axis=1, mode='train')
    # y_trueZ = unscale_axis_to_cm(y_trueZ, axis=2, mode='train')
    #
    # y_pred = Kb.concatenate([Kb.expand_dims(y_predX), Kb.expand_dims(y_predY), Kb.expand_dims(y_predZ)], axis=-1)
    # y_true = Kb.concatenate([Kb.expand_dims(y_trueX), Kb.expand_dims(y_trueY), Kb.expand_dims(y_trueZ)], axis=-1)

    # mean error in cm
    return Kb.mean(Kb.mean(Kb.sqrt(Kb.sum(Kb.square(y_pred - y_true), axis=-1)), axis=-1), axis=-1)


def huber_loss(y_true, y_pred):
    # y_pred = Reshape((numJoints, 3))(y_pred)
    # y_true = Reshape((numJoints, 3))(y_true)  # shape=(batch, numJoints, 3)

    # y_pred = unscale_to_cm(y_pred)
    # y_true = unscale_to_cm(y_true)

    clip_delta = 1.0  # 4.0
    error = y_true - y_pred
    cond = Kb.abs(error) < clip_delta

    squared_loss = 0.5 * Kb.square(error)
    linear_loss = clip_delta * (Kb.abs(error) - 0.5 * clip_delta)

    return Kb.mean(Kb.switch(cond, squared_loss, linear_loss), axis=-1)


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.0005
    drop = 0.8  # 0.5
    epochs_drop = 1.0
    lrate = initial_lrate * pow(drop,
                                np.floor(epoch / epochs_drop))  # pow(drop, np.floor((1 + epoch) / epochs_drop))
    if lrate < 0.00001:  # clip at 10^-5 to avoid getting stuck at local minima
        lrate = 0.00001
    return lrate


if __name__ == "__main__":
    Kb.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    model = PBPE()
    # model.summary(line_length=100)

    losses = {
        "output1": "mean_absolute_error",  # huber_loss, mean_squared_error "mean_absolute_error"
        "output2": "categorical_crossentropy",  # segmentation
    }

    get_custom_objects().update({'avg_error': avg_error, 'Kb': Kb})

    Adam = Adam(lr=0.0, decay=0.0)  # to be set in lrScheduler

    # learning schedule callback
    lrate = LearningRateScheduler(step_decay)

    # Tensorboard callback
    tbCallBack = keras.callbacks.TensorBoard(log_dir='data/tensorboard/original', histogram_freq=0,
                                             write_graph=True,
                                             write_images=True, write_grads=False, batch_size=32)
    callbacks_list = [lrate, tbCallBack]

    # Load training, validation data (only once)

    # load_poses(0)
    # scaler = np.asarray([pose_scaler.min_, pose_scaler.scale_])
    # np.save('data/pose_scaler.npy', scaler)

    # UBC_convert_pcl_files(index=5006, start=7, end=61, mode='train')
    # generate_regions_all(mode='train')
    # scale(mode='train')

    # UBC_convert_pcl_files(index=0, start=1, end=21, mode='valid')
    # generate_regions_all(mode='valid')
    # scale(mode='valid')  # with the same parameters as training set

    # TODO currently its multi-view(pcls from 3 cameras merged to 1)
    # TODO - single-view - export from matlab with generate_cloud_camera()

    # generate_regions_all(mode='valid')

    # with CustomObjectScope({'Kb': Kb}):
    #
    # model = load_model('data/models/10eps_mae_globalscaling_densenoact_lrdrop0.8_separatescaleposes_batchnormsegonly_nodropout.h5')

    # TODO Training
    dataset = 'UBC'

    [pcls_min, pcls_max] = np.load('data/' + dataset + '/train/pcls_minmax.npy')
    [poses_min, poses_max] = np.load('data/' + dataset + '/train/poses_minmax.npy')

    # UBC_convert_region_files(37037, 39, 61, 'train')

    # MHAD_loadpcls()
    # p = np.load('data/MHAD/train/notscaledpcl/010356.npy')
    # visualize_3D(p)

    # scale_poses(mode='test')
    # UBC_convert_pcl_files(index=0, start=1, end=21, mode='test')
    # generate_regions_all(mode='train', data='UBC')
    # scale(mode='train', data='UBC')

    model.compile(optimizer=Adam,
                  loss=losses, loss_weights=[1.0, 0.01],  # 0.1
                  metrics={'output1': avg_error, 'output2': 'accuracy'})

    workers = 8  # mp.cpu_count()

    train_generator = DataGenerator('data/' + dataset + '/train/', numPoints, numJoints, numRegions, steps=None,
                                    batch_size=batch_size,
                                    shuffle=True)
    valid_generator = DataGenerator('data/' + dataset + '/valid/', numPoints, numJoints, numRegions, steps=None,
                                    batch_size=batch_size,
                                    shuffle=False)

    model.fit_generator(generator=train_generator, epochs=10,  # validation_data=valid_generator,
                        callbacks=callbacks_list, initial_epoch=0, use_multiprocessing=True,
                        workers=workers, shuffle=True)
    # save the model
    model.save(
        'data/models/10eps_mae_globalscaling_densenoact_lrdrop0.8_separatescaleposes_batchnormsegonly_nodropout_weights1.02.h5')

    # show an example from train set
    # pcl = np.load('data/train/scaledpclglobal/05504.npy')
    # pose = np.load('data/train/posesglobalseparate/05504.npy')
    # region = np.load('data/train/region/05504.npy')
    # visualize_3D(pcl, regions=region, pose=pose)

    # TODO omit the segmentation sub-network at test time (? dont know how yet)
    # model.outputs = model.outputs[0]

    # [loss, output1_loss, output2_loss, output1_avg_error, output2_acc] = model.evaluate_generator(
    #   valid_generator, use_multiprocessing=True, workers=workers, verbose=1, steps=None)
    #
    # print('avg error: ', output1_avg_error, 'region accuracy: ', output2_acc)
    # predictions = model.predict_generator(valid_generator, use_multiprocessing=True, steps=1, workers=workers,
    #                                       verbose=1)
    # poses = predictions[0]  # output1
    # poses = np.reshape(poses, (poses.shape[0], numJoints, 3))
    # for p in range(25, 27):
    #     poses[p] = unscale_to_cm(poses[p], mode='train')
    # #  TODO try to unscale with validation set params instead (rather not since one-by-one pipeline)
    #     visualize_3D_pose(poses[p], pause=False)
    #     visualize_3D_pose(np.load('data/UBC/valid/notscaledpose/' + str(p).zfill(5) + '.npy'), title='Ground truth',
    #                       pause=False)
    # #  TODO visualize thoroughly the segmentation predictions
    # regions = predictions[1]  # output2
    # regs = np.argmax(regions, axis=-1)
    # for r in range(25, 27):
    #     pcl = np.load('data/UBC/valid/notscaledpcl/' + str(r).zfill(5) + '.npy')
    #     pose = np.load('data/UBC/valid/notscaledpose/' + str(r).zfill(5) + '.npy')
    #     region_gt = np.load('data/UBC/valid/region/' + str(r).zfill(5) + '.npy')
    #     visualize_3D(pcl, regions=region_gt, pose=pose, pause=False, title='ground truth body regions')
    #     visualize_3D(pcl, regions=regs[r], pose=pose, pause=False, title='predicted body regions')
