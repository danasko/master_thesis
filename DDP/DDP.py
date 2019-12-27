from keras.models import Sequential, load_model, Model
from keras.applications import vgg16
from keras.layers.core import Flatten, Dense, Dropout, Reshape
from keras.layers import Input, GaussianNoise, GaussianDropout, AlphaDropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import dot, BatchNormalization, Activation
from keras.optimizers import SGD, Adagrad, Adam, rmsprop
from keras.backend.tensorflow_backend import set_session
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers, models
from keras.utils import plot_model
from keras.utils.generic_utils import get_custom_objects
import keras.callbacks
import tensorflow as tf
import numpy as np
import keras.backend as Kb
from scipy.io import loadmat
from tensorflow import set_random_seed
from keras_contrib.optimizers import padam
from Adam_lr_mult import *
from keras_contrib.callbacks import DeadReluDetector
from visualize_weights import visualize_layer

from data_loader import data_loader
import cv2
from sklearn.utils import shuffle
import pose_visualizer
import k_means_clustering

meanpose, varpose = None, None

prototypes_file = 'data/DDP/prototype_poses.npy'
depth_maps_file = 'data/DDP/depth_maps.npy'
poses_file = 'data/DDP/poses.npy'
prototypes_vis_data_file = 'data/DDP/prototypes_vis_data.npy'
validation_samples_x_file = 'data/DDP/validation_samples_x.npy'
validation_samples_y_file = 'data/DDP/validation_samples_y.npy'
test_data_x_file = 'data/DDP/test_data_x.npy'
test_data_y_file = 'data/DDP/test_data_y.npy'
# mean_depth_file = 'data/DDP/mean_depth.npy'

# predictions_file = 'data/DDP/predictions_leak_128b_alpha0_3.npy'
# predictions_file2 = 'data/DDP/predictions_leak_dropouteach_evenlowestalpha.npy'
# C = None  # set of K prototypes (after clustering) - matrix
input_size = 100
batch_size = 64  # 1, 7, 257, 1799

# Hyperparameters:

# alpha = 0.08  # ITOP dataset 0.08
alpha = 0.1  # UBC3V dataset 0.01
# J = 15  # no. of joints ITOP dataset
J = 18  # no. of joints UBC3V dataset
K = 100  # no. of prototype poses
# o = 1.  # ITOP dataset
o = 0.8  # UBC3V dataset
thresh = 0.1  # for mean average precision (in meters)

# input depth maps size
# ITOP
# width = 320
# height = 240

# UBC3V
width = 512
height = 424

# Set random seed to make results reproducable
np.random.seed(21)
set_random_seed(21)


def vgg_custom():
    img_input = Input(shape=(input_size, input_size, 1))
    # Block 1
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_conv1')(x)
    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv1')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv2')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv1')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(K, activation='linear', name='predictions')(x)

    return models.Model(img_input, x, name='vgg_custom')


def DPP(weights=None):
    model = Sequential()
    # model.add(ZeroPadding2D((1, 1), )
    # glorot = keras.initializers.glorot_normal(seed=1)  # 1024
    model.add(Conv2D(filters=96, kernel_size=(7, 7), strides=(1, 1), input_shape=(input_size, input_size, 1),
                     kernel_initializer='glorot_normal', data_format="channels_last",
                     bias_initializer='zeros', activation='relu'
                     ))  # activation='relu'

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(
        Conv2D(filters=192, kernel_size=(5, 5), strides=(1, 1), kernel_initializer='glorot_normal',
               bias_initializer='zeros', activation='relu'
               ))  # strides=(2,2)

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(
        Conv2D(filters=512, kernel_size=(4, 4), strides=(1, 1), kernel_initializer='glorot_normal',
               bias_initializer='zeros', activation='relu'
               ))  # kernel_size=(3,3)
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # model.add(Dropout(0.4))

    # 6th CONV block
    model.add(
        Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='glorot_normal',
               bias_initializer='zeros', activation='relu'))  # kernel_size=(3,3)
    # model.add(BatchNormalization(momentum=0.9))
    # model.add(LeakyReLU(alpha=0.3))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2))) # TODO  commentout and add layers/increase kernel size
    # model.add(Dropout(0.4))
    ####

    # 7th CONV block
    model.add(
        Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='glorot_normal',
               bias_initializer='zeros', activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # model.add(Dropout(0.4))
    ####

    model.add(
        Conv2D(filters=1024, kernel_size=(2, 2), strides=(1, 1), kernel_initializer='glorot_normal',
               bias_initializer='zeros', activation='relu'
               ))  # strides=(2,2)

    # model.add(Dropout(0.4))

    model.add(
        Conv2D(filters=2048, kernel_size=(2, 2), strides=(1, 1), kernel_initializer='glorot_normal',
               bias_initializer='zeros',
               activation='relu'
               ))  # kernel_regularizer=keras.regularizers.l1(0.001)
    # model.add(Dropout(0.4))

    model.add(Flatten())
    # model.add(keras.layers.GlobalMaxPooling2D())  # instead of flatten
    model.add(Dense(1024, kernel_initializer='glorot_normal', bias_initializer='zeros', activation='relu'
                    # activity_regularizer=regularizers.l1(0.0001)
                    ))

    model.add(Dropout(0.4))  # 0.2

    model.add(Dense(256, kernel_initializer='glorot_normal',
                    bias_initializer='zeros', activation='relu'))  # kernel_regularizer=keras.regularizers.l2()
    # model.add(BatchNormalization(momentum=0.9))

    model.add(
        Dense(K, name='output', kernel_initializer='glorot_normal', bias_initializer='zeros'))

    if weights:
        model.load_weights(weights)

    return model


def loss_function(y_true, y_pred):
    # print(y_pred.shape)  # (batch_size, K)
    # print(y_true.shape)  # (15, batch_size, 3) ..should be.. but is actually (?, ?)

    # y = Kb.transpose(y_pred)[0]  # (batch_size,)
    # y = Kb.expand_dims(y, -1)  # (batch_size, 1)

    # p_e = Kb.constant(1, shape=(K, 1, J * 3))
    #
    # p = Kb.dot(y, p_e)  # (batch_size, K, J*3)
    # p = Reshape(target_shape=(K, J, 3))(p)
    # p = Kb.permute_dimensions(p, (0, 2, 1, 3))  # (batch_size, J, K, 3)

    # tc = Kb.ones_like(p) * C  # c_scaled

    # tc = Kb.permute_dimensions(tc, (0, 1, 3, 2))
    # tc = Reshape(target_shape=(J * 3, K))(tc)

    # c_pred = Kb.batch_dot(y_pred, tc, axes=[1, 2])  # p^

    # fixed
    # y_pred = y_pred * 2 - 1  # scale to range (-1,1)
    # y = Kb.stack([y_pred] * (J * 3), axis=-1)  # (batch, K, J*3)
    # yt = Kb.permute_dimensions(y, [0, 2, 1])  # same weight for each joint in particular prototype in column

    # normalize
    tci = Kb.transpose(tc) - meanpose
    tci = Kb.transpose(tci / np.sqrt(varpose))
    #######

    c_pred = y_pred @ Kb.transpose(tci)
    # c_pred = yt * tci  # (batch_size, J*3, K) # tci !!
    # c_pred = Kb.sum(c_pred, axis=-1)

    # back to real center and scale
    c_pred = c_pred * np.sqrt(varpose)
    c_pred = c_pred + meanpose
    # #######

    # # c_pred = Reshape(target_shape=(J, 3))(c_pred)  # expand last dimension back into two -> 15, 3
    y_true = Reshape(target_shape=(J * 3,))(y_true)  # flatten last two dimensions (15,3) into one

    p1 = (1 - alpha) * res_loss(c_pred, y_true)  # (batch_size,)
    reg_term = alpha * L2(y_pred)  # (batch_size,)
    return p1 + reg_term


def L1(v):  # L1 norm of ndarray v  # v of shape (batch_size, 70)
    return Kb.sum(Kb.abs(v), axis=-1)


def L2(v):
    return Kb.sqrt(Kb.sum(Kb.square(v), axis=-1))


def res_loss(pred, gtrue):
    r = gtrue - pred  # shape of both (batch, J*3)

    p = Kb.abs(r)
    # p = Kb.abs(r) + (0.5 / (o ** 2))

    # logic = Kb.less(Kb.abs(r), 1 / (o ** 2))
    # p = Kb.switch(logic, 0.5 * (o ** 2) * Kb.square(r), Kb.abs(r) - (0.5 / (o ** 2)))

    return Kb.sum(p, axis=-1)


def column(matrix, i):
    return np.transpose(matrix, (1, 0, 2))[i]


# def preprocess(depth_maps):
#     res = np.zeros((depth_maps.shape[0], input_size, input_size))
#     half = input_size // 2
#     center_y = round(depth_maps[0].shape[0] // 2)
#     center_x = round(depth_maps[0].shape[1] // 2)
#
#     for i in range(depth_maps.shape[0]):
#         res[i] = depth_maps[i][center_y - half: center_y + half,
#                  center_x - half:center_x + half]  # crop width to 100 around center
#         res[i] = cv2.normalize(res[i], None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#         # res[i] = np.array(res[i], dtype=np.float32)
#         # img = np.array(res[i] * 255, dtype=np.uint8)  # for visualization
#         # cv2.imshow("img", img)
#         # cv2.waitKey(0)
#         res[i] = rescale(res[i], 0, 1)
#
#     return res


def rescale(x, lw, up):
    return lw + ((x - Kb.min(x, axis=-1)) / (Kb.max(x, axis=-1) - Kb.min(x, axis=-1))) * (up - lw)


def random_prototype_poses(train_poses, img_coords=None):
    if img_coords is not None:
        [prototypes, img_coords] = shuffle(train_poses, img_coords)
        img_coords = img_coords[:K]

    else:
        prototypes = shuffle(train_poses)

    prototypes = prototypes[:K]
    prototypes = np.transpose(prototypes, (1, 0, 2))
    res = [prototypes, img_coords]  # each column represents one pose
    return res


def training_generator(files, label_files, batch_size=batch_size):
    while True:
        # Select files (paths/indices) for the batch
        idx = np.random.randint(0, files.shape[0] - batch_size)
        batch_x = files[idx: idx + batch_size]
        batch_y = label_files[idx: idx + batch_size]
        # batch_x = np.expand_dims(batch_x, -1)
        # batch_y = np.asarray(batch_y)
        yield batch_x, batch_y


def train_valid_split(train_x, train_y, vis_data, test_size):
    val_size = round(len(train_x) * test_size)
    new_train_x = train_x[:-1 - val_size]
    new_train_y = train_y[:-1 - val_size]

    new_vis_data = vis_data[:-1 - val_size]

    valid_x = train_x[-1 - val_size:-1]
    valid_y = train_y[-1 - val_size:-1]
    return [[new_train_x, new_train_y], [valid_x, valid_y], new_vis_data]


def validation_generator(files_x, files_y, batch_size=batch_size):
    while True:
        idx = np.random.randint(0, files_x.shape[0] - batch_size)
        batch_x = files_x[idx: idx + batch_size]
        batch_y = files_y[idx: idx + batch_size]
        # batch_x = np.expand_dims(batch_x, -1)
        batch_y = np.asarray(batch_y)
        yield batch_x, batch_y


def avg_error(y_true, y_pred):
    # y_true = unscale(y_true)
    # y_pred = unscale(y_pred)
    # y_pred = y_pred * 2 - 1

    # y = Kb.stack([y_pred] * (J * 3), axis=-1)
    # yt = Kb.permute_dimensions(y, [0, 2, 1])  # same weight for each joint in particular prototype in column

    # normalize
    tci = Kb.transpose(tc) - meanpose
    tci = Kb.transpose(tci / np.sqrt(varpose))
    #######

    # c_pred = yt * tci  # (batch_size, J*3, K) # tci
    # c_pred = Kb.sum(c_pred, axis=-1)
    c_pred = y_pred @ Kb.transpose(tci)

    # back to real center and scale
    c_pred = c_pred * np.sqrt(varpose)
    c_pred = c_pred + meanpose
    #######

    c_pred = Reshape(target_shape=(J, 3))(c_pred)  # expand last dimension back into two -> 15, 3
    # c_pred = unscale(c_pred)
    # c_pred = Reshape(target_shape=(J, 3))(y_pred)  # baseline model

    # Euclidean distance between predicted and ground-truth pose
    return Kb.mean(Kb.sqrt(Kb.sum(Kb.square(c_pred - y_true), axis=-1)), axis=-1)


def meanAveragePrecision(y_true, y_pred):
    # y_true = unscale(y_true)
    # y_pred = unscale(y_pred)
    # y_true = unscale(y_true)
    # y_pred = y_pred * 2 - 1

    # y = Kb.stack([y_pred] * (J * 3), axis=-1)
    # yt = Kb.permute_dimensions(y, [0, 2, 1])  # same weight for each joint in particular prototype in column

    # normalize
    tci = Kb.transpose(tc) - meanpose
    tci = Kb.transpose(tci / np.sqrt(varpose))
    #######

    # c_pred = yt * tci  # (batch_size, J*3, K) # tci
    # c_pred = Kb.sum(c_pred, axis=-1)
    c_pred = y_pred @ Kb.transpose(tci)

    # back to real center and scale
    c_pred = c_pred * np.sqrt(varpose)
    c_pred = c_pred + meanpose
    #######

    c_pred = Reshape(target_shape=(J, 3))(c_pred)  # expand last dimension back into two -> 15, 3
    # c_pred = Reshape(target_shape=(J, 3))(y_pred)  # baseline model
    # c_pred = unscale(c_pred)
    dist = Kb.sqrt(Kb.sum(Kb.square(c_pred - y_true), axis=-1))  # tensor of distances between joints pred and gtrue

    logic = Kb.less_equal(dist, thresh)

    res = Kb.switch(logic, Kb.ones_like(dist), Kb.zeros_like(dist))  # 1 if estimated correctly, else 0

    return Kb.mean(Kb.sum(res, axis=-1) / J, axis=-1)


def get_random_batch(files, label_files, batch_size=batch_size):
    idx = np.random.randint(0, files.shape[0] - batch_size)
    batch_x = files[idx: idx + batch_size]
    batch_y = label_files[idx: idx + batch_size]
    # batch_x = np.expand_dims(batch_x, -1)
    batch_y = np.asarray(batch_y)
    return batch_x, batch_y


def getfullpose(prediction):
    t = C
    # normalize
    t = np.transpose(np.transpose(t) - meanpose)
    t = np.transpose(np.transpose(t) / np.sqrt(varpose))

    pose = np.sum(prediction * t, axis=-1)

    # back to real center and scale
    pose = pose * np.sqrt(varpose)
    pose = pose + meanpose
    #######
    pose = np.reshape(pose, (15, 3))
    pose_visualizer.visualize_pose_2D(pose, pause=False, title='predicted pose', isnumpy=True)


def show_poses(preds, groundtruth, start):
    # t = Kb.permute_dimensions(C, (0, 2, 1))
    # t = Kb.reshape(t, (J * 3, K))
    t = C
    # normalize
    t = np.transpose(np.transpose(t) - meanpose)
    t = np.transpose(np.transpose(t) / np.sqrt(varpose))
    #######

    for idx in range(round(preds.shape[0] / 50)):  # predictions.shape[0]
        pose = np.sum(preds[idx * 50] * t, axis=-1)

        # back to real center and scale
        pose = pose * np.sqrt(varpose)
        pose = pose + meanpose
        #######

        # pose2 = Kb.sum(predictions2[idx*50] * t, axis=-1)
        # in 3D
        # pose = Kb.transpose(Kb.reshape(pose, (15, 3)))
        # pose_visualizer.visualize_pose_3D(pose, pause=False)
        # gt = groundtruth[start + (idx * 50)]
        # gt = Kb.transpose(gt)
        # pose_visualizer.visualize_pose_3D(gt, pause=True)
        # in 2D
        pose = np.reshape(pose, (15, 3))
        # pose2 = Kb.reshape(pose2, (15, 3))
        pose_visualizer.visualize_pose_2D(pose, pause=False, title='predicted pose', isnumpy=True)
        # pose_visualizer.visualize_pose_2D(pose2, pause=False, title='predicted pose a=0.000001')
        pose_visualizer.visualize_pose_2D(groundtruth[start + (idx * 50)], pause=True, isnumpy=True,
                                          title='ground truth')
        #     # depth map
        #     # depth_map = cv2.resize(test_data_x[start + idx], (244, 244))
        #     # img = np.array(depth_map * 255, dtype=np.uint8)  # for visualization
        #     # cv2.imshow('depth map', img)
        #     # print(idx)
        #     # cv2.waitKey(0)
        #
        #     # most relevant prototype poses #######
        #
        #     # sorted_indices = np.argsort(np.abs(predictions[idx]))
        #     # for p in range(5):
        #     #     pp = column(C, sorted_indices[-1-p])
        #     #     print(predictions[idx])
        #     #     if predictions[idx][sorted_indices[-1-p]] < 0:
        #     #         sign = "neg"
        #     #     else:
        #     #         sign = "pos"
        #     #
        #     #     pose_visualizer.visualize_pose_2D(pp, pause=True, isnumpy=True, title=str(p+1)+". most relevant - "+sign)


def normalize_meanstd(a, axis=None):
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean) ** 2).mean(axis=axis, keepdims=True))
    return (a - mean) / std


# # learning rate schedule
# def step_decay(epoch):
#     initial_lrate = 0.001
#     drop = 0.08
#     epochs_drop = 10.0
#     lrate = initial_lrate * (1. / (1. + drop * (epoch // epochs_drop)))
#     # lrate = initial_lrate * (1. / (1. + np.power(drop, np.floor((1 + epoch) / epochs_drop))))
#     return lrate


if __name__ == "__main__":
    Kb.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    # ITOP DATASET ########################################################
    #
    # train_dataset = data_loader('D:/skola/master/datasets/ITOP/depth_maps/ITOP_side_train_depth_map.h5',
    #                             'D:/skola/master/datasets/ITOP/labels/ITOP_side_train_labels.h5')
    #
    # # new_width = round(width / (height / input_size))
    # [D, P, posemin, posemax] = train_dataset.get_data(0, 0, input_size)
    #
    # # train_data_x = preprocess(train_data_x)  # depth maps (resized to input_size x input_size (crop),/preprocessed)
    #
    # # [train_samples, validation_samples, vis_data] = train_valid_split(train_data_x, train_data_y, vis_data,
    # #                                                                   test_size=0.25)
    #
    # # D = train_samples[0]
    # # P = train_samples[1]  # ground truth poses - normalized joint coords (zero mean //(and one standard deviation))
    #
    # D = np.expand_dims(D, -1)
    # meanpose = np.mean(P, axis=0)
    # varpose = np.var(P, axis=0)
    # meanpose = meanpose.flatten()
    # varpose = varpose.flatten()

    # TEST ######################################################

    # test_dataset = data_loader('D:/skola/master/datasets/ITOP/depth_maps/ITOP_side_test_depth_map.h5',
    #                            'D:/skola/master/datasets/ITOP/labels/ITOP_side_test_labels.h5')
    # # new_width = round(width / (height / input_size))
    # [test_data_x, test_data_y, _, _] = test_dataset.get_data(posemin, posemax, input_size, mode='test')
    # # test_data_x = preprocess(test_data_x)
    # test_data_x = np.expand_dims(test_data_x, -1)
    #
    # # normalize input data ##########################################
    #
    # mean_depth = np.mean(D, axis=0, keepdims=True)
    # D = D - mean_depth
    # test_data_x = test_data_x - mean_depth

    #################################################################

    D = np.load(depth_maps_file)
    P = np.load(poses_file)
    C = np.load(prototypes_file)

    # C = loadmat('data/DDP/centroids.mat')['sc']  # (45,70)

    test_data_x = np.load(test_data_x_file)
    test_data_y = np.load(test_data_y_file)
    meanpose = np.load('data/DDP/meanpose.npy')
    varpose = np.load('data/DDP/varpose.npy')
    # shuffle(D, P, random_state=10)
    # print(P.min(axis=(0, 1)))
    # print(P.max(axis=(0, 1)))
    # C = k_means_clustering.cluster_prototypes(np.concatenate((P, test_data_y)),
    #                                           K)  # [C, prototypes_vis_data] = random_prototype_poses(P, vis_data)
    tc = C

    # tc = Kb.permute_dimensions(C, [0, 2, 1])
    # tc = Kb.reshape(tc, (J * 3, K))

    # shuffle prototype poses
    # C = C[:, np.random.permutation(C.shape[1])]

    # prototypes_vis_data = np.load(prototypes_vis_data_file)  # deprecated
    # validation_samples = [np.load(validation_samples_x_file), np.load(validation_samples_y_file)]

    # np.save('data/DDP/meanpose.npy', meanpose)
    # np.save('data/DDP/varpose.npy', varpose)
    # # np.save('data/DDP/poseminmax.npy', [posemin, posemax])
    # np.save(depth_maps_file, D)
    # np.save(poses_file, P)
    # np.save(prototypes_file, C)
    # # np.save(prototypes_vis_data_file, prototypes_vis_data)
    # # np.save(validation_samples_x_file, validation_samples[0])
    # # np.save(validation_samples_y_file, validation_samples[1])
    # np.save(test_data_x_file, test_data_x)
    # np.save(test_data_y_file, test_data_y)

    # MODEL ##########################################################

    model = DPP()
    # model = vgg_custom()
    #
    # # adagrad = Adagrad(lr=0.001, decay=0.00002)
    # sgd = SGD(lr=0.001, decay=0.0005)
    # pdm = padam.Padam(lr=0.001, decay=0.00001, beta_1=0.8, beta_2=0.9)
    # rms = rmsprop(lr=0.001, rho=0.9, epsilon=None, decay=0.001)
    multipliers = {'conv2d_1': 1, 'conv2d_2': 1, 'conv2d_3': 1, 'conv2d_4': 1, 'conv2d_5': 1, 'conv2d_6': 1,
                   'conv2d_7': 1, 'dense_1': 1,
                   'dense_2': 1, 'output': 0.1}

    # adam = Adam(lr=0.0001, decay=0.000001, epsilon=0.0000000001)  # 0.0008 # 0.00088 # 0.002 # epsilon=0.0000000001

    adam_lr_mult = Adam_lr_mult(multipliers=multipliers, lr=0.0001, decay=0.000001, epsilon=0.0000000001)
    model.compile(loss=loss_function, optimizer=adam_lr_mult, metrics=[avg_error, meanAveragePrecision],
                  target_tensors=[Kb.placeholder((None, 15, 3))])

    # model.summary()
    plot_model(model, to_file='D:/skola/master/prehlady/screenshots/DDP/model_plot_final.png', show_shapes=True,
               show_layer_names=True)
    ##################################################################

    # num_training_samples = D.shape[0]
    # num_validation_samples = validation_samples[0].shape[0]

    # data_generator = training_generator(D, P, batch_size)
    # valid_generator = validation_generator(test_data_x, test_data_y, batch_size)

    get_custom_objects().update(
        {"loss_function": loss_function, "avg_error": avg_error, "meanAveragePrecision": meanAveragePrecision,
         "Adam_lr_mult": Adam_lr_mult})

    # model = load_model('data/DDP/model_training_64b_6dropouts0.4_7conv_6convnopool_6_7_filter3x3.h5')

    tbCallBack = keras.callbacks.TensorBoard(log_dir='data/DDP/Graph/leak', histogram_freq=0,
                                             write_graph=True,
                                             write_images=True, write_grads=False, batch_size=512)

    earlyStop = keras.callbacks.EarlyStopping(monitor='val_avg_error', min_delta=0, patience=7, verbose=0, mode='min',
                                              baseline=None,
                                              restore_best_weights=True)

    # print(model.targets)
    # tbCallBack.set_model(model)

    # tbwrapcallback = TensorBoardWrapper(validation_generator(test_data_x, test_data_y, batch_size=32), nb_steps
    # =100, log_dir='data/DDP/Graph/leak', histogram_freq=1,
    #                                     batch_size=batch_size, write_graph=True, write_grads=True)

    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                                     patience=5, min_lr=0.0001)

    # deadReluCheck = DeadReluDetector(D, verbose=True)

    # lrate = keras.callbacks.LearningRateScheduler(step_decay)

    # D, P = shuffle(D, P, random_state=3)
    #
    # train_history = model.fit(D, P, epochs=50, batch_size=batch_size,
    #                           validation_split=0.2,  # 0.02
    #                           # validation_data=(test_data_x, test_data_y),
    #                           callbacks=[tbCallBack], initial_epoch=0, shuffle=True)  # , earlyStop
    #
    # # np.save('data/DDP/train_history_cluster_fixednorm.npy', train_history)
    # model.save('data/DDP/model_training_64b_6dropouts0.4_7conv_6convnopool_6_7_filter3x3_50eps.h5')

    # for layer in model.layers:
    #     print(layer.get_config())

    # if len(model.layers[1].get_weights()) > 0:
    #     print('weights: ')
    # print(model.layers[1].get_weights()[0])
    # print('biases: ')
    # print(model.layers[1].get_weights()[1])

    # visualize_layer(model, 'conv2d_1')

    start = 0

    # predictions = model.predict(D[start:])
    # predictions_tst = model.predict(test_data_x)
    # predictions = model.predict(test_data_x[start:])

    # postprocess - get full pose

    # np.save(predictions_file, predictions)

    # predictions = np.load(predictions_file)
    # predictions2 = np.load(predictions_file2)
    # #
    # score = model.evaluate(D, P, batch_size=batch_size)
    # score2 = model2.evaluate(test_data_x, test_data_y, batch_size=batch_size)
    # print('Test loss:', score[0], ' ', score2[0])  # 0.3152455667544362 # 0.3125105603790911 #0.14...
    # print('Test average error:', score[1], ' ', score2[1])  # 0.07282457308264982 # 0.07163713314290304 #0.068...

    # print('Test loss:', score[0])
    # print('Test average error:', score[1])
    # #
    # synth_prediction = [-1,  0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                     0, 0, 0, 0, 0, 0, 0, 0]

    # synth_prediction = [1] + [0] * 29
    # # # visualize predictions
    #

    # pose = Kb.sum(synth_prediction * t, axis=-1)
    # pose = Kb.reshape(pose, (15, 3))
    # pose_visualizer.visualize_pose_2D(pose, pause=True)

    # show_poses(predictions, test_data_y, start)

    #####################################################################

    # visualization ##################################################
    # 2D poses

    # for p in prototypes_vis_data:  # not updated prototypes
    #     pose_visualizer.visualize_pose(p)  # [Press enter]

    # 3D prototype poses

    # 3D
    # cp = Kb.permute_dimensions(C, (1, 2, 0))
    # pose_visualizer.visualize_pose_3D(cp[10], pause=False)

    # 2D
    # cp = Kb.permute_dimensions(C, (1, 0, 2))
    # #
    # for p in range(cp.shape[0]):
    #         pose_visualizer.visualize_pose_3D(cp[p], pause=True)  # [Press Enter]
    #     pose_visualizer.visualize_pose_2D(cp[p], pause=True)

    # train data

    # [dp, pp1] = get_random_batch(D, P, batch_size=10)
    # # pp1 = Kb.constant(pp1)
    # # pp1 = Kb.permute_dimensions(pp1, (0, 2, 1))
    # for i in range(dp.shape[0]):
    #     # pose_visualizer.visualize_pose_2D(pp1[i], pause=False, isnumpy=True, title='train data')
    #     # depth_map = cv2.resize(dp[i], (244, 244))
    #     depth_map = np.array(dp[i] * 255)  # for visualization
    #     cv2.imshow('depth map', depth_map)
    #     cv2.waitKey(0)  # [Press any key]

    # joint coords area
    # pose_visualizer.visualize_pose_2D(P, pause=False, array=True, isnumpy=True, title='train data')

    # test data

    # test_data_x, test_data_y = shuffle(test_data_x, test_data_y)
    # [dp, pp2] = get_random_batch(test_data_x, test_data_y, batch_size=1000)

    # t = Kb.transpose(Kb.transpose(tc) - meanpose)
    # t = Kb.transpose(Kb.transpose(t) / np.sqrt(varpose))
    #
    # dp, pp2 = test_data_x, predictions
    # # pp2 = Kb.constant(pp2)
    # # pp2 = Kb.permute_dimensions(pp2, (0, 2, 1))
    # for p in range(dp.shape[0]):
    #     pose = Kb.sum(pp2[p] * t, axis=-1)
    #     # back to real center and scale
    #     pose = pose * np.sqrt(varpose)
    #     pose = pose + meanpose
    #     pose = Kb.reshape(pose, (15, 3))
    #     pose_visualizer.visualize_pose_2D(pose, pause=False, title='test data')
    #     pose_visualizer.visualize_pose_2D(test_data_y[p], isnumpy=True, pause=False, title='test data gt')
    #     depth_map = cv2.resize(dp[p], (244, 244))
    #     depth_map = np.array(depth_map * 255)  # for visualization
    #     cv2.imshow('depth map', depth_map)
    #     cv2.waitKey(0)  # [Press any key]

    # joint coords area
    # pose_visualizer.visualize_pose_2D(test_data_y, pause=False, isnumpy=True, array=True, title='test data')
