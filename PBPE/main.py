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
# from tensorflow import set_random_seed
# from Adam_lr_mult import *
# from keras_contrib.callbacks import DeadReluDetector
# from visualize_weights import visualize_layer
from sklearn.utils import shuffle
import logging
import h5py

from preprocess import *
from visualizer import *
from data_generator import *
from data_loader import *
from config import *
from ITOP_data_loader import load_ITOP_from_npy


# keras.backend.set_floatx('float64')


def tile(global_feature, numPoints):
    return Kb.repeat_elements(global_feature, numPoints, 1)


def my_model(poolTo1=False, globalAvg=True):
    input_points = Input(shape=(numPoints, 1, 4))
    local_feature1 = Conv2D(filters=512, kernel_size=(1, 1), input_shape=(numPoints, 1, 4),
                            kernel_initializer='glorot_normal',
                            activation='relu')(input_points)
    local_feature2 = Conv2D(filters=1024, kernel_size=(1, 1),
                            activation='relu', kernel_initializer='glorot_normal')(local_feature1)
    local_feature3 = Conv2D(filters=2048, kernel_size=(1, 1),
                            activation='relu', kernel_initializer='glorot_normal')(local_feature2)
    local_feature1_exp = Conv2D(filters=2048, kernel_size=(1, 1))(
        local_feature1)  # TODO try res connection wo global (wo maxpool)
    local_feature2_exp = Conv2D(filters=2048, kernel_size=(1, 1))(
        local_feature1)  # TODO try res connection wo global (wo maxpool)
    shortcut1 = keras.layers.add([local_feature1_exp, local_feature2_exp, local_feature3])  # add

    # res1 = keras.layers.Activation('relu')(res1)  # a bez tohto
    if poolTo1:
        shortcut1 = MaxPooling2D(pool_size=(2048, 1))(shortcut1)

    # local_feature2_exp = Conv2D(filters=2048, kernel_size=(1, 1))(local_feature2)
    # global_exp = Lambda(tile, arguments={'numPoints': 128})(
    #     global_feature)
    # global_feature = keras.layers.Activation('relu')(global_feature)
    f1 = Conv2D(filters=512, kernel_size=(1, 1),
                kernel_initializer='glorot_normal')(shortcut1)
    f1a = keras.layers.Activation('relu')(f1)
    # f = Conv2D(filters=256, kernel_size=(2, 1),
    #                         activation='relu', kernel_initializer='glorot_normal')(f)
    f2 = Conv2D(filters=256, kernel_size=(1, 1), kernel_initializer='glorot_normal')(f1a)
    # f2 = Conv2D(filters=512, kernel_size=(1, 1))(f2)
    # shortcut2 = keras.layers.add([f2, f1])
    f2 = keras.layers.Activation('relu')(f2)
    # f2 = Conv2D(filters=512, kernel_size=(1, 1))(f2)
    # res2 = keras.layers.concatenate([f1, f2])
    # f = MaxPooling2D(pool_size=(15, 1))(f)
    #  strides  # shape= (b, 1, 1, 2048)
    # global_feature_exp = Lambda(tile, arguments={'numPoints': 2046})(
    #     global_feature)  # shape= (b, numPoints=2048, 1, 2048)
    # f = concatenate([local_feature2, local_feature3, global_feature_exp], axis=-1)
    # f = Conv2D(filters=256, kernel_size=(1,1), activation='relu', kernel_initializer='glorot_normal')(global_feature)

    if globalAvg:
        f = keras.layers.GlobalAveragePooling2D()(f2)
    else:
        f = Flatten()(f2)

    f = Dense(512, kernel_initializer='glorot_normal', activation='relu')(f)
    f = Dense(256, kernel_initializer='glorot_normal', activation='relu')(f)
    # output1 = Dense(k, name='output1', activation='softmax', kernel_initializer='glorot_normal')(f)
    output1 = Dense(numJoints * 3, name='output1', kernel_initializer='glorot_normal')(f)

    model = Model(inputs=input_points, outputs=output1)
    return model


def seg_net():
    input_points = Input(shape=(numPoints, 1, 3))
    local_feature1_noact = Conv2D(filters=1024, kernel_size=(1, 1), input_shape=(numPoints, 1, 3),  # 512
                                  kernel_initializer='glorot_normal')(input_points)

    # shortcut1_1 = keras.layers.concatenate([local_feature1, input_points])

    local_feature1 = keras.layers.Activation('relu')(local_feature1_noact)

    # local_feature1 = BatchNormalization(momentum=0.9)(local_feature1)

    local_feature2 = Conv2D(filters=1024, kernel_size=(1, 1),
                            kernel_initializer='glorot_normal')(local_feature1)

    shortcut1_2 = keras.layers.add([local_feature1_noact, local_feature2])  # input_points

    local_feature2 = keras.layers.Activation('relu')(shortcut1_2)

    # local_feature2 = BatchNormalization(momentum=0.9)(local_feature2)

    local_feature3 = Conv2D(filters=1024, kernel_size=(1, 1),  # 2048
                            kernel_initializer='glorot_normal')(local_feature2)

    shortcut1_3 = keras.layers.add([local_feature1_noact, local_feature3])  # input_points

    local_feature3 = keras.layers.Activation('relu')(shortcut1_3)

    # d = Dropout(0.2)(local_feature3)

    # local_feature3 = BatchNormalization(momentum=0.9)(local_feature3)

    # local_feature4 = Conv2D(filters=2048, kernel_size=(1, 1),
    #                         activation='relu', kernel_initializer='glorot_normal')(local_feature3)

    global_feature = MaxPooling2D(pool_size=(numPoints, 1))(local_feature3)  # strides  # shape= (b, 1, 1, 2048)
    global_feature_exp = Lambda(tile, arguments={'numPoints': numPoints})(
        global_feature)  # shape= (b, numPoints=2048, 1, 2048)

    # Auxiliary part-segmentation network - only for training - removed at test time

    c = keras.layers.concatenate([global_feature_exp, local_feature1, local_feature2, local_feature3])

    conv1 = Conv2D(filters=512, kernel_size=(1, 1), kernel_initializer='glorot_normal')(c)  # 256

    c = keras.layers.Activation('relu')(conv1)

    c = BatchNormalization(momentum=0.9)(c)

    c = Dropout(0.2)(c)

    c = Conv2D(filters=512, kernel_size=(1, 1), kernel_initializer='glorot_normal')(c)  # 256

    shortcut2_1 = keras.layers.add([conv1, c])

    c = keras.layers.Activation('relu')(shortcut2_1)

    c = BatchNormalization(momentum=0.9)(c)

    c = Dropout(0.2)(c)

    c = Conv2D(filters=512, kernel_size=(1, 1), kernel_initializer='glorot_normal')(c)  # 256

    shortcut2_2 = keras.layers.add([shortcut2_1, c])

    c = keras.layers.Activation('relu')(shortcut2_2)

    c = BatchNormalization(momentum=0.9)(c)

    output = Conv2D(numRegions, (1, 1), activation='softmax', kernel_initializer='glorot_normal')(c)

    return Model(inputs=input_points, outputs=output)


def loss_func(y_true, y_pred):
    # y_pred shape = (?, k), clusters shape = (k, numJoints, 3)
    clusters = centers.reshape(k, numJoints * 3)
    preds = y_pred @ clusters
    # preds = Reshape((numJoints, 3))(preds)  # shape = (?, numJoints, 3)
    return Kb.mean(Kb.abs(y_true - preds), axis=-1)  # MAE


def PBPE_new():
    input_points = Input(shape=(numPoints, 1, 3))
    local_feature1 = Conv2D(filters=512, kernel_size=(1, 1), input_shape=(numPoints, 1, 3),
                            kernel_initializer='glorot_normal',
                            activation='relu')(input_points)
    local_feature2 = Conv2D(filters=1024, kernel_size=(1, 1),
                            activation='relu', kernel_initializer='glorot_normal')(local_feature1)
    local_feature3 = Conv2D(filters=2048, kernel_size=(1, 1),
                            activation='relu', kernel_initializer='glorot_normal')(local_feature2)
    # local_feature4 = Conv2D(filters=2048, kernel_size=(1, 1),
    #                         activation='relu', kernel_initializer='glorot_normal')(local_feature3)

    global_feature = MaxPooling2D(pool_size=(numPoints, 1))(
        local_feature3)  # strides  # shape= (b, 1, 1, 2048)
    global_feature_exp = Lambda(tile, arguments={'numPoints': numPoints})(
        global_feature)  # shape= (b, numPoints=2048, 1, 2048)

    f = Flatten(dtype=np.float64)(global_feature)
    f = Dense(256, kernel_initializer='glorot_normal', activation='relu')(f)
    f = Dense(256, kernel_initializer='glorot_normal', activation='relu')(f)

    # f = Dropout(0.2)(f)

    output1 = Dense(3 * numJoints, name='output1')(f)

    # Auxiliary part-segmentation network - only for training - removed at test time

    c = concatenate([global_feature_exp, local_feature1, local_feature2, local_feature3],
                    axis=-1)  # TODO try concatenating only last two local feats

    c = Conv2D(filters=256, kernel_size=(1, 1),
               activation='relu', kernel_initializer='glorot_normal')(c)

    c = BatchNormalization(momentum=0.9)(c)

    c = Dropout(0.2)(c)

    c = Conv2D(filters=256, kernel_size=(1, 1),
               activation='relu', kernel_initializer='glorot_normal')(c)

    c = BatchNormalization(momentum=0.9)(c)

    c = Dropout(0.2, dtype=np.float64)(c)

    c = Conv2D(filters=128, kernel_size=(1, 1),
               activation='relu', kernel_initializer='glorot_normal')(c)

    c = BatchNormalization(momentum=0.9)(c)

    output2 = Conv2D(numRegions, (1, 1), activation='softmax', kernel_initializer='glorot_normal',
                     name='output2')(c)

    model = Model(inputs=input_points, outputs=[output1, output2])
    test_model = Model(inputs=input_points, outputs=output1)
    return model, test_model


def PBPE():
    input_points = Input(shape=(numPoints, 1, 3))
    local_feature1 = Conv2D(filters=512, kernel_size=(1, 1), input_shape=(numPoints, 1, 3),
                            kernel_initializer='glorot_normal',
                            activation='relu')(input_points)
    # local_feature1 = LeakyReLU()(local_feature1)
    # local_feature1 = BatchNormalization(momentum=0.9)(x)  # momentum=0.9  # TODO batchnorm
    local_feature2 = Conv2D(filters=2048, kernel_size=(1, 1),
                            activation='relu', kernel_initializer='glorot_normal')(local_feature1)

    # local_feature2 = BatchNormalization(momentum=0.9)(x)  # shape = (b, numPoints=2048, 1, 2048)  # TODO batchnorm

    global_feature = MaxPooling2D(pool_size=(numPoints, 1))(
        local_feature2)  # strides  # shape= (b, 1, 1, 2048)
    global_feature_exp = Lambda(tile, arguments={'numPoints': numPoints})(
        global_feature)  # shape= (b, numPoints=2048, 1, 2048)

    f = Flatten()(global_feature)
    f = Dense(256, kernel_initializer='glorot_normal')(f)  # todo dense with activation
    # f = BatchNormalization(momentum=0.9)(f)  # TODO batchnorm

    f = Dense(256, kernel_initializer='glorot_normal')(f)  # todo dense with activation
    # f = BatchNormalization(momentum=0.9)(f)  # TODO batchnorm

    # f = Dropout(0.3)(f)

    output1 = Dense(3 * numJoints, name='output1')(f)

    # Auxiliary part-segmentation network - only for training - removed at test time

    c = concatenate([global_feature_exp, local_feature1, local_feature2], axis=-1)

    c = Conv2D(filters=256, kernel_size=(1, 1),
               activation='relu', kernel_initializer='glorot_normal')(c)

    c = BatchNormalization(momentum=0.9)(c)

    c = Dropout(0.2)(c)

    c = Conv2D(filters=256, kernel_size=(1, 1),
               activation='relu', kernel_initializer='glorot_normal')(c)

    c = BatchNormalization(momentum=0.9)(c)

    c = Dropout(0.2)(c)

    c = Conv2D(filters=128, kernel_size=(1, 1),
               activation='relu', kernel_initializer='glorot_normal')(c)

    c = BatchNormalization(momentum=0.9)(c)

    output2 = Conv2D(numRegions, (1, 1), activation='softmax', kernel_initializer='glorot_normal',
                     name='output2')(c)

    model = Model(inputs=input_points, outputs=[output1, output2])
    test_model = Model(inputs=input_points, outputs=output1)
    return model, test_model


def avg_error_proto(y_true, y_pred):
    clusters = centers.reshape(k, numJoints * 3)
    preds = y_pred @ clusters
    return avg_error(y_true, preds)


def avg_error(y_true, y_pred):  # shape=(batch, 3 * numJoints)
    y_pred = Reshape((numJoints, 3))(y_pred)
    y_true = Reshape((numJoints, 3))(y_true)  # shape=(batch, numJoints, 3)

    y_pred = unscale_to_cm(y_pred, data=dataset)
    y_true = unscale_to_cm(y_true, data=dataset)

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
    return Kb.mean(Kb.mean(Kb.sqrt(Kb.sum(Kb.square(y_pred - y_true), axis=-1)), axis=-1),
                   axis=-1)


def mean_avg_precision(y_true, y_pred):
    y_pred = Reshape((numJoints, 3))(y_pred)
    y_true = Reshape((numJoints, 3))(y_true)  # shape=(batch, numJoints, 3)

    y_pred = unscale_to_cm(y_pred, data=dataset)
    y_true = unscale_to_cm(y_true, data=dataset)

    dist = Kb.sqrt(Kb.sum(Kb.square(y_pred - y_true), axis=-1))  # tensor of distances between joints pred and gtrue

    logic = Kb.less_equal(dist, thresh)

    res = Kb.switch(logic, Kb.ones_like(dist), Kb.zeros_like(dist))  # 1 if estimated correctly, else 0

    return Kb.mean(Kb.sum(res, axis=-1) / numJoints, axis=-1)


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


def real_time_init(model, first_sample):
    """ build model and run on first sample beforehand to achieve higher speed (especially on one by one feed)
     than with model.predict """
    get_output = Kb.function([model.layers[0].input, Kb.learning_phase()], [model.layers[-1].output])
    model_output = get_output([first_sample, 0])[0]
    return get_output


def real_time_predict(test_x, get_output_func):
    # run on the rest of test samples
    # import time
    # tic = time.time()
    model_output = get_output_func([test_x, 0])[0]
    # tac = time.time() - tic
    # print(tac)
    return model_output


def run_segnet(generator, x, mode='test', save=True):
    # Predict regions from segnet and save
    segnet_model = load_model('data/models/' + dataset + '/20eps_segnet_lr0.001_4residuals_2.blockconvs512.h5')
    get_output = Kb.function([segnet_model.layers[0].input, Kb.learning_phase()], [segnet_model.layers[-1].output])
    if mode == 'train' and dataset != 'ITOP' and dataset != 'CMU':
        for b_num in range(numTrainSamples // batch_size):
            pcl_batch = np.load(
                'data/' + dataset + '/' + mode + '/scaledpclglobalbatches/' + str(b_num + 1).zfill(fill) + '.npy')
            # pred = segnet_model.predict(pcl_batch, batch_size=batch_size, steps=None)
            pred = get_output([pcl_batch, 0])[0]
            pred = np.argmax(pred, -1)
            pred = np.expand_dims(pred, -1)
            if save:
                np.save(
                    'data/' + dataset + '/' + mode + '/regions_predicted_batches/' + str(b_num + 1).zfill(
                        fill) + '.npy', pred)
    else:
        if dataset == 'ITOP' or dataset == 'CMU':
            pred1 = segnet_model.predict(x[:x.shape[0] // 2], batch_size=batch_size, verbose=1).argmax(axis=-1).astype(
                np.int)
            # pred1 = get_output([x[:x.shape[0] // 2], 0])[0].argmax(axis=-1).astype(np.int)
            # pred2 = get_output([x[x.shape[0] // 2:], 0])[0].argmax(axis=-1).astype(np.int)
            pred2 = segnet_model.predict(x[x.shape[0] // 2:], batch_size=batch_size, verbose=1).argmax(axis=-1).astype(
                np.int)
            pred = np.concatenate([pred1, pred2], axis=0)
            pred = np.expand_dims(pred, -1)
        else:
            pred = segnet_model.predict_generator(generator, use_multiprocessing=True, steps=None, workers=workers,
                                                  verbose=1)
        if save:
            np.save('data/' + dataset + '/' + mode + '/171204_pose6_predicted_regs.npy', pred)
        return pred


# learning rate schedule
def step_decay(epoch):
    # if dataset == 'ITOP':
    #     initial_lrate = 0.0005
    # else:
    if segnet or mymodel:
        initial_lrate = 0.001
    else:
        initial_lrate = 0.0005  # 4chan 0.001 PBPE 0.0005
    drop = 0.8  # 0.5 0.8
    epochs_drop = 1.0
    lrate = initial_lrate * pow(drop,
                                np.floor(epoch / epochs_drop))  # pow(drop, np.floor((1 + epoch) / epochs_drop))
    if lrate < 0.00001:  # clip at 10^-5 to avoid getting stuck at local minima
        lrate = 0.00001  # 0.00003 0.00001
    return lrate


if __name__ == "__main__":
    Kb.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    # model, test_model = PBPE()
    if segnet:
        model = seg_net()
    elif mymodel:
        model = my_model(poolTo1=poolTo1, globalAvg=globalAvg)
    else:
        model, test_model = PBPE_new()

    model.summary(line_length=100)

    losses = {
        "output1": "mean_absolute_error",  # huber_loss, mean_squared_error "mean_absolute_error"
        "output2": "categorical_crossentropy",  # segmentation
    }

    get_custom_objects().update(
        {'avg_error': avg_error, 'Kb': Kb, 'mean_avg_precision': mean_avg_precision,
         'loss_func': loss_func})  # 'avg_error_proto': avg_error_proto, 'huber_loss': huber_loss

    Adam = Adam(lr=0.0, decay=0.0)  # to be set in lrScheduler

    # learning schedule callback
    lrate = LearningRateScheduler(step_decay)

    # Tensorboard callback
    tbCallBack = keras.callbacks.TensorBoard(log_dir='data/tensorboard/' + dataset + '/' + name, histogram_freq=0,
                                             write_graph=True,
                                             write_images=True, write_grads=False, batch_size=batch_size)

    checkpoint = keras.callbacks.ModelCheckpoint('data/models/' + dataset + '/{epoch:02d}eps_' + name + '.h5',
                                                 verbose=1,
                                                 period=1)

    callbacks_list = [lrate, tbCallBack, checkpoint]

    # Load training, validation data (only once)

    # load_poses(0)
    # scaler = np.asarray([pose_scaler.min_, pose_scaler.scale_])
    # np.save('data/pose_scaler.npy', scaler)

    # UBC_convert_pcl_files(index=0, start=1, end=2, mode='train')
    # generate_regions_all(mode='train')
    # scale(mode='train')

    # UBC_convert_pcl_files(index=0, start=1, end=21, mode='valid')
    # generate_regions_all(mode='valid')
    # scale(mode='valid')  # with the same parameters as training set

    # TODO Training
    # validtotrain()

    # UBC_convert_region_files(37037, 39, 61, 'train')

    # mindex = 0
    # MHAD_loadpcls(mode='train', start=1, end=12, index=0, singleview=False)  # notscaledpcl
    # MHAD_load_poses(mode='train', start=1, end=12, index=0, singleview=False)  # notscaledpose

    # MHAD_random_split(0.25, folders=['notscaledpclSW', 'notscaledposeSW', 'regionSW'], start=0, end=281222)

    # MHAD_load_poses(mode='train', start=1, end=13, index=84081, repetitions=(5, 6))  # notscaledpose
    # generate_regions_all(mode='train', data=dataset)  # region
    # MHAD_random_split(0.25, folders=['notscaledpose35j'], start=84081, end=112584)

    # print('train/test split done')
    # find_minmax(data=dataset, mode='train', pcls=True)
    # find_minmax(data=dataset, mode='train', pcls=False)  # poses
    # scale(mode='train', data=dataset)
    # scale_poses(mode='train', data=dataset)
    # MHAD_loadpcls(mode='test', start=12, end=13, index=0, singleview=False)  # notscaledpcl
    # MHAD_load_poses(mode='test', start=12, end=13, index=0, singleview=False)  # notscaledpose
    # scale(mode='test', data=dataset)
    # # scale_poses(mode='valid', data=dataset)
    # scale_poses(mode='test', data=dataset)

    # generate_regions_all(mode='train', data=dataset, start=None, end=None)  # region TODO end
    # generate_regions_all(mode='test', data=dataset)
    # MHAD_loadpcls(mode='test', start=1, end=13, index=0)
    # MHAD_load_poses(mode='test', start=1, end=13, index=0, sameaspcls=True)
    # ITOP_load()

    # centers = cluster('MHAD', k, numTrainSamples, numJoints, batch_size, fill)
    # centers = np.load('data/' + dataset + '/train/pose_clusters50.npy')
    #
    # make_batch_files('train')

    if singleview:
        [pcls_min, pcls_max] = np.load('data/' + dataset + '/train/pcls_minmaxSW.npy')
        [poses_min, poses_max] = np.load('data/' + dataset + '/train/poses_minmaxSW.npy')
    elif test_method == '11subjects':
        [pcls_min, pcls_max] = np.load('data/' + dataset + '/train/pcls_minmax_11subs.npy')
        [poses_min, poses_max] = np.load('data/' + dataset + '/train/poses_minmax_11subs.npy')
    else:
        [pcls_min, pcls_max] = np.load('data/' + dataset + '/train/pcls_minmax.npy')
        [poses_min, poses_max] = np.load('data/' + dataset + '/train/poses_minmax.npy')

    # p = np.load('data/UBC/train/scaledpclglobalbatches/00588.npy')[30]  # TODO fix - some poses small range -0.2,-0.9
    # p = np.reshape(p, (2048, 3))
    # p = (p + 1) * (pcls_max - pcls_min) / 2 + pcls_min
    # pose = np.load('data/UBC/train/posesglobalseparatebatches/00588.npy')[30]
    # pose = np.reshape(pose, (numJoints, 3))
    # pose = unscale_to_cm(pose, data='UBC')
    # region = np.load('data/UBC/train/regionbatches/00588.npy')[30]
    # visualize_3D(p, pose=pose, regions=region, numJoints=numJoints)
    # visualize_3D_pose(pose, numJoints=29)

    # show an example from train set
    # pcl = np.load('data/MHAD/train/scaledpclglobalnoshift/075505.npy')
    # pcl = (pcl + 1) * (pcls_max - pcls_min) / 2 + pcls_min
    # pose = np.load('data/MHAD/train/posesglobalseparatenoshift/075505.npy')
    # pose = (pose + 1) * (poses_max - poses_min) / 2 + poses_min
    # region = np.load('data/MHAD/train/region/075505.npy')
    # visualize_3D(pcl, regions=region, pose=pose, numJoints=numJoints)
    if segnet:
        metrics = ['accuracy']
        lossf = 'categorical_crossentropy'
        lossw = [1.]
    elif mymodel:
        metrics = [avg_error, mean_avg_precision]
        lossf = 'mean_absolute_error'
        lossw = [1.]
    else:  # PBPE model
        metrics = {'output1': [avg_error, mean_avg_precision], 'output2': 'accuracy'}
        lossf = losses
        lossw = [1.0, 0.01]  # TODO try bigger, original 0.1
        test_model.compile(optimizer=Adam,
                           loss="mean_absolute_error", metrics=metrics)

    model.compile(optimizer=Adam,
                  loss=lossf, loss_weights=lossw,
                  metrics=metrics)
    # metrics=[avg_error_proto])

    # model = load_model(
    #     'data/models/'+dataset+'/10eps_mymodel_lr0.001_noproto_convs1x1_poolto1_512_256_1residual_globalavgpool_4chan_reg_preds.h5')

    model = load_model('data/models/' + dataset + '/06eps_SVsegnet_lr0.001_4residuals_2.blockconvs512.h5')

    # model = load_model('data/models/CMU/20eps_' + name + '.h5')

    # test_model = load_model(
    #     'data/models/MHAD/test_models/20eps_batches_11subs_fixeddatafull_mae_denserelu_bnsegonly_weights1.01_lrdrop0.8_lr0.0005_3localfeats.h5')

    if dataset == 'ITOP':
        train_x, train_y, train_regs, test_x, test_y, test_regs = load_ITOP_from_npy()
        pred_regs = np.load('data/ITOP/train/predicted_regs.npy')
        test_pred_regs = np.load('data/ITOP/test/predicted_regs.npy')
        test_x_exp = np.concatenate([test_x, test_pred_regs], axis=-1)
        train_x_exp = np.concatenate([train_x, pred_regs], axis=-1)
        model.fit(train_x_exp, train_y, batch_size=batch_size, epochs=10, callbacks=callbacks_list,
                  # validation_split=0.1
                  validation_data=(test_x_exp, test_y),
                  shuffle=True, initial_epoch=0)
        # [testloss, testavg_err] = test_model.evaluate(test_x, test_y, batch_size=batch_size)
        # print('test avg error: ', testavg_err)
        # predictions = test_model.predict(test_x, batch_size=batch_size)
    elif dataset == 'CMU':  #### CMU panoptic demo #### # todo rewrite as generator ?
        regs_train = np.load('data/CMU/train/regs_onehot.npy', allow_pickle=True)
        x_train = np.load('data/CMU/train/scaled_pcls_lzeromean.npy', allow_pickle=True)
        x_train = np.expand_dims(x_train, axis=2)
        y_train = np.load('data/CMU/train/scaled_poses_lzeromean.npy', allow_pickle=True)
        y_train = y_train.reshape((y_train.shape[0], numJoints * 3))
        # one-hot encoding
        # regs_train = np.eye(numRegions, dtype=np.int)[regs_train]
        # regs_train = regs_train.reshape((regs_train.shape[0], numPoints, 1, numRegions))
        # np.save('data/CMU/train/regs_onehot.npy', regs_train)
        if segnet:
            model.fit(x_train, regs_train, batch_size=batch_size,
                      epochs=20,
                      callbacks=callbacks_list,
                      validation_split=0.2, shuffle=True, initial_epoch=13)  # 0.2
        elif mymodel:
            regs_train_pred = run_segnet(None, x_train, mode='train', save=True)
            regs_train_pred = np.load('data/CMU/train/predicted_regs.npy', allow_pickle=True).astype(np.int)
            x_train = np.concatenate([x_train, regs_train_pred], axis=-1)
            model.fit(x_train, y_train, batch_size=batch_size,
                      epochs=20,
                      callbacks=callbacks_list,
                      validation_split=0.2, shuffle=True, initial_epoch=10)  # 0.2
            # pass
        else:  # PBPE
            model.fit(x_train, {'output1': y_train, 'output2': regs_train}, batch_size=batch_size,
                      epochs=10,
                      callbacks=callbacks_list,
                      validation_split=0.2, shuffle=True, initial_epoch=0)  # 0.2

        x_test = np.load('data/CMU/test/scaled_pcls_lzeromean.npy', allow_pickle=True)
        x_test = np.expand_dims(x_test, axis=2)
        y_test = np.load('data/CMU/test/scaled_poses_lzeromean.npy', allow_pickle=True)
        y_test = y_test.reshape((y_test.shape[0], numJoints * 3))

        # todo test on 171204_pose6 sequence - video results
        # x_test = np.load('data/CMU/test/171204_pose6_scaledpcls_lzeromean.npy', allow_pickle=True)
        # x_test = np.expand_dims(x_test, axis=2)
        # y_test = np.load('data/CMU/test/171204_pose6_scaledposes_lzeromean.npy', allow_pickle=True)
        # y_test = y_test.reshape((y_test.shape[0], numJoints * 3))

        if segnet:
            regs_test = np.load('data/CMU/test/regions.npy', allow_pickle=True)
            # regs_test = np.load('data/CMU/test/171204_pose6_regs.npy', allow_pickle=True) # todo
            regs_test = np.eye(numRegions, dtype=np.int)[regs_test]
            regs_test = regs_test.reshape((regs_test.shape[0], numPoints, 1, numRegions))
            test_metrics = model.evaluate(x_test, regs_test, batch_size=batch_size)
        elif mymodel:
            # regs_test_pred = run_segnet(None, x_test, mode='test', save=True)
            regs_test_pred = np.load('data/CMU/test/predicted_regs.npy', allow_pickle=True).astype(np.int)
            # regs_test_pred = np.load('data/CMU/test/171204_pose6_predicted_regs.npy', allow_pickle=True).astype(np.int)
            x_test = np.concatenate([x_test, regs_test_pred], axis=-1)
            test_metrics = model.evaluate(x_test, y_test, batch_size=batch_size)
            # test_preds = model.predict(x_test, batch_size=batch_size, verbose=1)
            # np.save('data/CMU/test/171204_pose6_predictions.npy', test_preds)
        else:  # PBPE
            test_metrics = test_model.evaluate(x_test, y_test, batch_size=batch_size)
    else:
        train_generator = DataGenerator('data/' + dataset + '/train/', numPoints, numJoints, numRegions, steps=steps,
                                        batch_size=batch_size,
                                        shuffle=True, fill=fill, loadBatches=True, singleview=singleview,
                                        elevensubs=(test_method == '11subjects'), segnet=segnet, four_channels=mymodel,
                                        predicted_regs=predicted_regs)
        if dataset == 'UBC':
            valid_generator = DataGenerator('data/' + dataset + '/valid/', numPoints, numJoints, numRegions,
                                            steps=steps,
                                            batch_size=batch_size, fill=fill, singleview=singleview,
                                            shuffle=False, segnet=segnet, four_channels=mymodel)
        # test_generator = DataGenerator('data/' + dataset + '/test/', numPoints, numJoints, numRegions, steps=None,
        #                                batch_size=batch_size, shuffle=False, fill=6, singleview=singleview)

        test_generator = DataGenerator('data/' + dataset + '/test/', numPoints, numJoints, numRegions, steps=steps,
                                       batch_size=batch_size, shuffle=False, fill=fill, singleview=singleview,
                                       test=True, elevensubs=(test_method == '11subjects'), segnet=segnet,
                                       four_channels=mymodel, predicted_regs=predicted_regs)

        model.fit_generator(generator=train_generator, epochs=10,
                            # validation_data=test_generator,
                            # validation_data=(valid_generator if dataset == 'UBC' else test_generator),
                            # TODO remove test generator from validation
                            callbacks=callbacks_list, initial_epoch=6, use_multiprocessing=True,  # False
                            workers=workers, shuffle=True, max_queue_size=10)  # 20
        # run_segnet(train_generator, None, 'train', True)

    # # # # save the model
    # model.save(
    #     'data/models/' + dataset + '/' + name + '.h5') # is saved during checkpoints

    # test_model.save('data/models/' + dataset + '/test_models/' + name + '.h5')

    # Evaluate model (only regression branch)
    #
    # eval_metrics = model.evaluate_generator(
    #     test_generator, use_multiprocessing=True, workers=workers, max_queue_size=10, verbose=1, steps=None)

    # TODO wrap these into separate functions
    # Evaluate model (both branches)
    #
    # [loss, output1_loss, output2_loss, output1_avg_error, output1_map, output2_acc] = model.evaluate_generator(
    #   test_generator, use_multiprocessing=False, workers=workers, max_queue_size=10, verbose=1, steps=None)
    # # # #
    # print('avg error: ', output1_avg_error)
    #
    # TODO run segnet and save predicted regions
    # pred_regs = run_segnet(generator=None, mode='train', save=True)


    # predictions = np.load('data/MHAD/test/predictions_testmodel_20eps.npy')
    # poses = predictions[0]  # output1
    # poses = np.reshape(poses, (poses.shape[0], numJoints, 3))

    # all test gt poses

    # arr = np.empty((numTestSamples, numJoints, 3))
    # for i in range(numTestSamples):
    #     p = np.load('data/MHAD/test/posesglobalseparate/' + str(i).zfill(6) + '.npy')
    #     arr[i] = p

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
