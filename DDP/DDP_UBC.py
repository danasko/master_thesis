import DDP
import keras.backend as Kb
import tensorflow as tf
from Adam_lr_mult import *
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
import numpy as np
from keras.models import load_model
from tensorflow import set_random_seed
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import LearningRateScheduler
from keras.layers.core import Reshape
import keras
from data_generator import DataGenerator
from sklearn.utils import shuffle
from scipy.io import loadmat  # from Matlab
import os, glob
import cv2
import k_means_clustering
import pose_visualizer
from scipy import misc

width = 512
height = 424
np.random.seed(21)
set_random_seed(21)
batch_size = DDP.batch_size
numJoints = DDP.J
K = DDP.K
thresh = 10
meanpose, varpose = None, None
input_size = DDP.input_size

numSubjects = {'train': 61, 'valid': 20, 'test': 21}

numSamples = {'train': 59059 * 3, 'valid': 19019 * 3, 'test': 19019 * 3}


def rescale(x, lw, up):
    return lw + ((x - np.min(x)) / (np.max(x) - np.min(x))) * (up - lw)


def load_UBC(mode='train'):
    allimgs = np.empty(shape=(numSamples[mode], input_size, input_size, 1))
    poses = loadmat('D:/skola/master/datasets/UBC3V/ubc3v-master/exported_poses_hard_' + mode + '.mat')['poses'][0]
    poses = np.asarray([poses[i][0] for i in range(poses.shape[0])])
    poses = np.repeat(poses, 3, axis=0)
    idx = 0
    for s in range(1, numSubjects[mode]):
        print('Loading subject ' + str(s) + '..')
        if s != 6 or mode == 'valid':
            dir = 'D:\\skola\\master\\datasets\\UBC3V\\hard-pose\\' + mode + '\\' + str(s) + '\\images\\depthRender\\'
            cdir = dir + 'Cam'
            files = os.listdir(cdir + str(1) + '\\')
            # imgs = glob.glob("*.png")
            for f in files:
                if f.endswith(".png"):
                    for cam in range(1, 4):
                        img = misc.imread(cdir + str(cam) + '\\' + f)[:, :, 0] / 255.0
                        strip = (width - height) // 2
                        img = img[:, strip:width - strip]
                        img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
                        img = rescale(img, 0, 1)
                        img = np.expand_dims(img, axis=-1)
                        allimgs[idx] = img
                        idx += 1
    print('All files loaded.')

    if mode == 'train':
        allimgs, poses = shuffle(allimgs, poses)
        np.save('data/DDP/UBC/train/meanimg.npy', np.mean(allimgs, axis=0, keepdims=True))
        np.save('data/DDP/UBC/train/meanpose.npy', np.mean(poses, axis=0))
        np.save('data/DDP/UBC/train/varpose.npy', np.var(poses, axis=0))
        allimgs -= np.mean(allimgs, axis=0, keepdims=True)
        centroids = k_means_clustering.cluster_prototypes(poses, K)
        np.save('data/DDP/UBC/train/centroids.npy', centroids)
    else:
        meanimg = np.load('data/DDP/UBC/train/meanimg.npy')
        allimgs -= meanimg

    b = np.empty((batch_size, input_size, input_size, 1))
    bpose = np.empty((batch_size, numJoints, 3))
    for i, img in enumerate(allimgs):
        idx = i % batch_size
        if not idx and i > 0:
            np.save('data/DDP/UBC/' + mode + '/depth/' + str(i // batch_size).zfill(6) + '.npy', b)
            np.save('data/DDP/UBC/' + mode + '/pose/' + str(i // batch_size).zfill(6) + '.npy', bpose)
        b[idx] = img
        bpose[idx] = poses[i]


def scale_poses(mode='train'):
    # global poses_min, poses_max
    poses_min = [1000000, 1000000, 1000000]
    poses_max = [-1000000, -1000000, -1000000]
    [poses_min, poses_max] = np.load('data/DDP/UBC/train/poses_minmax.npy')

    poses_file = \
        loadmat('D:/skola/master/datasets/UBC3V/ubc3v-master/exported_poses_hard_' + mode + '.mat')['poses'][0]
    poses = np.asarray([poses_file[i][0] for i in range(poses_file.shape[0])])

    # if mode == 'train':
    # poses_min = np.minimum(poses_min, np.min(poses, axis=(0, 1)))
    # poses_max = np.maximum(poses_max, np.max(poses, axis=(0, 1)))
    # np.save('data/DDP/UBC/train/poses_minmax.npy', [poses_min, poses_max])
    #
    # # cents = np.load('data/DDP/UBC/train/centroids.npy')
    # # cents = cents.reshape((numJoints, 3, K))
    # # cents = np.transpose(cents, axes=(1, 2))
    # # cents = 2 * (cents - poses_min) / (poses_max - poses_min) - 1
    # # cents = np.transpose(cents, axes=(1, 2))
    #
    poses = 2 * (poses - poses_min) / (poses_max - poses_min) - 1
    cents = k_means_clustering.cluster_prototypes(poses, K)
    cents = cents.reshape((numJoints * 3, K))
    np.save('data/DDP/UBC/train/centroids_scaled.npy', cents)
    # np.save('data/DDP/UBC/train/meanpose.npy', np.mean(poses, axis=0))  # TODO try axis=(0,1) in both
    # np.save('data/DDP/UBC/train/varpose.npy', np.var(poses, axis=0))
    # if mode == 'test' or mode == 'valid':
    #     [poses_min, poses_max] = np.load('data/DDP/UBC/train/poses_minmax.npy')
    # #
    # for i in range(numSamples[mode] // batch_size):
    #     ps = np.load('data/DDP/UBC/' + mode + '/pose/' + str(i + 1).zfill(6) + '.npy')
    #     # if mode == 'test':
    #     #     np.save('data/' + data + '/test/notscaledpose/' + str(i).zfill(5) + '.npy', p)
    #     # p[:, 0] = 2 * (p[:, 0] - poses_min[0]) / (poses_max[0] - poses_min[0]) - 1
    #     # p[:, 1] = 2 * (p[:, 1] - poses_min[1]) / (poses_max[1] - poses_min[1]) - 1
    #     # p[:, 2] = 2 * (p[:, 2] - poses_min[2]) / (poses_max[2] - poses_min[2]) - 1
    #     # ps = 2 * (ps - poses_min) / (poses_max - poses_min) - 1
    #     ps *= [-1, 1, 1]
    #     np.save('data/DDP/UBC/' + mode + '/pose/' + str(i + 1).zfill(6) + '.npy', ps)


def unscale(tensor):
    [posemin, posemax] = np.load('data/DDP/UBC/train/poses_minmax.npy')
    tensor2 = (tensor + 1) * (posemax - posemin) / 2 + posemin
    return tensor2


def meanAveragePrecision(y_true, y_pred):
    # y_pred = y_pred * 2 - 1
    y_true = unscale(y_true)
    # y_pred = unscale(y_pred)
    # y_true = unscale(y_true)
    # y = Kb.stack([y_pred] * (numJoints * 3), axis=-1)
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

    c_pred = Reshape(target_shape=(numJoints, 3))(c_pred)  # expand last dimension back into two -> 15, 3
    # c_pred = Reshape(target_shape=(J, 3))(y_pred)  # baseline model
    c_pred = unscale(c_pred)
    dist = Kb.sqrt(Kb.sum(Kb.square(c_pred - y_true), axis=-1))  # tensor of distances between joints pred and gtrue

    logic = Kb.less_equal(dist, thresh)

    res = Kb.switch(logic, Kb.ones_like(dist), Kb.zeros_like(dist))  # 1 if estimated correctly, else 0

    return Kb.mean(Kb.sum(res, axis=-1) / numJoints, axis=-1)


def avg_error(y_true, y_pred):
    # y_pred = y_pred * 2 - 1
    y_true = unscale(y_true)
    # y_pred = unscale(y_pred)

    # y = Kb.stack([y_pred] * (numJoints * 3), axis=-1)
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

    c_pred = Reshape(target_shape=(numJoints, 3))(c_pred)  # expand last dimension back into two -> 15, 3
    c_pred = unscale(c_pred)
    # c_pred = Reshape(target_shape=(J, 3))(y_pred)  # baseline model

    # Euclidean distance between predicted and ground-truth pose
    return Kb.mean(Kb.sqrt(Kb.sum(Kb.square(c_pred - y_true), axis=-1)), axis=-1)


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.0005  # 0.001 0.0005
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
    lrate = LearningRateScheduler(step_decay)
    model = DDP.DPP()
    # load_UBC(mode='test')
    # scale_poses('train')
    # load_UBC(mode='train')

    # load_UBC(mode='valid')

    [poses_min, poses_max] = np.load('data/DDP/UBC/train/poses_minmax.npy')
    #
    # cents = np.load('data/DDP/UBC/train/centroids.npy')
    # cents = cents.reshape((numJoints, 3, K))
    # cents = np.transpose(cents, axes=(0, 2, 1))
    # cents = 2 * (cents - poses_min) / (poses_max - poses_min) - 1
    # cents = np.transpose(cents, axes=(0, 2, 1))
    # cents = cents.reshape((numJoints * 3, K))
    # cents = np.asarray(cents, dtype=np.float32)
    # np.save('data/DDP/UBC/train/centroids_scaled.npy', cents)

    tc = np.load('data/DDP/UBC/train/centroids_scaled.npy')
    # tc = cents
    DDP.tc = tc
    DDP.C = tc
    meanpose = np.load('data/DDP/UBC/train/meanpose.npy').flatten()  # TODO update meanpose, varpose
    varpose = np.load('data/DDP/UBC/train/varpose.npy').flatten()
    DDP.meanpose = meanpose
    DDP.varpose = varpose
    DDP.thresh = 10  # in cm

    # img = np.load('data/DDP/UBC/train/depth/000002.npy')[3]
    # pose = np.load('data/DDP/UBC/train/pose/000002.npy')[3]
    # # print(pose)
    # pose_visualizer.visualize_pose_3D(pose, isnumpy=True)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # #
    # pose = tc[:, 63].reshape((numJoints, 3))
    # pose_visualizer.visualize_pose_3D(pose, isnumpy=True)

    # multipliers = {'conv2d_1': 1, 'conv2d_2': 1, 'conv2d_3': 1, 'conv2d_4': 1, 'conv2d_5': 1, 'conv2d_6': 1,
    #                'conv2d_7': 1, 'dense_1': 1,
    #                'dense_2': 1, 'output': 0.1}

    adam = Adam(lr=0.0001, decay=0.00005)

    # adam_lr_mult = Adam_lr_mult(multipliers=multipliers, lr=0.0005, decay=0.000001, epsilon=0.0000000001)
    model.compile(loss=DDP.loss_function, optimizer=adam, metrics=[avg_error, meanAveragePrecision],
                  target_tensors=[Kb.placeholder((None, numJoints, 3))])

    # model.summary()
    # plot_model(model, to_file='D:/skola/master/prehlady/screenshots/DDP/model_plot_final.png', show_shapes=True,
    #            show_layer_names=True)

    get_custom_objects().update(
        {"loss_function": DDP.loss_function, "avg_error": avg_error,
         "meanAveragePrecision": meanAveragePrecision,
         "Adam_lr_mult": Adam_lr_mult})

    # model = load_model('data/DDP/UBC/20eps_model_training_64b_6dropouts0.4_7conv_6convnopool_6_7_filter3x3_nolrsched.h5')

    name = '10eps_model_training_64b_6dropouts0.4_7conv_6convnopool_6_7_filter3x3_nolrschedlrdecay_100K_alpha0.1'
    tbCallBack = keras.callbacks.TensorBoard(log_dir='data/DDP/Graph/UBC/' + name, histogram_freq=0,
                                             write_graph=True,
                                             write_images=True, write_grads=False, batch_size=512)

    # earlyStop = keras.callbacks.EarlyStopping(monitor='val_avg_error', min_delta=0, patience=7, verbose=0, mode='min',
    #                                           baseline=None,
    #                                           restore_best_weights=True)

    train_generator = DataGenerator('data/DDP/UBC/train/', input_size=DDP.input_size, numJoints=numJoints, steps=None,
                                    batch_size=batch_size,
                                    shuffle=True, mode='train')
    valid_generator = DataGenerator('data/DDP/UBC/valid/', input_size=DDP.input_size, numJoints=numJoints, steps=None,
                                    batch_size=batch_size,
                                    shuffle=False, mode='valid')
    test_generator = DataGenerator('data/DDP/UBC/test/', input_size=DDP.input_size, numJoints=numJoints, steps=None,
                                   batch_size=batch_size, shuffle=False, mode='test')

    # training on |steps| random batches from the training set in each epoch
    model.fit_generator(generator=train_generator, epochs=10, validation_data=test_generator,
                        # validation_steps=num_validation_samples // batch_size,
                        use_multiprocessing=False, workers=6, max_queue_size=10, initial_epoch=0,
                        steps_per_epoch=None,  # np.shape(D)[0]//batch_size,
                        callbacks=[tbCallBack], shuffle=True)  # max: epochs = 1000

    # np.save('data/DDP/train_history_cluster_fixednorm.npy', train_history)
    model.save('data/DDP/UBC/' + name + '.h5')  # Todo try smaller kernels in convs - overfitting

    # eval_metrics = model.evaluate_generator(test_generator, verbose=1)
    # print(eval_metrics)
