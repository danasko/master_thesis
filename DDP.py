from keras.models import Sequential, load_model, Model
from keras.layers.core import Flatten, Dense, Dropout, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import dot, BatchNormalization
from keras.optimizers import SGD, Adagrad, Adam, rmsprop
from keras.backend.tensorflow_backend import set_session
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from keras.utils import plot_model
from keras.utils.generic_utils import get_custom_objects
import keras.callbacks
import tensorflow as tf
import numpy as np
import keras.backend as Kb
from data_loader import data_loader
import cv2
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import pose_visualizer
import k_means_clustering

prototypes_file = 'data/DDP/prototype_poses.npy'
depth_maps_file = 'data/DDP/depth_maps.npy'
poses_file = 'data/DDP/poses.npy'
prototypes_vis_data_file = 'data/DDP/prototypes_vis_data.npy'
validation_samples_x_file = 'data/DDP/validation_samples_x.npy'
validation_samples_y_file = 'data/DDP/validation_samples_y.npy'
eval_data_x_file = 'data/DDP/eval_data_x.npy'
eval_data_y_file = 'data/DDP/eval_data_y.npy'
test_data_x_file = 'data/DDP/test_data_x.npy'
predictions_file = 'data/DDP/predictions_leakrelu2.npy'
C = None  # set of K prototypes (after clustering) - matrix
batch_size = 64  # 1, 7, 257, 1799

# Hyperparameters:

alpha = 0.08  # ITOP dataset 0.08
# alpha = 0.01 # UBC3V dataset
J = 15  # no. of joints ITOP dataset
# J = 18  # no. of joints UBC3V dataset
K = 70  # no. of prototype poses
o = 1  # ITOP dataset
# o = 0.8  # UBC3V dataset

# input depth maps size
# ITOP
width = 320
height = 240


# UBC3V
# width = 512
# height = 424


def DPP(weights=None):
    model = Sequential()
    # model.add(ZeroPadding2D((1, 1), )
    glorot = keras.initializers.glorot_normal(seed=1024)
    model.add(Conv2D(96, (94, 94), input_shape=(100, 100, 1), kernel_initializer=glorot)) # activation='relu'
    # model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D((2, 2)))  # out: 7x7x96

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(192, (1, 1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D((2, 2)))  # out: 4x4x192

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (2, 2)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D((2, 2)))  # out: 3x3x512

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(1024, (2, 2)))
    model.add(LeakyReLU(alpha=0.3))

    # model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(2048, (1, 1)))
    model.add(LeakyReLU(alpha=0.3))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.2))  # 0.2
    model.add(Dense(256, name='second_dense'))
    model.add(LeakyReLU(alpha=0.3))

    # model.add(Dropout(0.2))

    model.add(Dense(K, name='output'))
    model.add(LeakyReLU(alpha=0.3))

    if weights:
        model.load_weights(weights)

    return model


def loss_function(y_true, y_pred):
    # print(y_pred.shape)  # (batch_size, K)
    # print(y_true.shape)  # (15, batch_size, 3) ..should be.. but is actually (?, ?)

    y = Kb.transpose(y_pred)[0]  # (batch_size,)
    y = Kb.expand_dims(y, -1)  # (batch_size, 1)

    p_e = Kb.constant(1, shape=(K, 1, J * 3))

    p = Kb.dot(y, p_e)  # (batch_size, K, J*3)
    p = Reshape(target_shape=(K, J, 3))(p)
    p = Kb.permute_dimensions(p, (0, 2, 1, 3))  # (batch_size, J, K, 3)
    # p = Kb.constant(1, shape=(batch_size, J, K, 3))

    tc = Kb.ones_like(p) * C

    tc = Kb.permute_dimensions(tc, (0, 1, 3, 2))
    tc = Reshape(target_shape=(J * 3, K))(tc)

    c_pred = Kb.batch_dot(y_pred, tc, axes=[1, 2])  # p^

    # c_pred = Reshape(target_shape=(15, 3))(c_pred)  # expand last dimension back into two -> 15, 3
    y_true = Reshape(target_shape=(45,))(y_true)  # flatten last two dimensions (15,3) into one

    p1 = (1 - alpha) * res_loss(c_pred, y_true)
    p2 = alpha * L1(y_pred)

    return p1 + p2


def L1(v):  # L1 form of ndarray v
    return Kb.sum(Kb.abs(v), axis=-1)


def res_loss(pred, gtrue):
    r = gtrue - pred  # shape of both (batch, 45)
    logic = Kb.less(Kb.abs(r), 1 / (o ^ 2))
    p = Kb.switch(logic, 0.5 * (o ^ 2) * Kb.square(r), Kb.abs(r) - (0.5 / (o ^ 2)))

    return Kb.sum(p, axis=-1)


def column(matrix, i):
    return np.transpose(matrix, (1, 0, 2))[i]


def preprocess(depth_maps):
    res = np.zeros((depth_maps.shape[0], 100, 100))
    center_y = round(depth_maps[0].shape[0] / 2)
    center_x = round(depth_maps[0].shape[1] / 2)

    for i in range(depth_maps.shape[0]):
        res[i] = depth_maps[i][center_y - 50: center_y + 50,
                 center_x - 50:center_x + 50]  # crop width to 100 around center

        # img = np.array(res[i] * 255, dtype=np.uint8)  # for visualization
        # cv2.imshow("img", img)
        # cv2.waitKey(0)

    return res


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
        batch_x = np.expand_dims(batch_x, -1)
        batch_y = np.asarray(batch_y)
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
        batch_x = np.expand_dims(batch_x, -1)
        batch_y = np.asarray(batch_y)
        yield batch_x, batch_y


def avg_error(y_true, y_pred):
    y = Kb.transpose(y_pred)[0]  # (batch_size,)
    y = Kb.expand_dims(y, -1)  # (batch_size, 1)

    p_e = Kb.constant(1, shape=(K, 1, J * 3))

    p = Kb.dot(y, p_e)  # (batch_size, K, J*3)
    p = Reshape(target_shape=(K, J, 3))(p)
    p = Kb.permute_dimensions(p, (0, 2, 1, 3))  # (batch_size, J, K, 3)
    # p = Kb.constant(1, shape=(batch_size, J, K, 3))

    tc = Kb.ones_like(p) * C

    tc = Kb.permute_dimensions(tc, (0, 1, 3, 2))
    tc = Reshape(target_shape=(J * 3, K))(tc)

    c_pred = Kb.batch_dot(y_pred, tc, axes=[1, 2])  # p^

    c_pred = Reshape(target_shape=(15, 3))(c_pred)  # expand last dimension back into two -> 15, 3

    # Euclidean distance between predicted and ground-truth pose
    return Kb.mean(Kb.sqrt(Kb.sum(Kb.square(c_pred - y_true), axis=-1)), axis=-1)


def get_random_batch(files, label_files, batch_size=batch_size):
    idx = np.random.randint(0, files.shape[0] - batch_size)
    batch_x = files[idx: idx + batch_size]
    batch_y = label_files[idx: idx + batch_size]
    # batch_x = np.expand_dims(batch_x, -1)
    batch_y = np.asarray(batch_y)
    return batch_x, batch_y


if __name__ == "__main__":
    Kb.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    # DATASET ########################################################
    #
    # train_dataset = data_loader('D:/skola/master/datasets/ITOP/depth_maps/ITOP_side_train_depth_map.h5',
    #                      'D:/skola/master/datasets/ITOP/labels/ITOP_side_train_labels.h5')
    #
    # new_width = round(width / (height / 100.0))
    # [train_data_x, train_data_y, vis_data] = train_dataset.get_data(new_width, 100)
    #
    # train_data_x = preprocess(train_data_x)  # depth maps (resized to 100x100 (crop),/preprocessed)
    #
    # # [train_samples, validation_samples, vis_data] = train_valid_split(train_data_x, train_data_y, vis_data,
    # #                                                                   test_size=0.25)
    #
    # # D = train_samples[0]
    # # P = train_samples[1]  # ground truth poses - normalized joint coords (zero mean //(and one standard deviation))
    # D = train_data_x
    # P = train_data_y
    # D = np.expand_dims(D, -1)
    # TEST ######################################################
    #
    # test_dataset = data_loader('D:/skola/master/datasets/ITOP/depth_maps/ITOP_side_test_depth_map.h5',
    #                     'D:/skola/master/datasets/ITOP/labels/ITOP_side_test_labels.h5')
    # new_width = round(width / (height / 100.0))
    # [eval_data_x, eval_data_y, eval_vis_data] = test_dataset.get_data(new_width, 100)
    # eval_data_x = preprocess(eval_data_x)
    # eval_data_x = np.expand_dims(eval_data_x, -1)
    #
    # [test_data_x, empty, test_vis_data] = test_dataset.get_data(new_width, 100, test=True)
    # test_data_x = preprocess(test_data_x)
    # test_data_x = np.expand_dims(test_data_x, -1)
    #################################################################

    D = np.load(depth_maps_file)
    P = np.load(poses_file)
    # shuffle(D, P, random_state=10)

    C = np.load(prototypes_file)
    # C = k_means_clustering.cluster_prototypes(P, K)  # [C, prototypes_vis_data] = random_prototype_poses(P, vis_data)

    # shuffle prototype poses
    # C = C[:, np.random.permutation(C.shape[1])]

    # prototypes_vis_data = np.load(prototypes_vis_data_file)  # deprecated
    # validation_samples = [np.load(validation_samples_x_file), np.load(validation_samples_y_file)]
    eval_data_x = np.load(eval_data_x_file)
    eval_data_y = np.load(eval_data_y_file)
    test_data_x = np.load(test_data_x_file)

    # np.save(depth_maps_file, D)
    # np.save(poses_file, P)
    # np.save(prototypes_file, C)
    # np.save(prototypes_vis_data_file, prototypes_vis_data)
    # np.save(validation_samples_x_file, validation_samples[0])
    # np.save(validation_samples_y_file, validation_samples[1])
    # np.save(eval_data_x_file, eval_data_x)
    # np.save(eval_data_y_file, eval_data_y)
    # np.save(test_data_x_file, test_data_x)
    # MODEL ##########################################################

    model = DPP()

    # adagrad = Adagrad(lr=0.001, decay=0.001)
    # rms = rmsprop(lr=0.001, rho=0.9, epsilon=None, decay=0.001)
    adam = Adam(lr=0.001, decay=0.003)  # - Adam optimizer decays the lr automatically  # 0.0008 # 0.00088 # 0.002
    model.compile(loss=loss_function,
                  optimizer=adam, metrics=[avg_error])

    # model.summary()
    # plot_model(model, to_file='data/DDP/model_plot.png', show_shapes=True, show_layer_names=True)
    ##################################################################

    num_training_samples = D.shape[0]
    # num_validation_samples = validation_samples[0].shape[0]

    # data_generator = training_generator(D, P, batch_size)
    # valid_generator = validation_generator(validation_samples[0], validation_samples[1], batch_size)
    get_custom_objects().update({"loss_function": loss_function, "avg_error": avg_error})

    # model = load_model('data/DDP/model_training_norm_smallersplit_dropout_smaller_batch_leakrelu2.h5')

    tbCallBack = keras.callbacks.TensorBoard(log_dir='data/DDP/Graph/norm_new_downscaled', histogram_freq=0,
                                             write_graph=True,
                                             write_images=True)

    # training on |steps| random batches from the training set in each epoch
    # train_history = model.fit_generator(generator=data_generator, epochs=200, validation_data=valid_generator,
    #                                     validation_steps=num_validation_samples // batch_size,
    #                                     # use_multiprocessing=True,  workers=6,
    #                                     steps_per_epoch=500,  # 2000, num_training_samples // batch_size
    #                                     callbacks=[tbCallBack])  # max: epochs = 1000

    # training on all the training samples in each epoch once
    # D = D[:-6]
    # P = P[:-6]  # val split 0.2 :-6
    # eval_data_x = eval_data_x[:-1]
    # eval_data_y = eval_data_y[:-1]

    # validation_samples[0] = np.expand_dims(validation_samples[0], -1)
    #
    train_history = model.fit(D, P, epochs=300, batch_size=batch_size,
                              validation_split=0.07,
                              callbacks=[tbCallBack])
    # # validation_data=validation_samples

    # np.save('data/DDP/train_history_cluster_fixednorm.npy', train_history)
    model.save('data/DDP/model_training_leak_64b.h5')

    start = 160
    # predictions = model.predict(test_data_x[start:])

    # np.save(predictions_file, predictions)

    # predictions = np.load(predictions_file)

    # score = model.evaluate(eval_data_x, eval_data_y, batch_size=batch_size)
    # print('Test loss:', score[0])  # 0.3152455667544362 # 0.3125105603790911
    # print('Test average error:', score[1])  # 0.07282457308264982 # 0.07163713314290304

    # synth_prediction = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0,
    #  0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0,
    #                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                     0, 0, 0, 0, 0, 0, 0, 0]

    # # visualize predictions

    # t = Kb.permute_dimensions(C, (0, 2, 1))
    # t = Kb.reshape(t, (J * 3, K))
    #
    # for idx in range(predictions.shape[0]):
    #     pose = Kb.sum(predictions[idx] * t, axis=-1)
    #     # in 3D
    #     # pose = Kb.transpose(Kb.reshape(pose, (15, 3)))
    #     # pose_visualizer.visualize_pose_3D(pose, pause=False)
    #     # in 2D
    #     pose = Kb.reshape(pose, (15, 3))
    #     pose_visualizer.visualize_pose_2D(pose, pause=False)
    #
    #     # depth map
    #     depth_map = cv2.resize(test_data_x[start + idx], (244, 244))
    #     img = np.array(depth_map * 255, dtype=np.uint8)  # for visualization
    #     cv2.imshow('depth map', img)
    #     cv2.waitKey(0)
    #####################################################################

    # visualization ##################################################
    # 2D poses

    # for p in prototypes_vis_data:  # not updated prototypes
    #     pose_visualizer.visualize_pose(p)  # [Press enter]

    # 3D prototype poses

    # cp = Kb.constant(C)
    # # 3D
    # cp = Kb.permute_dimensions(cp, (1, 2, 0))
    # # pose_visualizer.visualize_pose_3D(cp[0], pause=False)
    # # 2D
    # cp = Kb.permute_dimensions(cp, (1, 0, 2))

    # for p in range(cp.shape[0]):
    #     # pose_visualizer.visualize_pose_3D(cp[p], pause=False)  # [Press Enter]
    #     pose_visualizer.visualize_pose_2D(cp[p])

    # train data
    #
    # [dp, pp] = get_random_batch(D, P)
    # pp = Kb.constant(pp)
    # pp = Kb.permute_dimensions(pp, (0, 2, 1))
    # for i in range(dp.shape[0]):
    #     pose_visualizer.visualize_pose_3D(pp[i], pause=False)
    #     depth_map = cv2.resize(dp[i], (244, 244))
    #     img = np.array(depth_map * 255, dtype=np.uint8)  # for visualization
    #     cv2.imshow('depth map', img)
    #     cv2.waitKey(0)  # [Press any key]

    # validation data
    #
    # [dp, pp] = get_random_batch(validation_samples[0], validation_samples[1], batch_size=batch_size)
    # pp = Kb.constant(pp)
    # pp = Kb.permute_dimensions(pp, (0, 2, 1))
    # for p in range(dp.shape[0]):
    #     pose_visualizer.visualize_pose_3D(pp[p], pause=False)
    #     depth_map = cv2.resize(dp[p], (244, 244))
    #     img = np.array(depth_map * 255, dtype=np.uint8)  # for visualization
    #     cv2.imshow('depth map', img)
    #     cv2.waitKey(0)  # [Press any key]
