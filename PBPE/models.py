from keras.models import load_model, Model
from keras.layers.core import Flatten, Dense, Dropout, Reshape
from keras.layers import Input, Convolution2D, Lambda, Activation, add, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import dot, BatchNormalization, concatenate, PReLU, LeakyReLU
import keras.backend as Kb

from config import *


def tile(global_feature, numPoints):
    return Kb.repeat_elements(global_feature, numPoints, 1)


def SGPE(poolTo1=False, globalAvg=True):
    input_points = Input(shape=(numPoints, 1, 4))
    local_feature1 = Conv2D(filters=512, kernel_size=(1, 1), input_shape=(numPoints, 1, 4),
                            kernel_initializer='glorot_normal',
                            activation='relu')(input_points)
    local_feature2 = Conv2D(filters=1024, kernel_size=(1, 1),
                            activation='relu', kernel_initializer='glorot_normal')(local_feature1)
    local_feature3 = Conv2D(filters=2048, kernel_size=(1, 1),
                            activation='relu', kernel_initializer='glorot_normal')(local_feature2)
    local_feature1_exp = Conv2D(filters=2048, kernel_size=(1, 1))(
        local_feature1)
    local_feature2_exp = Conv2D(filters=2048, kernel_size=(1, 1))(
        local_feature1)
    shortcut1 = add([local_feature1_exp, local_feature2_exp, local_feature3])  # add

    # res1 = Activation('relu')(res1)  # a bez tohto
    if poolTo1:
        shortcut1 = MaxPooling2D(pool_size=(2048, 1))(shortcut1)

    # local_feature2_exp = Conv2D(filters=2048, kernel_size=(1, 1))(local_feature2)
    # global_exp = Lambda(tile, arguments={'numPoints': 128})(
    #     global_feature)
    # global_feature = Activation('relu')(global_feature)
    f1 = Conv2D(filters=512, kernel_size=(1, 1),
                kernel_initializer='glorot_normal')(shortcut1)
    f1a = Activation('relu')(f1)
    # f = Conv2D(filters=256, kernel_size=(2, 1),
    #                         activation='relu', kernel_initializer='glorot_normal')(f)
    f2 = Conv2D(filters=256, kernel_size=(1, 1), kernel_initializer='glorot_normal')(f1a)
    # f2 = Conv2D(filters=512, kernel_size=(1, 1))(f2)
    # shortcut2 = add([f2, f1])
    f2 = Activation('relu')(f2)
    # f2 = Conv2D(filters=512, kernel_size=(1, 1))(f2)
    # res2 = concatenate([f1, f2])
    # f = MaxPooling2D(pool_size=(15, 1))(f)
    #  strides  # shape= (b, 1, 1, 2048)
    # global_feature_exp = Lambda(tile, arguments={'numPoints': 2046})(
    #     global_feature)  # shape= (b, numPoints=2048, 1, 2048)
    # f = concatenate([local_feature2, local_feature3, global_feature_exp], axis=-1)
    # f = Conv2D(filters=256, kernel_size=(1,1), activation='relu', kernel_initializer='glorot_normal')(global_feature)

    if globalAvg:
        f = GlobalAveragePooling2D()(f2)
    else:
        f = Flatten()(f2)

    f = Dense(512, kernel_initializer='glorot_normal', activation='relu')(f)
    f = Dense(256, kernel_initializer='glorot_normal', activation='relu')(f)
    # output1 = Dense(k, name='output1', activation='softmax', kernel_initializer='glorot_normal')(f)
    output1 = Dense(numJoints * 3, name='output1', kernel_initializer='glorot_normal')(f)

    model = Model(inputs=input_points, outputs=output1)
    return model


def SGPE_segnet():
    input_points = Input(shape=(numPoints, 1, 3))
    local_feature1_noact = Conv2D(filters=1024, kernel_size=(1, 1), input_shape=(numPoints, 1, 3),  # 512
                                  kernel_initializer='glorot_normal')(input_points)

    # shortcut1_1 = concatenate([local_feature1, input_points])

    local_feature1 = Activation('relu')(local_feature1_noact)

    # local_feature1 = BatchNormalization(momentum=0.9)(local_feature1)

    local_feature2 = Conv2D(filters=1024, kernel_size=(1, 1),
                            kernel_initializer='glorot_normal')(local_feature1)

    shortcut1_2 = add([local_feature1_noact, local_feature2])  # input_points

    local_feature2 = Activation('relu')(shortcut1_2)

    # local_feature2 = BatchNormalization(momentum=0.9)(local_feature2)

    local_feature3 = Conv2D(filters=1024, kernel_size=(1, 1),  # 2048
                            kernel_initializer='glorot_normal')(local_feature2)

    shortcut1_3 = add([local_feature1_noact, local_feature3])  # input_points

    local_feature3 = Activation('relu')(shortcut1_3)

    # d = Dropout(0.2)(local_feature3)

    # local_feature3 = BatchNormalization(momentum=0.9)(local_feature3)

    # local_feature4 = Conv2D(filters=2048, kernel_size=(1, 1),
    #                         activation='relu', kernel_initializer='glorot_normal')(local_feature3)

    global_feature = MaxPooling2D(pool_size=(numPoints, 1))(local_feature3)  # strides  # shape= (b, 1, 1, 2048)
    global_feature_exp = Lambda(tile, arguments={'numPoints': numPoints})(
        global_feature)  # shape= (b, numPoints=2048, 1, 2048)

    # Auxiliary part-segmentation network - only for training - removed at test time

    c = concatenate([global_feature_exp, local_feature1, local_feature2, local_feature3])

    conv1 = Conv2D(filters=512, kernel_size=(1, 1), kernel_initializer='glorot_normal')(c)  # 256

    c = Activation('relu')(conv1)

    c = BatchNormalization(momentum=0.9)(c)

    c = Dropout(0.2)(c)

    c = Conv2D(filters=512, kernel_size=(1, 1), kernel_initializer='glorot_normal')(c)  # 256

    shortcut2_1 = add([conv1, c])

    c = Activation('relu')(shortcut2_1)

    c = BatchNormalization(momentum=0.9)(c)

    c = Dropout(0.2)(c)

    c = Conv2D(filters=512, kernel_size=(1, 1), kernel_initializer='glorot_normal')(c)  # 256

    shortcut2_2 = add([shortcut2_1, c])

    c = Activation('relu')(shortcut2_2)

    c = BatchNormalization(momentum=0.9)(c)

    output = Conv2D(numRegions, (1, 1), activation='softmax', kernel_initializer='glorot_normal')(c)

    return Model(inputs=input_points, outputs=output)


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

    f = Flatten()(global_feature)
    f = Dense(256, kernel_initializer='glorot_normal', activation='relu')(f)
    f = Dense(256, kernel_initializer='glorot_normal', activation='relu')(f)

    # f = Dropout(0.2)(f)

    output1 = Dense(3 * numJoints, name='output1')(f)

    # Auxiliary part-segmentation network - only for training - removed at test time

    c = concatenate([global_feature_exp, local_feature1, local_feature2, local_feature3],
                    axis=-1)

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


def PBPE():
    input_points = Input(shape=(numPoints, 1, 3))
    local_feature1 = Conv2D(filters=512, kernel_size=(1, 1), input_shape=(numPoints, 1, 3),
                            kernel_initializer='glorot_normal',
                            activation='relu')(input_points)
    # local_feature1 = BatchNormalization(momentum=0.9)(x)  # momentum=0.9
    local_feature2 = Conv2D(filters=2048, kernel_size=(1, 1),
                            activation='relu', kernel_initializer='glorot_normal')(local_feature1)

    # local_feature2 = BatchNormalization(momentum=0.9)(x)  # shape = (b, numPoints=2048, 1, 2048)

    global_feature = MaxPooling2D(pool_size=(numPoints, 1))(
        local_feature2)  # strides  # shape= (b, 1, 1, 2048)
    global_feature_exp = Lambda(tile, arguments={'numPoints': numPoints})(
        global_feature)  # shape= (b, numPoints=2048, 1, 2048)

    f = Flatten()(global_feature)
    f = Dense(256, kernel_initializer='glorot_normal')(f)
    # f = BatchNormalization(momentum=0.9)(f)

    f = Dense(256, kernel_initializer='glorot_normal')(f)
    # f = BatchNormalization(momentum=0.9)(f)

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
