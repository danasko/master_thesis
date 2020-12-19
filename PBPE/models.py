from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Reshape
from keras.layers import Input, Lambda, Activation, add, GlobalAveragePooling2D, BatchNormalization, concatenate, \
    LeakyReLU, multiply, LSTM
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D
import keras.backend as Kb
import numpy as np

from config import *
import utils


def tile(global_feature, numPoints):
    return Kb.repeat_elements(global_feature, numPoints, 1)


# returns train, inference_encoder and inference_decoder models
def define_lstm_models(n_units):
    # define training encoder
    encoder_inputs = Input(shape=(None, numPoints * 3))
    # revert input sequences
    reverted_inputs = Lambda(lambda x: x[:, ::-1, :])(encoder_inputs)
    # reverted_inputs = Lambda(lambda x: Kb.permute_dimensions(x, [1, 0, 2]))(reverted_inputs)  # TODO ?? from paper
    # encoder_inputs = Reshape([-1, numPoints * 3])(encoder_inputs)
    # encoder_inputs = Lambda(lambda x: np.split(x, seq_length, axis=0))(encoder_inputs)
    ##
    encoder_lstm = LSTM(n_units, return_state=True, activation='tanh',recurrent_activation='tanh',
                        kernel_initializer='glorot_normal')  # TODO recurrent dropout
    encoder_outputs, state_h, state_c = encoder_lstm(reverted_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, numJoints * 3))  # previous estimated poses (except last/current pose)
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True, activation='tanh',recurrent_activation='tanh',
                        kernel_initializer='glorot_normal')  # TODO rec. dropout, residual connections ?
    decoder_out, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    # decoder_dense = Dense(128, kernel_initializer='glorot_normal', activation='relu')
    decoder_dense = Dense(numJoints * 3, kernel_initializer='glorot_normal')
    decoder_outputs = decoder_dense(
        decoder_out)  # same as decoder_inputs, but shifted by one frame forward (dec_out[:,t,:] == dec_in[:,t+1,:])
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model


def temp_graph_SGPE():
    input_points = Input(shape=(seq_length, numPoints * 3))
    outputs = []
    for i in range(seq_length):
        pcl = Lambda(lambda x: x[:, i, :])(input_points)
        pcl = Reshape([numPoints, 3])(pcl)
        out = graph_SGPE_nobottleneck(pcl)
        # weighted sequence (the closer is the frame to the current frame - the higher the weight)
        # if i < seq_length - 1:
        #     coef = Lambda(lambda x: Kb.ones_like(x) * (temp_coeff * (0.05 ** (seq_length - 2 - i))))(out)
        # else:
        #     coef = Lambda(lambda x: Kb.ones_like(x))(out)
        # outputs.append(multiply([coef, out]))
        outputs.append(out)

    merged = concatenate([*outputs], axis=-1)

    f = Dense(512, kernel_initializer='glorot_normal', activation='relu')(merged)
    f = Dense(256, kernel_initializer='glorot_normal', activation='relu')(f)
    f = Dense(128, kernel_initializer='glorot_normal', activation='relu')(f)
    f = Dense(numJoints * 3, kernel_initializer='glorot_normal')(f)
    output = Reshape((1, -1))(f)

    model = Model(inputs=input_points, outputs=output)
    return model


def graph_SGPE_nobottleneck(input_points):
    graph1 = EdgeConv(input_points)

    # transform = Lambda(dgcnn_transform_net, arguments={'K': 3})(graph0)
    # point_cloud_transformed = Lambda(lambda x: Kb.batch_dot(x[0], x[1]))([input_points, transform])
    #
    # graph1 = EdgeConv(point_cloud_transformed)

    local_feature1 = Conv2D(filters=32, kernel_size=(1, 1),
                            kernel_initializer='glorot_normal',
                            activation='relu')(graph1)
    local_feature2 = Conv2D(filters=32, kernel_size=(1, 1),
                            activation='relu', kernel_initializer='glorot_normal')(local_feature1)

    net1 = Lambda(lambda x: Kb.max(x, axis=-2, keepdims=True))(local_feature2)
    graph2 = EdgeConv(net1, expand=False)

    local_feature3 = Conv2D(filters=32, kernel_size=(1, 1),
                            activation='relu', kernel_initializer='glorot_normal')(graph2)
    shortcut1 = add([local_feature1, local_feature2, local_feature3])

    if poolTo1:
        shortcut1 = MaxPooling2D(pool_size=(32, 1))(shortcut1)

    f1 = Conv2D(filters=32, kernel_size=(1, 1),
                kernel_initializer='glorot_normal')(shortcut1)
    f1a = Activation('relu')(f1)

    net2 = Lambda(lambda x: Kb.max(x, axis=-2, keepdims=True))(f1a)
    graph3 = EdgeConv(net2, expand=False)

    f2 = Conv2D(filters=32, kernel_size=(1, 1), kernel_initializer='glorot_normal')(graph3)

    f2 = Activation('relu')(f2)

    net3 = Lambda(lambda x: Kb.max(x, axis=-2, keepdims=True))(f2)
    graph4 = EdgeConv(net3, expand=False)

    if globalAvg:
        f = GlobalAveragePooling2D()(graph4)
    else:
        f = Flatten()(graph4)
    return f


def graph_SGPE(poolTo1=False, globalAvg=True):
    input_points = Input(shape=(numPoints, 3))

    graph0 = EdgeConv(input_points)

    transform = Lambda(dgcnn_transform_net, arguments={'K': 3})(graph0)
    point_cloud_transformed = Lambda(lambda x: Kb.batch_dot(x[0], x[1]))([input_points, transform])

    graph1 = EdgeConv(point_cloud_transformed)

    local_feature1 = Conv2D(filters=128, kernel_size=(1, 1),
                            kernel_initializer='glorot_normal',
                            activation='relu')(graph1)
    local_feature2 = Conv2D(filters=128, kernel_size=(1, 1),
                            activation='relu', kernel_initializer='glorot_normal')(local_feature1)

    net1 = Lambda(lambda x: Kb.max(x, axis=-2, keepdims=True))(local_feature2)
    graph2 = EdgeConv(net1, expand=False)

    local_feature3 = Conv2D(filters=128, kernel_size=(1, 1),
                            activation='relu', kernel_initializer='glorot_normal')(graph2)
    shortcut1 = add([local_feature1, local_feature2, local_feature3])

    if poolTo1:
        shortcut1 = MaxPooling2D(pool_size=(128, 1))(shortcut1)

    f1 = Conv2D(filters=128, kernel_size=(1, 1),
                kernel_initializer='glorot_normal')(shortcut1)
    f1a = Activation('relu')(f1)

    net2 = Lambda(lambda x: Kb.max(x, axis=-2, keepdims=True))(f1a)
    graph3 = EdgeConv(net2, expand=False)

    f2 = Conv2D(filters=64, kernel_size=(1, 1), kernel_initializer='glorot_normal')(graph3)

    f2 = Activation('relu')(f2)

    net3 = Lambda(lambda x: Kb.max(x, axis=-2, keepdims=True))(f2)
    graph4 = EdgeConv(net3, expand=False)

    if globalAvg:
        f = GlobalAveragePooling2D()(graph4)
    else:
        f = Flatten()(graph4)

    f = Dense(512, kernel_initializer='glorot_normal', activation='relu')(f)
    f = Dense(256, kernel_initializer='glorot_normal', activation='relu')(f)
    f = Dense(128, kernel_initializer='glorot_normal', activation='relu')(f)  # TODO ?
    output1 = Dense(numJoints * 3, name='output1', kernel_initializer='glorot_normal')(f)

    model = Model(inputs=input_points, outputs=output1)
    return model


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
    shortcut1 = add([local_feature1_exp, local_feature2_exp, local_feature3])

    if poolTo1:
        shortcut1 = MaxPooling2D(pool_size=(2048, 1))(shortcut1)

    f1 = Conv2D(filters=512, kernel_size=(1, 1),
                kernel_initializer='glorot_normal')(shortcut1)
    f1a = Activation('relu')(f1)

    f2 = Conv2D(filters=256, kernel_size=(1, 1), kernel_initializer='glorot_normal')(f1a)

    f2 = Activation('relu')(f2)

    if globalAvg:
        f = GlobalAveragePooling2D()(f2)
    else:
        f = Flatten()(f2)

    f = Dense(512, kernel_initializer='glorot_normal', activation='relu')(f)
    f = Dense(256, kernel_initializer='glorot_normal', activation='relu')(f)
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
    global_feature = MaxPooling2D(pool_size=(numPoints, 1))(
        local_feature3)  # shape= (b, 1, 1, 2048)
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


def TempConv():
    input_points = Input(shape=(seq_length, numJoints * 3))
    x = Conv1D(filters=C, kernel_size=W, dilation_rate=1, kernel_initializer='glorot_normal')(input_points)
    # x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    # x = Dropout(p)(x)

    for i, length in enumerate(tensor_slices):
        slice = Lambda(lambda x: x[:, -length:, :])(x)

        x = Conv1D(filters=C, kernel_size=W, dilation_rate=W ** (i + 1), kernel_initializer='glorot_normal')(x)
        # x = BatchNormalization(momentum=0.9)(x)  # paper momentum 0.1 to 0.001 ?
        x = Activation('relu')(x)
        x = Dropout(p)(x)

        # TODO remove 1x1 convs ? => fewer parameters
        # x = Conv1D(filters=C, kernel_size=1, dilation_rate=1, kernel_initializer='glorot_normal')(x)
        # # x = BatchNormalization(momentum=0.9)(x)
        # x = Activation('relu')(x)
        # x = Dropout(p)(x)

        x = add([x, slice])

    out = Conv1D(filters=3 * numJoints, kernel_size=1, dilation_rate=1, kernel_initializer='glorot_normal')(x)

    model = Model(inputs=input_points, outputs=out)

    return model


def PclTempConv():
    pass
    # input_points = Input(shape=(seq_length, numPoints * 3))
    # # x = Conv1D(filters=C, kernel_size=W, dilation_rate=1, kernel_initializer='glorot_normal')(input_points)
    # # # x = BatchNormalization(momentum=0.9)(x)
    # # x = Activation('relu')(x)
    # # x = Dropout(p)(x)
    # x = input_points
    #
    # for i, length in enumerate(tensor_slices):
    #     # TODO remove?
    #     slice = Lambda(lambda x: x[:, -length:, :])(x)
    #
    #     x = Reshape([-1, 3])(x)
    #     adj = Lambda(utils.pairwise_distance)(x)
    #     nn_idx = Lambda(utils.knn, arguments={'k': k_neighbors})(adj)
    #     edge_feature = Lambda(utils.get_edge_feature, arguments={'nn_idx': nn_idx, 'k': k_neighbors})(x)
    #
    #     x = Conv1D(filters=C, kernel_size=W, dilation_rate=W**(i + 1), kernel_initializer='glorot_normal')(edge_feature)
    #     # x = BatchNormalization(momentum=0.9)(x)  # paper momentum 0.1 to 0.001 ?
    #     x = Activation('relu')(x)
    #     x = Dropout(p)(x)
    #
    #     x = Lambda(lambda x: Kb.max(x, axis=-2, keepdims=True))(x)
    #
    #     x = Conv1D(filters=C, kernel_size=1, dilation_rate=1, kernel_initializer='glorot_normal')(x)
    #     # # x = BatchNormalization(momentum=0.9)(x)
    #     x = Activation('relu')(x)
    #     # x = Dropout(p)(x)
    #
    #     # TODO remove?
    #     x = add([x, slice])
    #
    # out = Conv1D(filters=3 * numJoints, kernel_size=1, dilation_rate=1, kernel_initializer='glorot_normal')(x)
    #
    # model = Model(inputs=input_points, outputs=out)
    #
    # return model


def lambda_function(x, K):
    weights = Kb.zeros([256, K * K], dtype=np.float32)
    biases = Kb.zeros([K * K], dtype=np.float32)

    biases += Kb.constant(np.eye(K).flatten(), dtype=np.float32)
    transform = Kb.dot(x, weights)
    transform = Kb.bias_add(transform, biases)

    transform = Reshape([K, K])(transform)
    return transform


def dgcnn_transform_net(x, K=3):
    """ Input (XYZ) Transform Net, input is bxNumPointsx3
      Return:
        Transformation matrix of size 3xK """
    # input_image = Kb.expand_dims(x, -1)

    x = Conv2D(64, [1, 1], activation='relu', kernel_initializer='glorot_normal')(x)
    x = Conv2D(128, [1, 1], activation='relu', kernel_initializer='glorot_normal')(x)

    x = Kb.max(x, axis=-2, keepdims=True)

    x = Conv2D(1024, [1, 1], activation='relu', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D([numPoints, 1])(x)

    x = Flatten()(x)

    x = Dense(512, activation='relu', kernel_initializer='glorot_normal')(x)
    x = Dense(256, activation='relu', kernel_initializer='glorot_normal')(x)

    out = Lambda(lambda_function, arguments={'K': K})(x)

    return out


def dgcnnet():  # PE
    input_points = Input(shape=(numPoints, 3))

    input_image = Lambda(lambda x: Kb.expand_dims(x, -2))(input_points)

    adj = Lambda(utils.pairwise_distance)(input_points)
    nn_idx = Lambda(utils.knn, arguments={'k': k_neighbors})(adj)
    edge_feature = Lambda(utils.get_edge_feature, arguments={'nn_idx': nn_idx, 'k': k_neighbors})(input_image)

    transform = Lambda(dgcnn_transform_net, arguments={'K': 3})(edge_feature)
    point_cloud_transformed = Lambda(lambda x: Kb.batch_dot(x[0], x[1]))([input_points, transform])

    input_image = Lambda(lambda x: Kb.expand_dims(x, -2))(point_cloud_transformed)
    adj = Lambda(utils.pairwise_distance)(point_cloud_transformed)
    nn_idx = Lambda(utils.knn, arguments={'k': k_neighbors})(adj)
    edge_feature = Lambda(utils.get_edge_feature, arguments={'nn_idx': nn_idx, 'k': k_neighbors})(input_image)

    out1 = Conv2D(64, [1, 1], activation='relu', kernel_initializer='glorot_normal')(edge_feature)
    out2 = Conv2D(64, [1, 1], activation='relu', kernel_initializer='glorot_normal')(out1)

    net_1 = Lambda(lambda x: Kb.max(x, axis=-2, keepdims=True))(out2)

    adj = Lambda(utils.pairwise_distance)(net_1)
    nn_idx = Lambda(utils.knn, arguments={'k': k_neighbors})(adj)
    edge_feature = Lambda(utils.get_edge_feature, arguments={'nn_idx': nn_idx, 'k': k_neighbors})(net_1)

    out3 = Conv2D(64, [1, 1], activation='relu', kernel_initializer='glorot_normal')(edge_feature)
    out4 = Conv2D(64, [1, 1], activation='relu', kernel_initializer='glorot_normal')(out3)

    net_2 = Lambda(lambda x: Kb.max(x, axis=-2, keepdims=True))(out4)

    # net_2 = add([net_1, net_2])

    adj = Lambda(utils.pairwise_distance)(net_2)
    nn_idx = Lambda(utils.knn, arguments={'k': k_neighbors})(adj)
    edge_feature = Lambda(utils.get_edge_feature, arguments={'nn_idx': nn_idx, 'k': k_neighbors})(net_2)

    out3_2 = Conv2D(64, [1, 1], activation='relu', kernel_initializer='glorot_normal')(edge_feature)
    out4_2 = Conv2D(64, [1, 1], activation='relu', kernel_initializer='glorot_normal')(out3_2)

    net_22 = Lambda(lambda x: Kb.max(x, axis=-2, keepdims=True))(out4_2)

    # net_22 = add([net_2, net_22])

    adj = Lambda(utils.pairwise_distance)(net_22)
    nn_idx = Lambda(utils.knn, arguments={'k': k_neighbors})(adj)
    edge_feature = Lambda(utils.get_edge_feature, arguments={'nn_idx': nn_idx, 'k': k_neighbors})(net_22)

    out3_3 = Conv2D(64, [1, 1], activation='relu', kernel_initializer='glorot_normal')(edge_feature)
    out4_3 = Conv2D(64, [1, 1], activation='relu', kernel_initializer='glorot_normal')(out3_3)

    net_23 = Lambda(lambda x: Kb.max(x, axis=-2, keepdims=True))(out4_3)

    # net_23 = add([net_22, net_23])

    adj = Lambda(utils.pairwise_distance)(net_23)
    nn_idx = Lambda(utils.knn, arguments={'k': k_neighbors})(adj)
    edge_feature = Lambda(utils.get_edge_feature, arguments={'nn_idx': nn_idx, 'k': k_neighbors})(net_23)

    out3_4 = Conv2D(64, [1, 1], activation='relu', kernel_initializer='glorot_normal')(edge_feature)
    out4_4 = Conv2D(64, [1, 1], activation='relu', kernel_initializer='glorot_normal')(out3_4)

    net_24 = Lambda(lambda x: Kb.max(x, axis=-2, keepdims=True))(out4_4)

    # net_24 = add([net_24, net_23])

    adj = Lambda(utils.pairwise_distance)(net_24)
    nn_idx = Lambda(utils.knn, arguments={'k': k_neighbors})(adj)
    edge_feature = Lambda(utils.get_edge_feature, arguments={'nn_idx': nn_idx, 'k': k_neighbors})(net_24)

    out5 = Conv2D(64, [1, 1], activation='relu', kernel_initializer='glorot_normal')(edge_feature)
    # out6 = Conv2D(64, [1, 1], activation='relu', kernel_initializer='glorot_normal')(out5)

    net_3 = Lambda(lambda x: Kb.max(x, axis=-2, keepdims=True))(out5)

    # add1 = add([net_3, net_24])

    concat1 = concatenate([net_1, net_2, net_3])

    out7 = Conv2D(1024, [1, 1], activation='relu', kernel_initializer='glorot_normal')(concat1)

    out_max = MaxPooling2D([numPoints, 1])(out7)

    expand = Lambda(tile, arguments={'numPoints': numPoints})(out_max)  # per-point output

    # categorical vector

    # one_hot_label_expand = Kb.reshape(input_label, [batch_size, 1, 1, cat_num])
    # one_hot_label_expand = tf_util.conv2d(one_hot_label_expand, 64, [1, 1],
    #                                       padding='VALID', stride=[1, 1],
    #                                       bn=True, is_training=is_training,
    #                                       scope='one_hot_label_expand', bn_decay=bn_decay, is_dist=True)
    # out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])
    # expand = tf.tile(out_max, [1, num_point, 1, 1])

    concat2 = concatenate([expand,
                           net_1,
                           net_2,
                           net_3], axis=3)

    net2 = MaxPooling2D([numPoints, 1])(concat2)  # TODO skusit presunut tento maxpool za fullyconnected vrstvy

    net2 = Conv2D(256, [1, 1], activation='relu', kernel_initializer='glorot_normal')(net2)  # 512
    # net2 = Dropout(0.4)(net2)
    net2 = Conv2D(256, [1, 1], activation='relu', kernel_initializer='glorot_normal')(net2)
    net2 = Dropout(0.3)(net2)
    net2 = Conv2D(128, [1, 1], activation='relu', kernel_initializer='glorot_normal')(net2)

    out = Conv2D(numJoints * 3, [1, 1], kernel_initializer='glorot_normal')(net2)
    out = Flatten()(out)

    model = Model(input_points, out)
    return model


def temp_dgcnnet():  # TODO temporal graph - input sequence
    input_points = Input(shape=(seq_length, numPoints * 3))
    # input_points_flatten = Reshape((seq_length * numPoints, 3))(input_points)

    features = []

    for i in range(seq_length):
        x = Lambda(lambda x: x[:, i, :])(input_points)
        x = Reshape([numPoints, 3])(x)
        feat = Lambda(dgcnnet_nobottleneck)(x)
        if i < seq_length - 1:
            coef = Lambda(lambda x: Kb.ones_like(x) * (temp_coeff * (0.05 ** (seq_length - 2 - i))))(feat)
        else:
            coef = Lambda(lambda x: Kb.ones_like(x))(feat)
        features.append(multiply([coef, feat]))

    seq_features = concatenate([*features], axis=1)

    seq_features = Reshape([seq_length * numPoints, 1, -1])(seq_features)
    # seq_features = MaxPooling2D([seq_length, 1])(seq_features)

    # seq_features = Reshape([numPoints, 1, -1])(seq_features)

    out7 = Conv2D(1024, [1, 1], activation='relu', kernel_initializer='glorot_normal')(seq_features)

    out_max = MaxPooling2D([seq_length * numPoints, 1])(out7)

    expand = Lambda(tile, arguments={'numPoints': seq_length * numPoints})(out_max)  # per-point output

    concat2 = concatenate([expand,
                           seq_features], axis=3)

    net2 = MaxPooling2D([seq_length * numPoints, 1])(concat2)

    net2 = Conv2D(256, [1, 1], activation='relu', kernel_initializer='glorot_normal')(net2)  # 512
    # net2 = Dropout(0.4)(net2)
    net2 = Conv2D(256, [1, 1], activation='relu', kernel_initializer='glorot_normal')(net2)
    net2 = Dropout(0.3)(net2)
    net2 = Conv2D(128, [1, 1], activation='relu', kernel_initializer='glorot_normal')(net2)

    out = Conv2D(numJoints * 3, [1, 1], kernel_initializer='glorot_normal')(net2)
    out = Reshape((1, -1))(out)

    model = Model(input_points, out)
    return model


def EdgeConv(input_pts, expand=True):
    if expand:
        input_pts = Lambda(lambda x: Kb.expand_dims(x, -2))(input_pts)
    adj = Lambda(utils.pairwise_distance)(input_pts)
    nn_idx = Lambda(utils.knn, arguments={'k': k_neighbors})(adj)
    edge_feature = Lambda(utils.get_edge_feature, arguments={'nn_idx': nn_idx, 'k': k_neighbors})(input_pts)
    return edge_feature


def dgcnnet_nobottleneck(input_pts):
    """passes input through first 3 graph layers of DGCNN"""
    input_image = Lambda(lambda x: Kb.expand_dims(x, -2))(input_pts)
    adj = Lambda(utils.pairwise_distance)(input_pts)
    nn_idx = Lambda(utils.knn, arguments={'k': k_neighbors})(adj)
    edge_feature = Lambda(utils.get_edge_feature, arguments={'nn_idx': nn_idx, 'k': k_neighbors})(input_image)

    # transform = Lambda(dgcnn_transform_net, arguments={'K': 3})(edge_feature)
    # point_cloud_transformed = Lambda(lambda x: Kb.batch_dot(x[0], x[1]))([input_pts, transform])

    # input_image = Lambda(lambda x: Kb.expand_dims(x, -2))(point_cloud_transformed)
    # adj = Lambda(utils.pairwise_distance)(point_cloud_transformed)
    # nn_idx = Lambda(utils.knn, arguments={'k': k_neighbors})(adj)
    # edge_feature = Lambda(utils.get_edge_feature, arguments={'nn_idx': nn_idx, 'k': k_neighbors})(input_image)

    out1 = Conv2D(64, [1, 1], activation='relu', kernel_initializer='glorot_normal')(edge_feature)
    out2 = Conv2D(64, [1, 1], activation='relu', kernel_initializer='glorot_normal')(out1)

    net_1 = Lambda(lambda x: Kb.max(x, axis=-2, keepdims=True))(out2)

    adj = Lambda(utils.pairwise_distance)(net_1)
    nn_idx = Lambda(utils.knn, arguments={'k': k_neighbors})(adj)
    edge_feature = Lambda(utils.get_edge_feature, arguments={'nn_idx': nn_idx, 'k': k_neighbors})(net_1)

    out3 = Conv2D(128, [1, 1], activation='relu', kernel_initializer='glorot_normal')(edge_feature)
    out4 = Conv2D(128, [1, 1], activation='relu', kernel_initializer='glorot_normal')(out3)

    net_2 = Lambda(lambda x: Kb.max(x, axis=-2, keepdims=True))(out4)

    adj = Lambda(utils.pairwise_distance)(net_2)
    nn_idx = Lambda(utils.knn, arguments={'k': k_neighbors})(adj)
    edge_feature = Lambda(utils.get_edge_feature, arguments={'nn_idx': nn_idx, 'k': k_neighbors})(net_2)

    out5 = Conv2D(128, [1, 1], activation='relu', kernel_initializer='glorot_normal')(edge_feature)
    # out6 = Conv2D(64, [1, 1], activation='relu', kernel_initializer='glorot_normal')(out5)

    net_3 = Lambda(lambda x: Kb.max(x, axis=-2, keepdims=True))(out5)

    concat1 = concatenate([net_1, net_2, net_3])

    return concat1
