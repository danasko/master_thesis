import keras.backend as Kb
import numpy as np
from keras.layers.core import Reshape

from config import *
from data_loader import unscale_to_cm


def loss_func_proto(y_true, y_pred):
    clusters = centers.reshape(k, numJoints * 3)
    preds = y_pred @ clusters
    return Kb.mean(Kb.abs(y_true - preds), axis=-1)


def avg_error_proto(y_true, y_pred):
    clusters = centers.reshape(k, numJoints * 3)
    preds = y_pred @ clusters
    return avg_error(y_true, preds)


def avg_error(y_true, y_pred):
    y_pred = Reshape((numJoints, 3))(y_pred)
    y_true = Reshape((numJoints, 3))(y_true)

    y_pred = unscale_to_cm(y_pred)
    y_true = unscale_to_cm(y_true)

    # mean L2 error in cm
    return Kb.mean(Kb.mean(Kb.sqrt(Kb.sum(Kb.square(y_pred - y_true), axis=-1)), axis=-1),
                   axis=-1)


def mean_avg_precision(y_true, y_pred):
    y_pred = Reshape((numJoints, 3))(y_pred)
    y_true = Reshape((numJoints, 3))(y_true)

    y_pred = unscale_to_cm(y_pred)
    y_true = unscale_to_cm(y_true)

    dist = Kb.sqrt(Kb.sum(Kb.square(y_pred - y_true), axis=-1))

    logic = Kb.less_equal(dist, thresh)

    res = Kb.SVitch(logic, Kb.ones_like(dist), Kb.zeros_like(dist))

    return Kb.mean(Kb.mean(res, axis=-1), axis=-1)


def huber_loss(y_true, y_pred):
    clip_delta = 1.0  # 4.0
    error = y_true - y_pred
    cond = Kb.abs(error) < clip_delta

    squared_loss = 0.5 * Kb.square(error)
    linear_loss = clip_delta * (Kb.abs(error) - 0.5 * clip_delta)

    return Kb.mean(Kb.SVitch(cond, squared_loss, linear_loss), axis=-1)


def per_joint_err(preds, gt, save=False):
    preds = preds.reshape((preds.shape[0], numJoints, 3))

    preds = unscale_to_cm(preds)
    gt = unscale_to_cm(gt)

    diff = np.mean(np.sqrt(np.sum(np.square(preds - gt), axis=-1)), axis=0)

    if save:
        np.savetxt('data/' + dataset + '/test/per_joint_err_' + ('SV' if singleview else 'MV') + (
            '_35j' if numJoints == 35 else '') + '.csv', diff, delimiter=',')

