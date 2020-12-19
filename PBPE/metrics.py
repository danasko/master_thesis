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


def avg_error_lstm(y_true, y_pred):
    y_pred = Reshape((-1, numJoints, 3))(y_pred)
    y_true = Reshape((-1, numJoints, 3))(y_true)

    y_pred = unscale_to_cm(y_pred)
    y_true = unscale_to_cm(y_true)

    # mean L2 error in cm
    return Kb.mean(Kb.mean(Kb.mean(Kb.sqrt(Kb.sum(Kb.square(y_pred - y_true), axis=-1)), axis=-1),
                   axis=-1), axis=-1)


def velocity_error(gt_poses, predictions):
    predictions = predictions.reshape((-1, numJoints, 3))
    gt_poses = gt_poses.reshape((-1, numJoints, 3))

    predictions = unscale_to_cm(predictions)
    gt_poses = unscale_to_cm(gt_poses)

    diff_predictions = np.diff(predictions, axis=0)
    diff_gt_poses = np.diff(gt_poses, axis=0)

    # MPJPE of the first derivative of 3d pose sequences (if test set comprise of only one sequence!)
    return np.mean(np.mean(np.sqrt(np.sum((diff_predictions - diff_gt_poses) ** 2, axis=-1)), axis=-1),
                   axis=-1)


def max_error(gt_poses, predictions):
    predictions = predictions.reshape((-1, numJoints, 3))
    gt_poses = gt_poses.reshape((-1, numJoints, 3))

    predictions = unscale_to_cm(predictions)
    gt_poses = unscale_to_cm(gt_poses)

    return np.max(np.max(np.sqrt(np.sum((predictions - gt_poses) ** 2, axis=-1)), axis=-1),
                  axis=-1)


def velocity_error_sequences(gt_poses, predictions, seq_idx):
    err = 0
    num_seq = len(seq_idx)
    seq_idx = seq_idx.insert(0)  # first sequence from the start frame
    for start_id in range(len(seq_idx) - 1):
        gt_seq = gt_poses[seq_idx[start_id]:seq_idx[start_id + 1]]
        pred_seq = predictions[seq_idx[start_id]:seq_idx[start_id + 1]]
        err += velocity_error(gt_seq, pred_seq)

    # mean velocity error over all sequences
    return err / num_seq


def temporal_avg_error(y_true, y_pred):
    y_pred = Reshape((numJoints, 3))(y_pred)
    y_true = Reshape((numJoints, 3))(y_true[:, -1, :])

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

    res = Kb.switch(logic, Kb.ones_like(dist), Kb.zeros_like(dist))

    return Kb.mean(Kb.mean(res, axis=-1), axis=-1)


def mean_avg_precision_lstm(y_true, y_pred):
    y_pred = Reshape((-1, numJoints, 3))(y_pred)
    y_true = Reshape((-1, numJoints, 3))(y_true)

    y_pred = unscale_to_cm(y_pred)
    y_true = unscale_to_cm(y_true)

    dist = Kb.sqrt(Kb.sum(Kb.square(y_pred - y_true), axis=-1))

    logic = Kb.less_equal(dist, thresh)

    res = Kb.switch(logic, Kb.ones_like(dist), Kb.zeros_like(dist))

    return Kb.mean(Kb.mean(Kb.mean(res, axis=-1), axis=-1), axis=-1)


def temporal_mean_avg_precision(y_true, y_pred):
    y_pred = Reshape((numJoints, 3))(y_pred)
    y_true = Reshape((numJoints, 3))(y_true[:, -1, :])

    y_pred = unscale_to_cm(y_pred)
    y_true = unscale_to_cm(y_true)

    dist = Kb.sqrt(Kb.sum(Kb.square(y_pred - y_true), axis=-1))

    logic = Kb.less_equal(dist, thresh)

    res = Kb.switch(logic, Kb.ones_like(dist), Kb.zeros_like(dist))

    return Kb.mean(Kb.mean(res, axis=-1), axis=-1)


def temporal_huber_loss(y_true,
                        y_pred):
    clip_delta = 1.0  # 4.0
    error = y_true[:, -1, :] - y_pred
    cond = Kb.abs(error) < clip_delta

    squared_loss = 0.5 * Kb.square(error)
    linear_loss = clip_delta * (Kb.abs(error) - 0.5 * clip_delta)

    error_temp = y_true[:, -2, :] - y_pred
    cond_temp = Kb.abs(error_temp) < clip_delta

    squared_loss_temp = 0.5 * Kb.square(error_temp)
    linear_loss_temp = clip_delta * (Kb.abs(error_temp) - 0.5 * clip_delta)

    # return spatial + temporal error
    return Kb.mean(Kb.switch(cond, squared_loss, linear_loss), axis=-1) + temp_coeff * Kb.mean(
        Kb.switch(cond_temp, squared_loss_temp, linear_loss_temp), axis=-1)


def temporal_mae_loss(y_true, y_pred):
    temp_coeffs = np.empty((seq_length - 1))
    c = temp_coeff
    for i in range(seq_length - 1, 0, -1):
        temp_coeffs[i - 1] = c
        c *= 0.05  # 0.5

    temp_coeffs = np.stack([temp_coeffs] * batch_size, axis=0)
    gt_pose = y_true[:, -1, :]
    previous_poses = y_true[:, :-1, :]
    y_pred = Reshape((numJoints * 3,))(y_pred)
    error = Kb.mean(Kb.abs(gt_pose - y_pred), axis=-1)
    y_pred = Kb.stack([y_pred] * (seq_length - 1), axis=1)

    temp_error = Kb.sum(temp_coeffs * Kb.mean(Kb.abs(previous_poses - y_pred), axis=-1), axis=-1)

    return error + temp_error


def huber_loss(y_true, y_pred):
    clip_delta = 1.0  # 4.0
    error = y_true - y_pred
    cond = Kb.abs(error) < clip_delta

    squared_loss = 0.5 * Kb.square(error)
    linear_loss = clip_delta * (Kb.abs(error) - 0.5 * clip_delta)

    return Kb.mean(Kb.switch(cond, squared_loss, linear_loss), axis=-1)


def per_joint_err(preds, gt, save=False):
    preds = preds.reshape((preds.shape[0], numJoints, 3))

    preds = unscale_to_cm(preds)
    gt = unscale_to_cm(gt)

    diff = np.mean(np.sqrt(np.sum(np.square(preds - gt), axis=-1)), axis=0)

    if save:
        np.savetxt('data/' + dataset + '/test/per_joint_err_' + ('SV' if singleview else 'MV') + (
            '_35j' if numJoints == 35 else '') + '.csv', diff, delimiter=',')


def st_deviation(gt_poses, predictions):
    predictions = predictions.reshape((-1, numJoints, 3))
    gt_poses = gt_poses.reshape((-1, numJoints, 3))

    predictions = unscale_to_cm(predictions)
    gt_poses = unscale_to_cm(gt_poses)

    return np.std(np.mean(np.sqrt(np.sum((predictions - gt_poses) ** 2, axis=-1)), axis=-1), axis=-1)
