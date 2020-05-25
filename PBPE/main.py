from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.backend.tensorflow_backend import set_session
from keras.utils import plot_model, HDF5Matrix, CustomObjectScope
from keras.utils.generic_utils import get_custom_objects
import keras.callbacks
from sklearn.utils import shuffle
import time

from models import *
from metrics import *
from visualizer import *
from data_generator import *
from data_loader import *
from config import *
from ITOP_data_loader import load_ITOP_from_npy


def real_time_init(model, first_sample):
    """ build model and run on first sample beforehand to achieve higher speed (especially on one by one feed)
     than with model.predict """
    get_output = Kb.function([model.layers[0].input, Kb.learning_phase()], [model.layers[-1].output])
    _ = get_output([first_sample, 0])[0]
    return get_output


def real_time_predict(test_x, get_output_func):
    """ run on the rest of test samples """
    # tic = time.time()
    model_output = get_output_func([test_x, 0])[0]
    # tac = time.time() - tic
    # print(tac)
    return model_output


def run_sgpe_seg(generator, x, mode='test', save=True):
    """ predict regions with segmentation model and save """
    jts = ('35j' if numJoints == 35 else '')
    subs = (('_11subs' + str(leaveout)) if test_method == '11subjects' else '')
    view = ('SV' if singleview else '')
    split = ''

    if test_method == '11subjects':
        if mode == 'train':
            num = len(os.listdir('data/' + dataset + '/' + mode + '/scaledpcls' + view + subs + 'batch/'))
        else:
            num = len(os.listdir('data/' + dataset + '/' + mode + '/scaledpcls' + view + subs))
    else:
        num = numTrainSamples // batch_size

    sgpe_seg_model = load_model(
        'data/models/' + dataset + '/10eps_' + (
            'SV' if singleview else '') + subs + jts + 'segnet.h5')
    get_output = Kb.function([sgpe_seg_model.layers[0].input, Kb.learning_phase()], [sgpe_seg_model.layers[-1].output])
    if mode == 'train' and dataset != 'ITOP' and dataset != 'CMU':
        for b_num in range(num):
            pcl_batch = np.load(
                'data/' + dataset + '/' + mode + '/scaledpcls' + view + subs + 'batch/' + str(
                    b_num + 1).zfill(fill) + '.npy')
            # pred = sgpe_seg_model.predict(pcl_batch, batch_size=batch_size, steps=None)
            pred = get_output([pcl_batch, 0])[0]
            pred = np.argmax(pred, -1)
            pred = np.expand_dims(pred, -1)
            if save:
                np.save(
                    'data/' + dataset + '/' + mode + '/region' + view + jts + subs + '_predicted_batch/' + str(
                        b_num + 1).zfill(
                        fill) + '.npy', pred)
    else:
        if dataset == 'ITOP' or dataset == 'CMU':
            pred1 = sgpe_seg_model.predict(x[:x.shape[0] // 2], batch_size=batch_size, verbose=1).argmax(
                axis=-1).astype(
                np.int)
            # pred1 = get_output([x[:x.shape[0] // 2], 0])[0].argmax(axis=-1).astype(np.int)
            # pred2 = get_output([x[x.shape[0] // 2:], 0])[0].argmax(axis=-1).astype(np.int)
            pred2 = sgpe_seg_model.predict(x[x.shape[0] // 2:], batch_size=batch_size, verbose=1).argmax(
                axis=-1).astype(
                np.int)
            pred = np.concatenate([pred1, pred2], axis=0)
            pred = np.expand_dims(pred, -1)
        else:
            if generator.split == 1:
                split = '1'
            elif generator.split == 2:
                split = '2'
            pred = sgpe_seg_model.predict_generator(generator, use_multiprocessing=True, steps=None, workers=workers,
                                                    verbose=1)
            pred = np.argmax(pred, axis=-1).astype(np.int)
            pred = np.expand_dims(pred, -1)
        if save:
            np.save('data/' + dataset + '/' + mode + '/predicted_regs' + view + jts + subs + '_p' + split + '.npy',
                    pred)
        return pred


def step_decay(epoch):
    """ learning rate schedule """
    if sgpe_reg and dataset == 'ITOP':
        initial_lrate = 0.0007
    elif sgpe_seg or sgpe_reg:
        initial_lrate = 0.001
    else:
        initial_lrate = 0.0005
    drop = 0.8
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

    Adam = Adam(lr=0.0, decay=0.0)  # to be set in lrScheduler

    if sgpe_seg:
        model = SGPE_segnet()
        metrics = ['accuracy']
        lossf = 'categorical_crossentropy'
        losSV = [1.]
    elif sgpe_reg:
        model = SGPE(poolTo1=poolTo1, globalAvg=globalAvg)
        metrics = [avg_error, mean_avg_precision]
        lossf = 'mean_absolute_error'
        losSV = [1.]
    else:
        model, test_model = PBPE_new()
        metrics = {'output1': [avg_error, mean_avg_precision], 'output2': 'accuracy'}
        lossf = {"output1": "mean_absolute_error",  # huber_loss, mean_squared_error
                 "output2": "categorical_crossentropy",
                 }
        losSV = [1.0, 0.01]  # original 0.1
        test_model.compile(optimizer=Adam,
                           loss="mean_absolute_error", metrics=[avg_error, mean_avg_precision])

    model.summary(line_length=100)

    get_custom_objects().update(
        {'avg_error': avg_error, 'Kb': Kb, 'mean_avg_precision': mean_avg_precision})
    #  'loss_func_proto': loss_func_proto,'avg_error_proto': avg_error_proto, 'huber_loss': huber_loss

    model.compile(optimizer=Adam,
                  loss=lossf, loss_weights=losSV,
                  metrics=metrics)

    # Callbacks
    lrate = LearningRateScheduler(step_decay)

    tbCallBack = keras.callbacks.TensorBoard(log_dir='data/tensorboard/' + dataset + '/' + name, histogram_freq=0,
                                             write_graph=True,
                                             write_images=True, write_grads=False, batch_size=batch_size)

    checkpoint = keras.callbacks.ModelCheckpoint('data/models/' + dataset + '/{epoch:02d}eps_' + name + '.h5',
                                                 verbose=1,
                                                 period=1)

    callbacks_list = [lrate, tbCallBack, checkpoint]

    # Load trained model

    # model = load_model('data/models/' + dataset + '/10eps_sgpe_reg.h5')

    # test_model = load_model(
    #     'data/models/' + dataset + '/test_models/20eps_net_PBPE_new.h5')

    if dataset == 'ITOP':
        train_x, train_y, train_regs, test_x, test_y, test_regs = load_ITOP_from_npy()
        test_pred_regs = np.load('data/ITOP/test/predicted_regs.npy')
        test_x_exp = np.concatenate([test_x, test_pred_regs], axis=-1)

        if run_training:
            pred_regs = np.load('data/ITOP/train/predicted_regs.npy')
            train_x_exp = np.concatenate([train_x, pred_regs], axis=-1)
            model.fit(train_x_exp, train_y, batch_size=batch_size, epochs=20, callbacks=callbacks_list,
                      # validation_split=0.1
                      validation_data=(test_x_exp, test_y),
                      shuffle=True, initial_epoch=0)
        # [testloss, testavg_err, tmap] = model.evaluate(test_x_exp, test_y, batch_size=batch_size)
        # print('Test avg error: ', testavg_err)
        preds = model.predict(test_x_exp, batch_size=batch_size)
        # np.save('data/ITOP/test/predictions.npy', preds)
    elif dataset == 'CMU':
        regs_train = np.load('data/CMU/train/regs_onehot.npy', allow_pickle=True)
        x_train = np.load('data/CMU/train/scaled_pcls_lzeromean.npy', allow_pickle=True)
        x_train = np.expand_dims(x_train, axis=2)
        y_train = np.load('data/CMU/train/scaled_poses_lzeromean.npy', allow_pickle=True)
        y_train = y_train.reshape((y_train.shape[0], numJoints * 3))
        # one-hot encoding
        # regs_train = np.eye(numRegions, dtype=np.int)[regs_train]
        # regs_train = regs_train.reshape((regs_train.shape[0], numPoints, 1, numRegions))
        # np.save('data/CMU/train/regs_onehot.npy', regs_train)

        if run_training:
            if sgpe_seg:
                model.fit(x_train, regs_train, batch_size=batch_size,
                          epochs=20,
                          callbacks=callbacks_list,
                          validation_split=0.2, shuffle=True, initial_epoch=0)
            elif sgpe_reg:
                # regs_train_pred = run_sgpe_seg(None, x_train, mode='train', save=True)
                regs_train_pred = np.load('data/CMU/train/predicted_regs.npy', allow_pickle=True).astype(np.int)
                x_train = np.concatenate([x_train, regs_train_pred], axis=-1)
                model.fit(x_train, y_train, batch_size=batch_size,
                          epochs=20,
                          callbacks=callbacks_list,
                          validation_split=0.2, shuffle=True, initial_epoch=0)
            else:  # PBPE
                model.fit(x_train, {'output1': y_train, 'output2': regs_train}, batch_size=batch_size,
                          epochs=10,
                          callbacks=callbacks_list,
                          validation_split=0.2, shuffle=True, initial_epoch=0)
        # load test data
        x_test = np.load('data/CMU/test/scaled_pcls_lzeromean.npy', allow_pickle=True)
        x_test = np.expand_dims(x_test, axis=2)
        y_test = np.load('data/CMU/test/scaled_poses_lzeromean.npy', allow_pickle=True)
        y_test = y_test.reshape((y_test.shape[0], numJoints * 3))

        # test on 171204_pose6 sequence - video results
        # x_test = np.load('data/CMU/test/171204_pose6_scaledpcls_lzeromean.npy', allow_pickle=True)
        # x_test = np.expand_dims(x_test, axis=2)
        # y_test = np.load('data/CMU/test/171204_pose6_scaledposes_lzeromean.npy', allow_pickle=True)
        # y_test = y_test.reshape((y_test.shape[0], numJoints * 3))

        if sgpe_seg:
            regs_test = np.load('data/CMU/test/regions.npy', allow_pickle=True)
            # regs_test = np.load('data/CMU/test/171204_pose6_regs.npy', allow_pickle=True)
            regs_test = np.eye(numRegions, dtype=np.int)[regs_test]
            regs_test = regs_test.reshape((regs_test.shape[0], numPoints, 1, numRegions))
            test_metrics = model.evaluate(x_test, regs_test, batch_size=batch_size)
        elif sgpe_reg:
            # regs_test_pred = run_sgpe_seg(None, x_test, mode='test', save=True)
            regs_test_pred = np.load('data/CMU/test/predicted_regs.npy', allow_pickle=True).astype(np.int)
            # regs_test_pred = np.load('data/CMU/test/171204_pose6_predicted_regs.npy', allow_pickle=True).astype(np.int)
            x_test = np.concatenate([x_test, regs_test_pred], axis=-1)
            test_metrics = model.evaluate(x_test, y_test, batch_size=batch_size)
            # preds = model.predict(x_test, batch_size=batch_size, verbose=1)
            # np.save('data/CMU/test/171204_pose6_predictions.npy', preds)
        else:  # PBPE
            test_metrics = test_model.evaluate(x_test, y_test, batch_size=batch_size)
    else:
        train_generator = DataGenerator('data/' + dataset + '/train/', numPoints, numJoints, numRegions, steps=stepss,
                                        batch_size=batch_size,
                                        shuffle=True, fill=fill, loadbatch=True, singleview=singleview,
                                        elevensubs=(test_method == '11subjects'), sgpe_seg=sgpe_seg,
                                        four_channels=sgpe_reg,
                                        predicted_regs=predicted_regs)
        if dataset == 'UBC':
            if not singleview:
                valid_generator = DataGenerator('data/' + dataset + '/valid/', numPoints, numJoints, numRegions,
                                                steps=stepss,
                                                batch_size=batch_size, fill=fill, singleview=singleview,
                                                shuffle=False, sgpe_seg=sgpe_seg, four_channels=sgpe_reg)

        test_generator = DataGenerator('data/' + dataset + '/test/', numPoints, numJoints, numRegions, steps=stepss,
                                       batch_size=batch_size, shuffle=False, fill=fill, singleview=singleview,
                                       test=True, elevensubs=(test_method == '11subjects'), sgpe_seg=sgpe_seg,
                                       four_channels=sgpe_reg, predicted_regs=predicted_regs, split=1)
        if run_training:
            model.fit_generator(generator=train_generator, epochs=20,
                                callbacks=callbacks_list, initial_epoch=0, use_multiprocessing=True,
                                workers=workers, shuffle=True, max_queue_size=0)
        if predict_on_segnet:
            run_sgpe_seg(test_generator, None, 'test', True)

    # Save model ##########

    # model.save(
    #     'data/models/' + dataset + '/' + name + '.h5') # is saved during checkpoints

    # test_model.save('data/models/' + dataset + '/test_models/' + name + '.h5')

    # Predict ###########

    # preds = model.predict_generator(test_generator, verbose=1, steps=None, use_multiprocessing=True, workers=workers)
    # preds = np.load('data/' + dataset + '/test/predictions.npy')
    # gt = y_test.reshape((y_test.shape[0], numJoints, 3))
    # gt = np.empty((preds.shape[0], numJoints, 3))
    # for i in range(preds.shape[0]):
    #     gt[i] = np.load('data/' + dataset + '/test/scaledposes/' + str(i).zfill(fill) + '.npy')

    # per_joint_err(preds, gt, save=False)
    # np.save('data/' + dataset + '/test/predictions.npy', preds)

    # Evaluate model (only regression branch) ###########

    # eval_metrics = model.evaluate_generator(test_generator, verbose=1, steps=None, use_multiprocessing=True,
    #                                         workers=workers)
    # print('Test mean error: ', eval_metrics[1])
    # print('Test mAP@10cm: ', eval_metrics[2])
