from keras.models import load_model
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler
from keras.backend.tensorflow_backend import set_session
from keras.utils import plot_model, HDF5Matrix, CustomObjectScope
from keras.utils.generic_utils import get_custom_objects
import keras.callbacks
from sklearn.utils import shuffle
import time

# import wandb
from wandb.keras import WandbCallback

# wandb.init(project="automatic-3d-human-pose-estimation-skeleton-tracking-and-body-measurements")

from models import *
from metrics import *
from visualizer import *
from data_generator import *
from data_generator_seq import *
from data_generator_lstm import *
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
        'data/models/' + dataset + '/20eps_' + (
            'SV' if singleview else '') + subs + jts + 'segnet_' + str(numPoints) + 'pts.h5')
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
            if save:
                if (temp_convs or pcl_temp_convs) and ordered:
                    np.save('data/' + dataset + '/' + mode + '/ordered/predicted_regs' + view + jts + subs + str(
                        numPoints) + 'pts.npy', pred)
                else:
                    np.save(
                        'data/' + dataset + '/' + mode + '/predicted_regs' + view + jts + subs + str(
                            numPoints) + 'pts.npy', pred)
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
                np.save(
                    'data/' + dataset + '/' + mode + '/predicted_regs' + view + jts + subs + '_p' + split + '_' + str(
                        numPoints) + 'pts.npy', pred)
        return pred


def predict_sequence(enc_input):
    decoder_input = np.zeros([batch_size, 1, numJoints * 3])  # start tag # TODO try ones instead - also while training
    states = encoder_model.predict(enc_input)
    out_seq = np.empty([batch_size, seq_length, numJoints * 3])
    for j in range(seq_length):
        decoder_output, h, c = decoder_model.predict([decoder_input] + states)
        out_seq[:, j, :] = decoder_output[:, 0, :]
        states = [h, c]
        decoder_input = decoder_output

    return out_seq


def lstm_inference():
    # TODO predikovat postupne sekvencie pcls z test set, decoder input bude vzdy po jednom predosly decoder output (pre 1. frame zeros)
    res = 0
    num_batches = len(test_generator_lstm)
    # test_err = model.evaluate_generator(test_generator_lstm, verbose=1)
    # print(test_err)

    for i in range(num_batches):
        [x_test, _], y_test = test_generator_lstm[i]
        # out_seq = model.predict([x_test, dec_input])
        out_seq = predict_sequence(x_test)

        err = Kb.eval(avg_error_lstm(Kb.constant(y_test), Kb.constant(out_seq)))
        print(err)
        res += err

        # print('avg error so far: ', res / (i + 1))

    print('avg error: ', res / num_batches)
    return res / num_batches


def cos_step_decay(epoch):
    initial_lr = 0.0005
    if temp_dgcnn or lstm:
        initial_lr = 0.0001
    clip = 0.00001
    lrate = 0.5 * initial_lr * (
            1 + np.cos(np.pi * epoch / float(num_eps)))
    if lrate < clip:
        lrate = clip
    return lrate


def step_decay(epoch):
    """ learning rate schedule """
    if sgpe_reg and dataset == 'ITOP':
        initial_lrate = 0.0007
    elif sgpe_seg or sgpe_reg:  # or temp_convs:
        initial_lrate = 0.001
    elif pcl_temp_convs:
        initial_lrate = 0.00005
    elif temp_convs:
        initial_lrate = 0.0001
    elif temp_dgcnn:
        initial_lrate = 0.0001
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

    Adam = Adam(lr=0.0, decay=0.0,
                amsgrad=(temp_convs or pcl_temp_convs))  # to be set in lrScheduler # TODO try dgcnn with amsgrad
    # SGD = SGD(0.1, momentum=.9)

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
    elif pcl_temp_convs:
        model = PclTempConv()
        metrics = [avg_error, mean_avg_precision]
        lossf = 'mean_absolute_error'
        losSV = [1.]
    elif temp_convs:
        model = TempConv()
        metrics = [temporal_avg_error, temporal_mean_avg_precision]
        if temporal_loss:
            # lossf = temporal_huber_loss
            lossf = temporal_mae_loss
        else:
            lossf = 'mean_absolute_error'
        losSV = [1.]
    elif temp_dgcnn:
        model = temp_dgcnnet()
        metrics = [temporal_avg_error, temporal_mean_avg_precision]
        if temporal_loss:
            lossf = temporal_mae_loss
        else:
            lossf = 'mean_absolute_error'
        losSV = [1.]
    elif dgcnn:
        model = dgcnnet()
        metrics = [avg_error, mean_avg_precision]
        lossf = 'mean_absolute_error'
        losSV = [1.]
    elif graph_sgpe:
        model = graph_SGPE()
        metrics = [avg_error, mean_avg_precision]
        lossf = 'mean_absolute_error'
        losSV = [1.]
    elif temp_graph_sgpe:
        model = temp_graph_SGPE()
        metrics = [temporal_avg_error, temporal_mean_avg_precision]
        if temporal_loss:
            lossf = temporal_mae_loss
        else:
            lossf = 'mean_absolute_error'
        losSV = [1.]
    elif lstm:
        model, encoder_model, decoder_model = define_lstm_models(n_units)
        metrics = [avg_error_lstm, mean_avg_precision_lstm]
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

    model.summary(line_length=200)

    get_custom_objects().update(
        {'avg_error': avg_error, 'Kb': Kb, 'mean_avg_precision': mean_avg_precision,
         'temporal_huber_loss': temporal_huber_loss, 'temporal_mae_loss': temporal_mae_loss,
         'temporal_avg_error': temporal_avg_error,
         'temporal_mean_avg_precision': temporal_mean_avg_precision, 'avg_error_lstm': avg_error_lstm,
         'mean_avg_precision_lstm': mean_avg_precision_lstm})
    #  'loss_func_proto': loss_func_proto,'avg_error_proto': avg_error_proto, 'huber_loss': huber_loss

    model.compile(optimizer=Adam,
                  loss=lossf, loss_weights=losSV,
                  metrics=metrics)

    # Callbacks
    lrate = LearningRateScheduler(cos_step_decay)  # TODO step_decay

    tbCallBack = keras.callbacks.TensorBoard(log_dir='data/tensorboard/' + dataset + '/' + name, histogram_freq=0,
                                             write_graph=True,
                                             write_images=True, write_grads=False, batch_size=batch_size)

    if dgcnn or graph_sgpe or temp_graph_sgpe:
        checkpoint = keras.callbacks.ModelCheckpoint('data/models/' + dataset + '/{epoch:02d}eps_' + name + '.h5',
                                                     verbose=1,
                                                     period=10, save_weights_only=True)
    else:
        checkpoint = keras.callbacks.ModelCheckpoint('data/models/' + dataset + '/{epoch:02d}eps_' + name + '.h5',
                                                     verbose=1,
                                                     period=10)

    callbacks_list = [lrate, tbCallBack, checkpoint]

    # Load trained model
    if load_trained_model:
        if dgcnn or graph_sgpe:
            model.load_weights('data/models/' + dataset + '/20eps_' + name + '.h5')
        elif temp_convs:
            model = load_model('data/models/' + dataset + '/80eps_' + name + '.h5')
        elif lstm:
            model = load_model('data/models/' + dataset + '/100eps_' + name + '.h5')
        else:
            model = load_model('data/models/' + dataset + '/10eps_' + name + '.h5')
        if not (sgpe_seg or sgpe_reg or temp_convs or pcl_temp_convs or dgcnn or lstm):
            test_model = load_model(
                'data/models/' + dataset + '/test_models/20eps_net_PBPE_new.h5')

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
    elif dataset == 'CMU':  # TODO refactor this spaghetti code !!
        if run_training:
            if pcl_temp_convs:
                # load ordered train data
                # x_train = np.load('data/CMU/train/ordered/preds_seq.npy', allow_pickle=True)
                # y_train = np.load('data/CMU/train/ordered/scaled_poses_lzeromean.npy', allow_pickle=True)
                train_seqs_idx = [21612, 35519, 42700, 69303, 79909, 88456]
                test_seqs_idx = [27082, 52288]
                train_generator_seq = DataGeneratorSeq('data/CMU/train/ordered/', numJoints, train_seqs_idx, batch_size,
                                                       pcls=True)
                model.fit_generator(train_generator_seq, epochs=num_eps, callbacks=callbacks_list,
                                    shuffle=False, initial_epoch=0, workers=workers)
            elif temp_convs or temp_dgcnn or temp_graph_sgpe:
                # load ordered train data
                # x_train = np.load('data/CMU/train/ordered/preds_seq.npy', allow_pickle=True)
                # y_train = np.load('data/CMU/train/ordered/scaled_poses_lzeromean.npy', allow_pickle=True)
                train_seqs_idx = [21612, 35519, 42700, 69303, 79909, 88456]
                train_generator_seq = DataGeneratorSeq('data/CMU/train/ordered/', numJoints, train_seqs_idx, batch_size,
                                                       gt_sequence=temporal_loss, pcls=(temp_dgcnn or temp_graph_sgpe))
                test_seqs_idx = [27082, 52288]
                test_generator_seq = DataGeneratorSeq('data/CMU/test/ordered/', numJoints, test_seqs_idx, batch_size,
                                                      gt_sequence=temporal_loss, pcls=(temp_dgcnn or temp_graph_sgpe))
                model.fit_generator(train_generator_seq, epochs=num_eps, callbacks=callbacks_list,
                                    shuffle=False, initial_epoch=0,
                                    workers=workers)  # validation_data=test_generator_seq)
            elif lstm:
                train_seqs_idx = [21612, 35519, 42700, 69303, 79909, 88456]
                train_generator_lstm = DataGeneratorLSTM('data/CMU/train/ordered/', numJoints, train_seqs_idx,
                                                         batch_size)
                test_seqs_idx = [27082, 52288]
                test_generator_lstm = DataGeneratorLSTM('data/CMU/test/ordered/', numJoints, test_seqs_idx,
                                                        batch_size)
                test_generator_encoder = DataGeneratorLSTM('data/CMU/test/ordered/', numJoints, test_seqs_idx,
                                                           batch_size, encoder=True)
                model.fit_generator(train_generator_lstm, epochs=num_eps, callbacks=callbacks_list,
                                    shuffle=False, initial_epoch=0, validation_data=test_generator_lstm)
            elif dgcnn or graph_sgpe:
                x_train = np.load('data/CMU/train/scaled_pcls_lzeromean_' + str(numPoints) + 'pts.npy',
                                  allow_pickle=True)
                y_train = np.load('data/CMU/train/scaled_poses_lzeromean.npy', allow_pickle=True)
                y_train = y_train.reshape((-1, numJoints * 3))

                x_test = np.load('data/CMU/test/scaled_pcls_lzeromean_' + str(numPoints) + 'pts.npy', allow_pickle=True)
                y_test = np.load('data/CMU/test/scaled_poses_lzeromean.npy', allow_pickle=True)
                y_test = y_test.reshape((-1, numJoints * 3))

                model.fit(x_train, y_train, batch_size=batch_size,
                          epochs=num_eps,
                          callbacks=callbacks_list, shuffle=True, initial_epoch=0, validation_data=(x_test, y_test))

            else:
                regs_train = np.load('data/CMU/train/regs_onehot_' + str(numPoints) + 'pts.npy', allow_pickle=True)
                # regs_train = np.load('data/CMU/train/regions_' + str(numPoints) + 'pts.npy', allow_pickle=True).astype(
                #     np.int)
                x_train = np.load('data/CMU/train/scaled_pcls_lzeromean_' + str(numPoints) + 'pts.npy',
                                  allow_pickle=True)
                x_train = np.expand_dims(x_train, axis=2)
                y_train = np.load('data/CMU/train/scaled_poses_lzeromean.npy', allow_pickle=True)
                y_train = y_train.reshape((-1, numJoints * 3))
                # one-hot encoding
                # regs_train = np.eye(numRegions, dtype=np.int)[regs_train]
                # regs_train = regs_train.reshape((regs_train.shape[0], numPoints, 1, numRegions))
                # np.save('data/CMU/train/regs_onehot_' + str(numPoints) + 'pts.npy', regs_train)

                if sgpe_seg:
                    model.fit(x_train, regs_train, batch_size=batch_size,
                              epochs=num_eps,
                              callbacks=callbacks_list,
                              validation_split=0.2, shuffle=True, initial_epoch=0)
                elif sgpe_reg:
                    if predict_on_segnet:
                        regs_train_pred = run_sgpe_seg(None, x_train, mode='train', save=True)
                    else:
                        regs_train_pred = np.load('data/CMU/train/predicted_regs_' + str(numPoints) + 'pts.npy',
                                                  allow_pickle=True).astype(np.int)
                    x_train = np.concatenate([x_train, regs_train_pred], axis=-1)
                    model.fit(x_train, y_train, batch_size=batch_size,
                              epochs=num_eps,
                              callbacks=callbacks_list,
                              validation_split=0.2, shuffle=True, initial_epoch=0)

                else:  # PBPE
                    model.fit(x_train, {'output1': y_train, 'output2': regs_train}, batch_size=batch_size,
                              epochs=num_eps,
                              callbacks=callbacks_list,
                              validation_split=0.2, shuffle=True, initial_epoch=0)
        # load test data
        # x_test = np.load('data/CMU/test/scaled_pcls_lzeromean.npy', allow_pickle=True)
        # x_test = np.expand_dims(x_test, axis=2)
        # y_test = np.load('data/CMU/test/scaled_poses_lzeromean.npy', allow_pickle=True)
        # y_test = y_test.reshape((y_test.shape[0], numJoints * 3))

        # test on 171204_pose6 sequence - video results
        # x_test = np.load('data/CMU/test/171204_pose6_scaledpcls_lzeromean.npy', allow_pickle=True)
        # x_test = np.expand_dims(x_test, axis=2)
        # y_test = np.load('data/CMU/test/171204_pose6_scaledposes_lzeromean.npy', allow_pickle=True)
        # y_test = y_test.reshape((y_test.shape[0], numJoints * 3))
        if load_trained_model:
            # load ordered test data
            if ordered:
                ord = 'ordered/'
            else:
                ord = ''
            x_test = np.load('data/CMU/test/' + ord + 'scaled_pcls_lzeromean_' + str(numPoints) + 'pts.npy',
                             allow_pickle=True)
            x_test = np.expand_dims(x_test, axis=2)
            y_test = np.load('data/CMU/test/' + ord + 'scaled_poses_lzeromean.npy', allow_pickle=True)
            y_test = y_test.reshape((y_test.shape[0], numJoints * 3))

            if sgpe_seg:
                regs_test = np.load('data/CMU/test/' + ord + 'regions.npy', allow_pickle=True)
                # regs_test = np.load('data/CMU/test/171204_pose6_regs.npy', allow_pickle=True)
                regs_test = np.eye(numRegions, dtype=np.int)[regs_test]
                regs_test = regs_test.reshape((regs_test.shape[0], numPoints, 1, numRegions))
                test_metrics = model.evaluate(x_test, regs_test, batch_size=batch_size)
            elif sgpe_reg:
                if predict_on_segnet:
                    regs_test_pred = run_sgpe_seg(None, x_test, mode='test', save=True)
                else:
                    regs_test_pred = np.load('data/CMU/test/' + ord + 'predicted_regs.npy', allow_pickle=True).astype(
                        np.int)
                    # regs_test_pred = np.load('data/CMU/test/171204_pose6_predicted_regs.npy', allow_pickle=True).astype(np.int)
                x_test = np.concatenate([x_test, regs_test_pred], axis=-1)
                # test_metrics = model.evaluate(x_test, y_test, batch_size=batch_size)
                # preds = model.predict(x_test, batch_size=batch_size, verbose=1)

                # poses_min, poses_max = np.load('data/CMU/train/poses_minmax.npy')
                # pcls_min, pcls_max = np.load('data/CMU/train/pcls_minmax.npy')
                #
                # pose = (preds[0] + 1) * (poses_max - poses_min) / 2 + poses_min
                # pcl = x_test[0]
                # pcl = (pcl + 1) * (pcls_max - pcls_min) / 2 + pcls_min
                #
                # visualize_3D(coords=pcl, pose=pose, ms2=0.5, azim=-32, elev=11, title='')

                # np.save('data/CMU/test/ordered/predictions.npy', preds)
            elif temp_convs:
                # PE_pred_test = np.load('data/CMU/test/ordered/predictions.npy',
                #                        allow_pickle=True)
                test_seqs_idx = [27082, 52288]
                test_generator_seq = DataGeneratorSeq('data/CMU/test/ordered/', numJoints, test_seqs_idx, batch_size,
                                                      gt_sequence=temporal_loss)
                test_metrics = model.evaluate_generator(test_generator_seq, verbose=1)
                print('Temporal test loss: ', test_metrics[0])
                print('Test mean error: ', test_metrics[1])
                print('Test mAP@10cm: ', test_metrics[2])
            elif dgcnn:
                x_test = np.squeeze(x_test, axis=2)
                # test_metrics = model.evaluate(x_test, y_test)
                # print('Test loss: ', test_metrics[0])
                # print('Test mean error: ', test_metrics[1])
                # print('Test mAP@10cm: ', test_metrics[2])
                get_output = Kb.function([model.layers[0].input, Kb.learning_phase()],
                                         [model.layers[21].output])  # 13, 17
                out = get_output([x_test[200:200 + batch_size], 0])[0]
                adj_matrix = utils.pairwise_distance(out)
                # TODO visualize all points with feature distance from ref. point
                # nn_idx = utils.knn(adj_matrix, k_neighbors)
                visualize_features(x_test[200:200 + batch_size], adj_matrix, rnd_point_idx=100)

            elif lstm:
                test_seqs_idx = [27082, 52288]
                test_generator_encoder = DataGeneratorLSTM('data/CMU/test/ordered/', numJoints, test_seqs_idx,
                                                           batch_size, encoder=True)
                test_generator_decoder = DataGeneratorLSTM('data/CMU/test/ordered/', numJoints, test_seqs_idx,
                                                           batch_size, decoder=True)
                test_generator_lstm = DataGeneratorLSTM('data/CMU/test/ordered/', numJoints, test_seqs_idx,
                                                        batch_size)
                lstm_inference()

            elif not (temp_convs or pcl_temp_convs):  # PBPE
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

        if dataset == 'AMASS':
            test_generator = DataGenerator('data/' + dataset + '/test/', numPoints, numJoints, numRegions, steps=stepss,
                                           batch_size=batch_size, shuffle=False, fill=fill, singleview=singleview,
                                           test=True, elevensubs=(test_method == '11subjects'), sgpe_seg=sgpe_seg,
                                           four_channels=sgpe_reg, predicted_regs=predicted_regs, split=1,
                                           loadbatch=True)
        else:
            test_generator = DataGenerator('data/' + dataset + '/test/', numPoints, numJoints, numRegions, steps=stepss,
                                           batch_size=batch_size, shuffle=False, fill=fill, singleview=singleview,
                                           test=True, elevensubs=(test_method == '11subjects'), sgpe_seg=sgpe_seg,
                                           four_channels=sgpe_reg, predicted_regs=predicted_regs, split=1)
        if run_training:
            model.fit_generator(generator=train_generator, epochs=num_eps,
                                callbacks=callbacks_list, initial_epoch=0, use_multiprocessing=True,
                                workers=workers, shuffle=True, max_queue_size=0)
        if predict_on_segnet:
            run_sgpe_seg(train_generator, None, 'train', True)

        # Evaluate model (only regression branch) ###########
        if load_trained_model:
            eval_metrics = model.evaluate_generator(test_generator, verbose=1, steps=None, use_multiprocessing=True,
                                                    workers=workers)
            print('Test mean error: ', eval_metrics[1])
            print('Test mAP@10cm: ', eval_metrics[2])

    # Save model ##########

    # model.save(
    #     'data/models/' + dataset + '/' + name + '.h5') # is saved on checkpoints

    # test_model.save('data/models/' + dataset + '/test_models/' + name + '.h5')

    # Save model to wandb ##########
    # model.save(os.path.join(wandb.run.dir, name + ".h5"))

    # Predict ###########

    # preds = model.predict_generator(test_generator, verbose=1, steps=None, use_multiprocessing=True, workers=workers)
    # preds = np.load('data/' + dataset + '/test/predictions.npy')
    # gt = y_test.reshape((y_test.shape[0], numJoints, 3))
    # gt = np.empty((preds.shape[0], numJoints, 3))
    # for i in range(preds.shape[0]):
    #     gt[i] = np.load('data/' + dataset + '/test/scaledposes/' + str(i).zfill(fill) + '.npy')

    # per_joint_err(preds, gt, save=False)
    # np.save('data/' + dataset + '/test/predictions.npy', preds)
