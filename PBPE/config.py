dataset = 'CMU'
batch_size = 32
thresh = 10  # in cm
numPoints = 512  # number of points per pcl #2048
singleview = False
# test_method = '11subjects'
test_method = 'random25'
leaveout = 12

# do NOT set both to True at once
sgpe_seg = False
sgpe_reg = False

temp_convs = 0
pcl_temp_convs = 0
dgcnn = 0
temp_dgcnn = 0
graph_sgpe = 0
temp_graph_sgpe = 0
lstm = 1

stepss = None
num_eps = 80

run_training = 1
predict_on_segnet = 0
load_trained_model = 0

assert not sgpe_seg or not sgpe_reg

# Temporal Convolutions
if temp_convs or pcl_temp_convs:
    ordered = True  # whether to use ordered version of *CMU* dataset (subsequent frames)
    seq_length = 9  # 243, 81, 27, 9
    tensor_slices = [1]  # [235, 217, 163, 1] [73, 55, 1] [19, 1], [1]
    W = 3  # time window
    C = 2048  # 1024
    if pcl_temp_convs:
        C = 512
        seq_length = 27
        tensor_slices = [19, 1]
    p = 0.1  # 0.25
    temporal_loss = True
    temp_coeff = .25
###

# DGCNN
elif dgcnn:
    k_neighbors = 20
    temporal_loss = False
    ordered = False
    seq_length = 1
###
elif temp_dgcnn or temp_graph_sgpe:
    ordered = True
    k_neighbors = 20
    seq_length = 5
    temporal_loss = True
    temp_coeff = .25
elif graph_sgpe:
    ordered = False
    k_neighbors = 20
    temporal_loss = False
    seq_length = 1
elif lstm:
    n_units = 256  # TODO validate
    ordered = True
    seq_length = 32
else:
    temporal_loss = False
    ordered = False
    seq_length = 1
    n_units = None

if dataset == 'MHAD':
    poolTo1 = True
    globalAvg = False
    if singleview:
        numTrainSamples = 210917
        numTestSamples = 70305
    else:
        numTrainSamples = 105459
        numTestSamples = 35152
    numValSamples = 0

    numJoints = 29  # 35
    numRegions = numJoints
    fill = 6
elif dataset == 'UBC':
    poolTo1 = False
    globalAvg = True
    if singleview:
        numTrainSamples = 177177
        numValSamples = 57057
        numTestSamples = 57057
    else:
        numTrainSamples = 59059
        numValSamples = 19019
        numTestSamples = 19019

    numJoints = 18
    numRegions = 18
    fill = 5
elif dataset == 'ITOP':
    poolTo1 = False
    globalAvg = True
    numTrainSamples = 17991
    numValSamples = 0
    numTestSamples = 4863

    numJoints = 15
    numRegions = 15
    fill = 6
elif dataset == 'CMU':
    poolTo1 = False
    globalAvg = True
    numTrainSamples = 112596
    numValSamples = 0
    numTestSamples = 28148

    numJoints = 15
    numRegions = 15
    fill = 6
else:  # AMASS virtual dataset
    poolTo1 = False
    globalAvg = True
    numTrainSamples = 80224
    numTestSamples = 20032

    numJoints = 22
    numRegions = 22
    fill = 6

st = '_4000steps' if stepss is not None else ''
view = 'SV_' if singleview else ''
jts = '35j' if numJoints == 35 else ''
if sgpe_seg:
    name = view + ('11subs' + str(leaveout) + '_' if (
            dataset == 'MHAD' and test_method == '11subjects') else '') + jts + 'segnet_' + str(numPoints) + 'pts' + st
elif sgpe_reg:
    name = view + ('11subs' + str(leaveout) + '_' if (
            dataset == 'MHAD' and test_method == '11subjects') else '') + jts + 'sgpe_reg_' + str(numPoints) + 'pts'
    # 'poolto1' if poolTo1 else 'nomaxpool') + (
    # '_globalavgpool' if globalAvg else '')
elif temp_convs or pcl_temp_convs:
    pcl = ''
    if pcl_temp_convs:
        pcl = 'pcl_'
    temp_loss = ''
    if temporal_loss:
        temp_loss = '_temp_mae_loss_coeff_' + str(temp_coeff) + '_err_from_whole_seq_smooth_c0.05'
    name = pcl + 'temp_convs_seq' + str(seq_length) + '_' + str(len(tensor_slices)) + 'blocks_W' + str(W) + '_C' + str(
        C) + '_p' + str(p) + '_reset_seqs_nobatchnorm_lrdrop0.8_wo1x1convs_lr0.0001_wofirstDO_leakyrelu' + temp_loss
elif dgcnn:
    name = 'dgcnn_knn' + str(k_neighbors) + '_PE_mlp256_256_128_6convblocks_cosdecay_concatall'
elif temp_dgcnn:
    name = 'dgcnn_knn' + str(k_neighbors) + '_seq' + str(
        seq_length) + '_weightedsepbranches_v2_womaxpool_lr0.0001_orderedseq_cosdecay_wotransform_128convs'
elif graph_sgpe:
    name = 'graph_sgpe_knn' + str(k_neighbors) + '_3denses'
elif temp_graph_sgpe:
    name = 'temp_graph_sgpe_knn' + str(k_neighbors) + '_seq_length_' + str(seq_length) + '_3denses'
elif lstm:
    name = 'lstm_seq_length_' + str(seq_length) + '_' + str(n_units) + 'units_lr0.0001_tanh_recurrenttanh_Nonefirstdim'
else:
    name = view + ('11subs' + str(leaveout) + '_' if (
            dataset == 'MHAD' and test_method == '11subjects') else '') + jts + 'net_PBPE_new'

# use predicted regions when running 4-chan model - only applied when using generator
if sgpe_reg and (dataset in ['MHAD', 'UBC']):
    predicted_regs = True
else:
    predicted_regs = False

workers = 4  # mp.cpu_count()

# prototype clustering
k = 50  # num clusters
centers = None
