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

temp_convs = False
pcl_temp_convs = True

stepss = None

run_training = True
predict_on_segnet = False
load_trained_model = False

assert not sgpe_seg or not sgpe_reg

# Temporal Convolutions
ordered = True  # whether to use ordered version of *CMU* dataset (subsequent frames)
seq_length = 81  # TODO skusit iny pocet framov # 243, 81, 27, 9
tensor_slices = [73, 55, 1]  # [235, 217, 163, 1] [73, 55, 1] [19, 1], [1]
W = 3  # TODO skusit ine okno v case
C = 2048  # 1024
p = 0.2  # 0.25
###

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

    numJoints = 13
    numRegions = 13
    fill = 7

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
    name = pcl + 'temp_convs_seq' + str(seq_length) + '_' + str(len(tensor_slices)) + 'blocks_W' + str(W) + '_C' + str(
        C) + '_p' + str(p) + '_reset_seqs_nobatchnorm_lrdrop0.8_wo1x1convs_lr0.00005'
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
