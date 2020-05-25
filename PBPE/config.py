dataset = 'UBC'
batch_size = 32
thresh = 10  # in cm
numPoints = 2048  # number of points per pcl
singleview = False
# test_method = '11subjects'
test_method = 'random25'
leaveout = 12

# do NOT set both to True at once
sgpe_seg = True
sgpe_reg = False

stepss = None

run_training = False
predict_on_segnet = False

assert not sgpe_seg or not sgpe_reg

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
else:  # 'CMU'
    poolTo1 = False
    globalAvg = True
    numTrainSamples = 112596
    numValSamples = 0
    numTestSamples = 28148

    numJoints = 15
    numRegions = 15
    fill = 6

st = '_4000steps' if stepss is not None else ''
view = 'SV_' if singleview else ''
jts = '35j' if numJoints == 35 else ''
if sgpe_seg:
    name = view + ('11subs' + str(leaveout) + '_' if (
            dataset == 'MHAD' and test_method == '11subjects') else '') + jts + 'segnet' + st
elif sgpe_reg:
    name = view + ('11subs' + str(leaveout) + '_' if (
            dataset == 'MHAD' and test_method == '11subjects') else '') + jts + 'sgpe_reg'
    # 'poolto1' if poolTo1 else 'nomaxpool') + (
    # '_globalavgpool' if globalAvg else '')
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
