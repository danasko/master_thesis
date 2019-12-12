
dataset = 'ITOP'
batch_size = 32
k = 50
centers = None
thresh = 10  # in cm
numPoints = 2048  # number of points in each pcl
singleview = False
# test_method = '11subjects'
test_method = 'random25'

# do NOT set both to True at once
segnet = False
mymodel = True

if dataset == 'MHAD':
    if singleview:
        numTrainSamples = 210917
        numTestSamples = 70305
    elif test_method == '11subjects':
        numTrainSamples = 128865
        numTestSamples = 11746
    else:
        numTrainSamples = 105459
        numTestSamples = 35152
    numValSamples = 0

    numJoints = 29  # 35
    numRegions = 29  # 35
    fill = 6
elif dataset == 'UBC':
    numTrainSamples = 59059
    numValSamples = 19019
    numTestSamples = 19019

    numJoints = 18
    numRegions = 18
    fill = 5
else:  # ITOP
    numTrainSamples = 17991
    numValSamples = 0
    numTestSamples = 4863

    numJoints = 15
    numRegions = 15
    fill = 6
# scaler_minX, scaler_minY, scaler_minZ = None, None, None
# scaler_scaleX, scaler_scaleY, scaler_scaleZ = None, None, None

pcls_min = [1000000, 1000000, 1000000]
pcls_max = [-1000000, -1000000, -1000000]
# pcls_min = None
# pcls_max = None

poses_min = [1000000, 1000000, 1000000]
poses_max = [-1000000, -1000000, -1000000]

# name = 'segnet_lr0.001_4residuals_2.blockconvs512'
name = 'mymodel_lr0.0005_noproto_convs1x1_poolto1_512_256_1residual_nomaxpool_globalavgpool_4chan_reg_preds'

steps = None

# use predicted regions when running 4-chan model
predicted_regs = False

workers = 3  # mp.cpu_count()