import numpy as np
from scipy.spatial import distance
import visualizer
from sklearn.cluster import KMeans
import random
import rpy2
# print(rpy2.__version__)
# from rpy2.robjects.packages import importr
# rdist = importr('rdist')

color_map = [
    [255, 106, 0],
    [255, 0, 0],
    [255, 178, 127],
    [255, 127, 127],
    [182, 255, 0],
    [218, 255, 127],
    [255, 216, 0],
    [255, 233, 127],
    [0, 148, 255],
    [72, 0, 255],
    [48, 48, 48],
    [76, 255, 0],
    [0, 255, 33],
    [0, 255, 255],
    [0, 255, 144],
    [178, 0, 255],
    [127, 116, 63],
    [127, 63, 63],
    [127, 201, 255],
    [127, 255, 255],
    [165, 255, 127],
    [127, 255, 197],
    [214, 127, 255],
    [161, 127, 255],
    [107, 63, 127],
    [63, 73, 127],
    [63, 127, 127],
    [109, 127, 63],
    [255, 127, 237],
    [127, 63, 118],
    [0, 74, 127],
    [255, 0, 110],
    [0, 127, 70],
    [127, 0, 0],
    [33, 0, 127],
    [127, 0, 55],
    [38, 127, 0],
    [127, 51, 0],
    [64, 64, 64],
    [73, 73, 73],
    [0, 0, 0],
    [191, 168, 247],
    [192, 192, 192],
    [127, 63, 63],
    [127, 116, 63]
]


def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)


# def incremental_farthest_search(points, k):
#     remaining_points = points[:]
#     solution_set = []
#     solution_set.append(remaining_points.pop(
#         np.random.randint(0, len(remaining_points) - 1)))
#     for _ in range(k - 1):
#         distances = [distance(p, solution_set[0]) for p in remaining_points]
#         for i, p in enumerate(remaining_points):
#             for j, s in enumerate(solution_set):
#                 distances[i] = min(distances[i], distance(p, s))
#         solution_set.append(remaining_points.pop(distances.index(max(distances))))
#     return solution_set

def subsample_random(pcl, numPoints):
    indices = np.random.randint(0, pcl.shape[0], numPoints)
    return pcl[indices]


def subsample(pcl, numPoints, regions=None):
    farthest_pts = np.empty((numPoints, 3))

    p = np.random.randint(pcl.shape[0])
    farthest_pts[0] = pcl[p]

    distances = calc_distances(farthest_pts[0], pcl)

    for i in range(1, numPoints):
        farthest_pts[i] = pcl[np.argmax(distances)]
        # if regions is not None:
        #     newregions[i] = regions[np.argmax(distances)]

        distances = np.minimum(distances, calc_distances(farthest_pts[i], pcl))

    return farthest_pts  # , pcls_min, pcls_max


def region_mapping(regionPartition):  # shape (n, 3)
    """ convert initial RGB values to the region number"""
    colors = np.asarray(color_map)
    regs = np.repeat(-1, regionPartition.shape[0])
    for i in range(45):  # each region
        # print(np.where(abs(regionPartition[:, 0] - colors[i, 0]) < 3))
        cond = np.intersect1d(np.intersect1d(np.where(np.abs(regionPartition[:, 0] - colors[i, 0]) < 3), np.where(
            np.abs(regionPartition[:, 1] - colors[i, 1]) < 3)), np.where(
            np.abs(regionPartition[:, 2] - colors[i, 2]) < 3))
        regs[cond] = i

    regs[regs == -1] = 45  # points not belonging to any body part
    return regs


def automatic_annotation(pose, pcl):
    """automatic segmentation of the model to the specified number of body regions
        each point belongs to the region defined by the closest joint location"""
    regs = np.empty(shape=(pcl.shape[0], 1), dtype=int)
    # for each point find closest joint
    for i in range(regs.shape[0]):
        position = pcl[i]
        dists = np.sqrt(np.sum((pose - position) ** 2, axis=-1))
        regs[i] = np.argmin(dists)

    return regs


def interpolate_joints(pose):
    """generate new joints to overall number of 29 by interpolating the existing ones"""
    pose2 = np.empty((29, 3))
    pose2[:pose.shape[0]] = pose
    idx = pose.shape[0]

    pairs = [[2, 15, 50], [2, 12, 50], [6, 7, 40], [9, 10, 40], [8, 7, 40], [11, 10, 40], [15, 16, 25], [12, 13, 25],
             [13, 14, 25], [16, 17, 25], [1, 2, 20]]
    for i in range(len(pairs)):
        pose2[idx] = (pose[pairs[i][0]] - pose[pairs[i][1]]) / (100 / pairs[i][2])
    visualizer.visualize_3D_pose(pose)


def cluster(data, k, numSamples, numJoints, batch_size, fill):
    # load all train poses
    arr = np.empty((numSamples, numJoints * 3))
    for i in range(numSamples // batch_size):
        arr[i * batch_size:(i * batch_size) + batch_size] = np.load(
            'data/' + data + '/train/scaledposesbatch/' + str(i + 1).zfill(fill) + '.npy')

    # cluster into k clusters
    clusters = cluster_prototypes(arr, k)
    clusters = np.reshape(clusters, (k, numJoints, 3))
    np.save('data/' + data + '/train/pose_clusters.npy', clusters)
    return clusters


def cluster_prototypes(train_labels, k):  # input shape (#samples, J, 3) - train_data labels, K - no. of clusters
    kmeans = KMeans(n_clusters=k, random_state=128, init='k-means++')
    kmeans.fit(train_labels)
    kmeans.predict(train_labels)
    centers = kmeans.cluster_centers_

    # centers = np.reshape(centers, (k, j, 3))
    # centers = np.transpose(centers, (1, 0, 2))
    # centers = centers.T

    return centers.astype(np.float32)
