import numpy as np
from scipy.spatial import distance
from scipy.io import loadmat

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
    # return np.sqrt(np.sum((points - p0) ** 2, axis=-1))
    return ((p0 - points) ** 2).sum(axis=1)  # -1


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


def subsample(pcl, numPoints, pcls_min, pcls_max, regions=None):  # pose,
    farthest_pts = np.empty((numPoints, 3))
    if regions is not None:
        pcl = pcl[regions != 45]
        regions = regions[regions != 45]
    p = np.random.randint(pcl.shape[0])
    farthest_pts[0] = pcl[p]
    if regions is not None:
        newregions = np.empty((numPoints, 1))
        newregions[0] = regions[p]

    distances = calc_distances(farthest_pts[0], pcl)

    for i in range(1, numPoints):
        farthest_pts[i] = pcl[np.argmax(distances)]
        if regions is not None:
            newregions[i] = regions[np.argmax(distances)]

        distances = np.minimum(distances, calc_distances(farthest_pts[i], pcl))

    pcls_min = np.minimum(pcls_min, np.min(farthest_pts, axis=0))
    pcls_max = np.maximum(pcls_max, np.max(farthest_pts, axis=0))

    if regions is not None:
        return farthest_pts, pcls_min, pcls_max, newregions

    return farthest_pts, pcls_min, pcls_max


def region_mapping(regionPartition):  # shape (n, 3)
    """ convert initial RGB values to the region number"""
    colors = np.asarray(color_map)
    regs = np.repeat(-1, regionPartition.shape[0])
    for i in range(45):  # each region
        # print(np.where(abs(regionPartition[:, 0] - colors[i, 0]) < 3))
        cond = np.intersect1d(np.intersect1d(np.where(np.abs(regionPartition[:, 0] - colors[i, 0]) < 3), np.where(
            np.abs(regionPartition[:, 1] - colors[i, 1]) < 3)), np.where(
            np.abs(regionPartition[:, 2] - colors[i, 2]) < 3))
        # rgbstr = ''.join([str(j) for j in regionPartition[i]])
        # regs[i] = region_map[rgbstr]
        regs[cond] = i

    regs[regs == -1] = 45  # points not belonging to any body part
    # print(len(regs[regs == 45]))
    return regs


def automatic_annotation(pose, pcl):
    """automatic segmentation of the model to the specified number of body regions
        each point belongs to the region defined by the closest joint location"""
    regs = np.empty(shape=(pcl.shape[0], 1), dtype=int)
    # for each point find closest joint
    for i in range(regs.shape[0]):
        # region = None
        # mindist = None
        position = pcl[i]
        dists = np.sqrt(np.sum((pose - position) ** 2, axis=-1))
        regs[i] = np.argmin(dists)
        # for j in range(pose.shape[0]):
        #     dist = distance.euclidean(position, pose[j])
        #     if mindist is None or dist < mindist:
        #         mindist = dist
        #         region = j  # associated with j-th region
        # regs[i] = region
    return regs


def interpolate_joints(pose, numJoints):
    """generate new joints to overall number of {numJoints} by interpolating the existing ones"""
    # TODO
