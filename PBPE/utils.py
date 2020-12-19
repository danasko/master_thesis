import keras.backend as Kb
import numpy as np
import tensorflow as tf
import os

from config import *


def pairwise_distance(pcl):
    """Compute pairwise distance of a point cloud.
      Args:
        point_cloud: tensor (batch_size, num_points, num_dims)
      Returns:
        pairwise distance: (batch_size, num_points, num_points)"""
    og_batch_size = pcl.shape[0]
    if og_batch_size == 1:
        pcl = Kb.expand_dims(pcl, 0)
    if len(pcl.shape) == 4:
        pcl = Kb.squeeze(pcl, axis=-2)
    point_cloud_transpose = Kb.permute_dimensions(pcl, [0, 2, 1])
    point_cloud_inner = Kb.batch_dot(pcl, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = Kb.sum(Kb.square(pcl), axis=-1, keepdims=True)
    point_cloud_square_tranpose = Kb.permute_dimensions(point_cloud_square, [0, 2, 1])
    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


# def top_k(input, k):
#     """Top k max pooling
#     Args:
#         input(ndarray): convolutional feature in heigh x width x channel format
#         k(int): if k==1, it is equal to normal max pooling
#     Returns:
#         ndarray: k x (height x width)
#     """
#     # input = Kb.reshape(input, [-1, input.shape[-1]])
#     print(input.shape)
#     input = np.sort(input, axis=-1)[::-1, :][:k, :]
#     return input


def knn(adj_matrix, k):
    """Get KNN based on the pairwise distance.
     Args:
       pairwise distance: (batch_size, num_points, num_points)
       k: int
     Returns:
       nearest neighbors: (batch_size, num_points, k)
     """
    neg_adj = -adj_matrix
    _, nn_idx = tf.nn.top_k(neg_adj, k=k)
    return nn_idx


def get_edge_feature(point_cloud, nn_idx, k):
    """Construct edge feature for each point
     Args:
       point_cloud: (batch_size, num_points, 1, num_dims)
       nn_idx: (batch_size, num_points, k)
       k: int
     Returns:
       edge features: (batch_size, num_points, k, num_dims)
     """
    og_batch_size = point_cloud.shape[0]
    point_cloud = Kb.squeeze(point_cloud, axis=-2)
    if og_batch_size == 1:
        point_cloud = Kb.expand_dims(point_cloud, 0)

    point_cloud_central = point_cloud

    num_dims = point_cloud.shape[-1]
    num_pts = point_cloud.shape[1]
    idx_b = Kb.zeros_like(nn_idx)
    idx_b = Kb.reshape(idx_b, [-1, num_pts * k])
    idx_b = idx_b[:, 0]

    idx_b.values = Kb.arange(batch_size) * num_pts
    idx_b = Kb.reshape(idx_b, [-1, 1, 1])

    # print(idx_b.shape, nn_idx.shape)
    point_cloud_flat = Kb.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = Kb.gather(point_cloud_flat, nn_idx + idx_b)
    point_cloud_central = Kb.expand_dims(point_cloud_central, axis=-2)

    point_cloud_central = Kb.tile(point_cloud_central, [1, 1, k, 1])
    edge_feature = Kb.concatenate([point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    # print(point_cloud_neighbors.shape)
    return edge_feature
