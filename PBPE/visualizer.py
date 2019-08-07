import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import keras.backend as Kb

UBC_bone_list = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [5, 9], [6, 7], [7, 8], [9, 10], [1, 12], [10, 11],
                 [1, 15], [12, 13], [13, 14], [15, 16], [16, 17]]
numRegions = 45


def visualize_3D(coords, pause=True, array=False, regions=None, pose=None, numJoints=18,
                 title='Visualized pointcloud'):  # coords with shape (numPoints, 3)
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.text2D(0.05, 0.95, title, transform=ax.transAxes)
    if array:
        # x = coords[:, :, 0]
        # y = coords[:, :, 1]
        # z = coords[:, :, 2]
        coords = np.reshape(coords, (coords.shape[0], 3))
        # else:
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    if pose is not None:
        pose = np.reshape(pose, (numJoints, 3))
        ax.scatter(pose[:, 0], pose[:, 2], pose[:, 1], c='r', marker='x')
    if regions is not None:

        C = np.stack([regions] * 3,
                     axis=-1)  # shape = (numPoints, 1)  (number of corresponding joint representing the region)
        C = np.reshape(C, (regions.shape[0], 3))
        for j in range(numRegions):
            #     C[C == [j, j, j]] = (j * 7)
            color = np.random.randint(256, size=3)
            for a in range(C.shape[0]):
                if np.array_equal(C[a], [j, j, j]):
                    C[a] = color
        ax.scatter(x, z, y, c=C / 255.0, marker='o', s=3)
    else:
        ax.scatter(x, z, y, c='r', marker='o', s=3)
    ax.set_xlabel('x axis')
    ax.set_ylabel('z axis')
    ax.set_zlabel('y axis')

    # Fix aspect ratio
    max_range = np.array([x.max() - x.min(), y.max() - y.min(),
                          z.max() - z.min()]).max() / 2.0
    mean_x = x.mean()
    mean_z = y.mean()
    mean_y = z.mean()
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)
    #
    # plt.xlim(-100, 100)
    # ax.set_zlim3d(-100, 100)
    # plt.ylim(-100, 100)

    plt.show()
    if pause:
        plt.pause(0.001)
        input("Press [enter] to show next pcl.")


def visualize_3D_pose(pose, pause=True, numJoints=18,
                      title='Visualized pose'):  # coords with shape (numPoints, 3)
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.text2D(0.05, 0.95, title, transform=ax.transAxes)
    if pose is not None:
        pose = np.reshape(pose, (numJoints, 3))
        ax.scatter(pose[:, 0], pose[:, 2], pose[:, 1], c='r', marker='x')

    ax.set_xlabel('x axis')
    ax.set_ylabel('z axis')
    ax.set_zlabel('y axis')

    # Fix aspect ratio
    max_range = np.array([pose[:, 0].max() - pose[:, 0].min(), pose[:, 1].max() - pose[:, 1].min(),
                          pose[:, 2].max() - pose[:, 2].min()]).max() / 2.0
    mean_x = pose[:, 0].mean()
    mean_z = pose[:, 1].mean()
    mean_y = pose[:, 2].mean()
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)
    #
    # plt.xlim(-100, 100)
    # ax.set_zlim3d(-100, 100)
    # plt.ylim(-100, 100)
    if pose.shape[0] == 18:  # ITOP dataset
        for bone in UBC_bone_list:
            ax.plot([pose[:, 0][bone[0]], pose[:, 0][bone[1]]],
                    [pose[:, 2][bone[0]], pose[:, 2][bone[1]]], [pose[:, 1][bone[0]], pose[:, 1][bone[1]]], 'r')

    for i in range(numJoints):
        ax.text(pose[i, 0], pose[i, 2], pose[i, 1], '%s' % (str(i)), size=10, zorder=1,
                color='k')

    plt.show()
    if pause:
        plt.pause(0.001)
        input("Press [enter] to show next pose.")
