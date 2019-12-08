import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import keras.backend as Kb


def visualize_3D(coords, pause=True, pose=None, numJoints=18,
                 title='Visualized pointcloud', ms1=10, ms2=0.03, noaxes=False):  # coords with shape (numPoints, 3)
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.text2D(0.05, 0.95, title, transform=ax.transAxes)

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    if pose is not None:
        pose = np.reshape(pose, (numJoints, 3))
        ax.scatter(pose[:, 0], pose[:, 2], pose[:, 1], c='r', marker='x', s=ms1)
    # if regions is not None:
    #
    #     C = np.stack([regions] * 3,
    #                  axis=-1)  # shape = (numPoints, 1)  (number of corresponding joint representing the region)
    #     C = np.reshape(C, (regions.shape[0], 3))
    #     for j in range(numRegions):
    #         #     C[C == [j, j, j]] = (j * 7)
    #         color = np.random.randint(256, size=3)
    #         for a in range(C.shape[0]):
    #             if np.array_equal(C[a], [j, j, j]):
    #                 C[a] = color
    #     ax.scatter(x, z, y, c=C / 255.0, marker='o', s=3)
    # else:
    # x = Kb.eval(x)
    # y = Kb.eval(y)
    # z = Kb.eval(z)
    ax.scatter(x, z, y, c='blue', marker='o', s=ms2)

    ax.set_xlabel('x axis')
    ax.set_ylabel('z axis')
    ax.set_zlabel('y axis')
    if noaxes:
        plt.axis('off')

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
                      title='Visualized pose', noaxes=False, color='green'):  # coords with shape (numPoints, 3)
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.text2D(0.05, 0.95, title, transform=ax.transAxes)
    if pose is not None:
        pose = np.reshape(pose, (numJoints, 3))
        ax.scatter(pose[:, 0], pose[:, 2], pose[:, 1], c=color, marker='o')

    ax.set_xlabel('x axis')
    ax.set_ylabel('z axis')
    ax.set_zlabel('y axis')

    if noaxes:
        plt.axis('off')

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

    # for i in range(numJoints):
    #     ax.text(pose[i, 0], pose[i, 2], pose[i, 1], '%s' % (str(i)), size=10, zorder=1,
    #             color='k')

    plt.show()
    if pause:
        plt.pause(0.001)
        input("Press [enter] to show next pose.")
