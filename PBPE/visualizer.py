import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import keras.backend as Kb
import config
import os
import data_loader

UBC_bone_list = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [5, 9], [6, 7], [7, 8], [9, 10], [1, 12], [10, 11],
                 [1, 15], [12, 13], [13, 14], [15, 16], [16, 17]]

MHAD_bone_list = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [3, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
                  [3, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [0, 19],
                  [19, 20], [20, 21], [21, 22], [22, 23], [0, 24], [24, 25], [25, 26], [26, 27], [27, 28]]

CMU_bone_list = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10], [10, 11], [2, 6], [6, 7], [7, 8], [2, 12],
                 [12, 13], [13, 14]]

AMASS_bone_list = [[0, 1], [0, 2], [0, 3], [3, 6], [9, 13], [9, 6], [14, 9], [9, 12], [12, 15], [14, 17], [17, 19],
                   [19, 21], [13, 16], [16, 18], [18, 20], [1, 4], [2, 5], [5, 8], [8, 11], [4, 7], [7, 10]]


def save_frames(predictions, data, gt_dir, numJoints=29, num=None, fill=6,
                noaxes=False, gt=False):
    azim_min = -160
    azim_max = 200
    if num is None:
        num = range(0, predictions.shape[0])
    azim = (azim_max - azim_min) / len(num)
    elev = 12.
    if data in ['CMU', 'ITOP']:
        pcls_gt = data_loader.unscale_to_cm(gt_dir)
        pcls_gt -= pcls_gt.mean(axis=(0, 1))
        predictions = predictions.reshape((predictions.shape[0], numJoints, 3))
        predictions = data_loader.unscale_to_cm(predictions)
        for i in num:
            visualize_3D(pcls_gt[i], pose=predictions[i], title='', noaxes=noaxes, save_fig=True,
                         name=str(i).zfill(6), azim=azim_min + (azim * (i + 1)), elev=elev, gt=gt, pause=False)
    else:
        for i in num:
            pcl_gt = np.load('data/' + data + '/test/' + gt_dir + '/' + str(i).zfill(fill) + '.npy')
            # pcl_mean = np.load('data/' + data + '/test/notscaledpcl_11subs/' + str(i).zfill(fill) + '.npy').mean(axis=0)
            # pcl_gt = (pcl_gt + 1) * (pcls_max - pcls_min) / 2 + pcls_min

            # pcl_gt += pcl_mean

            # if gt:
            pose_gt = np.load(
                'data/' + data + '/test/notscaledpose/' + str(i).zfill(fill) + '.npy')  # gt
            # else:
            pose_pred = predictions[i].reshape(numJoints, 3)
            # pose_mean = np.load('data/' + data + '/test/notscaledpose_11subs/' + str(i).zfill(fill) + '.npy').mean(
            #     axis=0)
            # pose_pred = data_loader.unscale_to_cm(pose_pred, data=data)

            # pose_pred += pose_mean

            visualize_3D(pcl_gt, pose=pose_pred, title='', noaxes=noaxes, save_fig=True,
                         name=str(i).zfill(6), azim=azim_min + (azim * (i + 1)), elev=elev, gt=pose_gt, pause=False)


def visualize_3D(coords, pause=True, array=False, regions=None, pose=None,
                 title='Visualized pointcloud', ms1=10, ms2=0.03, noaxes=False, azim=-77, elev=12., save_fig=False,
                 name='samplefig', gt=None, dpi=500):  # coords with shape (numPoints, 3)
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.text2D(0.05, 0.95, title, transform=ax.transAxes)
    if array:
        coords = np.reshape(coords, (coords.shape[0], 3))

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    if pose is not None:
        pose = np.reshape(pose, (config.numJoints, 3))
        ax.scatter(pose[:, 0], pose[:, 2], pose[:, 1], c='magenta', marker='o', s=ms1)
    if gt is not None:
        gt = np.reshape(gt, (config.numJoints, 3))
        ax.scatter(gt[:, 0], gt[:, 2], gt[:, 1], c='green', marker='o', s=ms1)
    if regions is not None:
        c = np.zeros((regions.shape[0], 3))
        for j in range(config.numRegions):
            color = np.random.randint(256, size=3)
            for reg in range(regions.shape[0]):
                if regions[reg] == j:
                    c[reg] = color
        ax.scatter(x, z, y, c=c / 255.0, marker='o', s=3)
    elif coords is not None:
        ax.scatter(x, z, y, c='blue', marker='o', s=ms2)
    ax.set_xlabel('x axis')
    ax.set_ylabel('z axis')
    ax.set_zlabel('y axis')
    if noaxes:
        plt.axis('off')

    # Fix aspect ratio
    if coords is not None:
        max_range = np.max(np.array([x.max() - x.min(), y.max() - y.min(),
                                     z.max() - z.min()])) / 2.0
        mean_x = x.mean()
        mean_z = y.mean()
        mean_y = z.mean()
        ax.set_xlim(mean_x - max_range, mean_x + max_range)
        ax.set_ylim(mean_y + max_range, mean_y - max_range)
        ax.set_zlim(mean_z - max_range, mean_z + max_range)
    elif pose is not None:
        x = pose[:, 0]
        y = pose[:, 1]
        z = pose[:, 2]
        max_range = np.max(np.array([x.max() - x.min(), y.max() - y.min(),
                                     z.max() - z.min()])) / 2.0
        mean_x = x.mean()
        mean_z = y.mean()
        mean_y = z.mean()
        ax.set_xlim(mean_x - max_range, mean_x + max_range)
        ax.set_ylim(mean_y + max_range, mean_y - max_range)
        ax.set_zlim(mean_z - max_range, mean_z + max_range)

    bone_list = None
    if pose is not None or gt is not None:
        if config.dataset == 'UBC':
            bone_list = UBC_bone_list
        elif config.dataset == 'MHAD':
            bone_list = MHAD_bone_list
        elif config.dataset == 'CMU':
            bone_list = CMU_bone_list
        elif config.dataset == 'AMASS':
            bone_list = AMASS_bone_list

    if bone_list is not None:
        if pose is not None:
            for bone in bone_list:
                ax.plot([pose[:, 0][bone[0]], pose[:, 0][bone[1]]],
                        [pose[:, 2][bone[0]], pose[:, 2][bone[1]]], [pose[:, 1][bone[0]], pose[:, 1][bone[1]]],
                        'magenta')

        if gt is not None:
            for bone in bone_list:
                ax.plot([gt[:, 0][bone[0]], gt[:, 0][bone[1]]],
                        [gt[:, 2][bone[0]], gt[:, 2][bone[1]]], [gt[:, 1][bone[0]], gt[:, 1][bone[1]]],
                        'green')

    if save_fig:
        ax.view_init(elev=elev, azim=azim)
        plt.savefig('data/' + config.dataset + '/test/figures/' + name + '.png', dpi=dpi)
        plt.close(fig)
    else:
        plt.show()
        if pause:
            plt.pause(0.001)
            input("Press [enter] to show next pcl.")


def visualize_3D_pose(pose, pause=True,
                      title='Visualized pose', noaxes=False, color='green'):  # coords with shape (numPoints, 3)
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.text2D(0.05, 0.95, title, transform=ax.transAxes)
    if pose is not None:
        pose = np.reshape(pose, (config.numJoints, 3))
        ax.scatter(pose[:, 0], pose[:, 2], pose[:, 1], c=color, marker='o', )
        # show node ID labels
        # for node in range(pose.shape[0]):
        #     ax.text(pose[node, 0] + .03, pose[node, 2] + .03, pose[node, 1] + .03, str(node), fontsize=9)

    ax.set_xlabel('x axis')
    ax.set_ylabel('z axis')
    ax.set_zlabel('y axis')

    if noaxes:
        plt.axis('off')

    # Fix aspect ratio
    max_range = np.max(np.array([pose[:, 0].max() - pose[:, 0].min(), pose[:, 1].max() - pose[:, 1].min(),
                                 pose[:, 2].max() - pose[:, 2].min()])) / 2.0
    mean_x = pose[:, 0].mean()
    mean_z = pose[:, 1].mean()
    mean_y = pose[:, 2].mean()
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)

    if pose.shape[0] == 18:  # UBC dataset
        for bone in UBC_bone_list:
            ax.plot([pose[:, 0][bone[0]], pose[:, 0][bone[1]]],
                    [pose[:, 2][bone[0]], pose[:, 2][bone[1]]], [pose[:, 1][bone[0]], pose[:, 1][bone[1]]], 'r')
    elif pose.shape[0] == 29:  # MHAD
        for bone in MHAD_bone_list:
            ax.plot([pose[:, 0][bone[0]], pose[:, 0][bone[1]]],
                    [pose[:, 2][bone[0]], pose[:, 2][bone[1]]], [pose[:, 1][bone[0]], pose[:, 1][bone[1]]], color)
    elif config.dataset == 'CMU':
        for bone in CMU_bone_list:
            ax.plot([pose[:, 0][bone[0]], pose[:, 0][bone[1]]],
                    [pose[:, 2][bone[0]], pose[:, 2][bone[1]]], [pose[:, 1][bone[0]], pose[:, 1][bone[1]]], color)

    plt.show()
    if pause:
        plt.pause(0.001)
        input("Press [enter] to show next pose.")


def visualize_features(pcl, nn_idx, rnd_point_idx=256, noaxes=True):
    """ Args:
       pcl: (batch_size, num_points, 3)
       nn_idx: (batch_size, num_points, num_points)
    """
    for b in range(pcl.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.text2D(0.05, 0.95, title, transform=ax.transAxes)
        coords = pcl[b]

        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        rnd_point = coords[rnd_point_idx]
        ax.scatter(rnd_point[0], rnd_point[2], rnd_point[1], c='red', marker='o', s=20)
        knn = Kb.eval(nn_idx[b, rnd_point_idx])
        knn_ordered = np.argsort(knn)
        ax.scatter(x[knn_ordered[1:]], z[knn_ordered[1:]], y[knn_ordered[1:]], c=-np.sort(knn)[1:], marker='o', s=20)

        # ax.scatter(x, z, y, c='blue', marker='o', s=0.3)
        ax.set_xlabel('x axis')
        ax.set_ylabel('z axis')
        ax.set_zlabel('y axis')

        if noaxes:
            plt.axis('off')

        # Fix aspect ratio
        if coords is not None:
            max_range = np.max(np.array([x.max() - x.min(), y.max() - y.min(),
                                         z.max() - z.min()])) / 2.0
            mean_x = x.mean()
            mean_z = y.mean()
            mean_y = z.mean()
            ax.set_xlim(mean_x - max_range, mean_x + max_range)
            ax.set_ylim(mean_y + max_range, mean_y - max_range)
            ax.set_zlim(mean_z - max_range, mean_z + max_range)

        plt.show()
        plt.pause(0.001)
        input("Press [enter] to show next pcl.")


if __name__ == "__main__":
    poses_min, poses_max = np.load('data/CMU/train/poses_minmax.npy')
    pcls_min, pcls_max = np.load('data/CMU/train/pcls_minmax.npy')
    pcl = np.load('data/CMU/test/scaledpcls.npy')
    pcl = pcl.reshape((2048, 3))
    pcl = (pcl + 1) * (pcls_max - pcls_min) / 2 + pcls_min
    # reg = np.load('data/MHAD/train/region/004011.npy')[15]
    # pose = np.load('data/MHAD/train/scaledposesbatch/004011.npy')[15]
    pose = np.load('data/CMU/test/predictions.npy')[2]
    pose = pose.reshape((config.numJoints, 3))
    pose = (pose + 1) * (poses_max - poses_min) / 2 + poses_min

    visualize_3D(coords=pcl, pose=pose, ms2=0.5, azim=-32, elev=11, title='')
