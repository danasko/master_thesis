import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import keras.backend as Kb
import config
import data_loader

UBC_bone_list = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [5, 9], [6, 7], [7, 8], [9, 10], [1, 12], [10, 11],
                 [1, 15], [12, 13], [13, 14], [15, 16], [16, 17]]

MHAD_bone_list = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [3, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
                  [3, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [0, 19],
                  [19, 20], [20, 21], [21, 22], [22, 23], [0, 24], [24, 25], [25, 26], [26, 27], [27, 28]]

CMU_bone_list = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10], [10, 11], [2, 6], [6, 7], [7, 8], [2, 12],
                 [12, 13], [13, 14]]


def save_frames(predictions, data, gt_dir, pcls_min=None, pcls_max=None, numJoints=29, num=None, fill=6,
                noaxes=False, gt=False):
    azim_min = -160
    azim_max = 200
    if num is None:
        num = range(0, predictions.shape[0])
    azim = (azim_max - azim_min) / len(num)
    elev = 12.
    if data == 'CMU':
        pcls_gt = np.load(gt_dir, allow_pickle=True)
        pcls_gt = data_loader.unscale_to_cm(pcls_gt, data='CMU')
        predictions = data_loader.unscale_to_cm(predictions, data='CMU')
        for i in num:
            visualize_3D(pcls_gt[i], pose=predictions[i], title='', noaxes=noaxes, save_fig=True,
                         name=str(i).zfill(6), azim=azim_min + (azim * (i + 1)), elev=elev, gt=gt, pause=False)
    else:
        for i in num:
            pcl_gt = np.load('data/' + data + '/test/' + gt_dir + '/' + str(i).zfill(fill) + '.npy')
            pcl_mean = np.load('data/' + data + '/test/notscaledpcl_11subs/' + str(i).zfill(fill) + '.npy').mean(axis=0)
            pcl_gt = (pcl_gt + 1) * (pcls_max - pcls_min) / 2 + pcls_min

            pcl_gt += pcl_mean

            if gt:
                pose_pred = np.load(
                    'data/' + data + '/test/posesglobalseparate_11subs/' + str(i).zfill(fill) + '.npy')  # gt
            else:
                pose_pred = predictions[i].reshape(numJoints, 3)
            pose_mean = np.load('data/' + data + '/test/notscaledpose_11subs/' + str(i).zfill(fill) + '.npy').mean(
                axis=0)
            # pose_pred = (pose_pred + 1) * (poses_max - poses_min) / 2 + poses_min
            pose_pred = data_loader.unscale_to_cm(pose_pred, data=data)

            pose_pred += pose_mean

            visualize_3D(pcl_gt, pose=pose_pred, title='', noaxes=noaxes, save_fig=True,
                         name=str(i).zfill(6), azim=azim_min + (azim * (i + 1)), elev=elev, gt=gt, pause=False)


def visualize_3D(coords, pause=True, array=False, regions=None, pose=None,
                 title='Visualized pointcloud', ms1=10, ms2=0.03, noaxes=False, azim=-77, elev=12., save_fig=False,
                 name='samplefig', gt=False):  # coords with shape (numPoints, 3)
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
    if coords is not None:
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
    if gt:
        color = 'green'
    else:
        # color = 'magenta'
        color = 'darkgrey'

    if pose is not None:
        pose = np.reshape(pose, (config.numJoints, 3))
        ax.scatter(pose[:, 0], pose[:, 2], pose[:, 1], c='blue', marker='o', s=ms1)
    if regions is not None:
        # C = np.stack([regions] * 3,
        #              axis=-1)  # shape = (numPoints, 1)  (number of corresponding joint representing the region)
        # C = np.reshape(C, (regions.shape[0], 3))
        C = np.zeros((regions.shape[0], 3))
        for j in range(config.numRegions):
            #     C[C == [j, j, j]] = (j * 7)
            color = np.random.randint(256, size=3)
            for reg in range(regions.shape[0]):
                # if np.array_equal(C[a], [j, j, j]):
                if regions[reg] == j:
                    # C[a] = color
                    C[reg] = color
        ax.scatter(x, z, y, c=C / 255.0, marker='o', s=3)
    elif coords is not None:
        ax.scatter(x, z, y, c='blue', marker='o', s=ms2)
    ax.set_xlabel('x axis')
    ax.set_ylabel('z axis')
    ax.set_zlabel('y axis')
    if noaxes:
        plt.axis('off')

    # Fix aspect ratio
    if coords is not None:
        max_range = np.array([x.max() - x.min(), y.max() - y.min(),
                              z.max() - z.min()]).max() / 2.0
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
        max_range = np.array([x.max() - x.min(), y.max() - y.min(),
                              z.max() - z.min()]).max() / 2.0
        mean_x = x.mean()
        mean_z = y.mean()
        mean_y = z.mean()
        ax.set_xlim(mean_x - max_range, mean_x + max_range)
        ax.set_ylim(mean_y + max_range, mean_y - max_range)
        ax.set_zlim(mean_z - max_range, mean_z + max_range)
    # ax.invert_yaxis()
    if pose is not None:
        if pose.shape[0] == 18:  # UBC dataset
            for bone in UBC_bone_list:
                ax.plot([pose[:, 0][bone[0]], pose[:, 0][bone[1]]],
                        [pose[:, 2][bone[0]], pose[:, 2][bone[1]]], [pose[:, 1][bone[0]], pose[:, 1][bone[1]]], color)
        elif pose.shape[0] == 29:  # MHAD
            for bone in MHAD_bone_list:
                ax.plot([pose[:, 0][bone[0]], pose[:, 0][bone[1]]],
                        [pose[:, 2][bone[0]], pose[:, 2][bone[1]]], [pose[:, 1][bone[0]], pose[:, 1][bone[1]]], color)
        elif config.dataset == 'CMU':
            for bone in CMU_bone_list:
                ax.plot([pose[:, 0][bone[0]], pose[:, 0][bone[1]]],
                        [pose[:, 2][bone[0]], pose[:, 2][bone[1]]], [pose[:, 1][bone[0]], pose[:, 1][bone[1]]], color)

    # plt.xlim(-100, 100)
    # ax.set_zlim3d(-100, 100)
    # plt.ylim(-100, 100)
    if save_fig:
        ax.view_init(elev=elev, azim=azim)
        plt.savefig('data/' + config.dataset + '/test/figures/' + name + '.png', dpi=1200)
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

    # for i in range(numJoints):
    #     ax.text(pose[i, 0], pose[i, 2], pose[i, 1], '%s' % (str(i)), size=10, zorder=1,
    #             color='k')

    plt.show()
    if pause:
        plt.pause(0.001)
        input("Press [enter] to show next pose.")


if __name__ == "__main__":
    # idx = 100200
    # pcl, pose = np.load('data/CMU/train/171204_pose4.npy', allow_pickle=True)[idx]
    # regions = np.load('data/CMU/train/regions.npy')[idx]
    # pcl = np.load('data/CMU/train/scaled_pcls.npy', allow_pickle=True)[idx]  # todo check from 98162 to end (val split)
    # pose = np.load('data/CMU/train/scaled_poses.npy', allow_pickle=True)[idx]
    # regions = np.load('data/CMU/train/regions/')

    # preds = np.load('data/CMU/test/171204_pose6_predictions.npy', allow_pickle=True)
    # poses_gt = np.load('data/CMU/test/171204_pose6_scaledposes_lzeromean.npy', allow_pickle=True)
    # gt_dir = 'data/CMU/test/171204_pose6_scaledpcls_lzeromean.npy'
    # save_frames(preds, 'CMU', gt_dir, numJoints=15, noaxes=True, gt=False, num=range(9250, 9630))  # 7100
    # pcl = np.load('data/UBC/train/scaledpclglobal/00002.npy').reshape((2048, 3))
    # reg = np.load('data/UBC/train/region/00002.npy')
    #pose = np.load('data/UBC/train/posesglobalseparate/00002.npy')
    pcl = np.load('data/MHAD/train/scaledpclglobalSWbatches/000022.npy')[0]
    pcl = pcl.reshape((2048,3))
    reg = np.load('data/MHAD/train/region35jbatches/000022.npy')[0]
    visualize_3D(coords=pcl, regions=reg, ms2=1, azim=-32, elev=11, title='')
