import h5py
import numpy as np
# import cv2
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
import codecs, json
# import pose_visualizer
import preprocess
import visualizer
from scipy.io import loadmat

joint_id_to_name = {
    0: 'Head',
    1: 'Neck',
    2: 'R Shoulder',
    3: 'L Shoulder',
    4: 'R Elbow',
    5: 'L Elbow',
    6: 'R Hand',
    7: 'L Hand',
    8: 'Torso',
    9: 'R Hip',
    10: 'L Hip',
    11: 'R Knee',
    12: 'L Knee',
    13: 'R Foot',
    14: 'L Foot',
}


class DataLoader:
    def __init__(self, pcls_path, labels_path):
        # self.depth_maps = h5py.File(depth_maps_path, 'r')
        self.pointclouds = h5py.File(pcls_path, 'r')
        self.labels = h5py.File(labels_path, 'r')

    # def show(self):
    #
    #     for i in range(self.depth_maps['data'].shape[0]):
    #         if self.labels['is_valid'][i]:
    #             depth_map = self.depth_maps['data'][i].astype(np.float64)
    #             joints = self.labels['image_coordinates'][i]
    #             img = self.depth_map_to_image(depth_map, joints)
    #             cv2.imshow("Image", img)
    #             cv2.waitKey(0)
    #             # ...
    #     return 0

    # def depth_map_to_image(self, depth_map, joints=None):
    #     img = cv2.normalize(depth_map, depth_map, 0, 1, cv2.NORM_MINMAX)
    #     img = np.array(img * 255, dtype=np.uint8)
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #     img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
    #     for j in range(15):
    #         x, y = joints[j, 0], joints[j, 1]
    #         cv2.circle(img, (x, y), 1, (255, 255, 255), thickness=2)
    #         cv2.putText(img, joint_id_to_name[j], (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
    #     return img

    def get_data(self, numPoints=2048, start=0, mode='train'):
        numSamples = np.sum(self.labels['is_valid'])
        numJoints = self.labels['real_world_coordinates'][0].shape[0]
        x = np.empty((numSamples, numPoints, 3))
        loadedx = loadmat(
            'G:/skola/master/datasets/ITOP/pointclouds/' + mode + '_x.mat')['clouds'][0]
        y = np.empty((numSamples, numJoints, 3))
        regions = np.empty((numSamples, numPoints, 1), dtype=np.int32)
        # xarr = []
        # yarr = []
        # regionsarr = []
        idx = 0
        print('Unpacking dataset...')
        # print(self.pointclouds['data'].shape[0])
        # print(np.sum(self.labels['is_valid'])) #17991
        for i in range(start, self.pointclouds['data'].shape[0]):  # each shape (76800, 3)
            if self.labels['is_valid'][i]:
                # joint locations
                pose = self.labels['real_world_coordinates'][i]  # shape (15,3)
                pose = np.asarray(pose, dtype=np.float32)

                # ca = pose[1]  # neck
                # cb = pose[8]  # torso
                # cjnt = (ca + cb) * 0.5

                # pointclouds
                # pcl = self.pointclouds['data'][i].astype(np.float32)
                # pcl = np.asarray(pcl)
                # pcl, _, _ = preprocess.subsample(pcl, numPoints, 0, 0)

                regs = preprocess.automatic_annotation(pose=pose, pcl=loadedx[idx])
                pose = pose - pose.mean(axis=0)  # centered
                pcl = loadedx[idx] - loadedx[idx].mean(axis=0)
                y[idx] = pose
                x[idx] = pcl
                regions[idx] = regs

                # yarr.append(pose)
                # xarr.append(loadedx)
                # regionsarr.append(regs)

                # print(i)
                # if not i % 100 and i > 0:
                #     print(str(i) + ' samples done.')
                #     np.save('data/ITOP/' + mode + '/' + mode + '_data_x_notscaled_centered_' + str(i // 100).zfill(
                #         5) + '.npy',
                #             x)
                #     np.save('data/ITOP/' + mode + '/' + mode + '_data_y_notscaled_centered_' + str(i // 100).zfill(
                #         5) + '.npy', y)
                #     np.save('data/ITOP/' + mode + '/' + mode + '_data_regs_' + str(i // 100).zfill(
                #         5) + '.npy', regions)
                idx += 1
        # x = np.asarray(x)
        # y = np.asarray(y)
        # regions = np.asarray(regions)

        # x, y = shuffle(x, y, random_state=42)  # shuffle data

        return [x, y, regions]


def rescale(x, lw, up):
    return lw + ((x - np.min(x)) / (np.max(x) - np.min(x))) * (up - lw)

# if __name__ == '__main__':
