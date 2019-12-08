import h5py
import numpy as np
import cv2
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
import codecs, json
import pose_visualizer

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


class data_loader:
    def __init__(self, depth_maps_path, labels_path):
        self.depth_maps = h5py.File(depth_maps_path, 'r')
        self.labels = h5py.File(labels_path, 'r')

    def show(self):

        for i in range(self.depth_maps['data'].shape[0]):
            if self.labels['is_valid'][i]:
                depth_map = self.depth_maps['data'][i].astype(np.float64)
                joints = self.labels['image_coordinates'][i]
                img = self.depth_map_to_image(depth_map, joints)
                cv2.imshow("Image", img)
                cv2.waitKey(0)
                # ...
        return 0

    def depth_map_to_image(self, depth_map, joints=None):
        img = cv2.normalize(depth_map, depth_map, 0, 1, cv2.NORM_MINMAX)
        img = np.array(img * 255, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
        for j in range(15):
            x, y = joints[j, 0], joints[j, 1]
            cv2.circle(img, (x, y), 1, (255, 255, 255), thickness=2)
            cv2.putText(img, joint_id_to_name[j], (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
        return img

    def get_data(self, posemin, posemax, size=None, mode='train'):
        # x = []
        # y = []
        x = np.empty((np.sum(self.labels['is_valid']), size, size), dtype=np.float32)
        y = np.empty((np.sum(self.labels['is_valid']), 15, 3), dtype=np.float32)
        idx = 0
        print('Unpacking dataset...')
        # poses = self.labels['real_world_coordinates'][np.argwhere(self.labels['is_valid']).flatten(), :, :].astype(
        #     np.float32)
        # print(poses.shape)
        # poses = poses - np.mean(poses, axis=(0, 1))
        # if mode == 'train':
        #     #     # posemin = [1000000, 1000000, 1000000]
        #     #     # posemax = [-1000000, -1000000, -1000000]
        #     posemin = np.min(poses, axis=(0, 1))
        #     posemax = np.max(poses, axis=(0, 1))
        # poses = 2 * (poses - posemin) / (posemax - posemin) - 1

        for i in range(self.depth_maps['data'].shape[0]):
            if self.labels['is_valid'][i]:
                # joint locations
                pose = self.labels['real_world_coordinates'][i]
                pose = np.asarray(pose, dtype=np.float32)

                ca = pose[1]  # neck
                cb = pose[8]  # torso
                cjnt = (ca + cb) * 0.5
                pose = pose - cjnt  # centered

                # if mode == 'train':
                #     posemin = np.minimum(posemin, pose.min(axis=0))
                #     posemax = np.maximum(posemax, pose.max(axis=0))
                # pose = 2 * (pose - posemin) / (posemax - posemin) - 1  # scale

                # y.append(pose)
                y[idx] = pose
                # depth map
                depth_map = self.depth_maps['data'][i].astype(np.float32)
                if size is not None:
                    depth_map = depth_map[:, 41:281]
                    depth_map = cv2.resize(depth_map, (size, size), interpolation=cv2.INTER_NEAREST)
                # img = cv2.normalize(depth_map, depth_map, 0, 1, cv2.NORM_MINMAX)
                img = rescale(depth_map, 0, 1)
                img = np.array(img, dtype=np.float32)  # depth values scaled to range [0,1]

                # pose_visualizer.visualize_pose_2D(j, isnumpy=True, pause=False)
                # img = np.array(img * 255, dtype=np.uint8)  # for visualization
                # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                # img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)

                # cv2.imshow("Image", img)
                # cv2.waitKey(0)
                # x.append(img)
                x[idx] = img
                idx += 1
        # x = np.array(x)
        # y = np.array(y)
        # y = 2 * (y - posemin) / (posemax - posemin) - 1
        # x, y = shuffle(x, y, random_state=42)  # shuffle data

        return [x, y, posemin, posemax]


def rescale(x, lw, up):
    return lw + ((x - np.min(x)) / (np.max(x) - np.min(x))) * (up - lw)


if __name__ == '__main__':
    # dataset = data_loader('D:/skola/master/datasets/ITOP/depth_maps/ITOP_side_train_depth_map.h5',
    #                'D:/skola/master/datasets/ITOP/labels/ITOP_side_train_labels.h5')
    # [x, y] = dataset.get_train_data()

    file_path = "Output/sampleBW.json"  # your path variable
    # img = x[0]
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)W
    # img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)

    img = cv2.imread("D:/skola/master/datasets/ITOP/bw.jpg")
    img = cv2.resize(img, (100, 100))
    img = img[:, :, 0]
    b = np.array(img.flatten()).tolist()
    json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True,
              indent=4)  # this saves the array in .json format
    #
    # print(x[0].shape)
    #

    # print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    #     x.shape, x.min(), x.max()))
    # print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    #     y.shape, y.min(), y.max()))

# main()
