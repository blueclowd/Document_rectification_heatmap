
import os
import keras
import numpy as np
import math
import cv2 as cv
import random


class MaskGenerator(keras.utils.Sequence):

    def __init__(self, directory, idxs, img_dict, labels_dict,
                 target_size=(96, 96), batch_size=32, augment=True,
                 transform_dict=None, shuffle=True):

        self.directory = directory
        self.idxs = idxs
        self.img_dict = img_dict
        self.labels_dict = labels_dict
        self.transform_dict = transform_dict
        self.target_size = target_size
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()

    # shuffle indices at the end of each epoch
    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.idxs)

    # return number of batches per epoch
    def __len__(self):

        if self.augment is True:
            multiplier = 5
        else:
            multiplier = 1

        return int(np.floor(len(self.idxs) * multiplier / self.batch_size))

    # check if transformed point is located within image boundaries
    def _checkBoundaries(self, p):

        # x dimension
        if p[0] < 0:
            px = 0
        elif p[0] > self.target_size[0]:
            px = self.target_size[0]
        else:
            px = p[0]

        # y dimension
        if p[1] < 0:
            py = 0
        elif p[1] > self.target_size[1]:
            py = self.target_size[1]
        else:
            py = p[1]

        return (int(px), int(py))

    # apply shifts, rotations, scaling and flips to original image and keypoints
    def _transform_image(self, img, keypoints):

        aug_keypoints = []

        c = (img.shape[0] // 2, img.shape[1] // 2)

        if self.transform_dict['Flip']:
            flip = random.choice([True, False])
            if flip:
                img = cv.flip(img, flipCode=1)

        if self.transform_dict['Rotate']:

            if self.transform_dict['Scale']:
                s = random.uniform(0.7, 1.0)
            else:
                s = 1.0

            r = random.randint(-10, 10)
            M_rot = cv.getRotationMatrix2D(center=(img.shape[0] // 2, img.shape[1] // 2), angle=r, scale=s)
            img = cv.warpAffine(img, M_rot, (img.shape[0], img.shape[1]), borderMode=cv.BORDER_CONSTANT, borderValue=1)

        if self.transform_dict['Shift']:
            tx = random.randint(-20, 20)
            ty = random.randint(-20, 20)
            M_shift = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
            img = cv.warpAffine(img, M_shift, (img.shape[0], img.shape[1]),
                                borderMode=cv.BORDER_CONSTANT, borderValue=1)

        # transform keypoints
        c = (img.shape[0] // 2, img.shape[1] // 2)

        for i in range(0, len(keypoints) - 1, 2):

            px = keypoints[i]
            py = keypoints[i + 1]
            p = np.array([px, py, 1], dtype=int)

            # apply flip
            if self.transform_dict['Flip'] and flip:
                p[0] = c[0] - (p[0] - c[0])

            # apply rotation
            if self.transform_dict['Rotate']:
                p = np.dot(M_rot, p)

            # apply horizontal / vertical shifts
            if self.transform_dict['Shift']:
                p[0] += tx
                p[1] += ty

            p = self._checkBoundaries(p)

            aug_keypoints.append(p[0])
            aug_keypoints.append(p[1])

        return img, aug_keypoints

    # load image from disk
    def _load_image(self, fn):

        img = cv.imread(filename=os.path.join(self.directory, fn))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img = cv.resize(img, self.target_size)
        img = np.float32(img) / 255

        return img

    # apply gaussian kernel to image
    def _gaussian(self, xL, yL, sigma, H, W):

        channel = [math.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
        channel = np.array(channel, dtype=np.float32)
        channel = np.reshape(channel, newshape=(H, W))

        return channel

    # convert original image to heatmap
    def _convertToHM(self, img, keypoints, sigma=5):

        H = img.shape[0]
        W = img.shape[1]
        nKeypoints = len(keypoints)

        # One keypoint has two values: x, y
        n_channels = nKeypoints // 2

        img_hm = np.zeros(shape=(H, W, n_channels), dtype=np.float32)

        # Generate heatmaps for each keypoint
        for i in range(0, n_channels):
            x = keypoints[i * 2]
            y = keypoints[1 + 2 * i]

            channel_hm = self._gaussian(x, y, sigma, H, W)

            img_hm[:, :, i] = channel_hm

        img_hm = np.reshape(img_hm, newshape=(img_hm.shape[0] * img_hm.shape[1] * n_channels, 1))

        return img_hm

    # generate batches of scaled images and bounding boxes
    def _data_generation(self, idxs):

        x = []
        y = []

        for idx in idxs:
            img = self._load_image(self.img_dict[idx])
            keypoints = self.labels_dict[idx]

            if self.augment is True and self.transform_dict:
                img, keypoints = self._transform_image(img, keypoints)

            img = np.reshape(img, (self.target_size[0], self.target_size[1], 1))
            img_hm = self._convertToHM(img, keypoints)

            x.append(img)
            y.append(img_hm)

        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

    # return indices for train batches
    def _get_train_idxs(self, idx):

        # number of batches in original train set
        N = int(np.floor(len(self.idxs) / self.batch_size))

        # idx exceeds original image indices
        if idx > N:

            # reset start idx
            if idx % N == 0:
                reset_idx = 0  # ((idx - 1) % N) + 1
            else:
                reset_idx = idx % N - 1

            start = reset_idx * self.batch_size

            # end idx
            if (reset_idx + 1) * self.batch_size > len(self.idxs):
                end = len(self.idxs)
            else:
                end = (reset_idx + 1) * self.batch_size

        # idx is within in original train set
        else:
            start = idx * self.batch_size
            end = (idx + 1) * self.batch_size

        return start, end

    # return indices for val batches
    def _get_val_idxs(self, idx):

        if (idx + 1) * self.batch_size > len(self.idxs):
            end = len(self.idxs)
        else:
            end = (idx + 1) * self.batch_size

        return idx * self.batch_size, end

    # return batch of image data and labels
    def __getitem__(self, idx):

        if self.augment is True:
            start_batch_idx, end_batch_idx = self._get_train_idxs(idx)
        else:
            start_batch_idx, end_batch_idx = self._get_val_idxs(idx)

        idxs = self.idxs[start_batch_idx:end_batch_idx]
        batch_x, batch_y = self._data_generation(idxs)

        return batch_x, batch_y