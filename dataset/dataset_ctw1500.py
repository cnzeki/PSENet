# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import cv2
import numpy as np
from deeploader.dataset.dataset_base import ArrayDataset

import util


def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print(img_path)
        raise
    return img


def get_bboxes(img, gt_path):
    h, w = img.shape[0:2]
    lines = util.io.read_lines(gt_path)
    bboxes = []
    tags = []
    for line in lines:
        line = util.str.remove_all(line, '\xef\xbb\xbf')
        gt = util.str.split(line, ',')

        x1 = np.int(gt[0])
        y1 = np.int(gt[1])

        bbox = [np.int(gt[i]) for i in range(4, 32)]
        bbox = np.asarray(bbox) + ([x1 * 1.0, y1 * 1.0] * 14)
        bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * 14)

        bboxes.append(bbox)
        tags.append(True)
    return np.array(bboxes), tags


class CTW1500Dataset(ArrayDataset):
    def __init__(self, ctw_root='.', split='train', **kargs):
        ArrayDataset.__init__(self, **kargs)
        ctw_root_dir = ctw_root + '/data/ctw1500/'
        ctw_train_data_dir = ctw_root_dir + 'train/text_image/'
        ctw_train_gt_dir = ctw_root_dir + 'train/text_label_curve/'
        ctw_test_data_dir = ctw_root_dir + 'test/text_image/'
        ctw_test_gt_dir = ctw_root_dir + 'test/text_label_curve/'
        if split == 'train':
            data_dirs = [ctw_train_data_dir]
            gt_dirs = [ctw_train_gt_dir]
        else:
            data_dirs = [ctw_test_data_dir]
            gt_dirs = [ctw_test_gt_dir]

        self.img_paths = []
        self.gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = util.io.ls(data_dir, '.jpg')
            img_names.extend(util.io.ls(data_dir, '.png'))
            # img_names.extend(util.io.ls(data_dir, '.gif'))

            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)

                gt_name = img_name.split('.')[0] + '.txt'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)

        # self.img_paths = self.img_paths[440:]
        # self.gt_paths = self.gt_paths[440:]

    def size(self):
        return len(self.img_paths)

    def getData(self, index):
        """
        Load CTW1500 data
        :param index: zero-based data index
        :return: A dict like { img: RGB, bboxes: nxkx2 np array, tags: n }
        """
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]
        # RGB
        img = get_img(img_path)
        # bbox normed to 0~1
        bboxes, tags = get_bboxes(img, gt_path)
        # scale it back to pixel coord
        bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 14),
                            (bboxes.shape[0], int(bboxes.shape[1] / 2), 2)).astype('int32')
        item = {'img': img, 'type': 'contour', 'bboxes': bboxes, 'tags': tags,
                'path': img_path}
        return item
