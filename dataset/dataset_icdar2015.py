# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import cv2
import numpy as np
from deeploader.dataset.dataset_base import ArrayDataset

import util
from dataset.data_util import get_img


def get_bboxes(img, gt_path):
    h, w = img.shape[0:2]
    lines = util.io.read_lines(gt_path)
    bboxes = []
    tags = []
    for line in lines:
        line = util.str.remove_all(line, '\xef\xbb\xbf')
        gt = util.str.split(line, ',')
        if gt[-1][0] == '#':
            tags.append(False)
        else:
            tags.append(True)
        box = [int(gt[i]) for i in range(8)]
        box = np.asarray(box).reshape((4, 2)).tolist()
        bboxes.append(box)
    return bboxes, tags


class ICDAR2015Dataset(ArrayDataset):
    def __init__(self, data_root='.', split='train', **kargs):
        ArrayDataset.__init__(self, **kargs)
        self.split = split
        ic15_root_dir = data_root+'/ICDAR2015/Challenge4/'
        train_data_dir = ic15_root_dir + 'ch4_training_images/'
        train_gt_dir = ic15_root_dir + 'ch4_training_localization_transcription_gt/'
        test_data_dir = ic15_root_dir + 'ch4_test_images/'
        test_gt_dir = ic15_root_dir + 'ch4_test_localization_transcription_gt/'
        if split == 'train':
            data_dirs = [train_data_dir]
            gt_dirs = [train_gt_dir]
        else:
            data_dirs = [test_data_dir]
            gt_dirs = [test_gt_dir]

        self.img_paths = []
        self.gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = util.io.ls(data_dir, '.jpg')
            img_names.extend(util.io.ls(data_dir, '.png'))
            # img_names.extend(util.io.ls(data_dir, '.gif'))
            img_names.sort()
            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)

                gt_name = 'gt_' + img_name.split('.')[0] + '.txt'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)

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
        item = {'img': img, 'type': 'quad', 'bboxes': bboxes, 'tags': tags,
                'path': img_path}
        return item
