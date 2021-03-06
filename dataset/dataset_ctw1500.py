# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import cv2
import numpy as np
from deeploader.dataset.dataset_base import ArrayDataset
from deeploader.util.fileutil import read_lines

import util
from dataset.data_util import get_img


def get_bboxes_det(img, gt_path):
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
        bbox = np.asarray(bbox).reshape((14, 2)).tolist()

        bboxes.append(bbox)
        tags.append(True)
    return bboxes, tags


def get_bboxes_rec(img, gt_path):
    h, w = img.shape[0:2]
    lines = read_lines(gt_path)
    bboxes = []
    tags = []
    trans = []
    for line in lines[1:]:
        #line = util.str.remove_all(line, '\xef\xbb\xbf')
        gt = util.str.split(line, ',')

        bbox = [np.int(gt[i]) for i in range(0, 28)]
        # bbox = np.asarray(bbox)
        bbox = np.asarray(bbox).reshape((14, 2)).tolist()
        tag = True
        text = '###'
        texts = line.split('\"')
        if len(texts) > 2:
            text = texts[1]
        if not text or text[0] == '#':
            tag = False
        tags.append(tag)
        trans.append(text)
        bboxes.append(bbox)
    return bboxes, tags, trans


class CTW1500Dataset(ArrayDataset):
    def __init__(self, ctw_root='.', split='train', **kargs):
        ArrayDataset.__init__(self, **kargs)
        ctw_root_dir = ctw_root + '/data/ctw1500/'
        ctw_train_data_dir = ctw_root_dir + 'train/text_image/'
        ctw_train_gt_dir = ctw_root_dir + 'train/ctw1500_e2e_train/'
        ctw_test_data_dir = ctw_root_dir + 'test/text_image/'
        ctw_test_gt_dir = ctw_root_dir + 'test/ctw1500_e2e_test/'
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

            img_names.sort()
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
        bboxes, tags, trans = get_bboxes_rec(img, gt_path)

        item = {'img': img, 'type': 'contour', 'bboxes': bboxes, 'tags': tags,
                'trans': trans,
                'path': img_path}
        return item
