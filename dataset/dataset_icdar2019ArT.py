# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import json
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


def parse_gt(gt):
    bboxes = []
    tags = []
    langs = []
    trans = []
    for line in gt:
        points = line['points']
        transcription = line['transcription']
        language = line['language']
        tag = line['illegibility']
        tags.append(tag)
        #box = np.asarray(points)
        bboxes.append(points)
        langs.append(language)
        trans.append(transcription)
    return bboxes, tags, langs, trans


class ICDAR2019ARTDataset(ArrayDataset):
    def __init__(self, data_root='.', split='train', **kargs):
        ArrayDataset.__init__(self, **kargs)
        self.split = split
        root_dir = data_root+'/ArT2019/'
        train_data_dir = root_dir + 'train_images/'
        train_gt_path = root_dir + 'train_labels.json'
        # not gt for test set
        test_data_dir = root_dir + 'test_images/'
        test_gt_path = ''
        if split == 'train':
            data_dir = train_data_dir
            gt_path = train_gt_path
        else:
            data_dir = test_data_dir
            gt_path = test_gt_path

        # scan images
        self.img_paths = []
        img_names = util.io.ls(data_dir, '.jpg')
        img_names.extend(util.io.ls(data_dir, '.png'))

        img_paths = []
        for idx, img_name in enumerate(img_names):
            img_path = data_dir + img_name
            img_paths.append(img_path)

        self.img_paths.extend(img_paths)
        self.gt = None
        if gt_path:
            with open(gt_path, 'r') as f:
                gt = json.load(f)
                self.gt = gt

    def size(self):
        return len(self.img_paths)

    def getData(self, index):
        """
        Load ICDAR2019ArT data
        :param index: zero-based data index
        :return: A dict like { img: RGB, bboxes: nxkx2 np array, tags: n }
        """
        img_path = self.img_paths[index]
        # RGB
        img = get_img(img_path)
        if self.split == 'test':
            return {'img': img, 'path': img_path}
        # get gt
        img_name = os.path.basename(img_path).split('.')[0]
        gt = self.gt[img_name]
        bboxes, tags, langs, trans = parse_gt(gt)
        # scale it back to pixel coord
        print(type(bboxes))
        item = {'img': img, 'type': 'quad', 'bboxes': bboxes,
                'tags': tags, 'path': img_path}
        return item
