# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
from deeploader.dataset.dataset_base import ArrayDataset

import util
from dataset.data_util import get_img


def parse_gt(gt_path, target='lines'):
    with open(gt_path, 'r') as f:
        gt = json.load(f)
        bboxes = []
        tags = []
        trans = []
        for line in gt[target]:
            points = line['points']
            transcription = line['transcription']
            ignore = line['ignore']
            tag = not ignore
            tags.append(tag)
            bbox = np.asarray(points).reshape((-1, 2)).tolist()
            bboxes.append(bbox)
            trans.append(transcription)
        return bboxes, tags, trans


class ICDAR2019ReCTSDataset(ArrayDataset):
    def __init__(self, data_root='.', split='train', **kargs):
        ArrayDataset.__init__(self, **kargs)
        self.split = split
        root_dir = data_root + '/ReCTS2019/'
        train_data_dir = root_dir + 'train/img/'
        train_gt_path = root_dir + 'train/gt/'
        # not gt for test set
        test_data_dir = root_dir + 'ReCTS_test_part1/Task3_and_Task4/img/'
        test_gt_path = ''
        if split == 'train':
            data_dir = train_data_dir
            gt_dir = train_gt_path
        else:
            data_dir = test_data_dir
            gt_dir = test_gt_path

        self.random_scale = np.array([0.5, 0.7, 1.0, 1.2])
        # scan images
        self.img_paths = []
        img_names = util.io.ls(data_dir, '.jpg')
        img_names.extend(util.io.ls(data_dir, '.png'))
        img_names.sort()
        img_paths = []
        gt_paths = []
        for idx, img_name in enumerate(img_names):
            img_path = data_dir + img_name
            img_paths.append(img_path)
            if gt_dir:
                gt_name = img_name.split('.')[0] + '.json'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

        self.img_paths = img_paths
        self.gt_paths = gt_paths

    def size(self):
        return len(self.img_paths)

    def getData(self, index):
        """
        Load ICDAR2019ReCTS data
        :param index: zero-based data index
        :return: A dict like { img: RGB, bboxes: nxkx2 np array, tags: n }
        """
        img_path = self.img_paths[index]
        # RGB
        img = get_img(img_path)
        if self.split == 'test':
            return {'img': img, 'path': img_path}
        # get gt
        gt_path = self.gt_paths[index]
        bboxes, tags, trans = parse_gt(gt_path)
        item = {'img': img, 'type': 'quad', 'bboxes': bboxes,
                'tags': tags, 'path': img_path}
        return item
