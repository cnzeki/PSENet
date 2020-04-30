# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from deeploader.dataset.dataset_base import ArrayDataset

import util
from dataset.data_util import get_img


def rotate(angle, x, y):
    """
    基于原点的弧度旋转

    :param angle:   弧度
    :param x:       x
    :param y:       y
    :return:
    """
    rotatex = math.cos(angle) * x - math.sin(angle) * y
    rotatey = math.cos(angle) * y + math.sin(angle) * x
    return rotatex, rotatey


def xy_rorate(theta, x, y, centerx, centery):
    """
    针对中心点进行旋转

    :param theta:
    :param x:
    :param y:
    :param centerx:
    :param centery:
    :return:
    """
    r_x, r_y = rotate(theta, x - centerx, y - centery)
    return centerx + r_x, centery + r_y


def rbox2quad(x, y, width, height, theta):
    """
    传入矩形的x,y和宽度高度，弧度，转成QUAD格式
    :param x:
    :param y:
    :param width:
    :param height:
    :param theta:
    :return:
    """
    centerx = x + width / 2
    centery = y + height / 2

    x1, y1 = xy_rorate(theta, x, y, centerx, centery)
    x2, y2 = xy_rorate(theta, x + width, y, centerx, centery)
    x3, y3 = xy_rorate(theta, x + width, y + height, centerx, centery)
    x4, y4 = xy_rorate(theta, x, y + height, centerx, centery)

    return [x1, y1, x2, y2, x3, y3, x4, y4]


def get_bboxes(img, gt_path):
    lines = util.io.read_lines(gt_path)
    bboxes = []
    tags = []
    for line in lines:
        line = util.str.remove_all(line, '\xef\xbb\xbf')
        gt = util.str.split(line, ' ')

        diff = np.int(gt[1])
        x, y, w, h = np.int(gt[2]), np.int(gt[3]), np.int(gt[4]), np.int(gt[5])
        angle = np.float(gt[-1])

        bbox = rbox2quad(x, y, w, h, angle)
        bbox = np.array(bbox).reshape((4, 2)).tolist()

        bboxes.append(bbox)
        if diff == 1:
            tags.append(False)
        else:
            tags.append(True)
    return bboxes, tags


class MSRATD500Dataset(ArrayDataset):
    def __init__(self, ctw_root='.', split='train', **kargs):
        ArrayDataset.__init__(self, **kargs)
        ctw_root_dir = ctw_root + '/MSRA-TD500/'
        ctw_train_data_dir = ctw_root_dir + 'train/'
        ctw_train_gt_dir = ctw_root_dir + 'train/'
        ctw_test_data_dir = ctw_root_dir + 'test/'
        ctw_test_gt_dir = ctw_root_dir + 'test/'
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

            img_names.sort()
            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)

                gt_name = img_name.split('.')[0] + '.gt'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)

    def size(self):
        return len(self.img_paths)

    def getData(self, index):
        """
        Load MSRA-TD500 data
        :param index: zero-based data index
        :return: A dict like { img: RGB, bboxes: nxkx2 np array, tags: n }
        """
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]
        # RGB
        img = get_img(img_path)
        # bbox normed to 0~1
        bboxes, tags = get_bboxes(img, gt_path)

        item = {'img': img, 'type': 'contour', 'bboxes': bboxes, 'tags': tags,
                'path': img_path}
        return item
