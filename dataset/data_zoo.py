# -*- coding:utf-8 -*-  
from __future__ import print_function

import os

from easydict import EasyDict as edict
import numpy as np
from deeploader.dataset.dataset_base import ArrayDataset


datasets = edict()
dataset = datasets

# ctw1500
ctw1500 = edict()
ctw1500.name = 'ctw1500'
ctw1500.dataset_path = '.'
ctw1500.split = 'train'
dataset.ctw1500 = ctw1500

# icdar2015
icdar2015 = edict()
icdar2015.name = 'icdar2015'
icdar2015.dataset_path = '.'
icdar2015.split = 'train'
dataset.icdar2015 = icdar2015

# icdar2019mlt
icdar2019mlt = edict()
icdar2019mlt.name = 'icdar2019mlt'
icdar2019mlt.dataset_path = '.'
icdar2019mlt.split = 'train'
dataset.icdar2019mlt = icdar2019mlt

# icdar2019Art
icdar2019art = edict()
icdar2019art.name = 'icdar2019art'
icdar2019art.dataset_path = '.'
icdar2019art.split = 'train'
dataset.icdar2019art = icdar2019art

# icdar2019ReCTS
icdar2019rects = edict()
icdar2019rects.name = 'icdar2019rects'
icdar2019rects.dataset_path = '.'
icdar2019rects.split = 'train'
dataset.icdar2019rects = icdar2019rects

# icdar2019ReCTS
msratd500 = edict()
msratd500.name = 'msratd500'
msratd500.dataset_path = '.'
msratd500.split = 'train'
dataset.msratd500 = msratd500


class SplitDataset(ArrayDataset):
    def __init__(self, dataset, filter=None, name='--'):
        """
        Split a dataset
        :param dataset: source dataset
        :param filter: sampling rule, like low-0.5, high-100
        :param name: dataset name
        :return selected part of the dataset.
        """
        ArrayDataset.__init__(self, name=name)
        self.dataset = dataset
        self.filter = filter
        total = dataset.size()
        if filter is None:
            self.ibuf = [i for i in range(dataset.size())]
        else:
            #
            self.ibuf = []
            segs = filter.split('-')
            pos = segs[0]
            num = float(segs[1])
            if num < 1:
                base = 100
                num = int(num * base)
                if num % 10 == 0:
                    base = 10
                    num /= 10

                if pos == 'low':
                    for i in range(total):
                        if i % base < num:
                            self.ibuf.append(i)
                else:
                    for i in range(total):
                        if i % base >= (base-num):
                            self.ibuf.append(i)
            else:
                num = int(num)
                if pos == 'low':
                    for i in range(total):
                        if i < num:
                            self.ibuf.append(i)
                else:
                    for i in range(total):
                        if i >= (total-num):
                            self.ibuf.append(i)
        # print(self.ibuf)

    def size(self):
        return len(self.ibuf)

    def getData(self, index):
        return self.dataset.getData(self.ibuf[index])


def get_dataset_by_name(name, *args, **kargs):
    names = []
    if isinstance(name, list):
        names = name
    elif isinstance(name, str):
        names = name.split(';')

    if len(names) > 1:
        from deeploader.dataset.dataset_multi import MultiDataset
        merged = MultiDataset();
        for item in names:
            ds = get_dataset_by_name(item, *args, **kargs)
            merged.add(ds, 0)
        return merged

    # dataset factory
    ds = getattr(dataset, name)
    _, ext = os.path.splitext(ds.dataset_path)
    data = None
    split = ds.split
    if 'split' in kargs:
        split = kargs['split']
    if name == 'ctw1500':
        from dataset_ctw1500 import CTW1500Dataset
        data = CTW1500Dataset(ds.dataset_path, split, name=ds.name)
    elif name == 'icdar2015':
        from dataset_icdar2015 import ICDAR2015Dataset
        data = ICDAR2015Dataset(ds.dataset_path, split, name=ds.name)
    elif name == 'icdar2019mlt':
        from dataset_icdar2019MLT import ICDAR2019MLTDataset
        data = ICDAR2019MLTDataset(ds.dataset_path, split, name=ds.name)
    elif name == 'icdar2019art':
        from dataset_icdar2019ArT import ICDAR2019ARTDataset
        data = ICDAR2019ARTDataset(ds.dataset_path, split, name=ds.name)
    elif name == 'icdar2019rects':
        from dataset_icdar2019ReCTS import ICDAR2019ReCTSDataset
        data = ICDAR2019ReCTSDataset(ds.dataset_path, split, name=ds.name)
    elif name == 'msratd500':
        from dataset_msratd500 import MSRATD500Dataset
        data = MSRATD500Dataset(ds.dataset_path, split, name=ds.name)

    if 'filter' in kargs and kargs['filter']:
        data = SplitDataset(data, kargs['filter'], data.name)
    return data
