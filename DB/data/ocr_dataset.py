import os
import sys

import cv2
import math
import numpy as np
import torch.utils.data as data

from concern.config import Configurable, State
# init path
the_dir = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.dirname(os.path.dirname(the_dir))
sys.path.insert(0, proj_dir)
#print(sys.path)
#print(proj_dir)
import dataset
from dataset import get_dataset_by_name


class OcrDataset(data.Dataset, Configurable):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''
    data_names = State()
    filter = State()
    is_training=State()
    processes = State(default=[])

    def __init__(self, data_names=None, filter=None, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.data_names = data_names or self.data_names
        self.filter = filter or self.filter
        self.is_training = False if not 'is_training' in kwargs else self.is_training
        self.debug = cmd.get('debug', False)

        # load dataset
        split = 'train' if self.is_training else 'test'
        dataset = get_dataset_by_name(self.data_names, filter=self.filter, split=split)
        dataset.verbose()
        self.dataset = dataset

    def __getitem__(self, index, retry=0):
        '''
        item = {'img': img, 'type': 'contour', 'bboxes': bboxes, 'tags': tags,
                'path': img_path}
        '''
        if index >= self.dataset.size():
            index = index % self.dataset.size()

        item = self.dataset.getData(index)

        data = {}
        image_path = item['path']
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        if self.is_training:
            data['filename'] = image_path
            data['data_id'] = image_path
        else:
            data['filename'] = image_path.split('/')[-1]
            data['data_id'] = image_path.split('/')[-1]
        data['image'] = img
        target = []
        num_targets = len(item['bboxes'])
        for idx in range(num_targets):
            text = 1234 if item['tags'][idx] else '###'
            target.append({'poly': item['bboxes'][idx], 'text': text})
        data['lines'] = target
        if self.processes is not None:
            for data_process in self.processes:
                data = data_process(data)
        return data

    def __len__(self):
        return self.dataset.size()
