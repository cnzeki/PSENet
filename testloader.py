import argparse
import os

import cv2
import numpy as np
from deeploader.util.fileutil import makedirs

from dataset import get_dataset_by_name
from dataset.data_loader import OcrDataLoader
from dataset import IC15Loader

def test_dataset(args):
    train_data = get_dataset_by_name(args.dataset, filter=args.filter)
    train_data.verbose()

    out_dir = 'outputs/ds_%s/' % (args.dataset)
    for i in range(args.n):
        item = train_data.getData(i)
        img = item['img']
        img = img[:, :, [2, 1, 0]].copy()
        path = item['path']
        img_name = os.path.basename(path)
        bboxes, tags = item['bboxes'], item['tags']
        num = len(bboxes)
        for idx, box in enumerate(bboxes):
            bboxes[idx] = np.array(box).astype('int32')

        # print(bboxes)
        gt_text = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        training_mask[:] = 255
        if num > 0:
            for i in range(num):
                cv2.drawContours(gt_text, [bboxes[i]], -1, 255, -1)
                img = cv2.drawContours(img, [bboxes[i]], -1, (0, 255, 0), 2)
                if not tags[i]:
                    cv2.drawContours(img, [bboxes[i]], -1, (0, 0, 255), 2)
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

        out_path = out_dir + img_name
        makedirs(out_path)
        cv2.imwrite(out_path, img)
        # cv2.imwrite(out_path+'.gt.jpg', gt_text)
        # cv2.imwrite(out_path+'.mask.jpg', training_mask)


def test_loader(args):
    kernel_num = 7
    min_scale = 0.4
    start_epoch = 0

    data_loader = OcrDataLoader(args, is_transform=True, img_size=args.img_size, \
                                kernel_num=kernel_num, min_scale=min_scale, debug=True)
    # data_loader = IC15Loader(is_transform=True, img_size=args.img_size, kernel_num=kernel_num, min_scale=min_scale, debug=True)

    # data_loader = CTW1500Loader(is_transform=True, img_size=args.img_size, kernel_num=kernel_num, min_scale=min_scale)
    out_dir = 'outputs/ld_%s/' % (args.dataset)
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    for i in range(args.n):
        # CHW
        img, gt_text, gt_kernals, training_mask = data_loader[i]
        img = img.cpu().numpy()
        img = ((img * std + mean) * 255).astype('uint8')
        # HWC
        img = img.transpose((1, 2, 0))
        print(img.shape)

        # gt
        gt_text = gt_text.cpu().numpy()
        gt_text = (gt_text * 255).astype('uint8')
        out_path = out_dir + '%d.jpg' % i
        makedirs(out_path)
        cv2.imwrite(out_path, img)
        cv2.imwrite(out_path + '.gt.jpg', gt_text)
        # cv2.imwrite(out_path+'.mask.jpg', training_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n', nargs='?', type=int, default=5,
                        help='# of the epochs')
    parser.add_argument('--img_size', nargs='?', type=int, default=640,
                        help='Height of the input image')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--filter', type=str, default='', help='dataset filter')
    args = parser.parse_args()
    test_dataset(args)
    test_loader(args)
    # exit(0)
