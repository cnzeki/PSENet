import argparse
import numpy as np
import cv2
from dataset import get_dataset_by_name
from deeploader.util.fileutil import makedirs
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n', nargs='?', type=int, default=5,
                        help='# of the epochs')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--filter', type=str, default='', help='dataset name')

    args = parser.parse_args()

    train_data = get_dataset_by_name(args.dataset, filter=args.filter)
    train_data.verbose()

    out_dir = 'outputs/dl_%s/' % (args.dataset)
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
                cv2.drawContours(img, [bboxes[i]], -1, (0, 255, 0), 2)
                if not tags[i]:
                    cv2.drawContours(img, [bboxes[i]], -1, (0, 0, 255), 2)
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

        out_path = out_dir + img_name
        makedirs(out_path)
        cv2.imwrite(out_path, img)
        # cv2.imwrite(out_path+'.gt.jpg', gt_text)
        # cv2.imwrite(out_path+'.mask.jpg', training_mask)
