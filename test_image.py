# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from deeploader.util.fileutil import read_lines
from torch.utils import data

from dataset import CTW1500TestLoader
from test_base import *

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
list_ext = ['txt', 'lst', 'list']


class VideoDataset(data.Dataset):
    def __init__(self, src, test_num=0):
        self.cap = cv2.VideoCapture(0 if src == 'webcam' else src)
        self.size = 1e6
        if src != 'webcam':
            self.size = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if test_num > 0:
            self.size = min(self.size, test_num)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        _, img = self.cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img


class ImgListLoader(data.Dataset):
    def __init__(self, img_list):
        self.img_list = img_list

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = cv2.imread(self.img_list[index])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img


def get_test_images(opt):
    input_ext = opt.demo[opt.demo.rfind('.') + 1:].lower()
    if os.path.isdir(opt.demo):
        image_names = []
        ls = os.listdir(opt.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(opt.demo, file_name))
    elif input_ext in list_ext:
        image_names = read_lines(opt.demo)
    else:
        image_names = [opt.demo]

    if opt.test_num > 0 and opt.test_num < len(image_names):
        image_names = image_names[:opt.test_num]

    return image_names


def get_test_dataset(opt):
    input_ext = opt.demo[opt.demo.rfind('.') + 1:].lower()
    if opt.demo == 'webcam' or input_ext in video_ext:
        dataset = VideoDataset(opt.demo, opt.test_num)
    else:
        images = get_test_images(opt)
        dataset = ImgListLoader(images)
    return dataset


def test(args):
    data_loader = CTW1500TestLoader(long_size=args.long_size, ctw_root=args.ctw_root)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    model = load_model(args)
    total_frame = 0.0
    total_time = 0.0
    for idx, (org_img, img) in enumerate(test_loader):
        print(('progress: %d / %d' % (idx, len(test_loader))))
        sys.stdout.flush()

        img = Variable(img.cuda(), volatile=True)
        org_img = org_img.numpy().astype('uint8')[0]
        text_box = org_img.copy()

        torch.cuda.synchronize()
        start = time.time()
        bboxes = run_PSENet(args, model, img, org_img.shape, out_type='contour')
        torch.cuda.synchronize()
        end = time.time()
        total_frame += 1
        total_time += (end - start)
        print(('fps: %.2f' % (total_frame / total_time)))
        sys.stdout.flush()

        for _bbox in bboxes:
            bbox = _bbox['bbox']
            cv2.drawContours(text_box, [bbox.reshape(int(bbox.shape[0] / 2), 2)], -1, (0, 255, 0), 2)

        image_name = data_loader.img_paths[idx].split('/')[-1].split('.')[0]
        write_pts_as_txt(image_name, bboxes, 'outputs/submit_ctw1500/')

        debug(idx, data_loader.img_paths, [[text_box]], 'outputs/vis_ctw1500/')

    cmd = 'cd %s;zip -j %s %s/*' % ('./outputs/', 'submit_ctw1500.zip', 'submit_ctw1500');
    print(cmd)
    sys.stdout.flush()
    util.cmd.cmd(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--binary_th', nargs='?', type=float, default=1.0,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=3,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--scale', nargs='?', type=int, default=1,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--long_size', nargs='?', type=int, default=1280,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--min_kernel_area', nargs='?', type=float, default=10.0,
                        help='min kernel area')
    parser.add_argument('--min_area', nargs='?', type=float, default=300.0,
                        help='min area')
    parser.add_argument('--min_score', nargs='?', type=float, default=0.93,
                        help='min score')
    parser.add_argument('--ctw_root', type=str, default='.',
                        help='ctw1500 data root dir')

    args = parser.parse_args()
    test(args)
