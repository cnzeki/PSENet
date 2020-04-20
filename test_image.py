# -*- coding:utf-8 -*-
# psenet-PyTorch-single-test
import argparse
import collections
import os
import sys
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

import models
# c++ version pse based on opencv 3+
from pse import pse


# python pse
# from pypse import pse as pypse

def load_psenet(filepath=None, cuda=torch.cuda.is_available()):
    model = models.resnet50(pretrained=True, num_classes=7, scale=1)
    for param in model.parameters():
        param.requires_grad = False

    model = model.cuda() if cuda else model

    if filepath is not None:
        if os.path.isfile(filepath):
            print("Loading model and optimizer from checkpoint '{}'".format(filepath))
            checkpoint = torch.load(filepath) if cuda else torch.load(filepath,
                                                                      map_location=lambda storage,
                                                                                          loc: storage)

            # model.load_state_dict(checkpoint['state_dict'])
            d = collections.OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(filepath, checkpoint['epoch']))
            sys.stdout.flush()
        else:
            print("No checkpoint found at '{}'".format(filepath))
            sys.stdout.flush()
    else:
        print("You must specify a filepath")
        sys.stdout.flush()
    return model.eval()


def use_psenet(img, model, precession=960, kernel_num=7, min_kernel_area=5.0, min_area=800,
               min_score=0.93, cuda=torch.cuda.is_available()):
    org_img = img[:, :, [2, 1, 0]]
    h, w = org_img.shape[0:2]
    scale = precession * 1.0 / max(h, w)
    scaled_img = cv2.resize(org_img, dsize=None, fx=scale, fy=scale)
    scaled_img = Image.fromarray(scaled_img)
    scaled_img = scaled_img.convert('RGB')
    scaled_img = transforms.ToTensor()(scaled_img)
    scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
        scaled_img)
    scaled_img = scaled_img.unsqueeze(0)
    text_box = org_img.copy()

    with torch.no_grad():
        if cuda:
            img = Variable(scaled_img.cuda())
            torch.cuda.synchronize()
            outputs = model(img)
            score = torch.sigmoid(outputs[:, 0, :, :])
            outputs = (torch.sign(outputs - 1) + 1) / 2
            text = outputs[:, 0, :, :]
            kernels = outputs[:, 0:kernel_num, :, :] * text
            score = score.data.cpu().numpy()[0].astype(np.float32)
            text = text.data.cpu().numpy()[0].astype(np.uint8)
            kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)

        else:
            img = Variable(scaled_img)
        outputs = model(img)
        score = torch.sigmoid(outputs[:, 0, :, :])
        outputs = (torch.sign(outputs - 1) + 1) / 2
        text = outputs[:, 0, :, :]
        kernels = outputs[:, 0:kernel_num, :, :] * text
        score = score.data.numpy()[0].astype(np.float32)
        text = text.data.numpy()[0].astype(np.uint8)
        kernels = kernels.data.numpy()[0].astype(np.uint8)

    # c++ version pse
    pred = pse(kernels, min_kernel_area)
    # python version pse
    # pred = pypse(kernels, min_kernel_area)

    scale = (org_img.shape[1] * 1.0 / pred.shape[1], org_img.shape[0] * 1.0 / pred.shape[0])
    label = pred
    label_num = np.max(label) + 1

    bboxes = []
    for i in range(1, label_num):
        points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

        if points.shape[0] < min_area:
            continue

        score_i = np.mean(score[label == i])
        if score_i < min_score:
            continue

        rect = cv2.minAreaRect(points)
        bbox = cv2.boxPoints(rect) * scale
        bbox = bbox.astype('int32')
        bboxes.append(bbox.reshape(-1))

    if cuda:
        torch.cuda.synchronize()
    return bboxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--filepath', nargs='?', type=str, default='test.jpg',
                        help='Image file for PSENet detection')
    parser.add_argument('--modelpath', nargs='?', type=str,
                        default="./checkpoints_1108_best/XH_resnet50_bs_8_ep_50/checkpoint.pth.tar",
                        help='Path to restore trained model')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=3,
                        help='Number of kernels')
    parser.add_argument('--precession', nargs='?', type=int, default=960,
                        help='Long size of intermediate image')
    parser.add_argument('--min_kernel_area', nargs='?', type=float, default=5.0,
                        help='min kernel area')
    parser.add_argument('--min_area', nargs='?', type=float, default=80.0,
                        help='min area')
    parser.add_argument('--min_score', nargs='?', type=float, default=0.93,
                        help='min score')
    parser.add_argument('--Polygon', nargs='?', type=bool, default=False,
                        help='Parallelogram outputs or Polygon outpus')

    args = parser.parse_args()

    model = load_psenet(args.modelpath)
    img = cv2.imread(args.filepath)
    # full version
    # bboxes = use_psenet(img,model,args.precession,args.kernel_num,args.min_kernel_area,args.min_area,args.min_score)
    # simple version
    t0 = time.time()
    bboxes = use_psenet(img, model, args.precession)
    t1 = time.time()
    print(t1 - t0)
    # print(bboxes)

    for bbox in bboxes:
        cv2.drawContours(img, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)
    cv2.imwrite('res_enjoy_' + args.filepath, img)
