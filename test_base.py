# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import os
import sys

import Polygon as plg
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

import models
import util
# c++ version pse based on opencv 3+
from pse import pse


# python pse
# from pypse import pse as pypse

def extend_3c(img):
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.concatenate((img, img, img), axis=2)
    return img


def debug(idx, img_paths, imgs, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    col = []
    for i in range(len(imgs)):
        row = []
        for j in range(len(imgs[i])):
            # img = cv2.copyMakeBorder(imgs[i][j], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
            row.append(imgs[i][j])
        res = np.concatenate(row, axis=1)
        col.append(res)
    res = np.concatenate(col, axis=0)
    img_name = img_paths[idx].split('/')[-1]
    print(idx, '/', len(img_paths), img_name)
    cv2.imwrite(output_root + img_name, res)


def write_quad_as_txt(image_name, bboxes, path):
    filename = util.io.join_path(path, 'res_%s.txt' % (image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = "%d, %d, %d, %d, %d, %d, %d, %d\n" % tuple(values)
        lines.append(line)
    util.io.write_lines(filename, lines)


def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    resBoxes = np.empty([1, 8], dtype='int32')
    resBoxes[0, 0] = int(points[0])
    resBoxes[0, 4] = int(points[1])
    resBoxes[0, 1] = int(points[2])
    resBoxes[0, 5] = int(points[3])
    resBoxes[0, 2] = int(points[4])
    resBoxes[0, 6] = int(points[5])
    resBoxes[0, 3] = int(points[6])
    resBoxes[0, 7] = int(points[7])
    pointMat = resBoxes[0].reshape([2, 4]).T
    return plg.Polygon(pointMat)


def load_model(args):
    # Setup Model
    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=7, scale=args.scale)

    for param in model.parameters():
        param.requires_grad = False

    model = model.cuda()

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(("Loading model and optimizer from checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)

            # model.load_state_dict(checkpoint['state_dict'])
            d = collections.OrderedDict()
            for key, value in list(checkpoint['state_dict'].items()):
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)

            print(("Loaded checkpoint '{}' (epoch {})"
                   .format(args.resume, checkpoint['epoch'])))
            sys.stdout.flush()
        else:
            print(("No checkpoint found at '{}'".format(args.resume)))
            sys.stdout.flush()

    model.eval()
    return model


def run_PSENet(args, model, img, org_shape, out_type='rect'):
    outputs = model(img)

    score = torch.sigmoid(outputs[:, 0, :, :])
    outputs = (torch.sign(outputs - args.binary_th) + 1) / 2

    text = outputs[:, 0, :, :]
    kernels = outputs[:, 0:args.kernel_num, :, :] * text

    score = score.data.cpu().numpy()[0].astype(np.float32)
    kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)

    # c++ version pse
    pred = pse(kernels, args.min_kernel_area / (args.scale * args.scale))
    # python version pse
    # pred = pypse(kernels, args.min_kernel_area / (args.scale * args.scale))

    # scale = (org_img.shape[0] * 1.0 / pred.shape[0], org_img.shape[1] * 1.0 / pred.shape[1])
    scale = (org_shape[1] * 1.0 / pred.shape[1], org_shape[0] * 1.0 / pred.shape[0])
    label = pred
    label_num = np.max(label) + 1
    bboxes = []
    for i in range(1, label_num):
        points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

        if points.shape[0] < args.min_area / (args.scale * args.scale):
            continue

        score_i = np.mean(score[label == i])
        if score_i < args.min_score:
            continue

        if out_type == 'rect':
            rect = cv2.boundingRect(points)
            x1, y1 = rect[0], rect[1]
            x2, y2 = x1 + rect[2] - 1, y1 + rect[3] - 1
            pts = [x1, y1, x2, y1, x2, y2, x1, y2]
            bbox = np.array(pts).reshape(-1, 2) * scale
            bbox = bbox.astype('int32')
        elif out_type == 'rbox':
            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect) * scale
            bbox = bbox.astype('int32')
        elif out_type == 'contour':
            binary = np.zeros(label.shape, dtype='uint8')
            binary[label == i] = 1
            _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0]
            # epsilon = 0.01 * cv2.arcLength(contour, True)
            # bbox = cv2.approxPolyDP(contour, epsilon, True)
            bbox = contour

            if bbox.shape[0] <= 2:
                continue

            bbox = bbox * scale
            bbox = bbox.astype('int32')

        bboxes.append({'type': out_type, 'bbox': bbox.reshape(-1)})
    return bboxes


def img_preprocess(img, precession=960, cuda=torch.cuda.is_available()):
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

    if cuda:
        img = Variable(scaled_img.cuda())
        torch.cuda.synchronize()
    else:
        img = Variable(scaled_img)
    return img


def default_parser():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--binary_th', nargs='?', type=float, default=1.0,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=7,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--scale', nargs='?', type=int, default=1,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--long_size', nargs='?', type=int, default=2240,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--min_kernel_area', nargs='?', type=float, default=5.0,
                        help='min kernel area')
    parser.add_argument('--min_area', nargs='?', type=float, default=800.0,
                        help='min area')
    parser.add_argument('--min_score', nargs='?', type=float, default=0.93,
                        help='min score')
