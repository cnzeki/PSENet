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
from dataset import get_dataset_by_name

# python pse
# from pypse import pse as pypse

def extend_3c(img):
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.concatenate((img, img, img), axis=2)
    return img


def debug(idx, img_name, imgs, output_root):
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
    cv2.imwrite(output_root + img_name, res)


def write_quad_as_txt(image_name, bboxes, path):
    filename = util.io.join_path(path, 'res_%s.txt' % (image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = "%d, %d, %d, %d, %d, %d, %d, %d\n" % tuple(values)
        lines.append(line)
    util.io.write_lines(filename, lines)


def write_pts_as_txt(image_name, bboxes, path):
    if not os.path.exists(path):
        os.makedirs(path)

    filename = util.io.join_path(path, '%s.txt' % (image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox['bbox']]
        line = "%d" % values[0]
        for v_id in range(1, len(values)):
            line += ", %d" % values[v_id]
        line += '\n'
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


def polygon_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()


def polygon_union(pD, pG):
    areaA = pD.area();
    areaB = pG.area();
    return areaA + areaB - polygon_intersection(pD, pG);


def eval_score(pred_list, gt_list, gt_tag_list, th=0.5):
    tp, fp, npos = 0, 0, 0
    nign = 0
    # loop all images
    num = len(pred_list)
    for i in range(num):
        # load prediction & gt
        preds = pred_list[i]
        gts = gt_list[i]
        tags = gt_tag_list[i]

        # npos += len(gts)
        gt_polys = []
        for gt_id, gt in enumerate(gts):
            gt = np.array(gt).reshape(-1)
            gt = gt.reshape(int(gt.shape[0] / 2), 2)
            gt_p = plg.Polygon(gt)
            gt_polys.append(gt_p)
            if tags[gt_id]:
                npos += 1
        # match predictions for the image
        # print(preds)
        cover = set()
        for pred_id, pred in enumerate(preds):
            pred = np.array(pred).reshape(-1)
            pred = pred.reshape(int(pred.shape[0] / 2), 2)
            # if pred.shape[0] <= 2:
            #     continue
            pred_p = plg.Polygon(pred)

            flag = False
            matched_to_ignore = False
            for gt_id, gt_p in enumerate(gt_polys):
                union = polygon_union(pred_p, gt_p)
                inter = polygon_intersection(pred_p, gt_p)
                #  IoU(pred, gt) > th => accept
                if inter * 1.0 / union >= th:
                    if gt_id not in cover:
                        flag = True
                        cover.add(gt_id)
                        if not tags[gt_id]:
                            matched_to_ignore = True
                        break
            if matched_to_ignore:
                nign += 1.0
                # print('matched to ignore gt')
                continue
            if flag:
                tp += 1.0
            else:
                fp += 1.0

    # ok, finish the job
    # print(tp, fp, npos, nign)
    precision = tp / (tp + fp)
    recall = tp / npos
    hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, hmean, (tp, fp, npos, nign)


def load_model(args):
    epoch = 0
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
            epoch = checkpoint['epoch']
            print(("Loaded checkpoint '{}' (epoch {})"
                   .format(args.resume, checkpoint['epoch'])))
            sys.stdout.flush()
        else:
            print(("No checkpoint found at '{}'".format(args.resume)))
            sys.stdout.flush()

    model.eval()
    return model, epoch


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
            ret = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # print(ret)
            # _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = ret[-2]
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


def img_preprocess(org_img, precession=960, cuda=torch.cuda.is_available()):
    # org_img = img[:, :, [2, 1, 0]]
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


def test_model(args, model, data_loader):
    # collect labels
    gt_bbox_list = []
    gt_tag_list = []
    pred_list = []
    for idx, item in enumerate(data_loader):
        # read in RGB
        org_img = item['img']
        gt_bboxes = item['bboxes']
        gt_tags = item['tags']
        gt_bbox_list.append(gt_bboxes)
        gt_tag_list.append(gt_tags)

        img = img_preprocess(org_img, args.long_size)

        torch.cuda.synchronize()
        pred_rets = run_PSENet(args, model, img, org_img.shape, out_type=args.out_type)
        torch.cuda.synchronize()
        pred_bboxes = []
        for item in pred_rets:
            pred_bboxes.append(item['bbox'])
        pred_list.append(pred_bboxes)

    return eval_score(pred_list, gt_bbox_list, gt_tag_list, th=0.5)


def run_tests(args, model, epoch, test_model_fn=test_model):
    # model single gpu  eval mode
    if hasattr(model, 'module'):
        model = model.module
    model.eval()
    torch.cuda.empty_cache()
    hmean_dict = {}
    print('\nTest @ epoch: %d' % (epoch + 1))
    for dataset in args.vals:
        # build loader
        data_loader = get_dataset_by_name(dataset, split='test')
        precision, recall, hmean, cnts = test_model_fn(args, model, data_loader)
        print('Val:%10s, imgs:%d, ' % (dataset, len(data_loader)), end='')
        print('tp:%4d, fp:%4d, npos:%4d, ' % (cnts[0], cnts[1], cnts[2]), end='')
        print('P: %.4f, R: %.4f, F1: %.4f' % (precision, recall, hmean))
        hmean_dict[dataset] = hmean
    # return target
    target = args.val_target if args.val_target else 'mean()'
    if target in ['mean()', 'sum()']:
        sum_hmean = 0.0
        num = 0
        for v in hmean_dict.values():
            sum_hmean += v
            num += 1
        target = sum_hmean / num
    else:
        target = hmean_dict[target]
    # back to training mode
    model.train()
    return target


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
