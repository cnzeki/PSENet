import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data

from data_util import *


class OcrDataLoader(data.Dataset):
    def __init__(self, dataset, is_transform=False, img_size=None, kernel_num=7, min_scale=0.4):
        # dataset
        self.dataset = dataset
        # transform params
        self.is_transform = is_transform
        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.kernel_num = kernel_num
        self.min_scale = min_scale

    def __len__(self):
        return self.dataset.size()

    def __getitem__(self, index):
        item = self.dataset[index]

        img = item['img']
        bboxes, tags = item['bboxes'], item['tags']
        num = len(bboxes)
        for idx, box in enumerate(bboxes):
            bboxes[idx] = np.array(box)
        if self.is_transform:
            img, scale = random_scale(img, self.img_size[0])
            for idx, box in enumerate(bboxes):
                sb = box * scale
                bboxes[idx] = sb.astype('int32')

        gt_text = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if num > 0:
            for i in range(num):
                cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)
                if not tags[i]:
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

        gt_kernals = []
        for i in range(1, self.kernel_num):
            rate = 1.0 - (1.0 - self.min_scale) / (self.kernel_num - 1) * i
            gt_kernal = np.zeros(img.shape[0:2], dtype='uint8')
            kernal_bboxes = shrink(bboxes, rate)
            for i in range(num):
                cv2.drawContours(gt_kernal, [kernal_bboxes[i]], -1, 1, -1)
            gt_kernals.append(gt_kernal)

        if self.is_transform:
            imgs = [img, gt_text, training_mask]
            imgs.extend(gt_kernals)

            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            imgs = random_crop(imgs, self.img_size)

            img, gt_text, training_mask, gt_kernals = imgs[0], imgs[1], imgs[2], imgs[3:]

        gt_text[gt_text > 0] = 1
        gt_kernals = np.array(gt_kernals)

        # '''
        if self.is_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).float()
        gt_kernals = torch.from_numpy(gt_kernals).float()
        training_mask = torch.from_numpy(training_mask).float()

        return img, gt_text, gt_kernals, training_mask
