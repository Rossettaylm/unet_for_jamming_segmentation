import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.unet import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image


class Unet(object):
    _defaults = {
        # 预测权值文件
        "model_path": "logs\ep040-loss0.128-val_loss0.081.pth",  # 脉内编码
        # "model_path": "logs\ep040-loss0.133-val_loss0.053.pth", # LFM

        # target + jamming + background
        "num_classes": 3,
        "input_shape": [512, 512],
        "blend": True,
        "cuda": False
    }

    # 初始化UNET
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # 画框设置不同的颜色
        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128,
                                                                      0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128,
                                                                         64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                           (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.)
                          for x in range(self.num_classes)]
            self.colors = list(
                map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        # 获得模型
        self.generate()

    # 获得所有的分类
    def generate(self):
        self.net = unet(num_classes=self.num_classes)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(
            self.model_path, map_location=device), strict=False)
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    # 检测图片
    def detect_image(self, image):
        # 转为RGB
        image = cvtColor(image)

        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        # 加上灰条
        image_data, nw, nh = resize_image(
            image, (self.input_shape[1], self.input_shape[0]))

        # 添加batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(
            np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h),
                            interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) *
                                 (self.colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) *
                                 (self.colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) *
                                 (self.colors[c][2])).astype('uint8')

        image = Image.fromarray(np.uint8(seg_img))

        if self.blend:
            image = Image.blend(old_img, image, 0.7)

        return image

    # 获取预测得到的干扰位置
    @staticmethod
    def get_jamming_pos(self, image_path):
        image = Image.open(image_path)
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        image_data, nw, nh = resize_image(
            image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(
            np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h),
                            interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

            jamming_pos = (~((pr == 2).astype(bool))).astype(int)
            jamming_pos = jamming_pos.tolist()
            return jamming_pos

    # 获取做miou评估的评测图片
    def get_miou_png(self, image):
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        image_data, nw, nh = resize_image(
            image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(
            np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h),
                            interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image
