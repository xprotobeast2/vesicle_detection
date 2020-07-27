'''
Uses pretrained YOLO with modified output layer.
'''

import numpy
import torch
from torch import nn
from yolov3 import models as yolov3_models


class Detector(nn.Module):
    def __init__(self, config_path, image_size, weights_path=None, n_classes=1):
        super().__init__()
        self.yolo_net = yolov3_models.Darknet(config_path, image_size)
        if weights_path is not None:
            self.yolo_net.load_darknet_weights(weights_path)
        self.class_transformer = nn.Linear(80, (n_classes))

    def forward(self, x):
        o = self.yolo_net(x).cuda()
        o_pos = o[..., :5]
        y = self.class_transformer(o[..., 5:])
        return o_pos, y


if __name__ == '__main__':
    raise NotImplementedError('Create example')
    model = Detector(config_path, image_size, weights_path, n_classes)
