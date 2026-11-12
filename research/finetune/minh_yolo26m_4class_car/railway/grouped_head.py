"""Preserve pretrained six-class logits while exposing four project groups.

Checkpoints containing this layer require this local module when loaded.
"""
import torch
from torch import nn

NAMES = ['person', 'two_wheeler', 'large_vehicle', 'car']
COCO_ROWS = [0, 1, 2, 3, 5, 7]


class GroupedClassifier(nn.Module):
    def __init__(self, source):
        super().__init__()
        assert isinstance(source, nn.Conv2d) and source.out_channels == 80
        self.classifier = nn.Conv2d(source.in_channels, 6, 1, bias=True)
        with torch.no_grad():
            self.classifier.weight.copy_(source.weight[COCO_ROWS])
            self.classifier.bias.copy_(source.bias[COCO_ROWS])
        self.out_channels = 4

    def forward(self, x):
        s = self.classifier(x)
        return torch.stack((s[:, 0], torch.maximum(s[:, 1], s[:, 3]),
                            torch.maximum(s[:, 4], s[:, 5]), s[:, 2]), dim=1)
