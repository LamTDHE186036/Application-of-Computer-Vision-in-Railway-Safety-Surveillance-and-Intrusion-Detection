"""Inference layers for the exported five-class fine-tuned checkpoint.

Training and ignore-aware validation remain in the research code. This module
has no dependency on the research folder, datasets or local training paths.
"""
import torch
from torch import nn
from ultralytics.nn.tasks import DetectionModel

NAMES = ['person', 'two_wheeler', 'large_vehicle', 'car', 'train']


class FiveGroupedClassifier(nn.Module):
    """Expose five classes using the same seven learned logits as training."""
    def __init__(self, old_classifier, train_classifier):
        super().__init__()
        assert old_classifier.out_channels == 6 and train_classifier.out_channels == 80
        assert old_classifier.in_channels == train_classifier.in_channels
        self.classifier = nn.Conv2d(old_classifier.in_channels, 7, 1, bias=True)
        with torch.no_grad():
            self.classifier.weight[:6].copy_(old_classifier.weight)
            self.classifier.bias[:6].copy_(old_classifier.bias)
            self.classifier.weight[6].copy_(train_classifier.weight[6])
            self.classifier.bias[6].copy_(train_classifier.bias[6])
        self.out_channels = 5

    def forward(self, x):
        s = self.classifier(x)
        return torch.stack((s[:, 0], torch.maximum(s[:, 1], s[:, 3]),
                            torch.maximum(s[:, 4], s[:, 5]), s[:, 2], s[:, 6]), dim=1)


class FiveDetectionModel(DetectionModel):
    """Portable inference model; resume training from the research checkpoint."""
    def init_criterion(self):
        raise RuntimeError('Checkpoint này dành cho web. Dùng code fine-tune và checkpoint nghiên cứu để huấn luyện.')
