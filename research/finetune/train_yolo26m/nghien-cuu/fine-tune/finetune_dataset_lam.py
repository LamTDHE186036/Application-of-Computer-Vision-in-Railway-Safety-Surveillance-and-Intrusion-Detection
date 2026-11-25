"""Fine-tune YOLO26m trên Dataset_Lam, giữ nguyên ID lớp.

Chạy --check-only để kiểm tra, hoặc --train để bắt đầu học từ COCO.
Đây là lượt thử theo split gốc sau lọc ảnh trùng byte. Điểm validation
có thể còn chịu ảnh hưởng của camera/nền chung và nhãn chưa rà đầy đủ.
"""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir, mkdtemp

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / 'data' / 'Dataset_Lam'
COCO = ROOT / 'Fine-tune-YOLO26m' / 'model' / 'yolo26m-coco.pt'
COCO_SHA256 = '401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7'
NAMES = ('car', 'motorcycle', 'person', 'truck-bus', 'train')
COCO_CLASSES = (('car',), ('motorcycle',), ('person',), ('truck', 'bus'), ('train',))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(gettempdir()) / 'railway-dataset-lam-yolo'))

import numpy as np
from PIL import Image
import torch
from torch import nn
import ultralytics
from ultralytics import YOLO
from ultralytics.data.augment import Compose, Format, LetterBox
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.models.yolo.detect import DetectionTrainer
import yaml


# Tham số của lượt 20261124-175059-luak_ppe. Bộ Lâm dùng loss YOLO chuẩn.
CONFIG = dict(
    epochs=20, patience=6, imgsz=640, batch=1, nbs=64, device=0, workers=0,
    optimizer='AdamW', lr0=1e-5, lrf=0.2, momentum=0.9, weight_decay=0.0005,
    warmup_epochs=1, warmup_bias_lr=1e-5, amp=False, seed=20261110, deterministic=True,
    freeze=6, cache=False, rect=False, nms=False, cos_lr=True, save=True, save_period=1,
    val=True, split='val', resume=False, close_mosaic=0, plots=False, verbose=False,
    cls_pw=0, compile=False, mosaic=0, mixup=0, cutmix=0, copy_paste=0, degrees=0,
    translate=0, scale=0, shear=0, perspective=0, flipud=0, fliplr=0, multi_scale=0,
    hsv_h=0, hsv_s=0, hsv_v=0, bgr=0,
)


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    image_hash: str
    label_hash: str
    shape: tuple[int, int]
    boxes: tuple[tuple[float, ...], ...]
    duplicate_boxes: int


def image_paths(folder):
    return sorted(path for path in folder.iterdir()
                  if path.is_file() and path.suffix[1:].lower() in IMG_FORMATS)


def read_sample(image, label):
    image_bytes = image.read_bytes()
    with Image.open(BytesIO(image_bytes)) as pixels:
        shape = pixels.height, pixels.width
        pixels.verify()
    if min(shape) < 10:
        raise ValueError(f'Ảnh quá nhỏ: {image}')
    label_bytes = label.read_bytes()
    boxes = []
    for number, line in enumerate(label_bytes.decode('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            box = tuple(map(float, line.split()))
            c, x, y, w, h = box
            if not all(map(math.isfinite, box)) or not c.is_integer() or not 0 <= c < len(NAMES):
                raise ValueError
            if min(w, h) <= 0 or min(x - w / 2, y - h / 2) < -1e-5 or max(x + w / 2, y + h / 2) > 1 + 1e-5:
                raise ValueError
        except ValueError as exc:
            raise ValueError(f'Nhãn YOLO sai tại {label}:{number}: {line}') from exc
        boxes.append(box)
    unique = tuple(dict.fromkeys(boxes))
    return Sample(image, label, sha256(image_bytes).hexdigest(), sha256(label_bytes).hexdigest(),
                  shape, unique, len(boxes) - len(unique))


def read_split(folder):
    images = image_paths(folder / 'images')
    labels = {p.stem: p for p in (folder / 'labels').glob('*.txt')}
    stems = {p.stem for p in images}
    if not images or len(stems) != len(images) or stems != labels.keys():
        raise ValueError(f'Cặp ảnh/nhãn sai tại {folder}. Thiếu nhãn: {sorted(stems - labels.keys())[:5]}. '
                         f'Nhãn lẻ: {sorted(labels.keys() - stems)[:5]}.')
    return [read_sample(p, labels[p.stem]) for p in images]


def plan_dataset(root):
    source = root / 'Dataset' / 'Dataset_all'
    data = yaml.safe_load((source / 'data.yaml').read_text(encoding='utf-8-sig'))
    names = data['names']
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if tuple(names) != NAMES or data['nc'] != len(NAMES):
        raise ValueError(f'Thứ tự lớp trong YAML không khớp Dataset_Lam: {names}')

    train = read_split(source / 'train')
    val = read_split(source / 'valid')
    test = defaultdict(list)
    for folder in sorted((root / 'Dataset_test' / 'dataset_test').glob('*/images')):
        for image in image_paths(folder):
            test[sha256(image.read_bytes()).hexdigest()].append(image.relative_to(root).as_posix())
    val_hashes = {sample.image_hash for sample in val}
    if len(val_hashes) != len(val) or val_hashes.intersection(test):
        raise ValueError('Validation có ảnh trùng byte nội bộ hoặc trùng benchmark Dataset_test.')

    groups = defaultdict(list)
    for sample in train:
        groups[sample.image_hash].append(sample)
    selected, held = [], []
    for digest, group in groups.items():
        if digest in val_hashes or digest in test:
            held.extend((sample, 'also_in_validation_or_test') for sample in group)
        elif any(set(sample.boxes) != set(group[0].boxes) for sample in group[1:]):
            held.extend((sample, 'identical_pixels_different_labels') for sample in group)
        else:
            selected.append(group[0])
            held.extend((sample, 'identical_image_and_labels') for sample in group[1:])

    for split, samples in (('train', selected), ('val', val)):
        counts = Counter(int(box[0]) for sample in samples for box in sample.boxes)
        if set(counts) != set(range(len(names))):
            raise ValueError(f'Tập {split} thiếu box của một hoặc nhiều lớp.')
        print(f'{split}: {len(samples)} ảnh, {dict(zip(names, (counts[i] for i in range(len(names)))))}', flush=True)
    print(f'Giữ riêng {len(held)} ảnh train: {dict(Counter(reason for _, reason in held))}', flush=True)
    print(f'Đã đối chiếu SHA-256 với {sum(map(len, test.values()))} ảnh Dataset_test.', flush=True)
    return {'train': selected, 'val': val}, held


def write_run_files(run, root, splits, held):
    data = dict(path=str(run), nc=len(NAMES), names=list(NAMES))
    for split, samples in splits.items():
        listing = run / f'{split}.txt'
        listing.write_text(''.join(f'{sample.image.as_posix()}\n' for sample in samples), encoding='utf-8')
        data[split] = listing.name
    config_file = run / 'data.yaml'
    config_file.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')

    def record(sample):
        return dict(image=sample.image.relative_to(root).as_posix(), image_sha256=sample.image_hash,
                    label=sample.label.relative_to(root).as_posix(), label_sha256=sample.label_hash,
                    boxes=len(sample.boxes), duplicate_label_rows=sample.duplicate_boxes)

    protocol = dict(dataset=str(root), classes=list(NAMES), source_weights_sha256=COCO_SHA256,
                    coco_initialization=dict(zip(NAMES, COCO_CLASSES)), config=CONFIG, coverage=False,
                    splits={split: [record(sample) for sample in samples] for split, samples in splits.items()},
                    held=[dict(record(sample), reason=reason) for sample, reason in held])
    (run / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return config_file


def initialize_from_coco():
    if sha256(COCO.read_bytes()).hexdigest() != COCO_SHA256:
        raise ValueError('Checkpoint YOLO26m COCO sai checksum.')
    yolo = YOLO(str(COCO))
    model = yolo.model
    by_name = {name: index for index, name in model.names.items()}
    head = model.model[-1]
    if len(by_name) != 80 or head.nc != 80 or head.one2one_cv3 is None or len(model.model) != 24:
        raise ValueError('Checkpoint không đúng kiến trúc YOLO26m gốc COCO.')

    with torch.no_grad():
        for branch in (head.cv3, head.one2one_cv3):
            for scale in branch:
                source = scale[-1]
                classifier = nn.Conv2d(source.in_channels, len(NAMES), 1, device=source.weight.device,
                                       dtype=source.weight.dtype)
                for i, names in enumerate(COCO_CLASSES):
                    rows = [by_name[name] for name in names]
                    classifier.weight[i].copy_(source.weight[rows].mean(dim=0))
                    classifier.bias[i].copy_(source.bias[rows].mean())
                scale[-1] = classifier
    model.names = dict(enumerate(NAMES))
    model.nc = head.nc = model.yaml['nc'] = len(NAMES)
    head.no = head.nc + 4 * head.reg_max
    model.end2end = True
    model.criterion = None
    return yolo


class LamDataset(YOLODataset):
    """Đọc các nhãn đã kiểm, không ghi cache vào thư mục dữ liệu gốc."""

    def __init__(self, *args, samples, **kwargs):
        self.samples = samples
        super().__init__(*args, **kwargs)

    def get_labels(self):
        labels = []
        self.label_files = []
        for path in self.im_files:
            sample = self.samples[path]
            boxes = np.asarray(sample.boxes, dtype=np.float32).reshape(-1, 5)
            labels.append(dict(im_file=path, shape=sample.shape, cls=boxes[:, :1], bboxes=boxes[:, 1:],
                               segments=[], keypoints=None, normalized=True, bbox_format='xywh'))
            self.label_files.append(str(sample.label))
        return labels

    def build_transforms(self, hyp=None):
        return Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=True),
                        Format(bbox_format='xywh', normalize=True, batch_idx=True, bgr=0)])


class LamTrainer(DetectionTrainer):
    def __init__(self, *args, samples, **kwargs):
        self.samples = samples
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is None or tuple(weights.names.values()) != NAMES:
            raise ValueError('Cần mô hình YOLO26m đã khởi tạo đúng năm lớp Dataset_Lam.')
        # Checkpoint COCO được nạp ở chế độ suy luận. Trainer áp dụng freeze=6 sau bước này.
        return deepcopy(weights).float().requires_grad_(True)

    def _model_train(self):
        super()._model_train()
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def build_dataset(self, img_path, mode='train', batch=None):
        return LamDataset(img_path, samples=self.samples, imgsz=self.args.imgsz, batch_size=batch,
                          augment=mode == 'train', hyp=self.args, rect=False, cache=False,
                          data=self.data, stride=32, pad=0, task='detect')


def verify_pipeline(yolo, run, data, samples, splits):
    device = CONFIG['device'] if torch.cuda.is_available() else 'cpu'
    options = dict(CONFIG, device=device, model=str(COCO), data=str(data), project=str(run), name='check')
    trainer = LamTrainer(samples=samples, overrides=options)
    trainer.model = trainer.get_model(weights=yolo.model)
    trainer._setup_train()
    for split, loader in (('train', trainer.train_loader), ('val', trainer.test_loader)):
        if len(loader.dataset) != len(splits[split]):
            raise ValueError(f'Bộ nạp {split} không khớp danh sách đã kiểm.')
    trainer._model_train()
    layers = trainer.model.model
    if any(p.requires_grad for layer in layers[:CONFIG['freeze']] for p in layer.parameters()):
        raise ValueError('Chưa khóa đủ sáu tầng đầu của backbone.')
    if any(module.training for module in trainer.model.modules() if isinstance(module, nn.BatchNorm2d)):
        raise ValueError('BatchNorm chưa được khóa thống kê chạy.')

    batch = trainer.preprocess_batch(next(batch for batch in trainer.train_loader if batch['cls'].numel()))
    if tuple(batch['img'].shape[1:]) != (3, 640, 640):
        raise ValueError('Kích thước ảnh train sai.')
    loss, _ = trainer.model(batch)
    loss.sum().backward()
    classifier = layers[-1].one2one_cv3[0][-1]
    if not torch.isfinite(loss).all() or classifier.weight.grad is None:
        raise ValueError('Loss hoặc gradient YOLO không hợp lệ.')

    with torch.inference_mode():
        trainer.model.eval()
        val_batch = trainer.preprocess_batch(next(iter(trainer.test_loader)))
        prediction, _ = trainer.model(val_batch['img'])
        if prediction.ndim != 3 or prediction.shape[-1] != 6 or not torch.isfinite(prediction).all():
            raise ValueError('Đầu ra YOLO26m sau khởi tạo không hợp lệ.')
    print('Đã kiểm train/val, khóa tầng, BatchNorm, loss và gradient. Optimizer chạy 0 bước.')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATASET, help='Thư mục Dataset_Lam.')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--train', action='store_true', help='Fine-tune từ checkpoint COCO gốc.')
    modes.add_argument('--check-only', action='store_true', help='Chỉ kiểm tra, mặc định.')
    args = parser.parse_args()
    if ultralytics.__version__ != '8.4.161':
        raise RuntimeError(f'Cần Ultralytics 8.4.161, đang có {ultralytics.__version__}.')
    if args.train and not torch.cuda.is_available():
        raise RuntimeError('Lượt train cần GPU NVIDIA và PyTorch hỗ trợ CUDA.')

    torch.set_num_threads(4)
    root = args.dataset.resolve()
    splits, held = plan_dataset(root)
    samples = {str(sample.image): sample for records in splits.values() for sample in records}
    yolo = initialize_from_coco()
    if not args.train:
        with TemporaryDirectory(prefix='lam-finetune-check-') as directory:
            run = Path(directory)
            data = write_run_files(run, root, splits, held)
            verify_pipeline(yolo, run, data, samples, splits)
        return

    runs = ROOT / 'ket-qua' / 'runs'
    runs.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix=datetime.now().strftime('dataset-lam-%Y%m%d-%H%M%S-'), dir=runs))
    data = write_run_files(run, root, splits, held)
    print(f'Lưu kết quả fine-tune tại {run}', flush=True)
    yolo.train(trainer=partial(LamTrainer, samples=samples), **CONFIG, data=str(data),
               project=str(run), name='training', exist_ok=False)


if __name__ == '__main__':
    main()
