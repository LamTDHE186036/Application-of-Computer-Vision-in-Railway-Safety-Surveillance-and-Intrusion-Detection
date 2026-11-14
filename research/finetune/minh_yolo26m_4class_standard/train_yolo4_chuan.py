"""Train YOLO26m with the ordinary Ultralytics detection pipeline.

Run normally to train; --check-only verifies inputs without training. No custom
head, loss, dataset subclass, class remapping, or deployment changes are used.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
from uuid import uuid4
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATASET = ROOT / 'data/curated/railway-yolo4-full-2004-20261113-v1'
PRETRAINED = HERE / 'models/yolo26m.pt'
EXPECTED_MODEL_SHA = '401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7'
EXPECTED_MANIFEST_SHA = '342ce9ebf2b8f0d0e451df66563845b746d20d4a66ebb631aceba7ec15a74a15'
EXPECTED_YAML_SHA = '72ecb855c54f8cdb30963cc7d25e592bb81cad84a1546f30509c1713f3b3179d'
NAMES = ['person', 'two_wheeler', 'car', 'train']
SPLITS = {'train': 1600, 'val': 200, 'test': 200}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if hasattr(obj, 'tolist'):
        return clean_json(obj.tolist())
    return obj


def write(path, obj):
    Path(path).write_text(json.dumps(clean_json(obj), ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile

    def write(self, text):
        self.console.write(text)
        self.logfile.write(text)
        return len(text)

    def flush(self):
        self.console.flush()
        self.logfile.flush()

    def isatty(self):
        return self.console.isatty()

    def __getattr__(self, name):
        return getattr(self.console, name)


@contextmanager
def capture_console(path):
    from ultralytics.utils import LOGGER

    before_out, before_err = sys.stdout, sys.stderr
    with Path(path).open('w', encoding='utf-8', buffering=1) as logfile:
        sys.stdout, sys.stderr = Tee(before_out, logfile), Tee(before_err, logfile)
        handlers = []
        for handler in LOGGER.handlers:
            stream = getattr(handler, 'stream', None)
            if stream is before_out or stream is before_err:
                handlers.append((handler, stream))
                handler.setStream(sys.stdout if stream is before_out else sys.stderr)
        try:
            yield
        finally:
            for handler, stream in handlers:
                handler.setStream(stream)
            sys.stdout, sys.stderr = before_out, before_err


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Chỉ kiểm tra, không train')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=float, default=0.5,
                        help='Mặc định AutoBatch 50%% VRAM; dùng 1 hoặc 2 nếu cần batch cố định')
    parser.add_argument('--device', choices=['0', 'cpu'], default='0',
                        help='Mặc định NVIDIA GPU 0; CPU chỉ dùng khi chọn rõ')
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.imgsz < 64 or args.imgsz % 32:
        parser.error('epochs/patience phải dương; imgsz >=64 và chia hết cho 32')
    if not (0 < args.batch < 1 or args.batch >= 1 and args.batch.is_integer()):
        parser.error('batch phải là số nguyên dương hoặc tỉ lệ VRAM giữa 0 và 1')
    if args.batch >= 1:
        args.batch = int(args.batch)
    if args.device == 'cpu' and args.batch < 1:
        args.batch = 1  # AutoBatch requires CUDA.
    return args


def prepare_runtime():
    runtime = HERE / '.settings-yolo4-standard'
    runtime.mkdir(exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(runtime)
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    from ultralytics.utils import USER_CONFIG_DIR
    font = Path(USER_CONFIG_DIR) / 'Arial.ttf'
    if not font.exists():
        shutil.copy2(ROOT / 'Ultralytics/Arial.ttf', font)


def preflight(args, run):
    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
    from ultralytics.data.utils import check_det_dataset

    if sys.version_info[:2] != (3, 11) or ultralytics.__version__ != '8.4.161' or torch.__version__.split('+')[0] != '2.7.1':
        raise RuntimeError('Môi trường khác cấu hình đã kiểm: cần Python 3.11, Ultralytics 8.4.161, PyTorch 2.7.1. Không tự cài/nâng thư viện.')
    if not PRETRAINED.is_file() or sha(PRETRAINED) != EXPECTED_MODEL_SHA:
        raise RuntimeError('Thiếu hoặc khác hash model pretrained models/yolo26m.pt. Không dùng model web thay thế.')
    assert sha(DATASET / 'manifest.json') == EXPECTED_MANIFEST_SHA, 'Manifest dataset đã thay đổi'
    assert sha(DATASET / 'data.yaml') == EXPECTED_YAML_SHA, 'data.yaml đã thay đổi'
    status = read(DATASET / 'STATUS.json')
    assert status['ready_for_standard_training'] and status['user_accepted_existing_annotations']
    assert status['split_images'] == SPLITS and status['boxes'] == 21825
    assert status['manifest_sha256'] == EXPECTED_MANIFEST_SHA
    assert status['data_yaml_sha256'] == EXPECTED_YAML_SHA
    config = check_det_dataset(str(DATASET / 'data.yaml'), autodownload=False)
    assert config['names'] == dict(enumerate(NAMES)) and config['nc'] == 4
    assert not config.get('masks_dir'), 'Không dùng semantic masks trong bản YOLO detection chuẩn'
    manifest = read(DATASET / 'manifest.json')
    assert len(manifest) == 2004 and len({r['index'] for r in manifest}) == 2004
    groups, hashes, pixels = defaultdict(set), defaultdict(set), defaultdict(set)
    counts, boxes = Counter(), Counter()
    for row in manifest:
        image, label = DATASET / row['image'], DATASET / row['label']
        assert sha(image) == row['image_sha256'], f'Ảnh đã thay đổi: {image.name}'
        assert sha(label) == row['label_sha256'], f'Nhãn đã thay đổi: {label.name}'
        if row['split'] == 'hold':
            continue
        split = row['split']
        assert image.parent == Path(config[split])
        assert label.parent == DATASET / 'labels' / split
        counts[split] += 1
        groups[row['source_group']].add(split)
        hashes[row['image_sha256']].add(split)
        pixels[row['pixel_hash']].add(split)
        actual_boxes = 0
        for line in label.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            fields = line.split()
            assert len(fields) == 5
            c = int(fields[0]); x, y, w, h = map(float, fields[1:])
            assert c in range(4) and all(math.isfinite(v) for v in (x, y, w, h))
            assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
            assert min(x - w / 2, y - h / 2) >= -1e-6 and max(x + w / 2, y + h / 2) <= 1 + 1e-6
            boxes[split, c] += 1
            actual_boxes += 1
        assert actual_boxes == row['box_count']
        assert actual_boxes > 0 or row['full_instance_qa_complete'], 'Ảnh nhãn rỗng chưa xác nhận nền'
    assert counts == SPLITS and sum(boxes.values()) == 21825
    assert all(len(v) == 1 for mapping in (groups, hashes, pixels) for v in mapping.values())
    assert all(boxes[s, c] > 0 for s in SPLITS for c in range(4))
    for split in SPLITS:
        assert len(list((DATASET / 'images' / split).iterdir())) == SPLITS[split]
        assert len(list((DATASET / 'labels' / split).glob('*.txt'))) == SPLITS[split]
    if args.device == '0':
        if not torch.cuda.is_available():
            raise RuntimeError('Chưa dùng được NVIDIA CUDA. Kiểm tra driver/GPU; không tự chuyển sang CPU để train nhiều giờ.')
        probe = torch.ones(1, device='cuda:0')
        torch.cuda.synchronize()
        del probe
        torch.cuda.empty_cache()
    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       ultralytics=ultralytics.__version__, cuda=torch.version.cuda,
                       device=args.device, gpu=[dict(name=torch.cuda.get_device_name(i),
                           vram_gib=round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2))
                           for i in range(torch.cuda.device_count())])
    # Ordinary Ultralytics owns the 80 -> 4 class head adaptation at train setup.
    model = YOLO(str(PRETRAINED), task='detect')
    assert len(model.names) == 80 and model.names[0] == 'person' and model.names[6] == 'train'
    assert all(type(m).__module__.startswith(('torch.', 'ultralytics.')) for m in model.model.modules())
    train_args = dict(data=str(DATASET / 'data.yaml'), epochs=args.epochs, patience=args.patience,
                      imgsz=args.imgsz, batch=args.batch, device=0 if args.device == '0' else 'cpu',
                      workers=0, optimizer='auto', freeze=None, pretrained=True,
                      seed=20261113, deterministic=True, amp=args.device != 'cpu',
                      cache=False, resume=False, val=True, split='val', save=True,
                      save_period=1, plots=True, project=str(run), name='train', exist_ok=False,
                      compile=False, single_cls=False, fraction=1.0)
    resolved = vars(get_cfg(overrides=train_args))
    # Do not silently disable normal augmentation or introduce a custom trainer.
    assert resolved['freeze'] is None and resolved['mosaic'] == DEFAULT_CFG_DICT['mosaic']
    protocol = dict(environment=environment, pretrained=str(PRETRAINED.relative_to(ROOT)),
                    pretrained_sha256=EXPECTED_MODEL_SHA, dataset=str(DATASET.relative_to(ROOT)),
                    manifest_sha256=EXPECTED_MANIFEST_SHA, data_yaml_sha256=EXPECTED_YAML_SHA,
                    status_sha256=sha(DATASET / 'STATUS.json'), split_images=SPLITS,
                    train_arguments=train_args, resolved_ultralytics_config=resolved,
                    annotation_qa_complete=False, user_accepted_existing_annotations=True,
                    test_is_new_blind_set=False, test_after_best_selection_only=True,
                    custom_trainer=False, custom_loss=False, additional_layer_freezing=False,
                    upstream_fixed_dfl_parameters_may_remain_frozen=True,
                    training_started=False, web_model_changed=False)
    write(run / 'protocol.json', protocol)
    for name in ('data.yaml', 'STATUS.json', 'manifest.json'):
        shutil.copy2(DATASET / name, run / ('dataset-' + name))
    (run / 'ultralytics-defaults.yaml').write_text(yaml.safe_dump(DEFAULT_CFG_DICT, sort_keys=False), encoding='utf-8')
    protected = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'ung-dung/railway-web/models').iterdir() if p.is_file()}
    write(run / 'protected-web-models.json', protected)
    print(f'Kiểm tra đạt: 1.600 train / 200 val / 200 test; 4 lớp; 21.825 box. GPU: {environment["gpu"]}')
    return model, train_args, protected


def train(args, run, model, train_args):
    import gc
    import torch
    from ultralytics import YOLO

    def record_train_setup(trainer):
        assert trainer.args.freeze is None
        assert [trainer.model.names[i] for i in range(4)] == NAMES
        frozen = [name for name, p in trainer.model.named_parameters() if not p.requires_grad]
        assert all('.dfl' in name for name in frozen), f'Tầng bị khóa ngoài quy tắc Ultralytics: {frozen}'
        write(run / 'effective-training-setup.json', dict(
            model_class=type(trainer.model).__module__ + '.' + type(trainer.model).__name__,
            dataset_class=type(trainer.train_loader.dataset).__module__ + '.' + type(trainer.train_loader.dataset).__name__,
            actual_batch=trainer.batch_size, trainable_parameters=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
            frozen_parameter_names=frozen, optimizer=type(trainer.optimizer).__name__,
            optimizer_groups=[{k: v for k, v in g.items() if k != 'params'} for g in trainer.optimizer.param_groups],
            arguments=vars(trainer.args)))

    model.add_callback('on_pretrain_routine_end', record_train_setup)
    write(run / 'STATE.json', dict(stage='training', training_started=True, training_completed=False, web_model_changed=False))
    model.train(**train_args)
    best = Path(model.trainer.best)
    last = Path(model.trainer.last)
    assert best.is_file(), 'Không tìm thấy best.pt sau train'
    write(run / 'STATE.json', dict(stage='evaluating_test', training_completed=True,
                                  best=str(best), test_completed=False, web_model_changed=False))
    # Release training optimizer/EMA before loading best for test on a 4-GB GPU.
    model.trainer = None
    model.to('cpu')
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Reload through the public loader: the checkpoint must be standalone.
    best_model = YOLO(str(best), task='detect')
    assert [best_model.names[i] for i in range(4)] == NAMES
    metrics = best_model.val(data=str(DATASET / 'data.yaml'), split='test', imgsz=args.imgsz,
                             batch=1, device=train_args['device'], workers=0, plots=True,
                             project=str(run), name='test', exist_ok=False)
    write(run / 'test-metrics.json', dict(results=metrics.results_dict, per_class_map50_95=metrics.box.maps,
                                        classes=NAMES, role='historical internal holdout; not a new blind benchmark'))
    write(run / 'MODEL.json', dict(best=str(best.relative_to(run)), best_sha256=sha(best),
                                  last=str(last.relative_to(run)), names=NAMES,
                                  normal_ultralytics_loader=True, custom_layers_required=False))
    write(run / 'STATE.json', dict(stage='complete', training_completed=True, test_completed=True,
                                  best=str(best), web_model_changed=False))
    print(f'ĐÃ XONG. Model: {best}\nBiểu đồ: {best.parent.parent}\nTest: {run / "test"}')


def main():
    args = parse_args()
    stamp = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%Y%m%d-%H%M%S')
    parent = ROOT / 'ket-qua' / ('reports' if args.check_only else 'runs')
    run = parent / f'yolo4-standard-{"check" if args.check_only else "train"}-{stamp}-{uuid4().hex[:6]}'
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, run / Path(__file__).name)
    (run / 'KET-QUA.txt').write_text('Lượt này dùng YOLO 4 lớp chuẩn. Xem STATE.json và console.log.\n'
                                    'Sau train: train/weights/best.pt, train/results.csv, train/results.png, test/.\n'
                                    'Không thay model web. Test là nguồn lịch sử, chưa phải đánh giá mù độc lập.\n', encoding='utf-8')
    prepare_runtime()
    with capture_console(run / 'console.log'):
        print(f'Thư mục kết quả: {run}')
        protected = None
        try:
            model, train_args, protected = preflight(args, run)
            if args.check_only:
                write(run / 'STATE.json', dict(stage='check_passed', training_started=False,
                                              training_completed=False, web_model_changed=False))
                print('CHỈ KIỂM TRA — chưa chạy train, AutoBatch hoặc đánh giá model.')
            else:
                print('BẮT ĐẦU TRAIN YOLO CHUẨN: 50% VRAM AutoBatch nếu batch chưa được đổi. Ctrl+C để dừng.')
                train(args, run, model, train_args)
        except BaseException as exc:
            old = read(run / 'STATE.json') if (run / 'STATE.json').exists() else {}
            write(run / 'STATE.json', dict(old, stage='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                          error=str(exc), error_type=type(exc).__name__))
            traceback.print_exc()
            print('Nếu hết VRAM: đóng ứng dụng dùng GPU rồi chạy với --batch 1. Không tự đổi dataset hoặc model web.')
            return 130 if isinstance(exc, KeyboardInterrupt) else 1
        finally:
            if protected is not None:
                unchanged = all(sha(ROOT / p) == h for p, h in protected.items())
                write(run / 'web-model-integrity.json', dict(unchanged=unchanged))
                if not unchanged:
                    raise RuntimeError('File model web đã thay đổi trong thời gian chạy; cần kiểm tra bên ngoài lượt train này.')
    return 0


if __name__ == '__main__':
    multiprocessing.freeze_support()
    interpreter = ROOT / 'ung-dung/.venv/Scripts/python.exe'
    if not interpreter.is_file():
        raise SystemExit('Không tìm thấy môi trường Python tại ung-dung/.venv/Scripts/python.exe')
    if Path(sys.executable).resolve() != interpreter.resolve():
        raise SystemExit(subprocess.call([str(interpreter), '-X', 'utf8', '-B', str(Path(__file__).resolve()), *sys.argv[1:]]))
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    raise SystemExit(main())
