"""Fine-tune YOLO26m trên dữ liệu thực và dữ liệu tổng hợp.

Mặc định: 700 ảnh thật + 700 ảnh AI, 125 ảnh validation thật, 3 epochs.
Mỗi lần chạy tạo thư mục riêng trong ket-qua/runs/finetune-that-ai-<ngày giờ>.
Không sửa checkpoint, báo cáo, dữ liệu cũ hoặc mô hình trên web; không đọc test v3.

File chứa toàn bộ logic huấn luyện, không import các module railway của dự án.
Vẫn cần thư viện trong .venv, model pretrained và dữ liệu đã chuẩn bị bên dưới.
Checkpoint mới cần file này (finetune_that_ai.py) nằm trên Python import path.

Tùy chọn terminal: --check-only (kiểm tra dữ liệu), --smoke (train rất ngắn).
Smoke chỉ kiểm tra code; KHÔNG dùng checkpoint smoke làm mô hình chính.
"""
from pathlib import Path
import sys
import os
import subprocess

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]

# Cấu hình huấn luyện.
MODEL_GOC = HERE / 'models/yolo26m.pt'
DU_LIEU = WORKSPACE / 'data/prepared/finetune-mixed-20261001'
NOI_LUU = HERE / 'runs'
SO_EPOCH = 3
KICH_THUOC_ANH = 640
BATCH = 8
LEARNING_RATE = 0.00005
SEED = 20261001
# ============================================================

# Sử dụng môi trường Python của dự án.
if __name__ == '__main__':
    python = HERE / '.venv/Scripts/python.exe'
    if python.is_file() and Path(sys.executable).resolve() != python.resolve():
        print('Chuyển sang môi trường Python của dự án...', flush=True)
        raise SystemExit(subprocess.call([str(python), '-X', 'utf8', str(Path(__file__).resolve()), *sys.argv[1:]]))
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')

import argparse
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import random
import shutil
import uuid
import math

os.environ.setdefault('YOLO_CONFIG_DIR', str(HERE / '.settings'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
import torch
from torch import nn
import numpy as np
import yaml
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss, E2ELoss

# Tên module cố định để checkpoint có thể được nạp lại.
MODULE = 'finetune_that_ai'
if __name__ == '__main__':
    sys.modules[MODULE] = sys.modules[__name__]

NAMES = ['person', 'two_wheeler', 'large_vehicle', 'car']
COCO_ROWS = [0, 1, 2, 3, 5, 7]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coverage_mask(files, device):
    choices = {'person': [1,0,0,0], 'vehicle': [0,1,1,1], 'full': [1,1,1,1]}
    masks = []
    for f in files:
        prefix = Path(f).name.split('_')[0]
        if prefix not in choices:
            raise ValueError('Thiếu quy ước lớp đã gán nhãn: ' + str(f))
        masks.append(choices[prefix])
    return torch.tensor(masks, device=device, dtype=torch.float32)[:,None,:]


class PartialDetectionLoss(v8DetectionLoss):
    def get_assigned_targets_and_loss(self, preds, batch):
        self.class_weights = coverage_mask(batch['im_file'], self.device)
        assert self.nc == 4 and self.class_weights.shape[0] == preds['scores'].shape[0]
        return super().get_assigned_targets_and_loss(preds, batch)


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
        return torch.stack((s[:,0], torch.maximum(s[:,1],s[:,3]),
                            torch.maximum(s[:,4],s[:,5]), s[:,2]), dim=1)


class PreservedHeadTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        assert weights is not None and list(weights.names.values()) == NAMES
        assert isinstance(weights.model[-1].one2one_cv3[0][-1], GroupedClassifier)
        return deepcopy(weights).float()

    def _model_train(self):
        super()._model_train()
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


for cls in (GroupedClassifier, PartialDetectionLoss, PreservedHeadTrainer):
    cls.__module__ = MODULE


def initialize_grouped(original):
    assert list(original.names.values())[:4] == ['person','bicycle','car','motorcycle']
    grouped = deepcopy(original).float().cpu()
    head = grouped.model[-1]
    for branch in (head.cv3, head.one2one_cv3):
        for scale in branch:
            scale[-1] = GroupedClassifier(scale[-1])
    head.nc = 4
    head.no = 4 + head.reg_max * 4
    grouped.nc = 4
    grouped.names = dict(enumerate(NAMES))
    grouped.yaml['nc'] = 4
    original.eval()
    grouped.eval()
    checks = {}
    with torch.no_grad():
        x = torch.rand(1,3,128,128)
        old, new = original(x)[1], grouped(x)[1]
        for branch in ('one2many','one2one'):
            z = old[branch]['scores']
            expected = torch.stack((z[:,0], z[:,[1,3]].amax(1),z[:,[5,7]].amax(1),z[:,2]),dim=1)
            box_error = float((old[branch]['boxes']-new[branch]['boxes']).abs().max())
            cls_error = float((expected-new[branch]['scores']).abs().max())
            assert box_error < 1e-5 and cls_error < 1e-5
            checks[branch] = dict(box_error=box_error, grouped_logit_error=cls_error)
    return grouped, checks


def on_train_start(trainer):
    for p in trainer.model.parameters():
        p.requires_grad_(False)
    for m in trainer.model.modules():
        if isinstance(m, GroupedClassifier):
            for p in m.parameters():
                p.requires_grad_(True)
    assert sum(p.numel() for p in trainer.model.parameters() if p.requires_grad) == 9252
    for key in ('mosaic','mixup','cutmix','copy_paste'):
        assert getattr(trainer.args,key) == 0
    trainer.model.criterion = E2ELoss(trainer.model, PartialDetectionLoss)


def log_epoch(trainer):
    write(trainer.save_dir/'progress.json',dict(epoch=trainer.epoch+1,epochs=trainer.epochs,metrics=trainer.metrics))


def select_data():
    rows = json.loads((DU_LIEU/'manifest.json').read_text(encoding='utf-8'))
    real = [r for r in rows if r['split']=='train' and r['source_dataset']!='synthetic']
    ai = [r for r in rows if r['split']=='train' and r['source_dataset']=='synthetic']
    # The shared data archive already contains the original selected 700+700 rows.
    if len(real)==700 and len(ai)==700:
        mixed = [r for r in rows if r['split']=='train']
    else:
        rng = random.Random(SEED)
        rng.shuffle(real)
        rng.shuffle(ai)
        half_real = []
        for source in sorted({r['source_dataset'] for r in real}):
            group = [r for r in real if r['source_dataset']==source]
            half_real.extend(group[:len(group)//2])
        assert len(half_real)==700 and len(ai)>=700, 'Cần đủ 700 ảnh thật và 700 ảnh AI'
        mixed = half_real + ai[:700]
        # Giữ đúng thứ tự RNG của thử nghiệm cũ (đã shuffle nhánh real-only trước).
        rng.shuffle(real)
        rng.shuffle(mixed)
    val = [r for r in rows if r['split']=='val']
    assert len(val)==125 and all(r['source_dataset']!='synthetic' for r in val)
    train_hashes, val_hashes = set(), set()
    for split, items in [('train',mixed),('val',val)]:
        seen = set()
        for r in items:
            image, label = DU_LIEU/r['image'], DU_LIEU/r['label']
            if image.name in seen:
                raise ValueError('Trùng tên ảnh: '+image.name)
            seen.add(image.name)
            for path,key in [(image,'image_sha256'),(label,'label_sha256')]:
                if not path.is_file() or sha(path)!=r[key]:
                    raise ValueError('Dữ liệu thay đổi hoặc thiếu: '+str(path))
            expected = coverage_mask([image.name],'cpu').flatten().tolist()
            for line in label.read_text(encoding='utf-8').splitlines():
                fields = list(map(float,line.split()))
                if len(fields)!=5 or not all(math.isfinite(v) for v in fields):
                    raise ValueError('Nhãn YOLO không hợp lệ: '+str(label))
                c,x,y,w,h = fields
                if c!=int(c) or int(c) not in range(4) or not expected[int(c)] or not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):
                    raise ValueError('Lớp/toạ độ nhãn không hợp lệ: '+str(label))
            (train_hashes if split=='train' else val_hashes).add(r['image_sha256'])
    if train_hashes & val_hashes:
        raise ValueError('Ảnh train trùng nội dung với validation')
    return mixed, val


def prepare_run_data(out, train, val):
    # Bản sao riêng: cache và mọi thao tác của thư viện đều nằm trong lượt chạy mới.
    for split, items in [('train',train),('val',val)]:
        for kind in ('images','labels'):
            (out/'data'/split/kind).mkdir(parents=True)
        for row in items:
            for key,kind in [('image','images'),('label','labels')]:
                src = DU_LIEU/row[key]
                shutil.copy2(src,out/'data'/split/kind/src.name)
    # Giữ thứ tự danh sách train như công thức trộn dữ liệu cũ.
    listing = out/'train.txt'
    listing.write_text('\n'.join((out/'data/train/images'/Path(r['image']).name).as_posix() for r in train)+'\n',encoding='utf-8')
    config = dict(path=(out/'data').as_posix(),train=listing.as_posix(),val='val/images',names=NAMES)
    path = out/'data.yaml'
    path.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    return path


def main():
    global MODEL_GOC, DU_LIEU, NOI_LUU
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--ignore-data',type=Path,help='Bộ ảnh vd1 toàn cảnh có mặt nạ riêng; bắt buộc dùng adapter spatial_ignore.py')
    parser.add_argument('--pretrained', type=Path, help='YOLO26m COCO checkpoint')
    parser.add_argument('--data-root', type=Path, help='Original full dataset or prepared 1400-image subset')
    parser.add_argument('--output', type=Path, help='New training run directory parent')
    args = parser.parse_args()
    if args.pretrained: MODEL_GOC = args.pretrained.resolve()
    if args.data_root: DU_LIEU = args.data_root.resolve()
    if args.output: NOI_LUU = args.output.resolve()
    if args.ignore_data:
        from spatial_ignore import validate_bundle,check_version
        check_version()
        validate_bundle(args.ignore_data)
        if args.check_only:
            from verify_spatial_ignore import run_checks
            run_checks(args.ignore_data)
            print('IGNORE CHECK OK — không huấn luyện, không cập nhật trọng số.');return
        if args.smoke:
            raise ValueError('Không dùng --smoke với vd1. Dùng --check-only để kiểm tra không cập nhật trọng số.')
    if not MODEL_GOC.is_file():
        raise FileNotFoundError(MODEL_GOC)
    train, val = select_data()
    # Kiểm tra loss không phạt các nhóm chưa được gán nhãn trong từng nguồn.
    x = torch.zeros((3,5,4),requires_grad=True)
    mask = coverage_mask(['person_a.jpg','vehicle_a.jpg','full_a.jpg'],'cpu')
    (torch.nn.functional.binary_cross_entropy_with_logits(x,torch.zeros_like(x),reduction='none')*mask).sum().backward()
    assert torch.all(x.grad[0,:,1:]==0) and torch.all(x.grad[1,:,0]==0) and torch.all(x.grad[2]!=0)
    print('Đã kiểm tra: 700 ảnh thật + 700 ảnh AI; 125 ảnh validation; nhãn, hash và chia tập hợp lệ.',flush=True)
    if args.check_only:
        print('CHECK OK — chưa huấn luyện, chưa tạo lượt chạy.');return
    if not torch.cuda.is_available():
        raise RuntimeError('Chưa có CUDA. Hãy dùng môi trường .venv với PyTorch CUDA và GPU NVIDIA.')
    if args.smoke:
        train = [r for r in train if r['source_dataset']!='synthetic'][:8]+[r for r in train if r['source_dataset']=='synthetic'][:8]
        val = val[:8]
    stamp = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%Y%m%d-%H%M%S')
    out = NOI_LUU / (('smoke-that-ai-' if args.smoke else 'finetune-that-ai-')+stamp+'-'+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    print('KẾT QUẢ MỚI:',out,flush=True)
    try:
        shutil.copy2(Path(__file__),out/'finetune_that_ai.py')
        config = prepare_run_data(out,train,val)
        trainer_type=PreservedHeadTrainer
        if args.ignore_data:
            from spatial_ignore import append_ignore_data,SpatialPreservedHeadTrainer
            ignore_summary=append_ignore_data(out,config,args.ignore_data)
            trainer_type=SpatialPreservedHeadTrainer
            shutil.copy2(HERE/'spatial_ignore.py',out/'spatial_ignore.py')
            write(out/'spatial-ignore-protocol.json',ignore_summary)
        write(out/'protocol.json',dict(smoke=args.smoke,model=str(MODEL_GOC),model_sha256=sha(MODEL_GOC),
            script_sha256=sha(Path(__file__)),ultralytics=ultralytics.__version__,torch=torch.__version__,
            device=torch.cuda.get_device_name(0),seed=SEED,train=train,validation=val,
            selection='best.pt selected by validation only; no test loaded',
            adaptation='Only 9252 classifier parameters trained; grouped maximum of pretrained COCO logits',
            limitation='Short classifier-only adaptation; synthetic images are not real railway observations'))
        torch.manual_seed(SEED)
        model = YOLO(str(MODEL_GOC))
        grouped, checks = initialize_grouped(model.model.float().cpu())
        model.model = grouped
        initial = out/'preserved-init.pt'
        model.save(str(initial))
        write(out/'initialization-checks.json',checks)
        del grouped, model
        model = YOLO(str(initial))
        model.add_callback('on_train_start',on_train_start)
        if args.ignore_data:
            from spatial_ignore import enable_spatial_ignore
            model.add_callback('on_train_start',enable_spatial_ignore)
        model.add_callback('on_fit_epoch_end',log_epoch)
        write(out/'status.json',dict(status='training',smoke=args.smoke))
        model.train(trainer=trainer_type,data=str(config),
            epochs=1 if args.smoke else SO_EPOCH,imgsz=160 if args.smoke else KICH_THUOC_ANH,
            batch=2 if args.smoke else BATCH,device=0,workers=0,
            project=str(out),name='training',exist_ok=False,freeze=23,
            # Smoke dùng bước cập nhật lớn hơn để kiểm chứng gradient/lưu trọng số
            # trong 8 batches; KHÔNG phải thí nghiệm chất lượng.
            optimizer='AdamW',lr0=.001 if args.smoke else LEARNING_RATE,lrf=.2,weight_decay=.0005,
            nbs=2 if args.smoke else 64,
            warmup_epochs=1,warmup_bias_lr=.001 if args.smoke else LEARNING_RATE,patience=3,
            amp='bf16',seed=SEED,deterministic=True,cache=False,
            mosaic=0,mixup=0,cutmix=0,copy_paste=0,close_mosaic=0,
            scale=0,translate=0,hsv_h=0,hsv_s=0,hsv_v=0,fliplr=0,
            plots=not args.smoke,save_period=1,cos_lr=True,verbose=False,val=True)
        best = Path(model.trainer.best)
        saved = YOLO(str(best)).model.float().state_dict()
        start = YOLO(str(initial)).model.float().state_dict()
        changed = [k for k in start if not torch.equal(start[k].half(),saved[k].half())]
        frozen = [k for k in changed if '.classifier.' not in k]
        delta = max((float((start[k]-saved[k]).abs().max()) for k in frozen if start[k].is_floating_point()),default=0.)
        assert delta<.01, 'Trọng số cố định thay đổi quá mức cho phép'
        assert any('.classifier.' in k for k in changed), 'Chưa ghi nhận cập nhật trọng số phân loại'
        write(out/'weight-checks.json',dict(changed_tensors=changed,frozen_max_abs_delta=delta))
        # Nạp checkpoint trong tiến trình khác để chắc chắn không phụ thuộc __main__.
        check = "import sys; from ultralytics import YOLO; import numpy as np; m=YOLO(sys.argv[1]); assert list(m.names.values())==['person','two_wheeler','large_vehicle','car']; m.predict(np.zeros((160,160,3),np.uint8),imgsz=160,device='cpu',verbose=False); print('Checkpoint reload OK')"
        subprocess.run([sys.executable,'-X','utf8','-c',check,str(best)],cwd=out,check=True)
        write(out/'status.json',dict(status='complete',smoke=args.smoke,best=str(best),sha256=sha(best),default_model_changed=False))
        print('\nHOÀN THÀNH. Checkpoint:',best,flush=True)
        print('Kết quả cũ và model trên web được giữ nguyên.',flush=True)
    except BaseException as exc:
        write(out/'status.json',dict(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc)))
        raise


if __name__=='__main__':
    main()
