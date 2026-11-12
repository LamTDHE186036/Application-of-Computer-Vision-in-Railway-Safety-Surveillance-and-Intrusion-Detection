"""Spatial-ignore adapter for this project's pinned YOLO26/Ultralytics version.

This module never starts training. Masks affect negative classification anchors,
not the pixels seen by the backbone. Valid positive assignments always survive.
Box/DFL/L1 losses use ONLY retained GT; withheld GT is absent from label files.
"""
from pathlib import Path
import hashlib, json, shutil
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import ultralytics
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose, Format, LetterBox
from ultralytics.utils.loss import v8DetectionLoss, E2ELoss
from ultralytics.utils.tal import make_anchors
from finetune_that_ai import PreservedHeadTrainer, coverage_mask, NAMES

SUPPORTED_VERSION = '8.4.161'
FORBIDDEN = ('mosaic','mixup','cutmix','copy_paste','degrees','translate','scale',
             'shear','perspective','flipud','fliplr','multi_scale','hsv_h','hsv_s','hsv_v','bgr')

def check_version():
    if ultralytics.__version__ != SUPPORTED_VERSION:
        raise RuntimeError(f'Ignore adapter tested on Ultralytics {SUPPORTED_VERSION}; got {ultralytics.__version__}. Revalidate before use.')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate_bundle(root):
    root=Path(root).resolve()
    doc=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert doc['schema']==1 and doc['names']==NAMES
    assert len(doc['images'])==100 and len({r['file'] for r in doc['images']})==100
    for r in doc['images']:
        name=r['file'];assert Path(name).name==name and name.startswith('vd1_')
        ip=root/'images'/name;lp=root/'labels'/Path(name).with_suffix('.txt');mp=root/'ignore-masks'/Path(name).with_suffix('.png')
        for p,key in [(ip,'image_sha256'),(lp,'label_sha256'),(mp,'mask_sha256')]:
            if sha(p)!=r[key]:raise ValueError(f'File changed; regenerate/review bundle: {p}')
        mask=cv2.imdecode(np.fromfile(mp,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        assert mask.shape==(r['height'],r['width']) and set(np.unique(mask))<={0,255}
        lines=lp.read_text(encoding='utf-8').splitlines()
        assert len(lines)==len(r['retained_boxes'])
        for line,b in zip(lines,r['retained_boxes']):
            c,x,y,w,h=map(float,line.split());assert c==b['cls'] and c in range(4)
            xy=np.array([(x-w/2)*r['width'],(y-h/2)*r['height'],(x+w/2)*r['width'],(y+h/2)*r['height']])
            assert np.max(np.abs(xy-np.array(b['xyxy'])))<.001
        assert not ({b['id'] for b in r['retained_boxes']} & {b['id'] for b in r['withheld_boxes']})
    return doc

class IgnoreLetterBox(LetterBox):
    """One shared set of resize/pad parameters for image, boxes and binary mask."""
    def __call__(self, labels):
        path=Path(labels['im_file'])
        mp=path.parent.parent/'ignore-masks'/path.with_suffix('.png').name
        if path.name.startswith('vd1_'):
            if not mp.is_file():raise FileNotFoundError(f'Cannot train vd1 without mask: {mp}')
            mask=cv2.imdecode(np.fromfile(mp,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape!=tuple(labels['ori_shape']):raise ValueError('Mask/original shape mismatch')
            if not set(np.unique(mask))<={0,255}:raise ValueError('Mask must contain only 0 and 255')
            cov=torch.ones(4)
        else:
            if mp.exists():raise ValueError('Unexpected mask for legacy data')
            cov=coverage_mask([path.name],'cpu')[0,0]
            mask=np.zeros(labels['ori_shape'],np.uint8)
        # YOLO load_image may resize once before transforms. Use exactly that size.
        ih,iw=labels['img'].shape[:2]
        mask=cv2.resize(mask,(iw,ih),interpolation=cv2.INTER_NEAREST)
        params=self.get_params(labels)
        mask=cv2.resize(mask,params['new_unpad'],interpolation=cv2.INTER_NEAREST)
        mask=cv2.copyMakeBorder(mask,params['top'],params['bottom'],params['left'],params['right'],cv2.BORDER_CONSTANT,value=0)
        labels=self.apply_image(labels,params)
        labels=self.apply_instances(labels,params)
        labels=self.apply_semantic(labels,params)
        assert mask.shape==labels['img'].shape[:2]
        labels['ignore_mask']=torch.from_numpy(mask>0)
        labels['class_coverage']=cov
        return labels

class IgnoreDataset(YOLODataset):
    def build_transforms(self,hyp=None):
        check_version()
        for key in FORBIDDEN:
            if getattr(hyp,key,0):raise ValueError(f'{key} must be 0 until mask-aware augmentation is implemented')
        if self.rect or getattr(hyp,'augmentations',None):raise ValueError('rect/custom augmentation unsupported by ignore adapter')
        return Compose([IgnoreLetterBox(new_shape=(self.imgsz,self.imgsz),scaleup=True),
                        Format(bbox_format='xywh',normalize=True,batch_idx=True,bgr=0)])

    @staticmethod
    def collate_fn(batch):
        result=YOLODataset.collate_fn(batch)
        result['ignore_mask']=torch.stack(result['ignore_mask'])
        result['class_coverage']=torch.stack(result['class_coverage'])
        return result

class IgnoreBCE(nn.Module):
    def __init__(self):
        super().__init__();self.ignore=None;self.last_effective_ignore=None
    def forward(self,logits,targets):
        if self.ignore is None or self.ignore.shape!=logits.shape[:2]:raise ValueError('Missing/misaligned anchor ignore mask')
        # All class logits at a retained positive anchor remain supervised.
        effective=self.ignore & ~targets.sum(-1).gt(0)
        self.last_effective_ignore=effective.detach()
        return F.binary_cross_entropy_with_logits(logits,targets,reduction='none')*(~effective).unsqueeze(-1)

class SpatialIgnoreLoss(v8DetectionLoss):
    def __init__(self,*args,**kwargs):
        check_version();super().__init__(*args,**kwargs);self.bce=IgnoreBCE()
    def get_assigned_targets_and_loss(self,preds,batch):
        n=preds['scores'].shape[0]
        cov=batch.get('class_coverage')
        if cov is None:
            if any(Path(p).name.startswith('vd1_') for p in batch['im_file']):raise ValueError('vd1 requires custom dataloader')
            cov=coverage_mask(batch['im_file'],self.device)[:,0,:]
        self.class_weights=cov.to(self.device).view(n,1,4)
        anchors,strides=make_anchors(preds['feats'],self.stride,.5)
        points=anchors*strides
        mask=batch.get('ignore_mask')
        if mask is None:
            if any(Path(p).name.startswith('vd1_') for p in batch['im_file']):raise ValueError('Missing spatial-ignore mask')
            self.bce.ignore=torch.zeros((n,len(points)),dtype=torch.bool,device=self.device)
        else:
            h,w=batch['img'].shape[-2:]
            if mask.shape!=(n,h,w):raise ValueError('Mask does not match model input; transformed incorrectly')
            expected_h=preds['feats'][0].shape[-2]*int(self.stride[0]);expected_w=preds['feats'][0].shape[-1]*int(self.stride[0])
            if (h,w)!=(expected_h,expected_w):raise ValueError('Feature grid/input size mismatch')
            xi=points[:,0].long().clamp(0,w-1);yi=points[:,1].long().clamp(0,h-1)
            self.bce.ignore=mask.to(self.device,dtype=torch.bool)[:,yi,xi]
        return super().get_assigned_targets_and_loss(preds,batch)

class SpatialPreservedHeadTrainer(PreservedHeadTrainer):
    def build_dataset(self,img_path,mode='train',batch=None):
        if mode!='train':return super().build_dataset(img_path,mode,batch)
        if not self.data.get('spatial_ignore'):raise ValueError('Missing spatial_ignore data marker')
        return IgnoreDataset(img_path=img_path,imgsz=self.args.imgsz,batch_size=batch,
            augment=True,hyp=self.args,rect=False,cache=None,single_cls=False,
            stride=32,pad=0,task='detect',data=self.data,fraction=1.0)
    def preprocess_batch(self,batch):
        if self.args.multi_scale:raise ValueError('multi_scale would desynchronize mask')
        return super().preprocess_batch(batch)

def enable_spatial_ignore(trainer):
    check_version()
    for key in FORBIDDEN:
        if getattr(trainer.args,key,0):raise ValueError(f'Unsupported augmentation: {key}')
    trainer.model.criterion=E2ELoss(trainer.model,SpatialIgnoreLoss)

def append_ignore_data(out,config,bundle):
    """Append full frames to the existing real+AI TRAIN split, never validation."""
    import yaml
    doc=validate_bundle(bundle);bundle=Path(bundle);out=Path(out)
    val_hashes={sha(p) for p in (out/'data/val/images').glob('*') if p.is_file()}
    if val_hashes & {r['image_sha256'] for r in doc['images']}:
        raise ValueError('vd1 contains an exact duplicate of a validation image')
    dest=out/'data/train'
    for folder in ['images','labels','ignore-masks']:(dest/folder).mkdir(exist_ok=True)
    listing=[]
    for r in doc['images']:
        for kind,suffix in [('images','.jpg'),('labels','.txt'),('ignore-masks','.png')]:
            name=Path(r['file']).with_suffix(suffix).name;target=dest/kind/name
            if target.exists():raise FileExistsError(target)
            shutil.copy2(bundle/kind/name,target)
        listing.append((dest/'images'/r['file']).resolve().as_posix())
    with (out/'train.txt').open('a',encoding='utf-8') as f:f.write('\n'.join(listing)+'\n')
    cfg=yaml.safe_load(Path(config).read_text(encoding='utf-8'));cfg['spatial_ignore']=True
    Path(config).write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    shutil.copy2(bundle/'manifest.json',out/'vd1-ignore-manifest.json')
    return doc['summary']
