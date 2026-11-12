"""Five-class warm start and strict partial/spatial-ignore integration.

Pinned to Ultralytics 8.4.161. Importing this module never trains or deploys.
Keep this module on PYTHONPATH to load its research checkpoints.
"""
from pathlib import Path
from copy import deepcopy,copy
import hashlib,json,math
import cv2,numpy as np,torch,ultralytics
from torch import nn
import torch.nn.functional as F
from PIL import Image
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose,Format,LetterBox
from ultralytics.models.yolo.detect import DetectionTrainer,DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import E2ELoss,v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.metrics import box_iou
from ultralytics.utils import ops
from torch.utils.data import DataLoader

NAMES=['person','two_wheeler','large_vehicle','car','train']
VERSION='8.4.161'
FORBIDDEN=('mosaic','mixup','cutmix','copy_paste','degrees','translate','scale','shear',
 'perspective','flipud','fliplr','multi_scale','hsv_h','hsv_s','hsv_v','bgr')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def version_check():
 if ultralytics.__version__!=VERSION:raise RuntimeError(f'Five-class adapter requires {VERSION}; got {ultralytics.__version__}')

class FiveGroupedClassifier(nn.Module):
 """Keep all six learned logits and append a seventh logit for COCO train."""
 def __init__(self,old_classifier,train_classifier):
  super().__init__()
  assert old_classifier.out_channels==6 and train_classifier.out_channels==80
  assert old_classifier.in_channels==train_classifier.in_channels
  self.classifier=nn.Conv2d(old_classifier.in_channels,7,1,bias=True)
  with torch.no_grad():
   self.classifier.weight[:6].copy_(old_classifier.weight)
   self.classifier.bias[:6].copy_(old_classifier.bias)
   self.classifier.weight[6].copy_(train_classifier.weight[6])
   self.classifier.bias[6].copy_(train_classifier.bias[6])
  self.out_channels=5
 def forward(self,x):
  s=self.classifier(x)
  return torch.stack((s[:,0],torch.maximum(s[:,1],s[:,3]),torch.maximum(s[:,4],s[:,5]),s[:,2],s[:,6]),dim=1)

class FiveDetectionModel(DetectionModel):
 """Make EMA, validation and checkpoint reload use the same strict criterion."""
 def init_criterion(self):return E2ELoss(self,FiveDetectionLoss)

def expand_from_web(web,coco):
 version_check()
 assert list(web.names.values())==NAMES[:4] and web.model[-1].nc==4
 assert coco.names[6]=='train' and coco.model[-1].nc==80
 expanded=deepcopy(web).float().cpu();expanded.__class__=FiveDetectionModel
 for branch in ('cv3','one2one_cv3'):
  for i,scale in enumerate(getattr(expanded.model[-1],branch)):
   old=scale[-1];base=getattr(coco.model[-1],branch)[i][-1]
   assert hasattr(old,'classifier') and old.out_channels==4
   scale[-1]=FiveGroupedClassifier(old.classifier,base)
 head=expanded.model[-1];head.nc=5;head.no=5+4*head.reg_max
 expanded.nc=5;expanded.names=dict(enumerate(NAMES));expanded.yaml['nc']=5
 expanded.criterion=None
 assert not any(isinstance(m,FiveGroupedClassifier) for m in web.modules())
 return expanded

def freeze_for_continuation(model):
 """Conservative first configuration: classifier weights only; all BN frozen."""
 assert list(model.names.values())==NAMES and model.model[-1].nc==5
 for p in model.parameters():p.requires_grad_(False)
 heads=[m for m in model.modules() if isinstance(m,FiveGroupedClassifier)]
 assert len(heads)==6
 for head in heads:
  for p in head.parameters():p.requires_grad_(True)
 for m in model.modules():
  if isinstance(m,nn.BatchNorm2d):m.eval()
 return sum(p.numel() for p in model.parameters() if p.requires_grad)

def validate_contract(data,split=None):
 version_check();path=Path(data['five_class_annotation_contract']).resolve()
 doc=json.loads(path.read_text(encoding='utf8'))
 assert doc['schema']=='railway-five-class-v1' and doc['names']==NAMES
 assert data['nc']==5 and list(data['names'].values() if isinstance(data['names'],dict) else data['names'])==NAMES
 rows={};root=path.parent
 manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
 assert doc['entries']==manifest
 for row in doc['entries']:
  if split is not None and row['split']!=split:continue
  key=str(Path(row['image']).resolve());assert key not in rows
  assert Path(key).is_relative_to(root/row['split'])
  assert len(row['coverage'])==5 and all(c in (0,1) for c in row['coverage'])
  for field in ('image','label'):
   if sha(row[field])!=row[field+'_sha256']:raise ValueError(f'Reviewed {field} changed: '+row[field])
  for line in Path(row['label']).read_text(encoding='utf8').splitlines():
   if not line.strip():continue
   v=list(map(float,line.split()));assert len(v)==5 and all(math.isfinite(x) for x in v)
   c,x,y,w,h=v
   assert c==int(c) and 0<=c<5 and row['coverage'][int(c)]==1
   assert 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1
  with Image.open(key) as im:size=(im.height,im.width)
  for mask in row['masks']:
   raw=read_mask(mask,size)
  rows[key]=row
 if not rows:raise ValueError('No registered images for split '+str(split))
 if split:
  listing=Path(data['path'])/(split+'.txt')
  assert {str(Path(p).resolve()) for p in listing.read_text(encoding='utf8').splitlines() if p}==set(rows)
 return doc,rows

def read_mask(m,shape):
 if sha(m['path'])!=m['sha256']:raise ValueError('Missing or changed mask: '+m['path'])
 raw=cv2.imdecode(np.fromfile(m['path'],np.uint8),cv2.IMREAD_GRAYSCALE)
 if raw is None or raw.shape!=tuple(shape):raise ValueError('Mask/original size mismatch')
 if m['kind']=='spatial':
  if not set(np.unique(raw))<={0,255}:raise ValueError('Spatial mask must be binary')
 elif m['kind']=='classes':
  if raw.max()>31:raise ValueError('Class mask exceeds five bits')
 else:raise ValueError('Unknown mask kind '+str(m['kind']))
 return raw

def raw_masks(row,shape):
 # Per-class representation prevents a class-specific mask from suppressing others.
 mask=np.zeros((5,*shape),dtype=bool)
 for m in row['masks']:
  raw=read_mask(m,shape)
  allowed=m.get('applies_to_classes',list(range(5)))
  for c in allowed:
   if c not in range(5):raise ValueError('Invalid mask class')
   mask[c]|=(raw>0) if m['kind']=='spatial' else ((raw&(1<<c))!=0)
 return mask

class FiveLetterBox(LetterBox):
 def __init__(self,rows,**kw):super().__init__(**kw);self.rows=rows
 def __call__(self,labels):
  row=self.rows[str(Path(labels['im_file']).resolve())]
  masks=raw_masks(row,tuple(labels['ori_shape']))
  ih,iw=labels['img'].shape[:2];params=self.get_params(labels)
  output=[]
  for mask in masks:
   raw=cv2.resize(mask.astype(np.uint8),(iw,ih),interpolation=cv2.INTER_NEAREST)
   raw=cv2.resize(raw,params['new_unpad'],interpolation=cv2.INTER_NEAREST)
   raw=cv2.copyMakeBorder(raw,params['top'],params['bottom'],params['left'],params['right'],cv2.BORDER_CONSTANT,value=0)
   output.append(raw>0)
  labels=self.apply_image(labels,params);labels=self.apply_instances(labels,params);labels=self.apply_semantic(labels,params)
  labels['class_ignore_mask']=torch.from_numpy(np.stack(output))
  labels['class_coverage']=torch.tensor(row['coverage'],dtype=torch.float32)
  assert tuple(labels['class_ignore_mask'].shape)==(5,*labels['img'].shape[:2])
  return labels

class FiveDataset(YOLODataset):
 def __init__(self,*args,split,validated=None,**kwargs):
  self.contract_split=split;self.validated=validated
  super().__init__(*args,**kwargs)
 def get_labels(self):
  # No native cache repair/deduplication: frozen, reviewed labels remain verbatim.
  result=[];self.label_files=[]
  for name in self.im_files:
   p=Path(name);label=p.parent.parent/'labels'/p.with_suffix('.txt').name
   with Image.open(p) as im:shape=(im.height,im.width)
   a=np.asarray([list(map(float,l.split())) for l in label.read_text(encoding='utf8').splitlines() if l.strip()],dtype=np.float32).reshape(-1,5)
   result.append(dict(im_file=name,shape=shape,cls=a[:,:1],bboxes=a[:,1:],segments=[],keypoints=None,normalized=True,bbox_format='xywh'))
   self.label_files.append(str(label))
  return result
 def build_transforms(self,hyp=None):
  for field in FORBIDDEN:
   if getattr(hyp,field,0):raise ValueError('Mask-unsafe augmentation: '+field)
  if self.rect or getattr(hyp,'augmentations',None):raise ValueError('rect/custom transforms unsupported')
  _,self.ignore_rows=self.validated or validate_contract(self.data,self.contract_split)
  if {str(Path(p).resolve()) for p in self.im_files}!=set(self.ignore_rows):raise ValueError('Dataset/contract split mismatch')
  return Compose([FiveLetterBox(self.ignore_rows,new_shape=(self.imgsz,self.imgsz),scaleup=True),Format(bbox_format='xywh',normalize=True,batch_idx=True,bgr=0)])
 @staticmethod
 def collate_fn(batch):
  result=YOLODataset.collate_fn(batch)
  for k in ('class_ignore_mask','class_coverage'):result[k]=torch.stack(result[k])
  return result

class FiveMaskedBCE(nn.Module):
 def __init__(self):
  super().__init__();self.coverage=None;self.ignored=None;self.last_weights=None;self.last_targets=None
 def forward(self,logits,targets):
  if self.coverage is None or self.ignored is None:raise ValueError('Missing five-class mask metadata')
  if self.ignored.shape!=logits.shape or self.coverage.shape!=(logits.shape[0],1,5):raise ValueError('Mask/logit shape mismatch')
  # A retained positive class wins over spatial/class ignores. Unannotated classes
  # are unknown everywhere; GT validation prohibits positives of unknown classes.
  weights=self.coverage*((~self.ignored)|targets.gt(0)).to(logits.dtype)
  self.last_weights=weights.detach();self.last_targets=targets.detach()
  return F.binary_cross_entropy_with_logits(logits,targets,reduction='none')*weights

class FiveDetectionLoss(v8DetectionLoss):
 def __init__(self,*args,**kwargs):
  version_check();super().__init__(*args,**kwargs)
  assert self.nc==5;self.class_weights=None;self.bce=FiveMaskedBCE()
 def get_assigned_targets_and_loss(self,preds,batch):
  n,_,h,w=batch['img'].shape
  cov=batch.get('class_coverage');masks=batch.get('class_ignore_mask')
  if cov is None or masks is None:raise ValueError('Use FiveDataset: coverage/ignore metadata is mandatory')
  if cov.shape!=(n,5) or masks.shape!=(n,5,h,w) or masks.dtype!=torch.bool:raise ValueError('Five-class masks misaligned')
  if not torch.all((cov==0)|(cov==1)):raise ValueError('Coverage must be binary')
  anchors,strides=make_anchors(preds['feats'],self.stride,.5);points=anchors*strides
  if (h,w)!=(preds['feats'][0].shape[-2]*int(self.stride[0]),preds['feats'][0].shape[-1]*int(self.stride[0])):raise ValueError('Feature/input grid mismatch')
  xi=points[:,0].long().clamp(0,w-1);yi=points[:,1].long().clamp(0,h-1)
  self.bce.coverage=cov.to(self.device).view(n,1,5)
  self.bce.ignored=masks.to(self.device)[:,:,yi,xi].permute(0,2,1)
  return super().get_assigned_targets_and_loss(preds,batch)

def filter_metric_predictions(pred,gt,row):
 """Ignore unknown-class predictions; mask >=50% area, preserve IoU>=.5 GT errors."""
 n=len(pred['cls']);keep=torch.ones(n,dtype=torch.bool,device=pred['cls'].device)
 if not n:return pred,keep
 coverage=torch.tensor(row['coverage'],device=keep.device)
 keep&=coverage[pred['cls'].long()].bool()
 boxes=ops.scale_boxes(gt['imgsz'],pred['bboxes'].clone(),gt['ori_shape'],gt['ratio_pad']).cpu()
 gb=ops.scale_boxes(gt['imgsz'],gt['bboxes'].clone(),gt['ori_shape'],gt['ratio_pad']).cpu()
 protected=box_iou(boxes,gb).amax(1)>=.5 if len(gb) else torch.zeros(n,dtype=torch.bool)
 if row['masks']:
  masks=raw_masks(row,tuple(gt['ori_shape']));h,w=masks.shape[-2:]
  for i,box in enumerate(boxes):
   if protected[i]:continue
   x1,y1,x2,y2=box.tolist();x1,y1=max(0,math.floor(x1)),max(0,math.floor(y1));x2,y2=min(w,math.ceil(x2)),min(h,math.ceil(y2))
   if x2>x1 and y2>y1 and masks[int(pred['cls'][i]),y1:y2,x1:x2].mean()>=.5:keep[i]=False
 return {k:v[keep] if isinstance(v,torch.Tensor) and v.ndim and len(v)==n else v for k,v in pred.items()},keep

class FiveValidator(DetectionValidator):
 def get_dataloader(self,dataset_path,batch_size):
  ds=FiveDataset(dataset_path,split=self.args.split,imgsz=self.args.imgsz,batch_size=batch_size,augment=False,hyp=self.args,rect=False,cache=False,data=self.data,stride=32,pad=0,task='detect')
  return DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=ds.collate_fn)
 def preprocess(self,batch):
  if 'class_coverage' not in batch or 'class_ignore_mask' not in batch:raise ValueError('Validation needs the same explicit ignore-aware loader')
  return super().preprocess(batch)
 def update_metrics(self,preds,batch):
  if not hasattr(self,'metric_rows'):_,self.metric_rows=validate_contract(self.data,self.args.split)
  filtered=[]
  for i,pred in enumerate(preds):
   gt=self._prepare_batch(i,batch);row=self.metric_rows[str(Path(gt['im_file']).resolve())]
   selected,_=filter_metric_predictions(pred,gt,row);filtered.append(selected)
  return super().update_metrics(filtered,batch)

def install_loss(trainer):
 freeze_for_continuation(trainer.model)
 trainer.model.criterion=trainer.model.init_criterion()
 start=int(getattr(trainer,'start_epoch',0))
 if start:trainer.model.criterion.updates=start-1;trainer.model.criterion.update()
 optimized={id(p) for group in trainer.optimizer.param_groups for p in group['params']}
 assert all(id(p) in optimized for p in trainer.model.parameters() if p.requires_grad)

class FiveTrainer(DetectionTrainer):
 def __init__(self,*args,allow_training=False,**kwargs):
  self.explicit_training=allow_training
  super().__init__(*args,**kwargs);self.add_callback('on_train_start',install_loss)
 def train(self):
  if not self.explicit_training:raise RuntimeError('Check-only mode: no training was requested')
  return super().train()
 def get_model(self,cfg=None,weights=None,verbose=True):
  if weights is None or not isinstance(weights,FiveDetectionModel):raise ValueError('Use the prepared five-class warm start, not a new random model')
  model=deepcopy(weights).float();model.criterion=None;freeze_for_continuation(model);return model
 def _model_train(self):
  super()._model_train()
  for m in self.model.modules():
   if isinstance(m,nn.BatchNorm2d):m.eval()
 def build_dataset(self,img_path,mode='train',batch=None):
  return FiveDataset(img_path,split='train' if mode=='train' else 'val',imgsz=self.args.imgsz,batch_size=batch,augment=mode=='train',hyp=self.args,rect=False,cache=False,data=self.data,stride=32,pad=0,task='detect')
 def preprocess_batch(self,batch):
  if self.args.multi_scale:raise ValueError('multi_scale would misalign ignore masks')
  return super().preprocess_batch(batch)
 def get_validator(self):
  return FiveValidator(self.test_loader,save_dir=self.save_dir,args=copy(self.args),_callbacks=self.callbacks)
