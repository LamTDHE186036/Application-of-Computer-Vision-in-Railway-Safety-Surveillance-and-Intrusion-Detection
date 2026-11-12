"""Kiểm tra dữ liệu và fine-tune mô hình YOLO26m năm lớp.

Học tiếp checkpoint YOLO26m fine-tune 4 lớp đã chốt, giữ 4 lớp và thêm train.
Mặc định huấn luyện vào thư mục mới, lưu checkpoint mỗi epoch.
--check-only chỉ kiểm tra dữ liệu và cấu hình.
Không thay model/config của web. railway_five_class.py phải nằm cạnh file này.
"""
from pathlib import Path
import os,sys,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
if __name__=='__main__':
 py=HERE/'.venv/Scripts/python.exe'
 if py.is_file() and Path(sys.executable).resolve()!=py.resolve():
  raise SystemExit(subprocess.call([str(py),'-X','utf8','-B',str(Path(__file__).resolve()),*sys.argv[1:]]))
 for s in (sys.stdout,sys.stderr):
  if hasattr(s,'reconfigure'):s.reconfigure(encoding='utf8')
sys.path.insert(0,str(HERE))
os.environ.setdefault('YOLO_AUTOINSTALL','false')
os.environ.setdefault('YOLO_CONFIG_DIR',str(HERE/'.settings'))
import argparse,copy,json,shutil,functools
from datetime import datetime,timezone,timedelta
import numpy as np,torch,yaml
from PIL import Image,ImageDraw
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from railway_five_class import (NAMES,FORBIDDEN,FiveGroupedClassifier,FiveDetectionModel,
 FiveDetectionLoss,FiveMaskedBCE,FiveDataset,FiveTrainer,FiveValidator,
 expand_from_web,freeze_for_continuation,validate_contract,raw_masks,read_mask,
 filter_metric_predictions,version_check,sha)

MODEL=HERE/'inputs/yolo26m-that-ai.pt'
MODEL_HASH='abe022e9d2b27f39f45507cdd289e7e152853a0255d776a3ad1b9c9aeed051de'
COCO=HERE/'models/yolo26m.pt'
DATA=ROOT/'data/prepared/finetune-five-class-train-20261110-reviewed'
PLAN=HERE/'runs/five-class-preflight'
REPORT=HERE/'runs/five-class-code-check'
TRAIN=dict(epochs=10,imgsz=640,batch=8,nbs=64,device=0,workers=0,optimizer='AdamW',
 lr0=1e-5,lrf=.2,weight_decay=.0005,warmup_epochs=1,warmup_bias_lr=1e-5,
 patience=4,amp=False,seed=20261110,deterministic=True,freeze=23,cache=False,
 rect=False,nms=False,cos_lr=True,save=True,save_period=1,val=True,resume=False,
 close_mosaic=0,plots=False,verbose=False,cls_pw=0,compile=False,
 **{k:0 for k in FORBIDDEN})

def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def raw_output(model,x):
 result=model(x)
 return result[1] if isinstance(result,tuple) else result
def dataset(data,split,size,validated):
 args=get_cfg(overrides=TRAIN);args.imgsz=size
 return FiveDataset(str(DATA/f'{split}.txt'),split=split,validated=validated,
  imgsz=size,batch_size=8,augment=split=='train',hyp=args,rect=False,cache=False,
  single_cls=False,stride=32,pad=0,task='detect',data=data)
def prepare_batch(ds,indices,trainer):
 return trainer.preprocess_batch(ds.collate_fn([ds[i] for i in indices]))

def verify_initialization(web,coco,expanded):
 old=web.state_dict();new=expanded.state_dict();changed=[]
 for name,t in old.items():
  if '.classifier.' in name:
   assert torch.equal(t,new[name][:6]),name
  else:assert torch.equal(t,new[name]),name
 for branch in ('cv3','one2one_cv3'):
  for i,scale in enumerate(getattr(expanded.model[-1],branch)):
   target=scale[-1].classifier;source=getattr(coco.model[-1],branch)[i][-1]
   assert torch.equal(target.weight[6],source.weight[6]) and torch.equal(target.bias[6],source.bias[6])
 web.eval();expanded.eval();torch.manual_seed(20261110)
 errors={}
 with torch.no_grad():
  x=torch.rand(1,3,160,160);a=raw_output(web,x);b=raw_output(expanded,x)
  for branch in ('one2many','one2one'):
   score=float((a[branch]['scores']-b[branch]['scores'][:,:4]).abs().max())
   box=float((a[branch]['boxes']-b[branch]['boxes']).abs().max())
   assert score<1e-5 and box<1e-5,(branch,score,box)
   errors[branch]=dict(old_four_logit_max_error=score,box_max_error=box)
 return dict(all_existing_parameter_values_preserved=True,new_class_initializer='COCO train row 6 only; all existing weights come from deployed fine-tuned model',branch_output_checks=errors,
  caveat='Preserved raw old-class outputs do not imply unchanged final detections: train competes during postprocessing.')

def save_initial(expanded):
 PLAN.mkdir(parents=True,exist_ok=True);init=PLAN/'init-five-class-untrained.pt'
 expanded.criterion=None
 if not init.exists():
  wrapper=YOLO(str(MODEL));wrapper.model=expanded.eval();wrapper.save(str(init));del wrapper
 loaded=YOLO(str(init)).model.float().eval()
 assert isinstance(loaded,FiveDetectionModel) and list(loaded.names.values())==NAMES
 for k,v in expanded.state_dict().items():assert torch.equal(v.half(),loaded.state_dict()[k].half()),k
 # Fresh interpreter reload validates the custom checkpoint dependency.
 code="from ultralytics import YOLO; import numpy as np,sys; m=YOLO(sys.argv[1]); assert list(m.names.values())==['person','two_wheeler','large_vehicle','car','train']; m.predict(np.zeros((160,160,3),np.uint8),imgsz=160,device='cpu',verbose=False); print('Five-class reload OK')"
 result=subprocess.run([sys.executable,'-X','utf8','-B','-c',code,str(init)],cwd=HERE,capture_output=True,text=True,encoding='utf8')
 if result.returncode:raise RuntimeError(result.stderr)
 return init,loaded

def check_direct_bce():
 bce=FiveMaskedBCE();scores=torch.zeros(1,4,5,requires_grad=True);target=torch.zeros_like(scores);target[0,0,4]=1
 bce.coverage=torch.tensor([[[1,1,1,1,1]]],dtype=torch.float32);bce.ignored=torch.ones_like(scores,dtype=torch.bool)
 bce.ignored[:,1,:]=False
 bce(scores,target).sum().backward()
 assert scores.grad[0,0,4]<0 and torch.count_nonzero(scores.grad[0,0,:4])==0
 assert torch.all(scores.grad[0,1]>0) and torch.count_nonzero(scores.grad[0,2:])==0
 bce.coverage[:,:,-1]=0;scores=torch.zeros_like(scores,requires_grad=True);bce(scores,torch.zeros_like(scores)).sum().backward()
 assert torch.count_nonzero(scores.grad[:,:,4])==0
 return dict(positive_overrides_ignore=True,ignored_negative_gradients_zero=True,unknown_train_gradients_zero=True)

def check_geometry(ds,rows):
 checks=[]
 # Every nonempty mask at the actual training resolution, including class bits.
 for i,p in enumerate(ds.im_files):
  row=rows[str(Path(p).resolve())]
  if not row['masks']:continue
  sample=ds[i];original=raw_masks(row,tuple(sample['ori_shape']))
  from railway_five_class import FiveLetterBox
  # Independently reconstruct expected normalized letterbox coordinates using
  # native loaded image dimensions; compare every class/pixel, not only a center.
  raw=ds.load_image(i)[0];h,w=raw.shape[:2];size=ds.imgsz
  ratio=min(size/h,size/w);nw,nh=round(w*ratio),round(h*ratio)
  left=round((size-nw)/2-.1);right=round((size-nw)/2+.1)
  top=round((size-nh)/2-.1);bottom=round((size-nh)/2+.1)
  import cv2
  expected=[]
  for c in range(5):
   a=cv2.resize(original[c].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
   a=cv2.resize(a,(nw,nh),interpolation=cv2.INTER_NEAREST)
   expected.append(cv2.copyMakeBorder(a,top,bottom,left,right,cv2.BORDER_CONSTANT,value=0)>0)
  assert np.array_equal(sample['class_ignore_mask'].numpy(),np.stack(expected)),p
  checks.append(p)
  if Path(p).name in ('rail_I342.jpg','rail_I359.jpg') or (Path(p).name.startswith('vd1_') and not (REPORT/'vd1-mask.jpg').exists()):
   img=Image.fromarray(sample['img'].permute(1,2,0).numpy());overlay=img.copy()
   arr=np.asarray(overlay).copy();mask=sample['class_ignore_mask'].any(0).numpy();arr[mask]=np.array([245,158,11],dtype=np.uint8)
   img=Image.blend(img,Image.fromarray(arr),.35)
   img.save(REPORT/('vd1-mask.jpg' if Path(p).name.startswith('vd1_') else Path(p).stem+'-mask.jpg'))
 return dict(masked_images_checked=len(checks),all_five_channels_pixel_aligned=True)

def score_branches(raw):return {k:raw[k]['scores'] for k in ('one2many','one2one')}
def check_real_loss(model,batch):
 model.train();count=freeze_for_continuation(model);assert count==10794
 model.criterion=model.init_criterion();before={k:t.clone() for k,t in model.state_dict().items()}
 raw=model(batch['img']);scores=score_branches(raw)
 for score in scores.values():score.retain_grad()
 loss,_=model.criterion(raw,batch);assert torch.isfinite(loss).all();loss.sum().backward()
 checks={}
 for name,criterion in [('one2many',model.criterion.one2many),('one2one',model.criterion.one2one)]:
  bce=criterion.bce;grad=scores[name].grad.permute(0,2,1)
  zero=bce.last_weights==0
  assert zero.any() and torch.count_nonzero(grad[zero])==0
  positive=bce.last_targets>0
  assert positive[:,:,4].any() and torch.all(bce.last_weights[positive]==1)
  assert grad[:,:,4].abs().sum()>0
  checks[name]=dict(zero_weight_logits=int(zero.sum()),zero_gradients_verified=True,
   train_positive_anchors=int(positive[:,:,4].sum()),train_logit_gradient=float(grad[:,:,4].abs().sum()))
 for k,t in before.items():assert torch.equal(t,model.state_dict()[k]),k
 assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
 # Empty-GT all-ignored actual feature grids: classification and regression
 # contributions both vanish on both branches; no optimizer is instantiated.
 def detached_preds():
  return {name:{k:([v.detach() for v in val] if isinstance(val,list) else val.detach().clone().requires_grad_(True) if k in ('scores','boxes') else val.detach()) for k,val in raw[name].items()} for name in ('one2many','one2one')}
 empty=dict(batch);empty.update(cls=torch.zeros((0,1)),bboxes=torch.zeros((0,4)),batch_idx=torch.zeros(0),class_ignore_mask=torch.ones_like(batch['class_ignore_mask']),class_coverage=torch.ones_like(batch['class_coverage']))
 pred=detached_preds();criterion=model.init_criterion();zero,_=criterion(pred,empty)
 assert float(zero.detach().sum())==0;zero.sum().backward()
 for name in pred:assert torch.count_nonzero(pred[name]['scores'].grad)==0
 # Retained GT with an all-ignored image still generates positive supervision.
 retained=dict(batch);retained['class_ignore_mask']=torch.ones_like(batch['class_ignore_mask'])
 pred=detached_preds();criterion=model.init_criterion();nonzero,_=criterion(pred,retained)
 assert torch.isfinite(nonzero).all() and float(nonzero.detach().sum())>0;nonzero.sum().backward()
 for name,c in [('one2many',criterion.one2many),('one2one',criterion.one2one)]:
  assert c.bce.last_targets.gt(0).any() and torch.all(c.bce.last_weights[c.bce.last_targets>0]==1)
  grad=pred[name]['scores'].grad.permute(0,2,1);assert torch.count_nonzero(grad[c.bce.last_weights==0])==0
 try:model.criterion(raw,{k:v for k,v in batch.items() if k!='class_coverage'})
 except ValueError:pass
 else:raise AssertionError('Missing coverage was silently accepted')
 for k,t in before.items():assert torch.equal(t,model.state_dict()[k]),k
 model.zero_grad(set_to_none=True);model.eval()
 return dict(trainable_parameters=count,branches=checks,empty_gt_all_ignore_loss_zero=True,
  positive_gt_survives_all_ignore=True,missing_metadata_rejected=True,all_weights_and_bn_buffers_unchanged=True,optimizer_steps=0)

def check_validation(model,ds,trainer):
 old=next(i for i,p in enumerate(ds.im_files) if Path(p).name.startswith('full_'))
 new=next(i for i,p in enumerate(ds.im_files) if Path(p).name.startswith('rail_'))
 validator=FiveValidator(args={**TRAIN,'imgsz':ds.imgsz,'plots':False},save_dir=REPORT/'validator-check')
 validator.device=torch.device('cpu');validator.data=trainer.data
 batch=validator.preprocess(ds.collate_fn([ds[old],ds[new]]))
 model.eval();model.criterion=None  # Exercise lazy criterion creation, also used by EMA validation.
 with torch.no_grad():loss,_=model.loss(batch)
 assert isinstance(model.criterion.one2one,FiveDetectionLoss)
 assert torch.isfinite(loss).all()
 assert batch['class_coverage'][0,4]==0 and batch['class_coverage'][1,4]==1
 # Feed explicit predictions through the real validator update_metrics hook.
 validator.init_metrics(model)
 preds=[]
 for i in range(2):
  gt=validator._prepare_batch(i,batch);assert len(gt['cls'])
  box=gt['bboxes'][0:1];cls=gt['cls'][0:1]
  if i==0:
   boxes=torch.cat([box,box,torch.tensor([[1.,1.,20.,20.]])]);classes=torch.cat([cls,(cls+1)%4,torch.tensor([4.])])
  else:boxes=box;classes=cls
  preds.append(dict(bboxes=boxes,cls=classes,conf=torch.ones(len(classes))*.9,extra=torch.zeros(len(classes),0)))
 validator.update_metrics(preds,batch)
 assert len(validator.metrics.stats['pred_cls'][0])==2
 assert 4 not in validator.metrics.stats['pred_cls'][0] and 4 in validator.metrics.stats['pred_cls'][1]
 return dict(actual_validation_loss_finite=True,lazy_ema_criterion_is_ignore_aware=True,unknown_train_predictions_excluded_from_old_val=True,
  observed_train_predictions_scored=True,wrong_class_foreground_predictions_retained=True)

def check_metric_ignore(train_rows):
 row=next(r for r in train_rows.values() if r.get('source_image_id')==342)
 with Image.open(row['image']) as im:h,w=im.height,im.width
 decisions=DATA/'review/decisions.json'
 removed=read(decisions if decisions.is_file() else HERE/'preflight-fixtures.json')['342']['withheld_boxes'][0]
 x,y,bw,bh=removed['xywh'];box=torch.tensor([[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]])
 gt=dict(imgsz=(h,w),ori_shape=(h,w),ratio_pad=((1.,1.),(0.,0.)),bboxes=torch.zeros(0,4))
 p=dict(bboxes=box,cls=torch.tensor([float(removed['class_id'])]),conf=torch.tensor([.9]),extra=torch.zeros(1,0))
 filtered,keep=filter_metric_predictions(p,gt,row);assert not keep.any()
 # Same overlapping retained GT protects a foreground classification error.
 gt['bboxes']=box.clone();filtered,keep=filter_metric_predictions(p,gt,row);assert keep.all()
 return dict(uncertain_region_prediction_ignored=True,retained_gt_overlap_protects_errors=True)

def preflight():
 torch.set_num_threads(4);version_check();REPORT.mkdir(parents=True,exist_ok=True)
 assert sha(MODEL)==MODEL_HASH
 protected={str(MODEL):sha(MODEL),str(COCO):sha(COCO),str(DATA/'manifest.json'):sha(DATA/'manifest.json'),str(DATA/'annotation-contract.json'):sha(DATA/'annotation-contract.json')}
 data=yaml.safe_load((DATA/'data.yaml').read_text(encoding='utf8'))
 train_validated=validate_contract(data,'train');val_validated=validate_contract(data,'val')
 web=YOLO(str(MODEL)).model.float().eval();coco=YOLO(str(COCO)).model.float().eval()
 expanded=expand_from_web(web,coco);initialization=verify_initialization(web,coco,expanded)
 del web,coco
 init,model=save_initial(expanded);del expanded
 trainer=FiveTrainer.__new__(FiveTrainer);trainer.args=get_cfg(overrides=TRAIN);trainer.device=torch.device('cpu');trainer.data=data
 # The production trainer's loader paths are used, not a stock YOLO loader.
 trainer.args.imgsz=320;model.args=trainer.args
 warm=trainer.get_model(weights=model)
 assert all(torch.equal(v,warm.state_dict()[k]) for k,v in model.state_dict().items())
 del model;model=warm
 ds=trainer.build_dataset(str(DATA/'train.txt'),'train',8)
 val=trainer.build_dataset(str(DATA/'val.txt'),'val',2)
 assert len(ds)==1779 and len(val)==175
 index={Path(p).name:i for i,p in enumerate(ds.im_files)}
 rows=train_validated[1]
 chosen=[index['rail_I342.jpg'],index['rail_I359.jpg']]
 for predicate in [
  lambda r:Path(r['image']).name.startswith('person_'),
  lambda r:Path(r['image']).name.startswith('vehicle_') and r['coverage'][4]==0,
  lambda r:r.get('train_absence_reviewed',False),
  lambda r:r['source_dataset']=='synthetic',
  lambda r:Path(r['image']).name.startswith('vd1_'),
  lambda r:any(m['kind']=='classes' for m in r['masks']),
 ]:
  chosen.append(next(i for i,p in enumerate(ds.im_files) if predicate(rows[str(Path(p).resolve())])))
 geometry={'320':check_geometry(ds,rows)}
 ds640=dataset(data,'train',640,train_validated)
 geometry['640']=check_geometry(ds640,rows);del ds640
 batch=prepare_batch(ds,chosen,trainer)
 direct=check_direct_bce();loss=check_real_loss(model,batch)
 validation=check_validation(model,val,trainer);metric_ignore=check_metric_ignore(rows)
 # Missing mask fails closed, without editing any reviewed files.
 bad=copy.deepcopy(next(m for r in rows.values() for m in r['masks']))
 bad['path']=str(REPORT/'deliberately-missing-mask.png')
 try:read_mask(bad,(10,10))
 except (FileNotFoundError,ValueError):missing=True
 else:raise AssertionError('Missing mask accepted')
 bad_args=copy.copy(trainer.args);bad_args.fliplr=.5
 try:ds.build_transforms(bad_args)
 except ValueError:augmentation=True
 else:raise AssertionError('Unsafe augmentation accepted')
 # Re-check actual source labels/masks after the entire forward/backward path.
 validate_contract(data,'train');validate_contract(data,'val')
 for path,digest in protected.items():assert sha(path)==digest,path
 config=dict(data,path=str(DATA),train=str(DATA/'train.txt'),val=str(DATA/'val.txt'),test=str(DATA/'test.txt'))
 (PLAN/'data.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf8')
 shutil.copy2(HERE/'railway_five_class.py',PLAN/'railway_five_class.py')
 protocol=dict(initialization=initialization,deployed_model_sha256=MODEL_HASH,coco_train_initializer_sha256=sha(COCO),
  init_checkpoint=str(init),init_checkpoint_sha256=sha(init),data_manifest_sha256=sha(DATA/'manifest.json'),
  annotation_contract_sha256=sha(DATA/'annotation-contract.json'),defaults=TRAIN,scope='10794 classifier parameters only; frozen backbone/neck/box regressors and all BN.',
  initialization_only=True,training_started=False,web_model_changed=False,
  selection='Validation with explicit class coverage and spatial-ignore filtering; test is not used for selection.')
 write(PLAN/'protocol.json',protocol)
 result=dict(passed=True,initialization=initialization,geometry=geometry,bce=direct,real_loss=loss,
  validation=validation,evaluation_masks=metric_ignore,missing_mask_rejected=missing,unsafe_augmentation_rejected=augmentation,
  fresh_process_checkpoint_reload=True,trainer_get_model_preserves_warm_start=True,train_images=len(ds),validation_images=len(val),
  mixed_batch=[str(Path(ds.im_files[i]).name) for i in chosen],optimizer_steps=0,training_started=False,
  deployed_model_unchanged=True,source_data_unchanged=True,trainer_integration_verified=True,
  limitation='Preflight verifies initialization/gradients/data flow, not trained model quality. Full epoch, GPU mixed precision and resume have not been exercised.')
 write(REPORT/'preflight.json',result)
 (REPORT/'README.md').write_text('# Kiểm tra code fine-tune 5 lớp\n\nChưa train: 0 optimizer steps. Các trọng số/buffer không đổi trong kiểm tra. init-five-class-untrained.pt chỉ là bản khởi tạo thêm lớp, chưa được fine-tune 5 lớp và không dùng thay model web.\n\n'+json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 return result

def train_later(resume=None):
 from functools import partial
 if resume:
  cp=torch.load(resume,map_location='cpu',weights_only=False)
  assert cp.get('optimizer') is not None and cp.get('epoch',-1)>=0
  assert isinstance(cp.get('ema'),FiveDetectionModel) or isinstance(cp.get('model'),FiveDetectionModel)
  cfg=dict(TRAIN);cfg['resume']=str(Path(resume).resolve())
  YOLO(str(resume)).train(trainer=partial(FiveTrainer,allow_training=True),**cfg)
  return
 result=preflight();assert result['passed']
 # Vietnam uses UTC+7; the fixed offset also works on Windows without tzdata.
 stamp=datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d-%H%M%S')
 out=RUNS/('finetune-five-class-'+stamp);out.mkdir(parents=True,exist_ok=False)
 for name in ('finetune_5_lop.py','railway_five_class.py'):shutil.copy2(HERE/name,out/name)
 shutil.copy2(PLAN/'protocol.json',out/'protocol.json')
 model=YOLO(str(PLAN/'init-five-class-untrained.pt'))
 model.train(trainer=partial(FiveTrainer,allow_training=True),data=str(PLAN/'data.yaml'),project=str(out),name='training',**TRAIN)
 assert sha(MODEL)==MODEL_HASH

def main(argv=None):
 global MODEL, MODEL_HASH, COCO, DATA, PLAN, REPORT, RUNS
 parser=argparse.ArgumentParser(description=__doc__)
 modes=parser.add_mutually_exclusive_group()
 modes.add_argument('--check-only',action='store_true',help='Chỉ kiểm tra, không bắt đầu huấn luyện')
 modes.add_argument('--train',action='store_true',help='Bắt đầu fine-tune (cũng là chế độ mặc định)')
 parser.add_argument('--resume',type=Path,help='Checkpoint có optimizer của lượt train 5 lớp')
 parser.add_argument('--source',type=Path,help='Grouped four-class checkpoint to continue')
 parser.add_argument('--pretrained',type=Path,help='YOLO26m COCO checkpoint for train initialization')
 parser.add_argument('--data-root',type=Path,help='Reviewed five-class dataset, with relocated metadata')
 parser.add_argument('--output',type=Path,help='Directory for new runs and preflight outputs')
 parser.add_argument('--expected-sha256',help='Optional expected SHA-256 of source checkpoint')
 args=parser.parse_args(argv)
 if args.source:
  MODEL=args.source.resolve();MODEL_HASH=args.expected_sha256 or sha(MODEL)
 elif args.expected_sha256:MODEL_HASH=args.expected_sha256
 if args.pretrained:COCO=args.pretrained.resolve()
 if args.data_root:DATA=args.data_root.resolve()
 RUNS=(args.output or HERE/'runs').resolve()
 PLAN=RUNS/'five-class-preflight';REPORT=RUNS/'five-class-code-check'
 if args.check_only and args.resume:parser.error('--resume không dùng cùng --check-only')
 if args.check_only:
  print('CHỈ KIỂM TRA — không cập nhật trọng số.',flush=True)
  print(json.dumps(preflight(),ensure_ascii=False,indent=2))
 else:
  print('FINE-TUNE 5 LỚP — lưu checkpoint từng epoch; không thay model trên web.',flush=True)
  train_later(args.resume)

if __name__=='__main__':main()
