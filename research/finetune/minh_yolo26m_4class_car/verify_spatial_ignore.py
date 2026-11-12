"""Checks only: image hashes, geometry, dataloader, losses and output gradients.
No optimizer, no training epoch and no checkpoint save.
--bundle PATH selects the dataset bundle.
"""
from pathlib import Path
import sys, os
ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
    py=Path(__file__).parent/'.venv/Scripts/python.exe'
    if py.is_file() and Path(sys.executable).resolve()!=py.resolve():
        import subprocess
        raise SystemExit(subprocess.call([str(py),'-X','utf8',str(Path(__file__).resolve()),*sys.argv[1:]]))
os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(__file__).parent/'.settings'))
os.environ.setdefault('YOLO_AUTOINSTALL','false')
import json, argparse, copy, tempfile
import torch
from torch import nn
import numpy as np
import cv2
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.instance import Instances
from finetune_that_ai import PartialDetectionLoss, NAMES
from spatial_ignore import (validate_bundle,sha,IgnoreBCE,IgnoreLetterBox,IgnoreDataset,
    SpatialIgnoreLoss,FORBIDDEN,check_version,append_ignore_data,SpatialPreservedHeadTrainer,enable_spatial_ignore)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__();self.p=nn.Parameter(torch.zeros(1));self.args=get_cfg()
        head=nn.Identity();head.stride=torch.tensor([8.,16.,32.]);head.nc=4;head.reg_max=1
        self.model=nn.ModuleList([head])

def prediction(size=128,batch=1):
    feats=[torch.zeros(batch,1,size//s,size//s) for s in [8,16,32]]
    n=sum(f.shape[-1]*f.shape[-2] for f in feats)
    return dict(feats=feats,boxes=torch.full((batch,4,n),2.,requires_grad=True),scores=torch.zeros(batch,4,n,requires_grad=True))

def blank_batch(size=128,batch=1,has_gt=False):
    return dict(img=torch.zeros(batch,3,size,size),im_file=['full_test.jpg']*batch,
        batch_idx=torch.tensor([0.] if has_gt else []),cls=torch.tensor([[0.]] if has_gt else []).reshape(-1,1),
        bboxes=torch.tensor([[.5,.5,.4,.4]] if has_gt else []).reshape(-1,4))

def check_loss_cases():
    results={};model=DummyModel()
    # Entire ignored image with no retained GT: all loss components and gradients zero.
    for case,has_gt in [('empty',False),('positive_inside_ignore',True)]:
        batch=blank_batch(has_gt=has_gt);batch['ignore_mask']=torch.ones(1,128,128,dtype=torch.bool)
        pred={b:prediction() for b in ['one2many','one2one']};loss=E2ELoss(model,SpatialIgnoreLoss)
        value,_=loss(pred,batch);value.sum().backward()
        branch_report={}
        for branch in ['one2many','one2one']:
            crit=getattr(loss,branch);ig=crit.bce.last_effective_ignore;g=pred[branch]['scores'].grad.permute(0,2,1)
            assert ig.any() and torch.all(g[ig]==0)
            if has_gt:
                assert (~ig).any() and g[~ig].abs().sum()>0
                assert pred[branch]['boxes'].grad.abs().sum()>0
            else:
                assert float(value.sum())==0 and torch.all(g==0)
                assert torch.all(pred[branch]['boxes'].grad==0)
            branch_report[branch]=dict(ignored_anchors=int(ig.sum()),positive_anchors=int((~ig).sum()),max_ignored_gradient=float(g[ig].abs().max()))
        results[case]=branch_report
    # No mask must reproduce existing partial-class behavior, loss AND gradients.
    for name in ['person_test.jpg','vehicle_test.jpg','full_test.jpg']:
        batch=blank_batch(has_gt=True);batch['im_file']=[name]
        if name.startswith('vehicle'):batch['cls'][:]=3
        p=prediction();q={k:([t.clone() for t in v] if isinstance(v,list) else v.detach().clone().requires_grad_(True)) for k,v in p.items()}
        a=PartialDetectionLoss(model);b=SpatialIgnoreLoss(model)
        va=a.loss(p,batch)[0];vb=b.loss(q,batch)[0];assert torch.equal(va,vb)
        va.sum().backward();vb.sum().backward()
        for k in ['scores','boxes']:assert torch.equal(p[k].grad,q[k].grad)
    results['legacy_loss_and_gradient_exact_match']=True
    # Batch masks independent; don't broadcast one image's mask to another.
    batch=blank_batch(batch=2);batch['ignore_mask']=torch.zeros(2,128,128,dtype=torch.bool);batch['ignore_mask'][0]=True
    p=prediction(batch=2);value=SpatialIgnoreLoss(model).loss(p,batch)[0];value.sum().backward()
    assert torch.all(p['scores'].grad[0]==0) and torch.all(p['scores'].grad[1]!=0)
    results['mixed_batch_isolation']=True
    # Direct boundary and foreground override of all class logits.
    bce=IgnoreBCE();bce.ignore=torch.tensor([[True,True,False]])
    logits=torch.zeros(1,3,4,requires_grad=True);target=torch.zeros_like(logits);target[0,1,0]=.8
    bce(logits,target).sum().backward()
    assert torch.all(logits.grad[0,0]==0) and torch.all(logits.grad[0,1]!=0) and torch.all(logits.grad[0,2]!=0)
    results['positive_override_all_classes']=True
    # Missing metadata fails closed.
    batch=blank_batch();batch['im_file']=['vd1_missing.jpg']
    try:SpatialIgnoreLoss(model).loss(prediction(),batch)
    except ValueError:results['missing_mask_rejected']=True
    else:raise AssertionError('Silent fallback to ordinary training')
    return results

def make_dataset(bundle,size=640):
    cfg=get_cfg()
    for key in FORBIDDEN:setattr(cfg,key,0)
    return IgnoreDataset(img_path=str(Path(bundle)/'images'),imgsz=size,batch_size=2,augment=True,
        hyp=cfg,rect=False,cache=None,single_cls=False,stride=32,pad=0,task='detect',
        data={'names':dict(enumerate(NAMES)),'nc':4},fraction=1)

def run_checks(bundle,real_model=True):
    torch.set_num_threads(4);check_version();bundle=Path(bundle).resolve()
    doc=validate_bundle(bundle);results={'bundle':str(bundle),'training_started':False,'optimizer_steps':0}
    reviewed=Path(doc['source'])
    for row in doc['images']:
        name=Path(row['file']).with_suffix('.txt').name
        assert (bundle/'labels'/name).read_bytes()==(reviewed/'labels'/name).read_bytes(), 'Reviewed labels must not be silently filtered'
        assert not row['withheld_boxes']
    results['all_100_reviewed_label_files_preserved_exactly']=True
    results['loss_checks']=check_loss_cases()
    ds=make_dataset(bundle);byfile={r['file']:r for r in doc['images']}
    for i in range(len(ds)):
        sample=ds[i];r=byfile[Path(sample['im_file']).name]
        assert sample['img'].shape==(3,640,640) and sample['ignore_mask'].shape==(640,640)
        # Independently reconstruct labels in the letterboxed frame.
        ih,iw=sample['resized_shape'];assert (ih,iw)==(640,640)
        # BaseDataset.load_image rounds its resized height UP with ceil.
        h0,w0=r['height'],r['width'];ratio=640/max(h0,w0)
        rh=int(np.ceil(h0*ratio));rw=int(np.ceil(w0*ratio));top=round((640-rh)/2-.1);left=round((640-rw)/2-.1)
        expected=[]
        for b in r['retained_boxes']:
            x,y,x2,y2=b['xyxy'];expected.append([(x+x2)/2/w0*rw+left,(y+y2)/2/h0*rh+top,(x2-x)/w0*rw,(y2-y)/h0*rh])
        assert np.max(np.abs(sample['bboxes'].numpy()*640-np.array(expected)))<.02
        raw=cv2.imdecode(np.fromfile(bundle/'ignore-masks'/Path(r['file']).with_suffix('.png'),np.uint8),0)
        expected_mask=cv2.resize(raw,(rw,rh),interpolation=cv2.INTER_NEAREST)>0
        expected_mask=np.pad(expected_mask,((top,640-rh-top),(left,640-rw-left)))
        assert np.array_equal(expected_mask,sample['ignore_mask'].numpy())
    results['all_100_label_and_mask_transforms_aligned_at_640']=True
    batch=ds.collate_fn([ds[0],ds[26]]);batch['img']=batch['img'].float()/255
    assert batch['ignore_mask'].shape==(2,640,640)
    pred={b:prediction(640,2) for b in ['one2many','one2one']};crit=E2ELoss(DummyModel(),SpatialIgnoreLoss)
    value,_=crit(pred,batch);assert torch.isfinite(value).all();value.sum().backward()
    for branch in pred:
        ignored=getattr(crit,branch).bce.last_effective_ignore;g=pred[branch]['scores'].grad.permute(0,2,1)
        assert ignored.any() and torch.all(g[ignored]==0) and g[~ignored].abs().sum()>0
    results['real_labels_collated_two_images_two_branches']=True
    # Reject geometric augmentations instead of accidentally misaligning masks.
    for key in ['mosaic','fliplr','translate','multi_scale']:
        cfg=get_cfg()
        for field in FORBIDDEN:setattr(cfg,field,0)
        setattr(cfg,key,1)
        try:ds.build_transforms(cfg)
        except ValueError:pass
        else:raise AssertionError(key)
    results['unsupported_augmentations_rejected']=True
    # Exercise the trainer's actual build/preprocess/callback wiring without train().
    trainer=SpatialPreservedHeadTrainer.__new__(SpatialPreservedHeadTrainer)
    trainer.args=get_cfg()
    for field in FORBIDDEN:setattr(trainer.args,field,0)
    trainer.args.imgsz=320;trainer.args.rect=False;trainer.device=torch.device('cpu')
    trainer.model=DummyModel();trainer.data={'spatial_ignore':True,'names':dict(enumerate(NAMES)),'nc':4}
    integrated=trainer.build_dataset(str(bundle/'images'),'train',2)
    mb=trainer.preprocess_batch(integrated.collate_fn([integrated[0],integrated[1]]))
    enable_spatial_ignore(trainer)
    assert isinstance(trainer.model.criterion.one2many,SpatialIgnoreLoss) and mb['ignore_mask'].shape==(2,320,320)
    assert mb['img'].max()<=1 and mb['ignore_mask'].dtype==torch.bool
    results['trainer_loader_preprocess_callback_integrated']=True
    if real_model:
        sys.path.insert(0,str(ROOT/'ung-dung/railway-web'))
        from ultralytics import YOLO
        mp=ROOT/'ung-dung/railway-web/models/yolo26m-that-ai.pt';before=sha(mp)
        model=YOLO(str(mp)).model.float().eval();model.args=get_cfg()
        state={k:t.detach().clone() for k,t in model.state_dict().items()}
        small=make_dataset(bundle,320);b=small.collate_fn([small[0]]);b['img']=b['img'].float()/255
        with torch.no_grad():raw=model(b['img'])
        raw=raw[1] if isinstance(raw,tuple) else raw
        pred={branch:{k:([t.detach() for t in v] if isinstance(v,list) else v.detach().clone().requires_grad_(k in ['boxes','scores'])) for k,v in raw[branch].items()} for branch in ['one2many','one2one']}
        crit=E2ELoss(model,SpatialIgnoreLoss);value,_=crit(pred,b);assert torch.isfinite(value).all();value.sum().backward()
        for branch in pred:
            ig=getattr(crit,branch).bce.last_effective_ignore;g=pred[branch]['scores'].grad.permute(0,2,1)
            assert ig.any() and torch.all(g[ig]==0) and g[~ig].abs().sum()>0
        assert sha(mp)==before and all(torch.equal(t,state[k]) for k,t in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
        results['actual_finetuned_yolo26_output_gradient_check_320']=True
        results['model_parameters_buffers_and_checkpoint_unchanged']=True
        results['checkpoint_sha256']=before
    results['passed']=True
    report=ROOT/'ket-qua/reports/vd1-ignore-20261028';report.mkdir(parents=True,exist_ok=True)
    (report/'code-validation.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2));return results

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,default=ROOT/'data/prepared/vd1-fullframe-ignore-20261028')
    run_checks(p.parse_args().bundle)
