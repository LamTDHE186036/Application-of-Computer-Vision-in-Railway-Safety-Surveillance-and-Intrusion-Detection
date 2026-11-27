"""Project image metric functions from evaluate_merged_vehicle.py and danh_gia.py."""
import types
import numpy as np
import torch
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import DetMetrics, box_iou

def overlaps(a,bs):
    a=np.asarray(a);bs=np.asarray(bs).reshape(-1,4)
    inter=np.maximum(0,np.minimum(a[2:],bs[:,2:])-np.maximum(a[:2],bs[:,:2])).prod(1)
    return inter/np.maximum(1e-9,(a[2:]-a[:2]).prod()+(bs[:,2:]-bs[:,:2]).prod(1)-inter)

def fixed(rows,names,threshold=.5):
    totals={n:dict(tp=0,fp=0,fn=0,gt=0) for n in names};audits=[]
    for row in rows:
        records=[]
        for c,n in enumerate(names):
            gs=[(j,g['xyxy']) for j,g in enumerate(row['ground_truth']) if g['cls']==c]
            ps=sorted([p for p in row['predictions'] if p['cls']==c and p['conf']>=threshold],key=lambda p:-p['conf'])
            used=set();tp=0
            for p in ps:
                ov=overlaps(p['xyxy'],[b for _,b in gs]);available=[j for j,v in enumerate(ov) if v>=.5 and j not in used]
                j=max(available,key=lambda j:ov[j]) if available else None
                if j is not None:used.add(j);tp+=1
                records.append(dict(p,gt_index=gs[j][0] if j is not None else None,matched=j is not None))
            for k,v in dict(tp=tp,fp=len(ps)-tp,fn=len(gs)-tp,gt=len(gs)).items():totals[n][k]+=v
        audits.append(dict(file=row['file'],predictions=records))
    totals['micro']={k:sum(v[k] for v in totals.values()) for k in ['tp','fp','fn','gt']}
    for v in totals.values():
        p=v['tp']/max(1,v['tp']+v['fp']);r=v['tp']/max(1,v['gt']);v.update(precision=p,recall=r,f1=2*p*r/max(1e-12,p+r))
    return totals,audits

def ap(rows,names):
    metrics=DetMetrics(names=dict(enumerate(names)))
    matcher=types.SimpleNamespace(iouv=torch.linspace(.5,.95,10))
    for r in rows:
        pb=torch.tensor([p['ap_xyxy'] for p in r['predictions']],dtype=torch.float32).reshape(-1,4)
        gb=torch.tensor([g['ap_xyxy'] for g in r['ground_truth']],dtype=torch.float32).reshape(-1,4)
        pc=torch.tensor([p['cls'] for p in r['predictions']],dtype=torch.float32)
        gc=torch.tensor([g['cls'] for g in r['ground_truth']],dtype=torch.float32)
        correct=BaseValidator.match_predictions(matcher,pc,gc,box_iou(gb,pb)).numpy()
        metrics.update_stats(dict(tp=correct,conf=np.array([p['conf'] for p in r['predictions']],dtype=np.float32),pred_cls=pc.numpy(),target_cls=gc.numpy(),target_img=np.unique(gc.numpy()),im_name=r['file']))
    metrics.process(plot=False)
    return dict(map50=float(metrics.box.map50),map50_95=float(metrics.box.map),per_class={names[int(c)]:dict(ap50=float(metrics.box.ap50[i]),ap50_95=float(metrics.box.ap[i])) for i,c in enumerate(metrics.box.ap_class_index)})
