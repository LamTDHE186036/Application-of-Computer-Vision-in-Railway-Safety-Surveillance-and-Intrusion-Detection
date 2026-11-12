"""Repackage learned grouped YOLO outputs using only built-in PyTorch modules.

No training or deployment. Exports a five-class equivalence control and a
four-class model whose car class is max(car, large_vehicle) BEFORE top-k.
"""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json,os,sys,subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
os.environ.setdefault('YOLO_AUTOINSTALL','false')
os.environ.setdefault('YOLO_CONFIG_DIR',str(HERE/'.settings'))
import numpy as np
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from training_utils import sha,write,verify_protected,install_legacy_aliases
install_legacy_aliases()

OUT=HERE/'runs/portable-car'
SOURCE=HERE/'inputs/yolo26m-5class-finetuned.pt'
DATA=ROOT/'data/prepared/finetune-five-class-train-20261110-reviewed'
NAMES5=['person','two_wheeler','large_vehicle','car','train']
NAMES4=['person','two_wheeler','car','train']


def export(source,names,selection,width,path):
    model=deepcopy(source.model).float().cpu().eval()
    original={k:v.clone() for k,v in model.state_dict().items()}
    replaced=[]
    for branch in ('cv3','one2one_cv3'):
        for i,scale in enumerate(getattr(model.model[-1],branch)):
            old=scale[-1];assert old.classifier.out_channels==7
            selector=nn.Conv2d(7,len(selection),1,bias=False)
            with torch.no_grad():
                selector.weight.zero_()
                for row,column in enumerate(selection):selector.weight[row,column,0,0]=1
            selector.requires_grad_(False)
            block=nn.Sequential(deepcopy(old.classifier),selector,
                                nn.Unflatten(1,(len(names),width)),
                                nn.MaxPool3d((width,1,1)),nn.Flatten(1,2))
            block.out_channels=len(names)
            scale[-1]=block;replaced.append(f'model.{len(model.model)-1}.{branch}.{i}.{len(scale)-1}')
    model.__class__=DetectionModel;model.criterion=None
    head=model.model[-1];head.nc=len(names);head.no=len(names)+4*head.reg_max
    model.nc=len(names);model.names=dict(enumerate(names));model.yaml['nc']=len(names)
    model.args=dict(model.args) if isinstance(model.args,dict) else vars(model.args).copy()
    model.args.update(nms=False,task='detect')
    state=model.state_dict()
    for key,tensor in original.items():
        destination=key
        for prefix in replaced:
            if key.startswith(prefix+'.classifier.'):
                destination=key.replace(prefix+'.classifier.',prefix+'.0.',1);break
        assert torch.equal(tensor,state[destination]),key
    assert all(type(m).__module__.startswith(('torch.','ultralytics.')) for m in model.modules())
    wrapper=deepcopy(source);wrapper.model=model;wrapper.save(str(path))
    loaded=YOLO(str(path)).model.float().eval()
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in state.items())
    return loaded,dict(original_state_tensors_preserved=len(original),extra_fixed_selector_tensors=len(replaced),
                       learned_weights_changed=False,optimizer_steps=0,names=names,sha256=sha(path))


def raw_check(old,new,merged):
    result=[];torch.manual_seed(20261112)
    import cv2
    from railway_five_class import NAMES
    manifest=DATA/'manifest.json'
    paths=json.loads(manifest.read_text(encoding='utf8')) if manifest.is_file() else []
    real=[r['image'] for r in paths if r['split']=='val'][:2]
    inputs=[('random',torch.rand(1,3,160,160))]
    for path in real:
        im=cv2.imdecode(np.fromfile(path,np.uint8),1)
        im=cv2.resize(im,(320,320))[:,:,::-1].copy()
        inputs.append((Path(path).name,torch.from_numpy(im).permute(2,0,1).unsqueeze(0).float()/255))
    with torch.inference_mode():
        for title,x in inputs:
            expected=old(x)[1];control=new(x)[1];actual=merged(x)[1]
            for branch in ('one2many','one2one'):
                a=expected[branch];b=control[branch];c=actual[branch]
                groups=torch.stack((a['scores'][:,0],a['scores'][:,1],
                                    torch.maximum(a['scores'][:,2],a['scores'][:,3]),a['scores'][:,4]),1)
                errors=dict(control_boxes=float((a['boxes']-b['boxes']).abs().max()),
                            control_logits=float((a['scores']-b['scores']).abs().max()),
                            merged_boxes=float((a['boxes']-c['boxes']).abs().max()),
                            merged_logits=float((groups-c['scores']).abs().max()))
                assert max(errors.values())<1e-6,errors
                result.append(dict(input=title,branch=branch,errors=errors))
    return result


def isolated(path,names):
    code="""import sys,json,numpy as np,torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
m=YOLO(sys.argv[1]); names=json.loads(sys.argv[2]);assert list(m.names.values())==names
assert type(m.model) is DetectionModel
assert all(type(x).__module__.startswith(('torch.','ultralytics.')) for x in m.model.modules())
r=m.predict(np.zeros((160,160,3),np.uint8),imgsz=160,device='cpu',verbose=False)
assert len(r)==1
assert not any(k in sys.modules for k in ('railway.five_class','railway_five_class','layers','inference'))
print(json.dumps(dict(passed=True,independent_python=True,names=names)))
"""
    r=subprocess.run([sys.executable,'-I','-X','utf8','-B','-c',code,str(path),json.dumps(names)],
                     cwd=path.parent,text=True,encoding='utf8',capture_output=True,check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    global SOURCE, OUT, DATA
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='Original grouped five-class checkpoint')
    parser.add_argument('--output',type=Path,required=True,help='New export directory')
    parser.add_argument('--data-root',type=Path,help='Optional five-class dataset for additional real-image checks')
    parser.add_argument('--expected-sha256',help='Optional expected source SHA-256')
    args=parser.parse_args()
    SOURCE=args.source.resolve();OUT=args.output.resolve()
    if args.data_root:DATA=args.data_root.resolve()
    if args.expected_sha256:assert sha(SOURCE)==args.expected_sha256
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    artifact=OUT/'models';artifact.mkdir(exist_ok=True)
    locks={str(SOURCE):sha(SOURCE)};write(OUT/'protected.json',locks)
    source=YOLO(str(SOURCE));assert list(source.names.values())==NAMES5
    control_path=artifact/'yolo26m-5class-portable.pt';merged_path=artifact/'yolo26m-4class-car.pt'
    control,a=export(source,NAMES5,[0,0,1,3,4,5,2,2,6,6],2,control_path)
    merged,b=export(source,NAMES4,[0,0,0,1,3,3,2,4,5,6,6,6],3,merged_path)
    old=source.model.float().cpu().eval()
    raw=raw_check(old,control,merged)
    a['isolated_load']=isolated(control_path,NAMES5)
    b['isolated_load']=isolated(merged_path,NAMES4)
    verify_protected(locks)
    write(OUT/'export-verification.json',dict(control=a,merged=b,raw_equivalence=raw,
          source_sha256=sha(SOURCE),web_unchanged=True,training=False,
          meaning='Built-in-module representation of the existing learned grouping, not a newly trained stock classifier head.',
          merging='max(car,large_vehicle) before native top-k; no additional NMS; train remains separate.'))
    print('EXPORTED',merged_path,flush=True)


if __name__=='__main__':main()
