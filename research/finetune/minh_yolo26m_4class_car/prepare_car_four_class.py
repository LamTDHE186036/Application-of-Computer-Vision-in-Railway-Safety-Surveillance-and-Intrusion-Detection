"""Prepare independent four-class labels and class-ignore masks; preserve sources."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import argparse,json,shutil
import cv2,numpy as np,yaml
from training_utils import ROOT,sha,write
DATA=ROOT/'data/prepared/finetune-five-class-train-20261110-reviewed'
from railway_five_class import raw_masks,validate_contract

DEST=Path(__file__).resolve().parent/'runs/car-four-class'
NAMES=['person','two_wheeler','car','train']
MAP={0:0,1:1,2:2,3:2,4:3}


def convert(rows,dest,training_contract=False):
    dest.mkdir(parents=True,exist_ok=True);result=[];counts={}
    for row in rows:
        old=deepcopy(row);split=old['split'] if training_contract else 'test'
        image=dest/split/'images'/Path(old['image']).name
        label=dest/split/'labels'/Path(old['label']).name
        image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        if not image.exists():shutil.copy2(old['image'],image)
        assert sha(image)==old['image_sha256']
        lines=[]
        for line in Path(old['label']).read_text(encoding='utf8').splitlines():
            if not line.strip():continue
            fields=line.split();fields[0]=str(MAP[int(fields[0])]);lines.append(' '.join(fields))
        label.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf8')
        new=deepcopy(old);new.update(image=str(image.resolve()),label=str(label.resolve()),
            image_sha256=sha(image),label_sha256=sha(label),origin_image=old['image'],origin_label=old['label'])
        if training_contract:
            coverage=old['coverage'];new['coverage']=[coverage[0],coverage[1],max(coverage[2:4]),coverage[4]]
            im=cv2.imdecode(np.fromfile(old['image'],np.uint8),1);assert im is not None
            masks=raw_masks(old,im.shape[:2])
            grouped=np.stack((masks[0],masks[1],masks[2]|masks[3],masks[4]))
            if coverage[2]!=coverage[3]:grouped[2]=True
            new['masks']=[]
            if grouped.any():
                bits=sum(grouped[i].astype(np.uint8)*(1<<i) for i in range(4))
                path=dest/'ignore-masks'/(image.stem+'.png');path.parent.mkdir(exist_ok=True)
                ok,encoded=cv2.imencode('.png',bits);assert ok;encoded.tofile(str(path))
                new['masks']=[dict(path=str(path.resolve()),sha256=sha(path),kind='classes',applies_to_classes=list(range(4)))]
        result.append(new)
        counts.setdefault(split,Counter()).update(int(l.split()[0]) for l in lines)
        # Every source box survives, with exactly the same coordinate strings.
        source_lines=[l.split() for l in Path(old['label']).read_text(encoding='utf8').splitlines() if l.strip()]
        assert len(source_lines)==len(lines)
        assert all(a[1:]==b.split()[1:] for a,b in zip(source_lines,lines))
        assert sha(old['image'])==old['image_sha256'] and sha(old['label'])==old['label_sha256']
    write(dest/'manifest.json',result)
    for split in counts:
        (dest/(split+'.txt')).write_text('\n'.join(r['image'] for r in result if (r['split']==split if training_contract else split=='test'))+'\n',encoding='utf8')
    config=dict(path=str(dest.resolve()),train='train.txt' if training_contract else 'test.txt',val='val.txt' if training_contract else 'test.txt',
                test='test.txt',nc=4,names=NAMES)
    if training_contract:
        contract=dest/'annotation-contract.json'
        write(contract,dict(schema='railway-four-car-v1',names=NAMES,entries=result,
             old_to_new=MAP,merge_ignore='union; partially annotated merged group ignores negatives everywhere; retained positives override ignore masks'))
        config['four_class_annotation_contract']=str(contract.resolve())
        config['training_authorized']=False
    (dest/'data.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf8')
    summary=dict(images_by_split=dict(Counter(r['split'] if training_contract else 'test' for r in result)),
                 labels_by_split={s:{NAMES[c]:n for c,n in v.items()} for s,v in counts.items()},
                 source_images_and_labels_unchanged=True,coordinates_and_box_count_preserved=True,
                 old_to_new=MAP,names=NAMES,training=False)
    write(dest/'conversion.json',summary)
    return result,summary


def main():
    global DATA,DEST
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--road-test',type=Path,help='Optional reviewed road benchmark')
    args=parser.parse_args();DATA=args.data_root.resolve();DEST=args.output.resolve()
    if DATA==DEST or DATA in DEST.parents or DEST in DATA.parents:
        raise ValueError('Output and source dataset directories must be separate')
    config=yaml.safe_load((DATA/'data.yaml').read_text(encoding='utf8'))
    validate_contract(config)
    rows=json.loads((DATA/'manifest.json').read_text(encoding='utf8'))
    _,summary=convert(rows,DEST,True)
    if args.road_test:
        road=json.loads((args.road_test.resolve()/'manifest.json').read_text(encoding='utf8'))
        convert(road,DEST/'road405',False)
    (DEST/'README.md').write_text('# Bốn lớp: person, two_wheeler, car, train\n\n'
       'Gộp large_vehicle và car thành car; giữ tàu riêng. Ảnh được sao chép, mọi box giữ nguyên tọa độ và số lượng. '
       'Nguồn, split và dữ liệu gốc không thay đổi. Class-ignore masks được gộp bằng phép hợp cho car.\n\n'
       'ID: 0 person; 1 two_wheeler; 2 car; 3 train. `car` bao gồm ô tô con, xe lớn, bus/truck/pickup/van theo quy ước cũ. '
       'road405/ là benchmark ba lớp người/xe; không chấm train vì nhãn tàu chưa đầy đủ. '
       '50 ảnh test trong bộ chính chỉ dùng chấm tàu riêng.\n\n'
       'Đây là bản chuyển nhãn, chưa train lại. Khi train, phải dùng loss/loader hiểu contract 4 lớp, coverage và masks; '
       'YOLO loader mặc định không tự xử lý các vùng chưa gán đủ nhãn.\n',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
