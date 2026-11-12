"""Local utilities for training/export scripts; not needed by the published model."""
from pathlib import Path
import hashlib
import json
import sys
import types

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    temporary.replace(path)


def verify_protected(locks):
    for path,digest in locks.items():
        if sha(path)!=digest:
            raise ValueError(f'Source changed: {path}')


def install_legacy_aliases():
    # The previously published five-class package used this pickle module path.
    from railway import five_class
    for name in ('models','models.minh_yolo26m_5class'):
        if name not in sys.modules:
            module=types.ModuleType(name);module.__path__=[]
            sys.modules[name]=module
    sys.modules.setdefault('models.minh_yolo26m_5class.layers',five_class)
