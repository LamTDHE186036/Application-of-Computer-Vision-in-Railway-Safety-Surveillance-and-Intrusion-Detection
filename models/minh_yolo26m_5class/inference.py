"""Load verified weights and expose explicit label mappings for integration."""
from pathlib import Path
import hashlib
import json

NATIVE_TO_LAM_ID = {0: 2, 1: 1, 2: 3, 3: 0, 4: 4}
NATIVE_TO_LAM_LABEL = {
    'person': 'person', 'two_wheeler': 'motorcycle',
    'large_vehicle': 'truck-bus', 'car': 'car', 'train': 'train',
}


def to_lam_class_id(class_id):
    return NATIVE_TO_LAM_ID[int(class_id)]


def to_lam_label(label):
    return NATIVE_TO_LAM_LABEL[label]


def load_model(device=None):
    from ultralytics import YOLO
    import ultralytics

    folder = Path(__file__).resolve().parent
    metadata = json.loads((folder / 'model.json').read_text(encoding='utf-8'))
    if ultralytics.__version__ != metadata['ultralytics']:
        raise RuntimeError(f"Requires ultralytics=={metadata['ultralytics']}; found {ultralytics.__version__}")
    weights = folder / metadata['file']
    if hashlib.sha256(weights.read_bytes()).hexdigest() != metadata['sha256']:
        raise ValueError('Checkpoint SHA-256 mismatch')
    model = YOLO(str(weights))
    if list(model.names.values()) != metadata['names']:
        raise ValueError('Unexpected model label order')
    model.overrides.update(imgsz=640, conf=0.50, nms=False, quantize=32)
    if device is not None:
        model.overrides['device'] = device
    return model
