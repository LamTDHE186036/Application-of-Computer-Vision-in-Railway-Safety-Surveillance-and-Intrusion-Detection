"""Load the contributed model and optionally predict one local image."""
from pathlib import Path
import argparse
import hashlib
import numpy as np
import ultralytics
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
MODEL = HERE / 'models/yolo26m-that-ai.pt'
EXPECTED_SHA256 = 'abe022e9d2b27f39f45507cdd289e7e152853a0255d776a3ad1b9c9aeed051de'
NAMES = ['person', 'two_wheeler', 'large_vehicle', 'car']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if ultralytics.__version__ != '8.4.161':
        raise RuntimeError('Use the isolated requirements.txt environment, ultralytics 8.4.161')
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError('Checkpoint checksum mismatch')
    if args.image and not args.image.is_file():
        raise FileNotFoundError(args.image)
    model = YOLO(str(MODEL))
    if list(model.names.values()) != NAMES:
        raise RuntimeError('Unexpected class order')
    image = str(args.image) if args.image else np.zeros((640, 640, 3), dtype=np.uint8)
    predictions = model.predict(image, imgsz=640, conf=0.35, device=args.device, verbose=False, save=False)
    print(f'Model loaded, classes={NAMES}, predicted_boxes={len(predictions[0].boxes)}')


if __name__ == '__main__':
    main()
