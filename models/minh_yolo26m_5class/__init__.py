"""Five-class YOLO26m inference package."""
from .inference import load_model, to_lam_class_id, to_lam_label

__all__ = ['load_model', 'to_lam_class_id', 'to_lam_label']
