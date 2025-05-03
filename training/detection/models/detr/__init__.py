# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from detection.models.detr import classify, detect, obb, pose, segment, world

from .model import DETECT

__all__ = "DETECT"
