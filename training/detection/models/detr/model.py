from detection.engine.model import Model
from detection.models import detr
from detection.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
)
from detection.utils import ROOT, yaml_load


class DETECT(Model):

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": detr.classify.ClassificationTrainer,
                "validator": detr.classify.ClassificationValidator,
                "predictor": detr.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": detr.detect.DetectionTrainer,
                "validator": detr.detect.DetectionValidator,
                "predictor": detr.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": detr.segment.SegmentationTrainer,
                "validator": detr.segment.SegmentationValidator,
                "predictor": detr.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": detr.pose.PoseTrainer,
                "validator": detr.pose.PoseValidator,
                "predictor": detr.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": detr.obb.OBBTrainer,
                "validator": detr.obb.OBBValidator,
                "predictor": detr.obb.OBBPredictor,
            },
        }

