import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

class InstanceSegmentationPredictor:

    def __init__(self, 
                 model_weights_path: str, 
                 class_names: list) :
        
        setup_logger()

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(cfg)

        MetadataCatalog.get("mice_metadata").thing_classes = class_names
        self.mice_metadata = MetadataCatalog.get("mice_metadata")


    def predict(self, file_image):
        try:

            gray_np = np.array(file_image.convert("L"))
            image_bgr = cv2.cvtColor(gray_np, cv2.COLOR_GRAY2BGR)

            output = self.predictor(image_bgr)
            instances = output["instances"].to("cpu")
            
            info = {
                'boxes': instances.pred_boxes.tensor.numpy().tolist(),
                'classes': instances.pred_classes.numpy().tolist(),
                'masks': instances.pred_masks.numpy().tolist(),
            }
            return info
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None