from typing import Optional, Union, Dict
from PIL import Image, ExifTags
import numpy as np
import cv2
import flyr
import tempfile
import shutil
import control
import matplotlib.pyplot as plt

class ProcessedImage:
    
    def __init__(self,
             original_image: Image.Image,
             animal_id: str,
             optical_image: Optional[Image.Image] = None,
             thermal_matrix: Optional[np.ndarray] = None,
             grayscale: Optional[Image.Image] = None,
             masks: Optional[Dict] = None,
             classes: Optional[Dict] = None,
             boxes: Optional[Dict] = None,
             metadata: Optional[Dict] = None):
        
        self.original_image = original_image
        self.animal_id = animal_id
        
        self.metadata = metadata or {}
        self.classes = classes or {}
        self.boxes = boxes or {}
        self.masks = masks or {}
        
        self.thermal_matrix = thermal_matrix
        self.optical_image = optical_image
        self.grayscale = grayscale

    def get_image_bboxes(self, class_id: int = 0):   
    
        bbox = control.crop_boxes_from_image(self.thermal_matrix, self.boxes[class_id])       
        fig, ax = plt.subplots()
        cax = ax.imshow(bbox, cmap='inferno')
        fig.colorbar(cax)
        return control.matplotlib_figure_to_image(fig)

    def get_metadata(self):

        formated_metadata = {}

        if self.metadata is not None:
            for tag_id, value in self.metadata.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag == 'MakerNote' or tag == 'ComponentsConfiguration':
                    continue
                formated_metadata[tag] = value

        return formated_metadata
    
    def get_thermal_stats(self):
        
        if self.thermal_matrix is None:
            return None
        
        stats = control.thermal_stats(self.thermal_matrix, self.metadata)
        return stats

    

    
        
