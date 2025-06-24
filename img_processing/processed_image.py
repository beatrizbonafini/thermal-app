from typing import Optional, Dict
from PIL import Image, ExifTags
import numpy as np
import control
import io
import matplotlib.pyplot as plt
from sqlalchemy import Column, Integer, String, LargeBinary, JSON
from sqlalchemy.ext.declarative import declarative_base
import json
from fractions import Fraction

Base = declarative_base()

class ProcessedImage(Base):
    
    __tablename__ = 'processed_image'

    id = Column(Integer, primary_key=True)
    animal_id = Column(String)
    original_image_data = Column(LargeBinary)
    optical_image_data = Column(LargeBinary, nullable=True)
    thermal_matrix_json = Column(JSON, nullable=True)
    grayscale_image_data = Column(LargeBinary, nullable=True)
    masks_json = Column(JSON, nullable=True)
    classes_json = Column(JSON, nullable=True)
    boxes_json = Column(JSON, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    
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
        
        self.animal_id = animal_id
        
        self.metadata = metadata or {}
        self.classes = classes or {}
        self.boxes = boxes or {}
        self.masks = masks or {}
        
        self.original_image = original_image
        self.thermal_matrix = thermal_matrix
        self.optical_image = optical_image
        self.grayscale = grayscale

        self.original_image_data = self.image_to_bytes(original_image)
        self.optical_image_data = self.image_to_bytes(optical_image) 
        self.thermal_matrix_json = self.ndarray_to_json(thermal_matrix) 
        self.grayscale_image_data = self.image_to_bytes(grayscale) 
        
        self.masks_json = masks
        self.classes_json = classes
        self.boxes_json = boxes
        self.metadata_json = self.make_json_serializable(metadata)

    @staticmethod
    def image_to_bytes(image_input) -> bytes:
        if image_input is None:
            return None

        if isinstance(image_input, np.ndarray):
            # Converte para uint8 se não for
            if image_input.dtype != np.uint8:
                image_input = np.clip(image_input, 0, 255).astype(np.uint8)
            image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("image_input deve ser PIL.Image.Image ou np.ndarray")

        buf = io.BytesIO()
        image.save(buf, format='PNG')
        return buf.getvalue()
    
    @staticmethod
    def ndarray_to_json(array: np.ndarray) -> str:
        return json.dumps(array.tolist())
    
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
    
    @staticmethod
    def make_json_serializable(obj):

        if isinstance(obj, dict):
            return {k: ProcessedImage.make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ProcessedImage.make_json_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(ProcessedImage.make_json_serializable(v) for v in obj)
        elif hasattr(obj, '__float__'):
            return float(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, bytes):
            return obj.decode(errors='ignore')
        elif isinstance(obj, Fraction):
            return float(obj)
        else:
            try:
                json.dumps(obj)
                return obj
            except Exception:
                return str(obj)
    

    
        
