from ultralytics import YOLO
from typing import Any, List, Dict, Optional
import torch
import numpy as np
import os

class DetectionModel:
    """Core detection model class for object detection using YOLOv8"""

    # Model information dictionary
    MODEL_INFO = {
        "yolov8n.pt": {
            "name": "YOLOv8n (Nano)",
            "description": "Fastest model with smallest size (3.2M parameters). Best for speed-critical applications.",
            "size_mb": 6,
            "inference_speed": "Very Fast"
        },
        "yolov8m.pt": {
            "name": "YOLOv8m (Medium)",
            "description": "Balanced model with good accuracy-speed tradeoff (25.9M parameters). Recommended for general use.",
            "size_mb": 25,
            "inference_speed": "Medium"
        },
        "yolov8x.pt": {
            "name": "YOLOv8x (XLarge)",
            "description": "Most accurate but slower model (68.2M parameters). Best for accuracy-critical applications.",
            "size_mb": 68,
            "inference_speed": "Slower"
        }
    }

    def __init__(self, model_name: str = 'yolov8m.pt', confidence: float = 0.25, iou: float = 0.25):
        """
        Initialize the detection model

        Args:
            model_name: Model name or path, default is yolov8m.pt
            confidence: Confidence threshold, default is 0.25
            iou: IoU threshold for non-maximum suppression, default is 0.45
        """
        self.model_name = model_name
        self.confidence = confidence
        self.iou = iou
        self.model = None
        self.class_names = {}
        self.is_model_loaded = False

        # Load model on initialization
        self._load_model()

    def _load_model(self):
        """Load the YOLO model"""
        try:
            print(f"Loading model: {self.model_name}")
            self.model = YOLO(self.model_name)
            self.class_names = self.model.names
            self.is_model_loaded = True
            print(f"Successfully loaded model: {self.model_name}")
            print(f"Number of classes the model can recognize: {len(self.class_names)}")
        except Exception as e:
            print(f"Error occurred when loading the model: {e}")
            self.is_model_loaded = False

    def change_model(self, new_model_name: str) -> bool:
        """
        Change the currently loaded model

        Args:
            new_model_name: Name of the new model to load

        Returns:
            bool: True if model changed successfully, False otherwise
        """
        if self.model_name == new_model_name and self.is_model_loaded:
            print(f"Model {new_model_name} is already loaded")
            return True

        print(f"Changing model from {self.model_name} to {new_model_name}")

        # Unload current model to free memory
        if self.model is not None:
            del self.model
            self.model = None

            # Clean GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Update model name and load new model
        self.model_name = new_model_name
        self._load_model()

        return self.is_model_loaded

    def reload_model(self):
        """Reload the model (useful for changing model or after error)"""
        if self.model is not None:
            del self.model
            self.model = None

            # Clean GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._load_model()

    def detect(self, image_input: Any) -> Optional[Any]:
        """
        Perform object detection on a single image

        Args:
            image_input: Image path (str), PIL Image, or numpy array

        Returns:
            Detection result object or None if error occurred
        """
        if self.model is None or not self.is_model_loaded:
            print("Model not found or not loaded. Attempting to reload...")
            self._load_model()
            if self.model is None or not self.is_model_loaded:
                print("Failed to load model. Cannot perform detection.")
                return None

        try:
            results = self.model(image_input, conf=self.confidence, iou=self.iou)
            return results[0]
        except Exception as e:
            print(f"Error occurred during detection: {e}")
            return None

    def get_class_names(self, class_id: int) -> str:
        """Get class name for a given class ID"""
        return self.class_names.get(class_id, "Unknown Class")

    def get_supported_classes(self) -> Dict[int, str]:
        """Get all supported classes as a dictionary of {id: class_name}"""
        return self.class_names

    @classmethod
    def get_available_models(cls) -> List[Dict]:
        """
        Get list of available models with their information

        Returns:
            List of dictionaries containing model information
        """
        models = []
        for model_file, info in cls.MODEL_INFO.items():
            models.append({
                "model_file": model_file,
                "name": info["name"],
                "description": info["description"],
                "size_mb": info["size_mb"],
                "inference_speed": info["inference_speed"]
            })
        return models

    @classmethod
    def get_model_description(cls, model_name: str) -> str:
        """Get description for a specific model"""
        if model_name in cls.MODEL_INFO:
            info = cls.MODEL_INFO[model_name]
            return f"{info['name']}: {info['description']} (Size: ~{info['size_mb']}MB, Speed: {info['inference_speed']})"
        return "Model information not available"
