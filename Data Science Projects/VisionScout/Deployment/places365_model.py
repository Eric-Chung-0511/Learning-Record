import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import logging

class Places365Model:
    """
    Places365 scene classification model wrapper for scene understanding integration.
    Provides scene classification and scene attribute prediction capabilities.
    """

    def __init__(self, model_name: str = 'resnet50_places365', device: Optional[str] = None):
        """
        Initialize Places365 model with configurable architecture and device.

        Args:
            model_name: Model architecture name (默認 resnet50)
            device: Target device for inference (auto-detected if None)
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Device configuration with fallback logic
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.model = None
        self.scene_classes = []
        self.scene_attributes = []

        # Model configuration mapping
        self.model_configs = {
            'resnet18_places365': {
                'arch': 'resnet18',
                'num_classes': 365,
                'url': 'http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar'
            },
            'resnet50_places365': {
                'arch': 'resnet50',
                'num_classes': 365,
                'url': 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
            },
            'densenet161_places365': {
                'arch': 'densenet161',
                'num_classes': 365,
                'url': 'http://places2.csail.mit.edu/models_places365/densenet161_places365.pth.tar'
            }
        }

        self._load_model()
        self._load_class_names()
        self._setup_scene_mapping()

    def _load_model(self):
        """載入與初始化 Places365 model"""
        try:
            if self.model_name not in self.model_configs:
                raise ValueError(f"Unsupported model name: {self.model_name}")

            config = self.model_configs[self.model_name]

            # Import model architecture
            if config['arch'].startswith('resnet'):
                import torchvision.models as models
                if config['arch'] == 'resnet18':
                    self.model = models.resnet18(num_classes=config['num_classes'])
                elif config['arch'] == 'resnet50':
                    self.model = models.resnet50(num_classes=config['num_classes'])
            elif config['arch'] == 'densenet161':
                import torchvision.models as models
                self.model = models.densenet161(num_classes=config['num_classes'])

            # Load pretrained weights
            checkpoint = torch.hub.load_state_dict_from_url(
                config['url'],
                map_location=self.device,
                progress=True
            )

            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Remove 'module.' prefix if present
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.logger.info(f"Places365 model {self.model_name} loaded successfully on {self.device}")

        except Exception as e:
            self.logger.error(f"Error loading Places365 model: {str(e)}")
            raise

    def _load_class_names(self):
        """Load Places365 class names and scene attributes."""
        try:
            # Load scene class names (365 categories)
            import urllib.request

            class_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            class_file = urllib.request.urlopen(class_url)

            self.scene_classes = []
            for line in class_file:
                class_name = line.decode('utf-8').strip().split(' ')[0][3:]  # Remove /x/ prefix
                self.scene_classes.append(class_name)

            # Load scene attributes (optional, for enhanced description)
            attr_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
            try:
                attr_file = urllib.request.urlopen(attr_url)
                self.scene_attributes = []
                for line in attr_file:
                    attr_name = line.decode('utf-8').strip()
                    self.scene_attributes.append(attr_name)
            except:
                self.logger.warning("Scene attributes not loaded, continuing with basic classification")
                self.scene_attributes = []

            self.logger.info(f"Loaded {len(self.scene_classes)} scene classes and {len(self.scene_attributes)} attributes")

        except Exception as e:
            self.logger.error(f"Error loading class names: {str(e)}")
            # Fallback to basic class names if download fails
            self.scene_classes = [f"scene_class_{i}" for i in range(365)]
            self.scene_attributes = []

    def _setup_scene_mapping(self):
        """Setup mapping from Places365 classes to common scene types."""
        # 建立Places365類別到通用場景類型的映射關係
        self.scene_type_mapping = {
            # Indoor scenes
            'living_room': 'living_room',
            'bedroom': 'bedroom',
            'kitchen': 'kitchen',
            'dining_room': 'dining_area',
            'bathroom': 'bathroom',
            'office': 'office_workspace',
            'conference_room': 'office_workspace',
            'classroom': 'educational_setting',
            'library': 'library',
            'restaurant': 'restaurant',
            'cafe': 'cafe',
            'bar': 'bar',
            'hotel_room': 'hotel_room',
            'hospital_room': 'medical_facility',
            'gym': 'gym',
            'supermarket': 'retail_store',
            'clothing_store': 'retail_store',

            # Outdoor urban scenes
            'street': 'city_street',
            'crosswalk': 'intersection',
            'parking_lot': 'parking_lot',
            'gas_station': 'gas_station',
            'bus_station': 'bus_stop',
            'train_station': 'train_station',
            'airport_terminal': 'airport',
            'subway_station': 'subway_station',
            'bridge': 'bridge',
            'highway': 'highway',
            'downtown': 'commercial_district',
            'shopping_mall': 'shopping_mall',

            # Natural outdoor scenes
            'park': 'park_area',
            'beach': 'beach',
            'forest': 'forest',
            'mountain': 'mountain',
            'lake': 'lake',
            'river': 'river',
            'ocean': 'ocean',
            'desert': 'desert',
            'field': 'field',
            'garden': 'garden',

            # Landmark and tourist areas
            'castle': 'historical_monument',
            'palace': 'historical_monument',
            'temple': 'temple',
            'church': 'church',
            'mosque': 'mosque',
            'museum': 'museum',
            'art_gallery': 'art_gallery',
            'tower': 'tourist_landmark',
            'monument': 'historical_monument',

            # Sports and entertainment
            'stadium': 'stadium',
            'basketball_court': 'sports_field',
            'tennis_court': 'sports_field',
            'swimming_pool': 'swimming_pool',
            'playground': 'playground',
            'amusement_park': 'amusement_park',
            'theater': 'theater',
            'concert_hall': 'concert_hall',

            # Transportation
            'airplane_cabin': 'airplane_cabin',
            'train_interior': 'train_interior',
            'car_interior': 'car_interior',

            # Construction and industrial
            'construction_site': 'construction_site',
            'factory': 'factory',
            'warehouse': 'warehouse'
        }

        # Indoor/outdoor classification helper
        self.indoor_classes = {
            'living_room', 'bedroom', 'kitchen', 'dining_room', 'bathroom', 'office',
            'conference_room', 'classroom', 'library', 'restaurant', 'cafe', 'bar',
            'hotel_room', 'hospital_room', 'gym', 'supermarket', 'clothing_store',
            'airplane_cabin', 'train_interior', 'car_interior', 'theater', 'concert_hall',
            'museum', 'art_gallery', 'shopping_mall'
        }

        self.outdoor_classes = {
            'street', 'crosswalk', 'parking_lot', 'gas_station', 'bus_station',
            'train_station', 'airport_terminal', 'bridge', 'highway', 'downtown',
            'park', 'beach', 'forest', 'mountain', 'lake', 'river', 'ocean',
            'desert', 'field', 'garden', 'stadium', 'basketball_court', 'tennis_court',
            'swimming_pool', 'playground', 'amusement_park', 'construction_site',
            'factory', 'warehouse', 'castle', 'palace', 'temple', 'church', 'mosque',
            'tower', 'monument'
        }

    def preprocess(self, image_pil: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL image for Places365 model inference.

        Args:
            image_pil: Input PIL image

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Places365 standard preprocessing
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Convert to RGB if needed
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        # Apply preprocessing
        input_tensor = transform(image_pil).unsqueeze(0)
        return input_tensor.to(self.device)

    def predict(self, image_pil: Image.Image) -> Dict[str, Any]:
        """
        Predict scene classification and attributes for input image.

        Args:
            image_pil: Input PIL image

        Returns:
            Dict containing scene predictions and confidence scores
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess(image_pil)

            # Model inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # 返回最有可能的項目
            top_k = min(10, len(self.scene_classes))  # Configurable top-k
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)

            # Extract results
            top_probs = top_probs.cpu().numpy()[0]
            top_indices = top_indices.cpu().numpy()[0]

            # Build prediction results
            predictions = []
            for i in range(top_k):
                class_idx = top_indices[i]
                confidence = float(top_probs[i])
                scene_class = self.scene_classes[class_idx]

                predictions.append({
                    'class_name': scene_class,
                    'class_index': class_idx,
                    'confidence': confidence
                })

            # Get primary prediction
            primary_prediction = predictions[0]
            primary_class = primary_prediction['class_name']

            # 確認是 indoor/outdoor
            is_indoor = self._classify_indoor_outdoor(primary_class)

            # Map to common scene type
            mapped_scene_type = self._map_places365_to_scene_types(primary_class)

            # Determine scene attributes (basic inference based on class)
            scene_attributes = self._infer_scene_attributes(primary_class)

            result = {
                'scene_label': primary_class,
                'mapped_scene_type': mapped_scene_type,
                'confidence': primary_prediction['confidence'],
                'is_indoor': is_indoor,
                'attributes': scene_attributes,
                'top_predictions': predictions,
                'all_probabilities': probabilities.cpu().numpy()[0].tolist()
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in Places365 prediction: {str(e)}")
            return {
                'scene_label': 'unknown',
                'mapped_scene_type': 'unknown',
                'confidence': 0.0,
                'is_indoor': None,
                'attributes': [],
                'top_predictions': [],
                'error': str(e)
            }

    def _classify_indoor_outdoor(self, scene_class: str) -> Optional[bool]:
        """
        Classify if scene is indoor or outdoor based on Places365 class.

        Args:
            scene_class: Places365 scene class name

        Returns:
            bool or None: True for indoor, False for outdoor, None if uncertain
        """
        if scene_class in self.indoor_classes:
            return True
        elif scene_class in self.outdoor_classes:
            return False
        else:
            # For ambiguous classes, use heuristics
            indoor_keywords = ['room', 'office', 'store', 'shop', 'hall', 'interior', 'indoor']
            outdoor_keywords = ['street', 'road', 'park', 'field', 'beach', 'mountain', 'outdoor']

            scene_lower = scene_class.lower()
            if any(keyword in scene_lower for keyword in indoor_keywords):
                return True
            elif any(keyword in scene_lower for keyword in outdoor_keywords):
                return False
            else:
                return None

    def _map_places365_to_scene_types(self, places365_class: str) -> str:
        """
        Map Places365 class to common scene type used by the system.

        Args:
            places365_class: Places365 scene class name

        Returns:
            str: Mapped scene type
        """
        # Direct mapping lookup
        if places365_class in self.scene_type_mapping:
            return self.scene_type_mapping[places365_class]

        # Fuzzy matching for similar classes
        places365_lower = places365_class.lower()

        # Indoor fuzzy matching
        if any(keyword in places365_lower for keyword in ['living', 'bedroom', 'kitchen']):
            return 'general_indoor_space'
        elif any(keyword in places365_lower for keyword in ['office', 'conference', 'meeting']):
            return 'office_workspace'
        elif any(keyword in places365_lower for keyword in ['dining', 'restaurant', 'cafe']):
            return 'dining_area'
        elif any(keyword in places365_lower for keyword in ['store', 'shop', 'market']):
            return 'retail_store'
        elif any(keyword in places365_lower for keyword in ['school', 'class', 'library']):
            return 'educational_setting'

        # Outdoor fuzzy matching
        elif any(keyword in places365_lower for keyword in ['street', 'road', 'crosswalk']):
            return 'city_street'
        elif any(keyword in places365_lower for keyword in ['park', 'garden', 'plaza']):
            return 'park_area'
        elif any(keyword in places365_lower for keyword in ['beach', 'ocean', 'lake']):
            return 'beach'
        elif any(keyword in places365_lower for keyword in ['mountain', 'forest', 'desert']):
            return 'natural_outdoor_area'
        elif any(keyword in places365_lower for keyword in ['parking', 'garage']):
            return 'parking_lot'
        elif any(keyword in places365_lower for keyword in ['station', 'terminal', 'airport']):
            return 'transportation_hub'

        # Landmark fuzzy matching
        elif any(keyword in places365_lower for keyword in ['castle', 'palace', 'monument', 'temple']):
            return 'historical_monument'
        elif any(keyword in places365_lower for keyword in ['tower', 'landmark']):
            return 'tourist_landmark'
        elif any(keyword in places365_lower for keyword in ['museum', 'gallery']):
            return 'cultural_venue'

        # Default fallback based on indoor/outdoor
        is_indoor = self._classify_indoor_outdoor(places365_class)
        if is_indoor is True:
            return 'general_indoor_space'
        elif is_indoor is False:
            return 'generic_street_view'
        else:
            return 'unknown'

    def _infer_scene_attributes(self, scene_class: str) -> List[str]:
        """
        Infer basic scene attributes from Places365 class.

        Args:
            scene_class: Places365 scene class name

        Returns:
            List[str]: Inferred scene attributes
        """
        attributes = []
        scene_lower = scene_class.lower()

        # Lighting attributes
        if any(keyword in scene_lower for keyword in ['outdoor', 'street', 'park', 'beach']):
            attributes.append('natural_lighting')
        elif any(keyword in scene_lower for keyword in ['indoor', 'room', 'office']):
            attributes.append('artificial_lighting')

        # Functional attributes
        if any(keyword in scene_lower for keyword in ['commercial', 'store', 'shop', 'restaurant']):
            attributes.append('commercial')
        elif any(keyword in scene_lower for keyword in ['residential', 'home', 'living', 'bedroom']):
            attributes.append('residential')
        elif any(keyword in scene_lower for keyword in ['office', 'conference', 'meeting']):
            attributes.append('workplace')
        elif any(keyword in scene_lower for keyword in ['recreation', 'park', 'playground', 'stadium']):
            attributes.append('recreational')
        elif any(keyword in scene_lower for keyword in ['educational', 'school', 'library', 'classroom']):
            attributes.append('educational')

        # Spatial attributes
        if any(keyword in scene_lower for keyword in ['open', 'field', 'plaza', 'stadium']):
            attributes.append('open_space')
        elif any(keyword in scene_lower for keyword in ['enclosed', 'room', 'interior']):
            attributes.append('enclosed_space')

        return attributes

    def get_scene_probabilities(self, image_pil: Image.Image) -> Dict[str, float]:
        """
        Get probability distribution over all scene classes.

        Args:
            image_pil: Input PIL image

        Returns:
            Dict mapping scene class names to probabilities
        """
        try:
            input_tensor = self.preprocess(image_pil)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

            probs = probabilities.cpu().numpy()[0]

            return {
                self.scene_classes[i]: float(probs[i])
                for i in range(len(self.scene_classes))
            }

        except Exception as e:
            self.logger.error(f"Error getting scene probabilities: {str(e)}")
            return {}
