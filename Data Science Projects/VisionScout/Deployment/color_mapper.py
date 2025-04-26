import numpy as np
from typing import Dict, List, Tuple, Union, Any

class ColorMapper:
    """
    A class for consistent color mapping of object detection classes
    Provides color schemes for visualization in both RGB and hex formats
    """
    
    # Class categories for better organization
    CATEGORIES = {
        "person": [0],
        "vehicles": [1, 2, 3, 4, 5, 6, 7, 8],
        "traffic": [9, 10, 11, 12],
        "animals": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        "outdoor": [13, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
        "sports": [34, 35, 36, 37, 38],
        "kitchen": [39, 40, 41, 42, 43, 44, 45],
        "food": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
        "furniture": [56, 57, 58, 59, 60, 61],
        "electronics": [62, 63, 64, 65, 66, 67, 68, 69, 70],
        "household": [71, 72, 73, 74, 75, 76, 77, 78, 79]
    }
    
    # Base colors for each category (in HSV for easier variation)
    CATEGORY_COLORS = {
        "person": (0, 0.8, 0.9),       # Red
        "vehicles": (210, 0.8, 0.9),   # Blue
        "traffic": (45, 0.8, 0.9),     # Orange
        "animals": (120, 0.7, 0.8),    # Green
        "outdoor": (180, 0.7, 0.9),    # Cyan
        "sports": (270, 0.7, 0.8),     # Purple
        "kitchen": (30, 0.7, 0.9),     # Light Orange
        "food": (330, 0.7, 0.85),      # Pink
        "furniture": (150, 0.5, 0.85), # Light Green
        "electronics": (240, 0.6, 0.9), # Light Blue
        "household": (60, 0.6, 0.9)    # Yellow
    }
    
    def __init__(self):
        """Initialize the ColorMapper with COCO class mappings"""
        self.class_names = self._get_coco_classes()
        self.color_map = self._generate_color_map()
    
    def _get_coco_classes(self) -> Dict[int, str]:
        """Get the standard COCO class names with their IDs"""
        return {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
            44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 
            49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
            54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 
            59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
            64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 
            79: 'toothbrush'
        }
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[int, int, int]:
        """
        Convert HSV color to RGB
        
        Args:
            h: Hue (0-360)
            s: Saturation (0-1)
            v: Value (0-1)
            
        Returns:
            Tuple of (R, G, B) values (0-255)
        """
        h = h / 60
        i = int(h)
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        
        return (int(r * 255), int(g * 255), int(b * 255))
    
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """
        Convert RGB color to hex color code
        
        Args:
            rgb: Tuple of (R, G, B) values (0-255)
            
        Returns:
            Hex color code (e.g. '#FF0000')
        """
        return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
    
    def _find_category(self, class_id: int) -> str:
        """
        Find the category for a given class ID
        
        Args:
            class_id: Class ID (0-79)
            
        Returns:
            Category name
        """
        for category, ids in self.CATEGORIES.items():
            if class_id in ids:
                return category
        return "other"  # Fallback
    
    def _generate_color_map(self) -> Dict:
        """
        Generate a color map for all 80 COCO classes
        
        Returns:
            Dictionary mapping class IDs and names to color values
        """
        color_map = {
            'by_id': {},      # Map class ID to RGB and hex
            'by_name': {},    # Map class name to RGB and hex
            'categories': {}  # Map category to base color
        }
        
        # Generate colors for categories
        for category, hsv in self.CATEGORY_COLORS.items():
            rgb = self._hsv_to_rgb(hsv[0], hsv[1], hsv[2])
            hex_color = self._rgb_to_hex(rgb)
            color_map['categories'][category] = {
                'rgb': rgb,
                'hex': hex_color
            }
        
        # Generate variations for each class within a category
        for class_id, class_name in self.class_names.items():
            category = self._find_category(class_id)
            base_hsv = self.CATEGORY_COLORS.get(category, (0, 0, 0.8))  # Default gray
            
            # Slightly vary the hue and saturation within the category
            ids_in_category = self.CATEGORIES.get(category, [])
            if ids_in_category:
                position = ids_in_category.index(class_id) if class_id in ids_in_category else 0
                variation = position / max(1, len(ids_in_category) - 1)  # 0 to 1
                
                # Vary hue slightly (±15°) and saturation
                h_offset = 30 * variation - 15  # -15 to +15
                s_offset = 0.2 * variation  # 0 to 0.2
                
                h = (base_hsv[0] + h_offset) % 360
                s = min(1.0, base_hsv[1] + s_offset)
                v = base_hsv[2]
            else:
                h, s, v = base_hsv
            
            rgb = self._hsv_to_rgb(h, s, v)
            hex_color = self._rgb_to_hex(rgb)
            
            # Store in both mappings
            color_map['by_id'][class_id] = {
                'rgb': rgb,
                'hex': hex_color,
                'category': category
            }
            
            color_map['by_name'][class_name] = {
                'rgb': rgb,
                'hex': hex_color,
                'category': category
            }
        
        return color_map
    
    def get_color(self, class_identifier: Union[int, str], format: str = 'hex') -> Any:
        """
        Get color for a specific class
        
        Args:
            class_identifier: Class ID (int) or name (str)
            format: Color format ('hex', 'rgb', or 'bgr')
            
        Returns:
            Color in requested format
        """
        # Determine if identifier is an ID or name
        if isinstance(class_identifier, int):
            color_info = self.color_map['by_id'].get(class_identifier)
        else:
            color_info = self.color_map['by_name'].get(class_identifier)
        
        if not color_info:
            # Fallback color if not found
            return '#CCCCCC' if format == 'hex' else (204, 204, 204)
        
        if format == 'hex':
            return color_info['hex']
        elif format == 'rgb':
            return color_info['rgb']
        elif format == 'bgr':
            # Convert RGB to BGR for OpenCV
            r, g, b = color_info['rgb']
            return (b, g, r)
        else:
            return color_info['rgb']
    
    def get_all_colors(self, format: str = 'hex') -> Dict:
        """
        Get all colors in the specified format
        
        Args:
            format: Color format ('hex', 'rgb', or 'bgr')
            
        Returns:
            Dictionary mapping class names to colors
        """
        result = {}
        for class_id, class_name in self.class_names.items():
            result[class_name] = self.get_color(class_id, format)
        return result
    
    def get_category_colors(self, format: str = 'hex') -> Dict:
        """
        Get base colors for each category
        
        Args:
            format: Color format ('hex', 'rgb', or 'bgr')
            
        Returns:
            Dictionary mapping categories to colors
        """
        result = {}
        for category, color_info in self.color_map['categories'].items():
            if format == 'hex':
                result[category] = color_info['hex']
            elif format == 'bgr':
                r, g, b = color_info['rgb']
                result[category] = (b, g, r)
            else:
                result[category] = color_info['rgb']
        return result
    
    def get_category_for_class(self, class_identifier: Union[int, str]) -> str:
        """
        Get the category for a specific class
        
        Args:
            class_identifier: Class ID (int) or name (str)
            
        Returns:
            Category name
        """
        if isinstance(class_identifier, int):
            return self.color_map['by_id'].get(class_identifier, {}).get('category', 'other')
        else:
            return self.color_map['by_name'].get(class_identifier, {}).get('category', 'other')
