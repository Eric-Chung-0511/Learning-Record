import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from typing import Any, List, Dict, Tuple, Optional
import io
from PIL import Image

class VisualizationHelper:
    """Helper class for visualizing detection results"""

    @staticmethod
    def visualize_detection(image: Any, result: Any, color_mapper: Optional[Any] = None,
                            figsize: Tuple[int, int] = (12, 12),
                            return_pil: bool = False,
                            filter_classes: Optional[List[int]] = None) -> Optional[Image.Image]:
        """
        Visualize detection results on a single image

        Args:
            image: Image path or numpy array
            result: Detection result object
            color_mapper: ColorMapper instance for consistent colors
            figsize: Figure size
            return_pil: If True, returns a PIL Image object

        Returns:
            PIL Image if return_pil is True, otherwise displays the plot
        """
        if result is None:
            print('No data for visualization')
            return None

        # Read image if path is provided
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image
            if len(img.shape) == 3 and img.shape[2] == 3:
                # Check if BGR format (OpenCV) and convert to RGB if needed
                if isinstance(img, np.ndarray):
                    # Assuming BGR format from OpenCV
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)

        # Get bounding boxes, classes and confidences
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        # Get class names
        names = result.names

        # Create a default color mapper if none is provided
        if color_mapper is None:
            # For backward compatibility, fallback to a simple color function
            from matplotlib import colormaps
            cmap = colormaps['tab10']
            def get_color(class_id):
                return cmap(class_id % 10)
        else:
            # Use the provided color mapper
            def get_color(class_id):
                hex_color = color_mapper.get_color(class_id)
                # Convert hex to RGB float values for matplotlib
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4)) + (1.0,)

        # Draw detection results
        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = box
            cls_id = int(cls)

            if filter_classes and cls_id not in filter_classes:
                continue

            cls_name = names[cls_id]

            # Get color for this class
            box_color = get_color(cls_id)

            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height

            # 根據框大小調整字體大小，但有限制
            adaptive_fontsize = max(10, min(14, int(10 + box_area / 10000)))


            ax.text(x1, y1 - 8, f'{cls_name}: {conf:.2f}',
                    color='white', fontsize=adaptive_fontsize, fontweight="bold",
                    bbox=dict(facecolor=box_color[:3], alpha=0.85, pad=3, boxstyle="round,pad=0.3"),
                    path_effects=[path_effects.withStroke(linewidth=1.5, foreground="black")])

            # Add bounding box
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                    fill=False, edgecolor=box_color[:3], linewidth=2))

        ax.axis('off')
        # ax.set_title('Detection Result')
        plt.tight_layout()

        if return_pil:
            # Convert plot to PIL Image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            pil_img = Image.open(buf)
            plt.close(fig)
            return pil_img
        else:
            plt.show()
            return None

    @staticmethod
    def create_summary(result: Any) -> Dict:
        """
        Create a summary of detection results

        Args:
            result: Detection result object

        Returns:
            Dictionary with detection summary statistics
        """
        if result is None:
            return {"error": "No detection result provided"}

        # Get classes and confidences
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        names = result.names

        # Count detections by class
        class_counts = {}
        for cls, conf in zip(classes, confidences):
            cls_name = names[int(cls)]
            if cls_name not in class_counts:
                class_counts[cls_name] = {"count": 0, "confidences": []}

            class_counts[cls_name]["count"] += 1
            class_counts[cls_name]["confidences"].append(float(conf))

        # Calculate average confidence for each class
        for cls_name, stats in class_counts.items():
            if stats["confidences"]:
                stats["average_confidence"] = float(np.mean(stats["confidences"]))
                stats.pop("confidences")  # Remove detailed confidences list to keep summary concise

        # Prepare summary
        summary = {
            "total_objects": len(classes),
            "class_counts": class_counts,
            "unique_classes": len(class_counts)
        }

        return summary
