import os
import numpy as np
import torch
import torch.nn as nn
import gradio as gr
import time
import timm
from torchvision.ops import nms, box_iou
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from breed_health_info import breed_health_info
from breed_noise_info import breed_noise_info
from dog_database import get_dog_description
from scoring_calculation_system import UserPreferences, calculate_compatibility_score
from recommendation_html_format import format_recommendation_html, get_breed_recommendations
from history_manager import UserHistoryManager
from search_history import create_history_tab, create_history_component
from styles import get_css_styles
from breed_detection import create_detection_tab
from breed_comparison import create_comparison_tab
from breed_recommendation import create_recommendation_tab
from breed_visualization import create_visualization_tab
from style_transfer import DogStyleTransfer, create_style_transfer_tab
from html_templates import (
    format_description_html,
    format_single_dog_result,
    format_multiple_breeds_result,
    format_unknown_breed_message,
    format_not_dog_message,
    format_hint_html,
    format_multi_dog_container,
    format_breed_details_html,
    get_color_scheme,
    get_akc_breeds_link
)
from model_architecture import BaseModel, dog_breeds
from urllib.parse import quote
from ultralytics import YOLO
import asyncio
import traceback

history_manager = UserHistoryManager()

MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/ConvNextV2Base_best_model.pth')

ASSETS_PATH = os.getenv('ASSETS_PATH', '/app/assets/example_images/')

class ModelManager:
    """
    Singleton class for managing model instances and device allocation
    specifically designed for Hugging Face Spaces deployment.
    """
    _instance = None
    _initialized = False
    _yolo_model = None
    _breed_model = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not ModelManager._initialized:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ModelManager._initialized = True

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device

    @property
    def yolo_model(self):
        if self._yolo_model is None:
            yolo_path = os.getenv('YOLO_MODEL_PATH', '/app/yolo_models/yolov8x.pt')
            if os.path.exists(yolo_path):
                self._yolo_model = YOLO(yolo_path)
            else:
                # 如果本地檔案不存在，回退到自動下載
                self._yolo_model = YOLO('yolov8x.pt')
        return self._yolo_model

    @property
    def breed_model(self):
        if self._breed_model is None:
            self._breed_model = BaseModel(
                num_classes=len(dog_breeds),
                device=self.device
            ).to(self.device)

            checkpoint = torch.load(
                MODEL_PATH,
                map_location=self.device
            )

            # Try to load with model_state_dict first, then base_model
            if 'model_state_dict' in checkpoint:
                self._breed_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'base_model' in checkpoint:
                self._breed_model.load_state_dict(checkpoint['base_model'], strict=False)
            else:
                # If neither key exists, raise a descriptive error
                available_keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else "not a dictionary"
                raise KeyError(f"Model checkpoint does not contain 'model_state_dict' or 'base_model' keys. Available keys: {available_keys}")

            self._breed_model.eval()
        return self._breed_model

# Initialize model manager
model_manager = ModelManager()

def preprocess_image(image):
    """Preprocesses images for model input"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform(image).unsqueeze(0)

def predict_single_dog(image):
    """Predicts dog breed for a single image"""
    image_tensor = preprocess_image(image).to(model_manager.device)

    with torch.no_grad():
        logits = model_manager.breed_model(image_tensor)[0]
        probs = F.softmax(logits, dim=1)

        top5_prob, top5_idx = torch.topk(probs, k=5)
        breeds = [dog_breeds[idx.item()] for idx in top5_idx[0]]
        probabilities = [prob.item() for prob in top5_prob[0]]

        sum_probs = sum(probabilities[:3])
        relative_probs = [f"{(prob/sum_probs * 100):.2f}%" for prob in probabilities[:3]]

        return probabilities[0], breeds[:3], relative_probs

def enhanced_preprocess(image, is_standing=False, has_overlap=False):
    """
    Enhanced image preprocessing function with special handling for different poses
    and overlapping cases.
    """
    target_size = 224
    w, h = image.size

    if is_standing:
        if h > w * 1.5:
            new_h = target_size
            new_w = min(target_size, int(w * (target_size / h)))
            new_w = max(new_w, int(target_size * 0.6))
    elif has_overlap:
        scale = min(target_size/w, target_size/h) * 0.95
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        scale = min(target_size/w, target_size/h)
        new_w = int(w * scale)
        new_h = int(h * scale)

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    final_image = Image.new('RGB', (target_size, target_size), (240, 240, 240))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    final_image.paste(resized, (paste_x, paste_y))

    return final_image

def detect_multiple_dogs(image, conf_threshold=0.3, iou_threshold=0.3):
    """
    Enhanced multiple dog detection with improved bounding box handling and
    intelligent boundary adjustments.
    """
    results = model_manager.yolo_model(image, conf=conf_threshold, iou=iou_threshold)[0]
    img_width, img_height = image.size
    detected_boxes = []

    # Phase 1: Initial detection and processing
    for box in results.boxes:
        if box.cls.item() == 16:  # Dog class
            xyxy = box.xyxy[0].tolist()
            confidence = box.conf.item()
            x1, y1, x2, y2 = map(int, xyxy)
            w = x2 - x1
            h = y2 - y1

            detected_boxes.append({
                'coords': [x1, y1, x2, y2],
                'width': w,
                'height': h,
                'center_x': (x1 + x2) / 2,
                'center_y': (y1 + y2) / 2,
                'area': w * h,
                'confidence': confidence,
                'aspect_ratio': w / h if h != 0 else 1
            })

    if not detected_boxes:
        return [(image, 1.0, [0, 0, img_width, img_height], False)]

    # Phase 2: Analysis of detection relationships
    avg_height = sum(box['height'] for box in detected_boxes) / len(detected_boxes)
    avg_width = sum(box['width'] for box in detected_boxes) / len(detected_boxes)
    avg_area = sum(box['area'] for box in detected_boxes) / len(detected_boxes)

    def calculate_iou(box1, box2):
        x1 = max(box1['coords'][0], box2['coords'][0])
        y1 = max(box1['coords'][1], box2['coords'][1])
        x2 = min(box1['coords'][2], box2['coords'][2])
        y2 = min(box1['coords'][3], box2['coords'][3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1['area']
        area2 = box2['area']
        return intersection / (area1 + area2 - intersection)

    # Phase 3: Processing each detection
    processed_boxes = []
    overlap_threshold = 0.2

    for i, box_info in enumerate(detected_boxes):
        x1, y1, x2, y2 = box_info['coords']
        w = box_info['width']
        h = box_info['height']
        center_x = box_info['center_x']
        center_y = box_info['center_y']

        # Check for overlaps
        has_overlap = False
        for j, other_box in enumerate(detected_boxes):
            if i != j and calculate_iou(box_info, other_box) > overlap_threshold:
                has_overlap = True
                break

        # Adjust expansion strategy
        base_expansion = 0.03
        max_expansion = 0.05

        is_standing = h > 1.5 * w
        is_sitting = 0.8 <= h/w <= 1.2
        is_abnormal_size = (h * w) > (avg_area * 1.5) or (h * w) < (avg_area * 0.5)

        if has_overlap:
            h_expansion = w_expansion = base_expansion * 0.8
        else:
            if is_standing:
                h_expansion = min(base_expansion * 1.2, max_expansion)
                w_expansion = base_expansion
            elif is_sitting:
                h_expansion = w_expansion = base_expansion
            else:
                h_expansion = w_expansion = base_expansion * 0.9

        # Position compensation
        if center_x < img_width * 0.2 or center_x > img_width * 0.8:
            w_expansion *= 0.9

        if is_abnormal_size:
            h_expansion *= 0.8
            w_expansion *= 0.8

        # Calculate final bounding box
        expansion_w = w * w_expansion
        expansion_h = h * h_expansion

        new_x1 = max(0, center_x - (w + expansion_w)/2)
        new_y1 = max(0, center_y - (h + expansion_h)/2)
        new_x2 = min(img_width, center_x + (w + expansion_w)/2)
        new_y2 = min(img_height, center_y + (h + expansion_h)/2)

        # Crop and process image
        cropped_image = image.crop((int(new_x1), int(new_y1),
                                  int(new_x2), int(new_y2)))

        processed_image = enhanced_preprocess(
            cropped_image,
            is_standing=is_standing,
            has_overlap=has_overlap
        )

        processed_boxes.append((
            processed_image,
            box_info['confidence'],
            [new_x1, new_y1, new_x2, new_y2],
            True
        ))

    return processed_boxes

def predict(image):
    """
    Main prediction function that handles both single and multiple dog detection.
    Args:
        image: PIL Image or numpy array
    Returns:
        tuple: (html_output, annotated_image, initial_state)
    """
    if image is None:
        return format_hint_html("Please upload an image to start."), None, None

    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # 檢測圖片中的物體
        dogs = detect_multiple_dogs(image)
        color_scheme = get_color_scheme(len(dogs) == 1)

        # 準備標註
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        dogs_info = ""

        # 處理每個檢測到的物體
        for i, (cropped_image, detection_confidence, box, is_dog) in enumerate(dogs):
            print(f"Predict processing - Object {i+1}:")
            print(f"    Is dog: {is_dog}")
            print(f"    Detection confidence: {detection_confidence:.4f}")

            # 如果是狗且進行品種預測
            if is_dog:
                top1_prob, topk_breeds, relative_probs = predict_single_dog(cropped_image)
                print(f"    Breed prediction - Top probability: {top1_prob:.4f}")
                print(f"    Top breeds: {topk_breeds[:3]}")
            color = color_scheme if len(dogs) == 1 else color_scheme[i % len(color_scheme)]

            # 繪製框和標籤
            draw.rectangle(box, outline=color, width=4)
            label = f"Dog {i+1}" if is_dog else f"Object {i+1}"
            label_bbox = draw.textbbox((0, 0), label, font=font)
            label_width = label_bbox[2] - label_bbox[0]
            label_height = label_bbox[3] - label_bbox[1]

            # 繪製標籤背景和文字
            label_x = box[0] + 5
            label_y = box[1] + 5
            draw.rectangle(
                [label_x - 2, label_y - 2, label_x + label_width + 4, label_y + label_height + 4],
                fill='white',
                outline=color,
                width=2
            )
            draw.text((label_x, label_y), label, fill=color, font=font)

            try:
                # 首先檢查是否為狗
                if not is_dog:
                    dogs_info += format_not_dog_message(color, i+1)
                    continue

                # 如果是狗，進行品種預測
                top1_prob, topk_breeds, relative_probs = predict_single_dog(cropped_image)
                combined_confidence = detection_confidence * top1_prob

                # 根據信心度決定輸出格式
                if combined_confidence < 0.15:
                    dogs_info += format_unknown_breed_message(color, i+1)
                elif top1_prob >= 0.4:
                    breed = topk_breeds[0]
                    description = get_dog_description(breed)
                    if description is None:
                        description = {
                            "Name": breed,
                            "Size": "Unknown",
                            "Exercise Needs": "Unknown",
                            "Grooming Needs": "Unknown",
                            "Care Level": "Unknown",
                            "Good with Children": "Unknown",
                            "Description": f"Identified as {breed.replace('_', ' ')}"
                        }
                    dogs_info += format_single_dog_result(breed, description, color)
                else:
                    dogs_info += format_multiple_breeds_result(
                        topk_breeds,
                        relative_probs,
                        color,
                        i+1,
                        lambda breed: get_dog_description(breed) or {
                            "Name": breed,
                            "Size": "Unknown",
                            "Exercise Needs": "Unknown",
                            "Grooming Needs": "Unknown",
                            "Care Level": "Unknown",
                            "Good with Children": "Unknown",
                            "Description": f"Identified as {breed.replace('_', ' ')}"
                        }
                    )
            except Exception as e:
                print(f"Error formatting results for dog {i+1}: {str(e)}")
                dogs_info += format_unknown_breed_message(color, i+1)

        # 包裝最終的HTML輸出
        html_output = format_multi_dog_container(dogs_info)

        # 準備初始狀態
        initial_state = {
            "dogs_info": dogs_info,
            "image": annotated_image,
            "is_multi_dog": len(dogs) > 1,
            "html_output": html_output
        }

        return html_output, annotated_image, initial_state

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return format_hint_html(error_msg), None, None


def show_details_html(choice, previous_output, initial_state):
    """
    Generate detailed HTML view for a selected breed.

    Args:
        choice: str, Selected breed option
        previous_output: str, Previous HTML output
        initial_state: dict, Current state information

    Returns:
        tuple: (html_output, gradio_update, updated_state)
    """
    if not choice:
        return previous_output, gr.update(visible=True), initial_state

    try:
        breed = choice.split("More about ")[-1]
        description = get_dog_description(breed)
        html_output = format_breed_details_html(description, breed)

        # Update state
        initial_state["current_description"] = html_output
        initial_state["original_buttons"] = initial_state.get("buttons", [])

        return html_output, gr.update(visible=True), initial_state

    except Exception as e:
        error_msg = f"An error occurred while showing details: {e}"
        print(error_msg)
        return format_hint_html(error_msg), gr.update(visible=True), initial_state
    
def load_example_images(image_dir):
    """動態載入範例圖片檔案"""
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    example_images = []
    
    if os.path.exists(image_dir):
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(supported_formats):
                example_images.append(os.path.join(image_dir, filename))
    
    return sorted(example_images)

def main():
    with gr.Blocks(css=get_css_styles()) as iface:
        # Header HTML

        gr.HTML("""
        <header style='text-align: center; padding: 20px; margin-bottom: 20px;'>
            <h1 style='font-size: 2.5em; margin-bottom: 10px; color: #2D3748;'>
                🐾 PawMatch AI
            </h1>
            <h2 style='font-size: 1.2em; font-weight: normal; color: #4A5568; margin-top: 5px;'>
                Your Smart Dog Breed Guide
            </h2>
            <div style='width: 50px; height: 3px; background: linear-gradient(90deg, #4299e1, #48bb78); margin: 15px auto;'></div>
            <p style='color: #718096; font-size: 0.9em;'>
                Powered by AI • Breed Recognition • Smart Matching • Companion Guide
            </p>
        </header>
        """)

        # 先創建歷史組件實例（但不創建標籤頁）
        history_component = create_history_component()

        # Initialize style transfor
        dog_style_transfer = DogStyleTransfer()

        with gr.Tabs():
            # 1. breed detection
            example_images = load_example_images(ASSETS_PATH)
            
            detection_components = create_detection_tab(predict, example_images)

            # 2. breed comparison
            comparison_components = create_comparison_tab(
                dog_breeds=dog_breeds,
                get_dog_description=get_dog_description,
                breed_health_info=breed_health_info,
                breed_noise_info=breed_noise_info
            )

            # 3. breed recommendation
            recommendation_components = create_recommendation_tab(
                UserPreferences=UserPreferences,
                get_breed_recommendations=get_breed_recommendations,
                format_recommendation_html=format_recommendation_html,
                history_component=history_component
            )

            # 4. Visualization Analysis
            with gr.Tab("Visualization Analysis"):
                create_visualization_tab(
                    dog_breeds=dog_breeds,
                    get_dog_description=get_dog_description,
                    calculate_compatibility_score=calculate_compatibility_score,
                    UserPreferences=UserPreferences
                )

            # 5. Style Transfer tab
            with gr.Tab("Style Transfer"):
                style_transfer_components = create_style_transfer_tab(dog_style_transfer)


            # 6. History Search
            create_history_tab(history_component)

        # Footer
        gr.HTML('''
            <div style="
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 20px;
                padding: 20px 0;
            ">
                <p style="
                    font-family: 'Arial', sans-serif;
                    font-size: 14px;
                    font-weight: 500;
                    letter-spacing: 2px;
                    background: linear-gradient(90deg, #555, #007ACC);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin: 0;
                    text-transform: uppercase;
                    display: inline-block;
                ">EXPLORE THE CODE →</p>
                <a href="https://github.com/Eric-Chung-0511/Learning-Record/tree/main/Data%20Science%20Projects/PawMatchAI" style="text-decoration: none;">
                    <img src="https://img.shields.io/badge/GitHub-PawMatch_AI-007ACC?logo=github&style=for-the-badge">
                </a>
            </div>
        ''')

    return iface

if __name__ == "__main__":
    iface = main()
    iface.launch(server_name="0.0.0.0", server_port=7860) 