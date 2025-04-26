import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import gradio as gr
import io
from PIL import Image, ImageDraw, ImageFont
import spaces
from typing import Dict, List, Any, Optional, Tuple
from ultralytics import YOLO

from detection_model import DetectionModel
from color_mapper import ColorMapper
from visualization_helper import VisualizationHelper
from evaluation_metrics import EvaluationMetrics
from style import Style


color_mapper = ColorMapper()
model_instances = {}

@spaces.GPU
def process_image(image, model_instance, confidence_threshold, filter_classes=None):
    """
    Process an image for object detection
    
    Args:
        image: Input image (numpy array or PIL Image)
        model_instance: DetectionModel instance to use
        confidence_threshold: Confidence threshold for detection
        filter_classes: Optional list of classes to filter results
        
    Returns:
        Tuple of (result_image, result_text, stats_data)
    """
    # initialize key variables
    result = None
    stats = {}
    temp_path = None
    
    try:
        # update confidence threshold
        model_instance.confidence = confidence_threshold
        
        # processing input image
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            pil_image = Image.fromarray(image_rgb)
        elif image is None:
            return None, "No image provided. Please upload an image.", {}
        else:
            pil_image = image
        
        # store temp files
        import uuid
        import tempfile
        
        temp_dir = tempfile.gettempdir()  # use system temp directory
        temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
        temp_path = os.path.join(temp_dir, temp_filename)
        pil_image.save(temp_path)
        
        # object detection
        result = model_instance.detect(temp_path)
        
        if result is None:
            return None, "Detection failed. Please try again with a different image.", {}
        
        # calculate stats
        stats = EvaluationMetrics.calculate_basic_stats(result)
            
        # add space calculation
        spatial_metrics = EvaluationMetrics.calculate_distance_metrics(result)
        stats["spatial_metrics"] = spatial_metrics
        
        if filter_classes and len(filter_classes) > 0:
            # get classes, boxes, confidence
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            
            mask = np.zeros_like(classes, dtype=bool)
            for cls_id in filter_classes:
                mask = np.logical_or(mask, classes == cls_id)
    
            filtered_stats = {
                "total_objects": int(np.sum(mask)),
                "class_statistics": {},
                "average_confidence": float(np.mean(confs[mask])) if np.any(mask) else 0,
                "spatial_metrics": stats["spatial_metrics"]  
            }
            
            # update stats 
            names = result.names
            for cls, conf in zip(classes[mask], confs[mask]):
                cls_name = names[int(cls)]
                if cls_name not in filtered_stats["class_statistics"]:
                    filtered_stats["class_statistics"][cls_name] = {
                        "count": 0,
                        "average_confidence": 0
                    }
                
                filtered_stats["class_statistics"][cls_name]["count"] += 1
                filtered_stats["class_statistics"][cls_name]["average_confidence"] = conf
            
            stats = filtered_stats
        
        viz_data = EvaluationMetrics.generate_visualization_data(
            result, 
            color_mapper.get_all_colors()
        )
        
        result_image = VisualizationHelper.visualize_detection(
            temp_path, result, color_mapper=color_mapper, figsize=(12, 12), return_pil=True
        )
        
        result_text = EvaluationMetrics.format_detection_summary(viz_data)
        
        return result_image, result_text, stats
        
    except Exception as e:
        error_message = f"Error Occurs: {str(e)}"
        import traceback
        traceback.print_exc()  
        print(error_message)
        return None, error_message, {}
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Cannot delete temp files {temp_path}: {str(e)}")

def format_result_text(stats):
    """
    Format detection statistics into readable text with improved spacing
    
    Args:
        stats: Dictionary containing detection statistics
        
    Returns:
        Formatted text summary
    """
    if not stats or "total_objects" not in stats:
        return "No objects detected."
        
    # 減少不必要的空行
    lines = [
        f"Detected {stats['total_objects']} objects.",  
        f"Average confidence: {stats.get('average_confidence', 0):.2f}",
        "Objects by class:"
    ]
    
    if "class_statistics" in stats and stats["class_statistics"]:
        # 按計數排序類別
        sorted_classes = sorted(
            stats["class_statistics"].items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        
        for cls_name, cls_stats in sorted_classes:
            count = cls_stats["count"]
            conf = cls_stats.get("average_confidence", 0)
            
            item_text = "item" if count == 1 else "items"
            lines.append(f"• {cls_name}: {count} {item_text} (avg conf: {conf:.2f})")
    else:
        lines.append("No class information available.")
    
    # 添加空間信息
    if "spatial_metrics" in stats and "spatial_distribution" in stats["spatial_metrics"]:
        lines.append("Object Distribution:")
        
        dist = stats["spatial_metrics"]["spatial_distribution"]
        x_mean = dist.get("x_mean", 0)
        y_mean = dist.get("y_mean", 0)
        
        # 描述物體的大致位置
        if x_mean < 0.33:
            h_pos = "on the left side"
        elif x_mean < 0.67:
            h_pos = "in the center"
        else:
            h_pos = "on the right side"
        
        if y_mean < 0.33:
            v_pos = "in the upper part"
        elif y_mean < 0.67:
            v_pos = "in the middle"
        else:
            v_pos = "in the lower part"
            
        lines.append(f"• Most objects appear {h_pos} {v_pos} of the image")
    
    return "\n".join(lines)

def format_json_for_display(stats):
    """
    Format statistics JSON for better display
    
    Args:
        stats: Raw statistics dictionary
        
    Returns:
        Formatted statistics structure for display
    """
    # Create a cleaner copy of the stats for display
    display_stats = {}
    
    # Add summary section
    display_stats["summary"] = {
        "total_objects": stats.get("total_objects", 0),
        "average_confidence": round(stats.get("average_confidence", 0), 3)
    }
    
    # Add class statistics in a more organized way
    if "class_statistics" in stats and stats["class_statistics"]:
        # Sort classes by count (descending)
        sorted_classes = sorted(
            stats["class_statistics"].items(),
            key=lambda x: x[1].get("count", 0),
            reverse=True
        )
        
        class_stats = {}
        for cls_name, cls_data in sorted_classes:
            class_stats[cls_name] = {
                "count": cls_data.get("count", 0),
                "average_confidence": round(cls_data.get("average_confidence", 0), 3)
            }
            
        display_stats["detected_objects"] = class_stats
    
    # Simplify spatial metrics
    if "spatial_metrics" in stats:
        spatial = stats["spatial_metrics"]
        
        # Simplify spatial distribution
        if "spatial_distribution" in spatial:
            dist = spatial["spatial_distribution"]
            display_stats["spatial"] = {
                "distribution": {
                    "x_mean": round(dist.get("x_mean", 0), 3),
                    "y_mean": round(dist.get("y_mean", 0), 3),
                    "x_std": round(dist.get("x_std", 0), 3),
                    "y_std": round(dist.get("y_std", 0), 3)
                }
            }
            
        # Add simplified size information
        if "size_distribution" in spatial:
            size = spatial["size_distribution"]
            display_stats["spatial"]["size"] = {
                "mean_area": round(size.get("mean_area", 0), 3),
                "min_area": round(size.get("min_area", 0), 3),
                "max_area": round(size.get("max_area", 0), 3)
            }
    
    return display_stats

def get_all_classes():
    """
    Get all available COCO classes from the currently active model or fallback to standard COCO classes
    
    Returns:
        List of tuples (class_id, class_name)
    """
    global model_instances
    
    # Try to get class names from any loaded model
    for model_name, model_instance in model_instances.items():
        if model_instance and model_instance.is_model_loaded:
            try:
                class_names = model_instance.class_names
                return [(idx, name) for idx, name in class_names.items()]
            except Exception:
                pass
    
    # Fallback to standard COCO classes
    return [
        (0, 'person'), (1, 'bicycle'), (2, 'car'), (3, 'motorcycle'), (4, 'airplane'),
        (5, 'bus'), (6, 'train'), (7, 'truck'), (8, 'boat'), (9, 'traffic light'),
        (10, 'fire hydrant'), (11, 'stop sign'), (12, 'parking meter'), (13, 'bench'), 
        (14, 'bird'), (15, 'cat'), (16, 'dog'), (17, 'horse'), (18, 'sheep'), (19, 'cow'),
        (20, 'elephant'), (21, 'bear'), (22, 'zebra'), (23, 'giraffe'), (24, 'backpack'),
        (25, 'umbrella'), (26, 'handbag'), (27, 'tie'), (28, 'suitcase'), (29, 'frisbee'),
        (30, 'skis'), (31, 'snowboard'), (32, 'sports ball'), (33, 'kite'), (34, 'baseball bat'),
        (35, 'baseball glove'), (36, 'skateboard'), (37, 'surfboard'), (38, 'tennis racket'), 
        (39, 'bottle'), (40, 'wine glass'), (41, 'cup'), (42, 'fork'), (43, 'knife'),
        (44, 'spoon'), (45, 'bowl'), (46, 'banana'), (47, 'apple'), (48, 'sandwich'), 
        (49, 'orange'), (50, 'broccoli'), (51, 'carrot'), (52, 'hot dog'), (53, 'pizza'),
        (54, 'donut'), (55, 'cake'), (56, 'chair'), (57, 'couch'), (58, 'potted plant'), 
        (59, 'bed'), (60, 'dining table'), (61, 'toilet'), (62, 'tv'), (63, 'laptop'),
        (64, 'mouse'), (65, 'remote'), (66, 'keyboard'), (67, 'cell phone'), (68, 'microwave'), 
        (69, 'oven'), (70, 'toaster'), (71, 'sink'), (72, 'refrigerator'), (73, 'book'),
        (74, 'clock'), (75, 'vase'), (76, 'scissors'), (77, 'teddy bear'), (78, 'hair drier'), 
        (79, 'toothbrush')
    ]

def create_interface():
    """創建 Gradio 界面，包含美化的視覺效果"""
    css = Style.get_css()

    # 獲取可用模型信息
    available_models = DetectionModel.get_available_models()
    model_choices = [model["model_file"] for model in available_models]
    model_labels = [f"{model['name']} - {model['inference_speed']}" for model in available_models]
    
    # 可用類別過濾選項
    available_classes = get_all_classes()
    class_choices = [f"{id}: {name}" for id, name in available_classes]
    
    # 創建 Gradio Blocks 界面
    with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="teal", secondary_hue="blue")) as demo:
        # 頁面頂部標題
        with gr.Group(elem_classes="app-header"):
            gr.HTML("""
                <div style="text-align: center; width: 100%;">
                    <h1 class="app-title">VisionScout</h1>
                    <h2 class="app-subtitle">Detect and identify objects in your images</h2>
                    <div class="app-divider"></div>
                </div>
            """)

        current_model = gr.State("yolov8m.pt")  # use medium size model as defualt
        
        # 主要內容區 - 輸入和輸出面板
        with gr.Row(equal_height=True):
            # 左側 - 輸入控制區(可上傳圖片)
            with gr.Column(scale=4, elem_classes="input-panel"):
                with gr.Group():
                    gr.HTML('<div class="section-heading">Upload Image</div>')
                    image_input = gr.Image(type="pil", label="Upload an image", elem_classes="upload-box")
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                choices=model_choices,
                                value="yolov8m.pt",  
                                label="Select Model",
                                info="Choose different models based on your needs for speed vs. accuracy"
                            )
                        
                        # display model info
                        model_info = gr.Markdown(DetectionModel.get_model_description("yolov8m.pt"))

                        confidence = gr.Slider(
                            minimum=0.1, 
                            maximum=0.9, 
                            value=0.25, 
                            step=0.05, 
                            label="Confidence Threshold",
                            info="Higher values show fewer but more confident detections"
                        )
                        
                        with gr.Accordion("Filter Classes", open=False):
                            # 常見物件類別快速選擇按鈕
                            gr.HTML('<div class="section-heading" style="font-size: 1rem;">Common Categories</div>')
                            with gr.Row():
                                people_btn = gr.Button("People", size="sm")
                                vehicles_btn = gr.Button("Vehicles", size="sm")
                                animals_btn = gr.Button("Animals", size="sm")
                                objects_btn = gr.Button("Common Objects", size="sm")
                            
                            # 類別選擇下拉框
                            class_filter = gr.Dropdown(
                                choices=class_choices,
                                multiselect=True,
                                label="Select Classes to Display",
                                info="Leave empty to show all detected objects"
                            )
                    
                    # detect buttom
                    detect_btn = gr.Button("Detect Objects", variant="primary", elem_classes="detect-btn")
                
                # 使用說明區
                with gr.Group(elem_classes="how-to-use"):
                    gr.HTML('<div class="section-heading">How to Use</div>')
                    gr.Markdown("""
                    1. Upload an image or use the camera
                    2. (Optional) Adjust settings like confidence threshold or model size (n, m, x)
                    3. Optionally filter to specific object classes
                    4. Click "Detect Objects" button
                    
                    The model will identify objects in your image and display them with bounding boxes.
                    
                    **Note:** Detection quality depends on image clarity and model settings.
                    """)
            
            # 右側 - 結果顯示區
            with gr.Column(scale=6, elem_classes="output-panel"):
                with gr.Tabs(elem_classes="tabs"):
                    with gr.Tab("Detection Result"):
                        result_image = gr.Image(type="pil", label="Detection Result")
                        
                        # 文本框的格式
                        with gr.Group(elem_classes="result-details-box"):
                            gr.HTML('<div class="section-heading">Detection Details</div>')
                            # 文本框設置，讓顯示會更寬
                            result_text = gr.Textbox(
                                label=None,
                                lines=12,
                                max_lines=15,
                                elem_classes="wide-result-text",
                                elem_id="detection-details",
                                container=False,  
                                scale=2,          
                                min_width=600     
                            )
                    
                    with gr.Tab("Statistics"):
                        with gr.Row():
                            with gr.Column(scale=3, elem_classes="plot-column"):
                                gr.HTML('<div class="section-heading">Object Distribution</div>')
                                plot_output = gr.Plot(
                                    label=None,  
                                    elem_classes="large-plot-container"
                                )
                            
                            # 右側放 JSON 數據比較清晰
                            with gr.Column(scale=2, elem_classes="stats-column"):
                                gr.HTML('<div class="section-heading">Detection Statistics</div>')
                                stats_json = gr.JSON(
                                    label=None,  # remove label
                                    elem_classes="enhanced-json-display"
                                )
        
        detect_btn.click(
            fn=lambda img, model, conf, classes: process_and_plot(img, model, conf, classes),
            inputs=[image_input, current_model, confidence, class_filter],
            outputs=[result_image, result_text, stats_json, plot_output]
        )

        # model option
        model_dropdown.change(
            fn=lambda model: (model, DetectionModel.get_model_description(model)),
            inputs=[model_dropdown],
            outputs=[current_model, model_info]
        )
        
        # each classes link
        people_classes = [0]  # 人
        vehicles_classes = [1, 2, 3, 4, 5, 6, 7, 8]  # 各種車輛
        animals_classes = list(range(14, 24))  # COCO 中的動物
        common_objects = [41, 42, 43, 44, 45, 67, 73, 74, 76]  # 常見家居物品
        
        # Linked the quik buttom
        people_btn.click(
            lambda: [f"{id}: {name}" for id, name in available_classes if id in people_classes],
            outputs=class_filter
        )
        
        vehicles_btn.click(
            lambda: [f"{id}: {name}" for id, name in available_classes if id in vehicles_classes],
            outputs=class_filter
        )
        
        animals_btn.click(
            lambda: [f"{id}: {name}" for id, name in available_classes if id in animals_classes],
            outputs=class_filter
        )
        
        objects_btn.click(
            lambda: [f"{id}: {name}" for id, name in available_classes if id in common_objects],
            outputs=class_filter
        )
        
        example_images = [
            "room_01.jpg",
            "street_01.jpg",
            "street_02.jpg",
            "street_03.jpg"
        ]
        
        # add example images
        gr.Examples(
            examples=example_images,
            inputs=image_input,
            outputs=None,  
            fn=None,  
            cache_examples=False,  
        )
        
        # 頁腳部分
        gr.HTML("""
            <div class="footer">
                <p>Powered by YOLOv8 and Ultralytics • Created with Gradio</p>
                <p>Model can detect 80 different classes of objects</p>
            </div>
        """)
    
    return demo

@spaces.GPU
def process_and_plot(image, model_name, confidence_threshold, filter_classes=None):
    """
    Process image and create plots for statistics with enhanced visualization
    
    Args:
        image: Input image
        model_name: Name of the model to use
        confidence_threshold: Confidence threshold for detection
        filter_classes: Optional list of classes to filter results
        
    Returns:
        Tuple of (result_image, result_text, formatted_stats, plot_figure)
    """
    global model_instances
    
    if model_name not in model_instances:
        print(f"Creating new model instance for {model_name}")
        model_instances[model_name] = DetectionModel(model_name=model_name, confidence=confidence_threshold, iou=0.45)
    else:
        print(f"Using existing model instance for {model_name}")
        model_instances[model_name].confidence = confidence_threshold
    
    class_ids = None
    if filter_classes:
        class_ids = []
        for class_str in filter_classes:
            try:
                # Extract ID from format "id: name"
                class_id = int(class_str.split(":")[0].strip())
                class_ids.append(class_id)
            except:
                continue
    
    # Execute detection
    result_image, result_text, stats = process_image(
        image, 
        model_instances[model_name], 
        confidence_threshold, 
        class_ids
    )
    
    # Format the statistics for better display
    formatted_stats = format_json_for_display(stats)
    
    if not stats or "class_statistics" not in stats or not stats["class_statistics"]:
        # Create the table
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No detection data available", 
                ha='center', va='center', fontsize=14, fontfamily='Arial')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plot_figure = fig
    else:
        # prepare visualization data
        viz_data = {
            "total_objects": stats.get("total_objects", 0),
            "average_confidence": stats.get("average_confidence", 0),
            "class_data": []
        }
        
        # get the color map
        color_mapper_instance = ColorMapper()
        
        # class data
        available_classes = dict(get_all_classes())
        for cls_name, cls_stats in stats.get("class_statistics", {}).items():
            # search class ID
            class_id = -1
            for id, name in available_classes.items():
                if name == cls_name:
                    class_id = id
                    break
            
            cls_data = {
                "name": cls_name,
                "class_id": class_id,
                "count": cls_stats.get("count", 0),
                "average_confidence": cls_stats.get("average_confidence", 0),
                "color": color_mapper_instance.get_color(class_id if class_id >= 0 else cls_name)
            }
            
            viz_data["class_data"].append(cls_data)
        
        # descending order
        viz_data["class_data"].sort(key=lambda x: x["count"], reverse=True)
        
        plot_figure = EvaluationMetrics.create_enhanced_stats_plot(viz_data)
    
    return result_image, result_text, formatted_stats, plot_figure


if __name__ == "__main__":
    import time
    
    demo = create_interface()
    demo.launch()
