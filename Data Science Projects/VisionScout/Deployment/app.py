import os
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from typing import Dict, List, Any, Optional, Tuple
import spaces

from detection_model import DetectionModel
from color_mapper import ColorMapper
from evaluation_metrics import EvaluationMetrics
from style import Style
from image_processor import ImageProcessor

# Initialize image processor
image_processor = ImageProcessor()

def get_all_classes():
    """
    Get all available COCO classes from the currently active model or fallback to standard COCO classes

    Returns:
        List of tuples (class_id, class_name)
    """
    # Try to get class names from any loaded model
    for model_name, model_instance in image_processor.model_instances.items():
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
    result_image, result_text, stats = image_processor.process_image(
        image,
        model_name,
        confidence_threshold,
        class_ids
    )

    # Format the statistics for better display
    formatted_stats = image_processor.format_json_for_display(stats)

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
        # Prepare visualization data
        available_classes = dict(get_all_classes())
        viz_data = image_processor.prepare_visualization_data(stats, available_classes)
        
        # Create plot
        plot_figure = EvaluationMetrics.create_enhanced_stats_plot(viz_data)

    return result_image, result_text, formatted_stats, plot_figure

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

        # 主要內容區 
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

                        # details summary
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
            fn=process_and_plot,
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

        # Footer
        gr.HTML("""
            <div class="footer">
                <p>Powered by YOLOv8 and Ultralytics • Created with Gradio</p>
                <p>Model can detect 80 different classes of objects</p>
            </div>
        """)

    return demo

if __name__ == "__main__":
    import time

    demo = create_interface()
    demo.launch()
