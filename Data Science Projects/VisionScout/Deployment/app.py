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
        Tuple of results including lighting conditions
    """
    try:
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

        # Extract scene analysis info
        scene_analysis = stats.get("scene_analysis", {})

        scene_desc = scene_analysis.get("description", "No scene analysis available.")
        scene_desc = scene_desc.strip()

        # HTML format
        scene_desc_html = f"""
        <div id='scene-desc-container' style='width:100%; padding:20px; text-align:center; background-color:#f5f9fc; border-radius:8px; margin:10px auto; min-height:200px; max-height:none; overflow-y:auto;'>
            <div style='width:100%; text-align:center; margin:0 auto; font-family:Arial, sans-serif; font-size:14px; line-height:1.8;'>
                {scene_desc}
            </div>
        </div>
        """

        # Extract lighting conditions
        lighting_conditions = scene_analysis.get("lighting_conditions",
                                               {"time_of_day": "unknown", "confidence": 0.0})

        # 準備活動列表
        activities = scene_analysis.get("possible_activities", [])
        if not activities:
            activities_data = [["No activities detected"]]
        else:
            activities_data = [[activity] for activity in activities]

        # 準備安全注意事項列表
        safety_concerns = scene_analysis.get("safety_concerns", [])
        if not safety_concerns:
            safety_data = [["No safety concerns detected"]]
        else:
            safety_data = [[concern] for concern in safety_concerns]

        # 功能區域
        zones = scene_analysis.get("functional_zones", {})

        return result_image, result_text, formatted_stats, plot_figure, scene_desc, activities_data, safety_data, zones, lighting_conditions

    except Exception as e:
        # 添加錯誤處理，確保即使出錯也能返回有效的數據
        import traceback
        error_msg = f"Error processing image: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)

        # 創建一個簡單的錯誤圖
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error: {str(e)}",
                ha='center', va='center', fontsize=14, fontfamily='Arial', color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # 返回有效的默認值
        return None, error_msg, "{}", fig, "Error processing image", [["No activities"]], [["No safety concerns"]], {}, {"time_of_day": "unknown", "confidence": 0}

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
        # 主頁頂部的標題
        with gr.Group(elem_classes="app-header"):
              gr.HTML("""
                    <div style="text-align: center; width: 100%; padding: 2rem 0 3rem 0; background: linear-gradient(135deg, #f0f9ff, #e1f5fe);">
                        <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem; background: linear-gradient(90deg, #38b2ac, #4299e1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: bold; font-family: 'Arial', sans-serif;">VisionScout</h1>

                        <h2 style="color: #4A5568; font-size: 1.2rem; font-weight: 400; margin-top: 0.5rem; margin-bottom: 1.5rem; font-family: 'Arial', sans-serif;">Detect and identify objects in your images</h2>

                        <div style="display: flex; justify-content: center; gap: 10px; margin: 0.5rem 0;">
                            <div style="height: 3px; width: 80px; background: linear-gradient(90deg, #38b2ac, #4299e1);"></div>
                        </div>

                        <div style="display: flex; justify-content: center; gap: 25px; margin-top: 1.5rem;">
                            <div style="padding: 8px 15px; border-radius: 20px; background: rgba(66, 153, 225, 0.15); color: #2b6cb0; font-weight: 500; font-size: 0.9rem;">
                                <span style="margin-right: 6px;">🔍</span> Object Detection
                            </div>
                            <div style="padding: 8px 15px; border-radius: 20px; background: rgba(56, 178, 172, 0.15); color: #2b6cb0; font-weight: 500; font-size: 0.9rem;">
                                <span style="margin-right: 6px;">🌐</span> Scene Understanding
                            </div>
                            <div style="padding: 8px 15px; border-radius: 20px; background: rgba(66, 153, 225, 0.15); color: #2b6cb0; font-weight: 500; font-size: 0.9rem;">
                                <span style="margin-right: 6px;">📊</span> Visual Analysis
                            </div>
                        </div>

                        <div style="margin-top: 20px; padding: 10px 15px; background-color: rgba(255, 248, 230, 0.9); border-left: 3px solid #f6ad55; border-radius: 6px; max-width: 600px; margin-left: auto; margin-right: auto; text-align: left;">
                            <p style="margin: 0; font-size: 0.9rem; color: #805ad5; font-weight: 500;">
                                <span style="margin-right: 5px;">📱</span> iPhone users: HEIC images are not supported.
                                <a href="https://cloudconvert.com/heic-to-jpg" target="_blank" style="color: #3182ce; text-decoration: underline;">Convert HEIC to JPG here</a> before uploading.
                            </p>
                        </div>
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
                                lines=15,
                                max_lines=20,
                                elem_classes="wide-result-text",
                                elem_id="detection-details",
                                container=False,
                                scale=2,
                                min_width=600
                            )

                    # Scene Analysis
                    with gr.Tab("Scene Understanding", elem_classes="scene-understanding-tab"):
                        with gr.Group(elem_classes="result-details-box"):
                            gr.HTML("""
                                <div class="section-heading">Scene Analysis</div>
                                <details class="info-details" style="margin: 5px 0 15px 0;">
                                    <summary style="padding: 8px; background-color: #f0f7ff; border-radius: 6px; border-left: 3px solid #4299e1; font-weight: bold; cursor: pointer; color: #2b6cb0;">
                                        🔍 The AI Vision Scout Report: Click for important notes about this analysis
                                    </summary>
                                    <div style="margin-top: 8px; padding: 10px; background-color: #f8f9fa; border-radius: 6px; border: 1px solid #e2e8f0;">
                                        <p style="font-size: 13px; color: #718096; margin: 0;">
                                            <b>About this analysis:</b> This analysis is the model's best guess based on visible objects.
                                            Like human scouts, it sometimes gets lost or sees things that aren't there (but don't we all?).
                                            Consider this an educated opinion rather than absolute truth. For critical applications, always verify with human eyes! 🧐
                                        </p>
                                    </div>
                                </details>
                            """)

                            # 使用更適合長文本的容器
                            with gr.Group(elem_classes="scene-description-container"):
                                scene_description = gr.HTML(
                                        value="<div id='scene-desc-container'></div>",
                                        label="Scene Description"
                                    )

                            with gr.Row():
                                with gr.Column(scale=2):
                                    activities_list = gr.Dataframe(
                                        headers=["Activities"],
                                        datatype=["str"],
                                        col_count=1,
                                        row_count=5,
                                        elem_classes="full-width-element"  
                                    )

                                with gr.Column(scale=2):
                                    safety_list = gr.Dataframe(
                                        headers=["Safety Concerns"],
                                        datatype=["str"],
                                        col_count=1,
                                        row_count=5,
                                        elem_classes="full-width-element"  
                                    )

                            gr.HTML('<div class="section-heading">Functional Zones</div>')
                            zones_json = gr.JSON(label=None, elem_classes="json-box")

                            gr.HTML('<div class="section-heading">Lighting Conditions</div>')
                            lighting_info = gr.JSON(label=None, elem_classes="json-box")

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
                outputs=[
                    result_image, result_text, stats_json, plot_output,
                    scene_description, activities_list, safety_list, zones_json,
                    lighting_info
                ]
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
            "room_02.jpg",
            "street_02.jpg",
            "street_04.jpg"
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
