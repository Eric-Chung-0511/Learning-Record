import re
import os
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from typing import Dict, List, Any, Optional, Tuple
import cv2
from PIL import Image
import tempfile
import uuid
import spaces

from detection_model import DetectionModel
from color_mapper import ColorMapper
from evaluation_metrics import EvaluationMetrics
from style import Style
from image_processor import ImageProcessor
from video_processor import VideoProcessor
from llm_enhancer import LLMEnhancer

# Initialize Processors with LLM support
image_processor = ImageProcessor(use_llm=True, llm_model_path="meta-llama/Llama-3.2-3B-Instruct")
video_processor = VideoProcessor(image_processor)

# Helper Function
def get_all_classes():
    """Gets all available COCO classes."""
    # Try to get from a loaded model first
    if image_processor and image_processor.model_instances:
         for model_instance in image_processor.model_instances.values():
              if model_instance and model_instance.is_model_loaded:
                   try:
                        # Ensure class_names is a dict {id: name}
                        if isinstance(model_instance.class_names, dict):
                             return sorted([(int(idx), name) for idx, name in model_instance.class_names.items()])
                   except Exception as e:
                        print(f"Error getting class names from model: {e}")

    # Fallback to standard COCO (ensure keys are ints)
    default_classes = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
        6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
        11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
        16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
        22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
        27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
        32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
        36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
        51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
        57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
        62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
        67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
        72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
        77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }
    return sorted(default_classes.items())

@spaces.GPU
def handle_image_upload(image, model_name, confidence_threshold, filter_classes=None, use_llm=True):
    """Processes a single uploaded image."""
    print(f"Processing image with model: {model_name}, confidence: {confidence_threshold}, use_llm: {use_llm}")
    try:
        image_processor.use_llm = use_llm
        if hasattr(image_processor, 'scene_analyzer'):
            image_processor.scene_analyzer.use_llm = use_llm
            print(f"Updated existing scene_analyzer use_llm setting to: {use_llm}")

        class_ids_to_filter = None
        if filter_classes:
            class_ids_to_filter = []
            available_classes_dict = dict(get_all_classes())
            name_to_id = {name: id for id, name in available_classes_dict.items()}
            for class_str in filter_classes:
                class_name_or_id = class_str.split(":")[0].strip()
                class_id = -1
                try:
                    class_id = int(class_name_or_id)
                    if class_id not in available_classes_dict:
                        class_id = -1
                except ValueError:
                    if class_name_or_id in name_to_id:
                        class_id = name_to_id[class_name_or_id]
                    elif class_str in name_to_id: # Check full string "id: name"
                        class_id = name_to_id[class_str]

                if class_id != -1:
                    class_ids_to_filter.append(class_id)
                else:
                    print(f"Warning: Could not parse class filter: {class_str}")
            print(f"Filtering image results for class IDs: {class_ids_to_filter}")

        # Call the existing image processing logic
        result_image, result_text, stats = image_processor.process_image(
            image,
            model_name,
            confidence_threshold,
            class_ids_to_filter
        )

        # Format stats for JSON display
        formatted_stats = image_processor.format_json_for_display(stats)

        # Prepare visualization data for the plot
        plot_figure = None
        if stats and "class_statistics" in stats and stats["class_statistics"]:
            available_classes_dict = dict(get_all_classes())
            viz_data = image_processor.prepare_visualization_data(stats, available_classes_dict)
            if "error" not in viz_data:
                 plot_figure = EvaluationMetrics.create_enhanced_stats_plot(viz_data)
            else:
                 fig, ax = plt.subplots(figsize=(8, 6))
                 ax.text(0.5, 0.5, viz_data["error"], ha='center', va='center', fontsize=12)
                 ax.axis('off')
                 plot_figure = fig
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No detection data for plot", ha='center', va='center', fontsize=12)
            ax.axis('off')
            plot_figure = fig

        # Extract scene analysis info
        scene_analysis = stats.get("scene_analysis", {})
        scene_desc = scene_analysis.get("description", "Scene analysis requires detected objects.")
        # Ensure scene_desc is a string before adding HTML
        if not isinstance(scene_desc, str):
            scene_desc = str(scene_desc)

        def clean_description(desc):
            if not desc:
                return ""

            # 先過濾問答格式
            if "Questions:" in desc:
                desc = desc.split("Questions:")[0].strip()
            if "Answers:" in desc:
                desc = desc.split("Answers:")[0].strip()

            # 然後按行過濾代碼和其他非敘述內容
            lines = desc.split('\n')
            clean_lines = []
            skip_block = False

            for line in lines:
                # 檢測問題格式
                if re.match(r'^\d+\.\s+(What|How|Why|When|Where|Who|The)', line):
                    continue

                # 檢查需要跳過的行
                if line.strip().startswith(':param') or line.strip().startswith('"""'):
                    continue
                if line.strip().startswith("Exercise") or "class SceneDescriptionSystem" in line:
                    skip_block = True
                    continue
                if ('def generate_scene_description' in line or
                    'def enhance_scene_descriptions' in line or
                    'def __init__' in line):
                    skip_block = True
                    continue
                if line.strip().startswith('#TEST'):
                    skip_block = True
                    continue

                if skip_block and line.strip() == "":
                    skip_block = False

                # 如果不需要跳過
                if not skip_block:
                    clean_lines.append(line)

            cleaned_text = '\n'.join(clean_lines)

            # 如果清理後為空，返回原始描述的第一段作為保險
            if not cleaned_text.strip():
                paragraphs = [p.strip() for p in desc.split('\n\n') if p.strip()]
                if paragraphs:
                    return paragraphs[0]
                return desc

            return cleaned_text

        # 獲取和處理場景描述
        scene_analysis = stats.get("scene_analysis", {})
        print("Processing scene_analysis:", scene_analysis.keys())

        # 獲取原始描述
        scene_desc = scene_analysis.get("description", "Scene analysis requires detected objects.")
        if not isinstance(scene_desc, str):
            scene_desc = str(scene_desc)

        print(f"Original scene description (first 50 chars): {scene_desc[:50]}...")

        # 確保使用的是有效的描述
        clean_scene_desc = clean_description(scene_desc)
        print(f"Cleaned scene description (first 50 chars): {clean_scene_desc[:50]}...")

        # 即使清理後為空也確保顯示原始內容
        if not clean_scene_desc.strip():
            clean_scene_desc = scene_desc

        # 創建原始描述的HTML
        scene_desc_html = f"<div>{clean_scene_desc}</div>"

        # 獲取LLM增強描述並且確保設置默認值為空字符串而非 None，不然會有None type Error
        enhanced_description = scene_analysis.get("enhanced_description", "")
        if enhanced_description is None:
            enhanced_description = ""

        if not enhanced_description or not enhanced_description.strip():
            print("WARNING: LLM enhanced description is empty!")

        # 準備徽章和描述標籤
        llm_badge = ""
        description_to_show = ""

        if use_llm and enhanced_description:
            llm_badge = '<span style="display:inline-block; margin-left:8px; padding:3px 10px; border-radius:12px; background: linear-gradient(90deg, #38b2ac, #4299e1); color:white; font-size:0.7rem; font-weight:bold; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2);">LLM Enhanced</span>'
            description_to_show = enhanced_description
            # 在 Original Scene Analysis 折疊區顯示原始的描述
        else:
            llm_badge = '<span style="display:inline-block; margin-left:8px; padding:3px 10px; border-radius:12px; background-color:#718096; color:white; font-size:0.7rem; font-weight:bold;">Basic</span>'
            description_to_show = clean_scene_desc
            # 不使用 LLM 時，折疊區不顯示內容

        # 使用LLM敘述時會有徽章標籤在標題上
        scene_description_html = f'''
        <div>
            <div class="section-heading" style="font-size:1.2rem; margin-top:15px;">Scene Description {llm_badge}
                <span style="font-size:0.8rem; color:#666; font-weight:normal; display:block; margin-top:2px;">
                    {('(Enhanced by AI language model)' if use_llm and enhanced_description else '(Based on object detection)')}
                </span>
            </div>
            <div style="padding:15px; background-color:#ffffff; border-radius:8px; border:1px solid #e2e8f0; margin-bottom:20px; box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                {description_to_show}
            </div>
        </div>
        '''

        # 原始描述只在使用 LLM 且有增強描述時在折疊區顯示
        original_desc_visibility = "block" if use_llm and enhanced_description else "none"
        original_desc_html = f'''
        <div id="original_scene_analysis_accordion" style="display: {original_desc_visibility};">
            <div style="padding:15px; background-color:#f0f0f0; border-radius:8px; border:1px solid #e2e8f0;">
                {clean_scene_desc}
            </div>
        </div>
        '''

        # Prepare activities list
        activities_list = scene_analysis.get("possible_activities", [])
        if not activities_list:
            activities_list_data = [["No specific activities inferred"]] # Data for Dataframe
        else:
            activities_list_data = [[activity] for activity in activities_list]

        # Prepare safety concerns list
        safety_concerns_list = scene_analysis.get("safety_concerns", [])
        if not safety_concerns_list:
            safety_data = [["No safety concerns detected"]] # Data for Dataframe
        else:
            safety_data = [[concern] for concern in safety_concerns_list]

        zones = scene_analysis.get("functional_zones", {})
        lighting = scene_analysis.get("lighting_conditions", {"time_of_day": "unknown", "confidence": 0})

        # 如果描述為空，記錄警告
        if not clean_scene_desc.strip():
            print("WARNING: Scene description is empty after cleaning!")
        if not enhanced_description.strip():
            print("WARNING: LLM enhanced description is empty!")

        return (result_image, result_text, formatted_stats, plot_figure,
            scene_description_html, original_desc_html, 
            activities_list_data, safety_data, zones, lighting)

    except Exception as e:
        print(f"Error in handle_image_upload: {e}")
        import traceback
        error_msg = f"Error processing image: {str(e)}\n{traceback.format_exc()}"
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Processing Error", color="red", ha="center", va="center")
        ax.axis('off')
        # Ensure return structure matches outputs even on error
        return (None, error_msg, {}, fig, f"<div>Error: {str(e)}</div>", "Error",
            [["Error"]], [["Error"]], {}, {"time_of_day": "error", "confidence": 0})

def download_video_from_url(video_url, max_duration_minutes=10):
    """
    Downloads a video from a YouTube URL and returns the local path to the downloaded file.

    Args:
        video_url (str): URL of the YouTube video to download
        max_duration_minutes (int): Maximum allowed video duration in minutes

    Returns:
        tuple: (Path to the downloaded video file or None, Error message or None)
    """
    try:
        # Create a temporary directory to store the video
        temp_dir = tempfile.gettempdir()
        output_filename = f"downloaded_{uuid.uuid4().hex}.mp4"
        output_path = os.path.join(temp_dir, output_filename)

        # Check if it's a YouTube URL
        if "youtube.com" in video_url or "youtu.be" in video_url:
            # Import yt-dlp here to avoid dependency if not needed
            import yt_dlp

            # Setup yt-dlp options
            ydl_opts = {
                'format': 'best[ext=mp4]/best',  # Best quality MP4 or best available format
                'outtmpl': output_path,
                'noplaylist': True,
                'quiet': False,  # Set to True to reduce output
                'no_warnings': False,
            }

            # First extract info to check duration
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print(f"Extracting info from YouTube URL: {video_url}")
                info_dict = ydl.extract_info(video_url, download=False)

                # Check if video exists
                if not info_dict:
                    return None, "Could not retrieve video information. Please check the URL."

                video_title = info_dict.get('title', 'Unknown Title')
                duration = info_dict.get('duration', 0)

                print(f"Video title: {video_title}")
                print(f"Video duration: {duration} seconds")

                # Check video duration
                if duration > max_duration_minutes * 60:
                    return None, f"Video is too long ({duration} seconds). Maximum duration is {max_duration_minutes} minutes."

                # Download the video
                print(f"Downloading YouTube video: {video_title}")
                ydl.download([video_url])

            # Verify the file exists and has content
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                return None, "Download failed: Empty or missing file."

            print(f"Successfully downloaded video to: {output_path}")
            return output_path, None
        else:
            return None, "Only YouTube URLs are supported at this time. Please enter a valid YouTube URL."

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error downloading video: {e}\n{error_details}")
        return None, f"Error downloading video: {str(e)}"


@spaces.GPU
def handle_video_upload(video_input, video_url, input_type, model_name, confidence_threshold, process_interval):
    """Handles video upload or URL input and calls the VideoProcessor."""

    print(f"Received video request: input_type={input_type}")
    video_path = None

    # Handle based on input type
    if input_type == "upload" and video_input:
        print(f"Processing uploaded video file")
        video_path = video_input
    elif input_type == "url" and video_url:
        print(f"Processing video from URL: {video_url}")
        # Download video from URL
        video_path, error_message = download_video_from_url(video_url)
        if error_message:
            error_html = f"<div class='video-summary-content-wrapper'><pre>{error_message}</pre></div>"
            return None, error_html, {"error": error_message}
    else:
        print("No valid video input provided.")
        return None, "<div class='video-summary-content-wrapper'><pre>Please upload a video file or provide a valid video URL.</pre></div>", {}

    print(f"Starting video processing with: model={model_name}, confidence={confidence_threshold}, interval={process_interval}")
    try:
        # Call the VideoProcessor method
        output_video_path, summary_text, stats_dict = video_processor.process_video_file(
            video_path=video_path,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            process_interval=int(process_interval) # Ensure interval is int
        )
        print(f"Video processing function returned: path={output_video_path}, summary length={len(summary_text)}")

        # Wrap processing summary in HTML tags for consistent styling with scene understanding page
        summary_html = f"<div class='video-summary-content-wrapper'><pre>{summary_text}</pre></div>"

        # Format statistics for better display
        formatted_stats = {}
        if stats_dict and isinstance(stats_dict, dict):
            formatted_stats = stats_dict

        return output_video_path, summary_html, formatted_stats

    except Exception as e:
        print(f"Error in handle_video_upload: {e}")
        import traceback
        error_msg = f"Error processing video: {str(e)}\n{traceback.format_exc()}"
        error_html = f"<div class='video-summary-content-wrapper'><pre>{error_msg}</pre></div>"
        return None, error_html, {"error": str(e)}


# Create Gradio Interface
def create_interface():
    """Creates the Gradio interface with Tabs."""
    css = Style.get_css()
    available_models = DetectionModel.get_available_models()
    model_choices = [model["model_file"] for model in available_models]
    class_choices_formatted = [f"{id}: {name}" for id, name in get_all_classes()] # Use formatted choices

    with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="teal", secondary_hue="blue")) as demo:

        # Header
        with gr.Group(elem_classes="app-header"):
              gr.HTML("""
                    <div style="text-align: center; width: 100%; padding: 2rem 0 3rem 0; background: linear-gradient(135deg, #f0f9ff, #e1f5fe);">
                        <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem; background: linear-gradient(90deg, #38b2ac, #4299e1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: bold; font-family: 'Arial', sans-serif;">VisionScout</h1>
                        <h2 style="color: #4A5568; font-size: 1.2rem; font-weight: 400; margin-top: 0.5rem; margin-bottom: 1.5rem; font-family: 'Arial', sans-serif;">Object Detection and Scene Understanding</h2>
                        <div style="display: flex; justify-content: center; gap: 10px; margin: 0.5rem 0;"><div style="height: 3px; width: 80px; background: linear-gradient(90deg, #38b2ac, #4299e1);"></div></div>
                        <div style="display: flex; justify-content: center; gap: 25px; margin-top: 1.5rem;">
                            <div style="padding: 8px 15px; border-radius: 20px; background: rgba(66, 153, 225, 0.15); color: #2b6cb0; font-weight: 500; font-size: 0.9rem;"><span style="margin-right: 6px;">🖼️</span> Image Analysis</div>
                            <div style="padding: 8px 15px; border-radius: 20px; background: rgba(56, 178, 172, 0.15); color: #2b6cb0; font-weight: 500; font-size: 0.9rem;"><span style="margin-right: 6px;">🎬</span> Video Analysis</div>
                        </div>
                         <div style="margin-top: 20px; padding: 10px 15px; background-color: rgba(255, 248, 230, 0.9); border-left: 3px solid #f6ad55; border-radius: 6px; max-width: 600px; margin-left: auto; margin-right: auto; text-align: left;">
                             <p style="margin: 0; font-size: 0.9rem; color: #805ad5; font-weight: 500;">
                                 <span style="margin-right: 5px;">📱</span> iPhone users: HEIC images may not be supported.
                                 <a href="https://cloudconvert.com/heic-to-jpg" target="_blank" style="color: #3182ce; text-decoration: underline;">Convert HEIC to JPG</a> before uploading if needed.
                             </p>
                         </div>
                    </div>
                """)

        # Main Content with Tabs
        with gr.Tabs(elem_classes="tabs"):

            # Tab 1: Image Processing
            with gr.Tab("Image Processing"):
                current_image_model = gr.State("yolov8m.pt") # State for image model selection
                with gr.Row(equal_height=False): # Allow columns to have different heights
                    # Left Column: Image Input & Controls
                    with gr.Column(scale=4, elem_classes="input-panel"):
                        with gr.Group():
                            gr.HTML('<div class="section-heading">Upload Image</div>')
                            image_input = gr.Image(type="pil", label="Upload an image", elem_classes="upload-box")

                            with gr.Accordion("Image Analysis Settings", open=False):
                                image_model_dropdown = gr.Dropdown(
                                    choices=model_choices,
                                    value="yolov8m.pt", # Default for images
                                    label="Select Model",
                                    info="Choose speed vs. accuracy (n=fast, m=balanced, x=accurate)"
                                )
                                # Display model info
                                image_model_info = gr.Markdown(DetectionModel.get_model_description("yolov8m.pt"))

                                image_confidence = gr.Slider(
                                    minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                                    label="Confidence Threshold",
                                    info="Minimum confidence for displaying a detected object"
                                )

                                use_llm = gr.Checkbox(
                                    label="Use LLM for enhanced scene descriptions",
                                    value=True,
                                    info="Provides more detailed and natural language descriptions (may increase processing time)"
                                )

                                with gr.Accordion("Filter Classes", open=False):
                                     gr.HTML('<div class="section-heading" style="font-size: 1rem;">Common Categories</div>')
                                     with gr.Row():
                                         people_btn = gr.Button("People", size="sm")
                                         vehicles_btn = gr.Button("Vehicles", size="sm")
                                         animals_btn = gr.Button("Animals", size="sm")
                                         objects_btn = gr.Button("Common Objects", size="sm")
                                     image_class_filter = gr.Dropdown(
                                         choices=class_choices_formatted, # Use formatted choices
                                         multiselect=True,
                                         label="Select Classes to Display",
                                         info="Leave empty to show all detected objects"
                                     )

                        image_detect_btn = gr.Button("Analyze Image", variant="primary", elem_classes="detect-btn")

                        with gr.Group(elem_classes="how-to-use"):
                             gr.HTML('<div class="section-heading">How to Use (Image)</div>')
                             gr.Markdown("""
                                    1. Upload an image or use the camera
                                    2. (Optional) Adjust settings like confidence threshold or model size (n, m=balanced, x=accurate)
                                    3. In Analysis Settings, you can uncheck "Use LLM for enhanced scene descriptions" if you prefer faster processing
                                    4. Optionally filter to specific object classes
                                    5. Click **Detect Objects** button
                                """)
                        # Image Examples
                        gr.Examples(
                            examples=[
                                "room_01.jpg",
                                "room_02.jpg",
                                "street_02.jpg",
                                "street_04.jpg"
                                ],
                            inputs=image_input,
                            label="Example Images"
                         )

                    # Right Column: Image Results
                    with gr.Column(scale=6, elem_classes="output-panel"):
                        with gr.Tabs(elem_classes="tabs"):
                            with gr.Tab("Detection Result"):
                                image_result_image = gr.Image(type="pil", label="Detection Result")
                                gr.HTML('<div class="section-heading">Detection Details</div>')
                                image_result_text = gr.Textbox(label=None, lines=10, elem_id="detection-details", container=False)

                            with gr.Tab("Scene Understanding"):
                                gr.HTML('<div class="section-heading">Scene Analysis</div>')
                                gr.HTML("""
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

                                gr.HTML('''
                                    <div style="margin-top: 5px; padding: 6px 10px; background-color: #f0f9ff; border-radius: 4px; border-left: 3px solid #63b3ed; font-size: 12px; margin-bottom: 10px;">
                                        <p style="margin: 0; color: #4a5568;">
                                            <b>Note:</b> AI descriptions may vary slightly with each generation, reflecting the creative nature of AI. This is similar to how a person might use different words each time they describe the same image. Processing time may be longer during first use or when analyzing complex scenes, as the LLM enhancement requires additional computational resources.
                                        </p>
                                    </div>
                                    ''')
                                image_scene_description_html = gr.HTML(label=None, elem_id="scene_analysis_description_text")  
                                
                                # 使用LLM增強敘述時也會顯示原本敘述內容
                                with gr.Accordion("Original Scene Analysis", open=False, elem_id="original_scene_analysis_accordion"):
                                    image_llm_description = gr.HTML(label=None, elem_id="original_scene_description_text")

                                with gr.Row():
                                     with gr.Column(scale=1):
                                         gr.HTML('<div class="section-heading" style="font-size:1rem; text-align:left;">Possible Activities</div>')
                                         image_activities_list = gr.Dataframe(headers=["Activity"], datatype=["str"], row_count=5, col_count=1, wrap=True)

                                     with gr.Column(scale=1):
                                         gr.HTML('<div class="section-heading" style="font-size:1rem; text-align:left;">Safety Concerns</div>')
                                         image_safety_list = gr.Dataframe(headers=["Concern"], datatype=["str"], row_count=5, col_count=1, wrap=True)

                                gr.HTML('<div class="section-heading">Functional Zones</div>')
                                image_zones_json = gr.JSON(label=None, elem_classes="json-box")

                                gr.HTML('<div class="section-heading">Lighting Conditions</div>')
                                image_lighting_info = gr.JSON(label=None, elem_classes="json-box")

                            with gr.Tab("Statistics"):
                                with gr.Row():
                                    with gr.Column(scale=3, elem_classes="plot-column"):
                                        gr.HTML('<div class="section-heading">Object Distribution</div>')
                                        image_plot_output = gr.Plot(label=None, elem_classes="large-plot-container")
                                    with gr.Column(scale=2, elem_classes="stats-column"):
                                        gr.HTML('<div class="section-heading">Detection Statistics</div>')
                                        image_stats_json = gr.JSON(label=None, elem_classes="enhanced-json-display")

            # Tab 2: Video Processing
            with gr.Tab("Video Processing"):
                with gr.Row(equal_height=False):
                    # Left Column: Video Input & Controls
                    with gr.Column(scale=4, elem_classes="input-panel"):
                        with gr.Group():
                            gr.HTML('<div class="section-heading">Video Input</div>')

                            # Add input type selection
                            video_input_type = gr.Radio(
                                ["upload", "url"],
                                label="Input Method",
                                value="upload",
                                info="Choose how to provide the video"
                            )

                            # File upload (will be shown/hidden based on selection)
                            with gr.Group(elem_id="upload-video-group"):
                                video_input = gr.Video(
                                    label="Upload a video file (MP4, AVI, MOV)",
                                    sources=["upload"],
                                    visible=True
                                )

                            # URL input (will be shown/hidden based on selection)
                            with gr.Group(elem_id="url-video-group"):
                                video_url_input = gr.Textbox(
                                    label="Enter video URL (YouTube or direct video link)",
                                    placeholder="https://www.youtube.com/watch?v=...",
                                    visible=False,
                                    elem_classes="custom-video-url-input"
                                )
                                gr.HTML("""
                                    <div style="padding: 8px; margin-top: 5px; background-color: #fff8f8; border-radius: 4px; border-left: 3px solid #f87171; font-size: 12px;">
                                        <p style="margin: 0; color: #4b5563;">
                                            Note: Currently only YouTube URLs are supported. Maximum video duration is 10 minutes. Due to YouTube's anti-bot protection, some videos may not be downloadable. For protected videos, please upload a local video file instead.
                                        </p>
                                    </div>
                                """)

                            with gr.Accordion("Video Analysis Settings", open=True):
                                video_model_dropdown = gr.Dropdown(
                                    choices=model_choices,
                                    value="yolov8n.pt", # Default 'n' for video
                                    label="Select Model (Video)",
                                    info="Faster models (like 'n') are recommended"
                                )
                                video_confidence = gr.Slider(
                                    minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                                    label="Confidence Threshold (Video)"
                                )
                                video_process_interval = gr.Slider(
                                    minimum=1, maximum=60, value=10, step=1, # Allow up to 60 frame interval
                                    label="Processing Interval (Frames)",
                                    info="Analyze every Nth frame (higher value = faster)"
                                )
                        video_process_btn = gr.Button("Process Video", variant="primary", elem_classes="detect-btn")

                        with gr.Group(elem_classes="how-to-use"):
                            gr.HTML('<div class="section-heading">How to Use (Video)</div>')
                            gr.Markdown("""
                            1. Choose your input method: Upload a file or enter a URL.
                            2. Adjust settings if needed (using a faster model and larger interval is recommended for longer videos).
                            3. Click "Process Video". **Processing can take a significant amount of time.**
                            4. The annotated video and summary will appear on the right when finished.
                            """)

                        # Add video examples
                        gr.HTML('<div class="section-heading">Example Videos</div>')
                        gr.HTML("""
                            <div style="padding: 10px; background-color: #f0f7ff; border-radius: 6px; margin-bottom: 15px;">
                                <p style="font-size: 14px; color: #4A5568; margin: 0;">
                                    Upload any video containing objects that YOLO can detect. For testing, find sample videos
                                    <a href="https://www.pexels.com/search/videos/street/" target="_blank" style="color: #3182ce; text-decoration: underline;">here</a>.
                                </p>
                            </div>
                        """)

                    # Right Column: Video Results
                    with gr.Column(scale=6, elem_classes="output-panel video-result-panel"):
                        gr.HTML("""
                            <div class="section-heading">Video Result</div>
                            <details class="info-details" style="margin: 5px 0 15px 0;">
                                <summary style="padding: 8px; background-color: #f0f7ff; border-radius: 6px; border-left: 3px solid #4299e1; font-weight: bold; cursor: pointer; color: #2b6cb0;">
                                    🎬 Video Processing Notes
                                </summary>
                                <div style="margin-top: 8px; padding: 10px; background-color: #f8f9fa; border-radius: 6px; border: 1px solid #e2e8f0;">
                                    <p style="font-size: 13px; color: #718096; margin: 0;">
                                        The processed video includes bounding boxes around detected objects. For longer videos,
                                        consider using a faster model (like YOLOv8n) and a higher frame interval to reduce processing time.
                                    </p>
                                </div>
                            </details>
                        """)
                        video_output = gr.Video(label="Processed Video", elem_classes="video-output-container") # Output for the processed video file

                        gr.HTML('<div class="section-heading">Processing Summary</div>')
                        # 使用HTML顯示影片的摘要
                        video_summary_text = gr.HTML(
                            label=None,
                            elem_id="video-summary-html-output"
                        )

                        gr.HTML('<div class="section-heading">Aggregated Statistics</div>')
                        video_stats_json = gr.JSON(label=None, elem_classes="video-stats-display") # Display statistics

        # Event Listeners
        # Image Model Change Handler
        image_model_dropdown.change(
            fn=lambda model: (model, DetectionModel.get_model_description(model)),
            inputs=[image_model_dropdown],
            outputs=[current_image_model, image_model_info] # Update state and description
        )

        # Image Filter Buttons
        available_classes_list = get_all_classes() # Get list of (id, name)
        people_classes_ids = [0]
        vehicles_classes_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        animals_classes_ids = list(range(14, 24))
        common_objects_ids = [39, 41, 42, 43, 44, 45, 56, 57, 60, 62, 63, 67, 73] # Bottle, cup, fork, knife, spoon, bowl, chair, couch, table, tv, laptop, phone, book

        people_btn.click(lambda: [f"{id}: {name}" for id, name in available_classes_list if id in people_classes_ids], outputs=image_class_filter)
        vehicles_btn.click(lambda: [f"{id}: {name}" for id, name in available_classes_list if id in vehicles_classes_ids], outputs=image_class_filter)
        animals_btn.click(lambda: [f"{id}: {name}" for id, name in available_classes_list if id in animals_classes_ids], outputs=image_class_filter)
        objects_btn.click(lambda: [f"{id}: {name}" for id, name in available_classes_list if id in common_objects_ids], outputs=image_class_filter)

        video_input_type.change(
            fn=lambda input_type: [
                # Show/hide file upload
                gr.update(visible=(input_type == "upload")),
                # Show/hide URL input
                gr.update(visible=(input_type == "url"))
            ],
            inputs=[video_input_type],
            outputs=[video_input, video_url_input]
        )

        image_detect_btn.click(
            fn=handle_image_upload,
            inputs=[image_input, image_model_dropdown, image_confidence, image_class_filter, use_llm],
            outputs=[
                image_result_image, image_result_text, image_stats_json, image_plot_output,
                image_scene_description_html, image_llm_description, image_activities_list, image_safety_list, image_zones_json,
                image_lighting_info
            ]
        )

        video_process_btn.click(
            fn=handle_video_upload,
            inputs=[
                video_input,
                video_url_input,
                video_input_type,
                video_model_dropdown,
                video_confidence,
                video_process_interval
            ],
            outputs=[video_output, video_summary_text, video_stats_json]
        )

        # Footer
        gr.HTML("""
             <div class="footer" style="padding: 25px 0; text-align: center; background: linear-gradient(to right, #f5f9fc, #e1f5fe); border-top: 1px solid #e2e8f0; margin-top: 30px;">
                 <div style="margin-bottom: 15px;">
                     <p style="font-size: 14px; color: #4A5568; margin: 5px 0;">Powered by YOLOv8, CLIP, Meta Llama3.2 and Ultralytics • Created with Gradio</p>
                 </div>
                 <div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-top: 15px;">
                     <p style="font-family: 'Arial', sans-serif; font-size: 14px; font-weight: 500; letter-spacing: 2px; background: linear-gradient(90deg, #38b2ac, #4299e1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; text-transform: uppercase; display: inline-block;">EXPLORE THE CODE →</p>
                     <a href="https://github.com/Eric-Chung-0511/Learning-Record/tree/main/Data%20Science%20Projects/VisionScout" target="_blank" style="text-decoration: none;">
                         <img src="https://img.shields.io/badge/GitHub-VisionScout-4299e1?logo=github&style=for-the-badge">
                     </a>
                 </div>
             </div>
         """)

    return demo


if __name__ == "__main__":
    demo_interface = create_interface()

    demo_interface.launch()
