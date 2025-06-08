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
import time
import traceback
import spaces

from detection_model import DetectionModel
from color_mapper import ColorMapper
from evaluation_metrics import EvaluationMetrics
from style import Style
from image_processor import ImageProcessor
from video_processor import VideoProcessor
from llm_enhancer import LLMEnhancer
from ui_manager import UIManager

# Initialize Processors with LLM support
image_processor = None
video_processor = None
ui_manager = None

def initialize_processors():
    """
    Initialize the image and video processors with LLM support.

    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global image_processor, video_processor

    try:
        print("Attempting to initialize ImageProcessor with LLM support...")
        image_processor = ImageProcessor(use_llm=True, llm_model_path="meta-llama/Llama-3.2-3B-Instruct")
        print("ImageProcessor initialized successfully with LLM")

        # 檢查狀態
        if hasattr(image_processor, 'scene_analyzer'):
            if image_processor.scene_analyzer is not None:
                print(f"scene_analyzer initialized: {type(image_processor.scene_analyzer)}")
                if hasattr(image_processor.scene_analyzer, 'use_llm'):
                    print(f"scene_analyzer.use_llm available: {image_processor.scene_analyzer.use_llm}")
            else:
                print("WARNING: scene_analyzer is None after initialization")
        else:
            print("WARNING: scene_analyzer attribute not found in image_processor")

        # 初始化獨立的VideoProcessor
        video_processor = VideoProcessor()
        print("VideoProcessor initialized successfully as independent module")
        return True

    except Exception as e:
        print(f"Error initializing processors with LLM: {e}")
        import traceback
        traceback.print_exc()

        # Create fallback processor without LLM
        try:
            print("Attempting fallback initialization without LLM...")
            image_processor = ImageProcessor(use_llm=False, enable_places365=False)
            video_processor = VideoProcessor()  
            print("Fallback processors initialized successfully without LLM and Places365")
            return True

        except Exception as fallback_error:
            print(f"Fatal error: Cannot initialize processors: {fallback_error}")
            import traceback
            traceback.print_exc()
            image_processor = None
            video_processor = None
            return False

def initialize_ui_manager():
    """
    Initialize the UI manager and set up references to processors.

    Returns:
        UIManager: Initialized UI manager instance
    """
    global ui_manager, image_processor

    ui_manager = UIManager()

    # Set image processor reference for dynamic class retrieval
    if image_processor:
        ui_manager.set_image_processor(image_processor)

    return ui_manager

@spaces.GPU(duration=180)
def handle_image_upload(image, model_name, confidence_threshold, filter_classes=None, use_llm=True, enable_landmark=True):
    """
    Processes a single uploaded image.

    Args:
        image: PIL Image object
        model_name: Name of the YOLO model to use
        confidence_threshold: Confidence threshold for detections
        filter_classes: List of class names/IDs to filter
        use_llm: Whether to use LLM for enhanced descriptions
        enable_landmark: Whether to enable landmark detection

    Returns:
        Tuple: (result_image, result_text, formatted_stats, plot_figure,
                scene_description_html, original_desc_html, activities_list_data,
                safety_data, zones, lighting)
    """
    # Enhanced safety check for image_processor
    if image_processor is None:
        error_msg = "Image processor is not initialized. Please restart the application or check system dependencies."
        print(f"ERROR: {error_msg}")

        # Create error plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Initialization Error\nProcessor Not Available",
                color="red", ha="center", va="center", fontsize=14, fontweight="bold")
        ax.axis('off')

        return (None, error_msg, {}, fig, f"<div style='color: red; font-weight: bold;'>Error: {error_msg}</div>",
                "<div style='color: red;'>Error: System not initialized</div>",
                [["System Error"]], [["System Error"]], {}, {"time_of_day": "error", "confidence": 0})

    # Additional safety check for processor attributes
    if not hasattr(image_processor, 'use_llm'):
        error_msg = "Image processor is corrupted. Missing required attributes."
        print(f"ERROR: {error_msg}")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Processor Error\nCorrupted State",
                color="red", ha="center", va="center", fontsize=14, fontweight="bold")
        ax.axis('off')

        return (None, error_msg, {}, fig, f"<div style='color: red; font-weight: bold;'>Error: {error_msg}</div>",
                "<div style='color: red;'>Error: Processor corrupted</div>",
                [["Processor Error"]], [["Processor Error"]], {}, {"time_of_day": "error", "confidence": 0})

    print(f"DIAGNOSTIC: Image upload handled with enable_landmark={enable_landmark}, use_llm={use_llm}")
    print(f"Processing image with model: {model_name}, confidence: {confidence_threshold}, use_llm: {use_llm}, enable_landmark: {enable_landmark}")

    try:
        image_processor.use_llm = use_llm

        # 確保 scene_analyzer 不是 None
        if hasattr(image_processor, 'scene_analyzer') and image_processor.scene_analyzer is not None:
            if hasattr(image_processor.scene_analyzer, 'use_llm'):
                image_processor.scene_analyzer.use_llm = use_llm
                print(f"Updated existing scene_analyzer use_llm setting to: {use_llm}")

            # 檢查並設置 landmark detection
            if hasattr(image_processor.scene_analyzer, 'use_landmark_detection'):
                # 設置所有相關標記
                image_processor.scene_analyzer.use_landmark_detection = enable_landmark
                image_processor.scene_analyzer.enable_landmark = enable_landmark

                # 確保處理器也設置了這選項(檢測地標用)
                image_processor.enable_landmark = enable_landmark

                # 檢查並設置更深層次的組件
                if hasattr(image_processor.scene_analyzer, 'scene_describer') and image_processor.scene_analyzer.scene_describer is not None:
                    image_processor.scene_analyzer.scene_describer.enable_landmark = enable_landmark

                # 檢查並設置CLIP Analyzer
                if hasattr(image_processor.scene_analyzer, 'clip_analyzer') and image_processor.scene_analyzer.clip_analyzer is not None:
                    if hasattr(image_processor.scene_analyzer.clip_analyzer, 'enable_landmark'):
                        image_processor.scene_analyzer.clip_analyzer.enable_landmark = enable_landmark

                # 檢查並設置LLM方面
                if hasattr(image_processor.scene_analyzer, 'llm_enhancer') and image_processor.scene_analyzer.llm_enhancer is not None:
                    if hasattr(image_processor.scene_analyzer.llm_enhancer, 'enable_landmark'):
                        image_processor.scene_analyzer.llm_enhancer.enable_landmark = enable_landmark
                        print(f"Updated LLM enhancer enable_landmark to: {enable_landmark}")

                print(f"Updated all landmark detection settings to: {enable_landmark}")
        else:
            print("WARNING: scene_analyzer is None or not available")
            if hasattr(image_processor, 'enable_landmark'):
                image_processor.enable_landmark = enable_landmark

                # 設置更深層次的組別
                if hasattr(image_processor.scene_analyzer, 'scene_describer'):
                    image_processor.scene_analyzer.scene_describer.enable_landmark = enable_landmark

                # 設置CLIP分析器上的標記
                if hasattr(image_processor.scene_analyzer, 'clip_analyzer'):
                    if hasattr(image_processor.scene_analyzer.clip_analyzer, 'enable_landmark'):
                        image_processor.scene_analyzer.clip_analyzer.enable_landmark = enable_landmark

                # 如果有LLM增強器，也設置它
                if hasattr(image_processor.scene_analyzer, 'llm_enhancer'):
                    image_processor.scene_analyzer.llm_enhancer.enable_landmark = enable_landmark
                    print(f"Updated LLM enhancer enable_landmark to: {enable_landmark}")

                print(f"Updated all landmark detection settings to: {enable_landmark}")

        class_ids_to_filter = None
        if filter_classes:
            class_ids_to_filter = []
            available_classes_dict = dict(ui_manager.get_all_classes())
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
        print(f"DEBUG: app.py 傳遞 enable_landmark={enable_landmark} 到 process_image")
        result_image, result_text, stats = image_processor.process_image(
            image,
            model_name,
            confidence_threshold,
            class_ids_to_filter,
            enable_landmark
        )

        # Format stats for JSON display
        formatted_stats = image_processor.format_json_for_display(stats)

        # Prepare visualization data for the plot
        plot_figure = None
        if stats and "class_statistics" in stats and stats["class_statistics"]:
            available_classes_dict = dict(ui_manager.get_all_classes())
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

        # determine original description
        clean_scene_desc = clean_description(scene_desc)
        print(f"Cleaned scene description (first 50 chars): {clean_scene_desc[:50]}...")

        if not clean_scene_desc.strip():
            clean_scene_desc = scene_desc

        scene_desc_html = f"<div>{clean_scene_desc}</div>"

        # 獲取LLM增強描述並且確保設置默認值為空字符串而非 None，不然會有None type Error
        enhanced_description = scene_analysis.get("enhanced_description", "")
        if enhanced_description is None:
            enhanced_description = ""

        if not enhanced_description or not enhanced_description.strip():
            print("WARNING: LLM enhanced description is empty!")

        # bedge & label
        llm_badge = ""
        description_to_show = ""

        # 在 Original Scene Analysis 折疊區顯示原始的描述
        if use_llm and enhanced_description:
            llm_badge = '<span style="display:inline-block; margin-left:8px; padding:3px 10px; border-radius:12px; background: linear-gradient(90deg, #38b2ac, #4299e1); color:white; font-size:0.7rem; font-weight:bold; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2);">LLM Enhanced</span>'
            description_to_show = enhanced_description

        else:
            llm_badge = '<span style="display:inline-block; margin-left:8px; padding:3px 10px; border-radius:12px; background-color:#718096; color:white; font-size:0.7rem; font-weight:bold;">Basic</span>'
            description_to_show = clean_scene_desc

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

        # 原始描述只在使用 LLM 且有增強敘述時會在折疊區顯示
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

def generate_basic_video_summary(analysis_results: Dict) -> str:
    """
    生成基本的視頻統計摘要
    
    Args:
        analysis_results (Dict): 新的分析結果結構
        
    Returns:
        str: 詳細的統計摘要
    """
    summary_lines = ["=== Video Analysis Summary ===", ""]
    
    # process info
    processing_info = analysis_results.get("processing_info", {})
    duration = processing_info.get("video_duration_seconds", 0)
    total_frames = processing_info.get("total_frames", 0)
    analyzed_frames = processing_info.get("frames_analyzed", 0)
    
    summary_lines.extend([
        f"Video Duration: {duration:.1f} seconds ({total_frames} total frames)",
        f"Frames Analyzed: {analyzed_frames} frames (every {processing_info.get('processing_interval', 1)} frames)",
        ""
    ])
    
    # object detected summary
    object_summary = analysis_results.get("object_summary", {})
    total_objects = object_summary.get("total_unique_objects_detected", 0)
    object_types = object_summary.get("object_types_found", 0)
    
    summary_lines.extend([
        f"Objects Detected: {total_objects} total objects across {object_types} categories",
        ""
    ])
    
    # detailed counting number
    detailed_counts = object_summary.get("detailed_counts", {})
    if detailed_counts:
        summary_lines.extend([
            "Object Breakdown:",
            *[f"  • {count} {name}(s)" for name, count in detailed_counts.items()],
            ""
        ])
    
    # 實用分析摘要
    practical_analytics = analysis_results.get("practical_analytics", {})
    
    # 物體密度分析
    density_info = practical_analytics.get("object_density", {})
    if density_info:
        objects_per_min = density_info.get("objects_per_minute", 0)
        peak_periods = density_info.get("peak_activity_periods", [])
        summary_lines.extend([
            f"Activity Level: {objects_per_min:.1f} objects per minute",
            f"Peak Activity Periods: {len(peak_periods)} identified",
            ""
        ])
    
    # 場景適合性
    scene_info = practical_analytics.get("scene_appropriateness", {})
    if scene_info.get("scene_detected", False):
        scene_name = scene_info.get("scene_name", "unknown")
        appropriateness = scene_info.get("appropriateness_score", 0)
        summary_lines.extend([
            f"Scene Type: {scene_name}",
            f"Object-Scene Compatibility: {appropriateness:.1%}",
            ""
        ])
    
    # 品質指標
    quality_info = practical_analytics.get("quality_metrics", {})
    if quality_info:
        quality_grade = quality_info.get("quality_grade", "unknown")
        overall_confidence = quality_info.get("overall_confidence", 0)
        summary_lines.extend([
            f"Detection Quality: {quality_grade.title()} (avg confidence: {overall_confidence:.3f})",
            ""
        ])
    
    summary_lines.append(f"Processing completed in {processing_info.get('processing_time_seconds', 0):.1f} seconds.")
    
    return "\n".join(summary_lines)

@spaces.GPU
def handle_video_upload(video_input, video_url, input_type, model_name, 
                       confidence_threshold, process_interval):
    """
    處理影片上傳的函數
    
    Args:
        video_input: 上傳的視頻文件
        video_url: 視頻URL（如果使用URL輸入）
        input_type: 輸入類型（"upload" 或 "url"）
        model_name: YOLO模型名稱
        confidence_threshold: 置信度閾值
        process_interval: 處理間隔（每N幀處理一次）
        
    Returns:
        Tuple: (output_video_path, summary_html, formatted_stats, object_details)
    """
    if video_processor is None:
        error_msg = "Error: Video processor not initialized."
        error_html = f"<div class='video-summary-content-wrapper'><pre>{error_msg}</pre></div>"
        empty_object_details = {}
        return None, error_html, {"error": "Video processor not available"}, empty_object_details

    video_path = None
    
    # 根據輸入類型處理
    if input_type == "upload" and video_input:
        video_path = video_input
        print(f"Processing uploaded video file: {video_path}")
    elif input_type == "url" and video_url:
        print(f"Processing video from URL: {video_url}")
        video_path, error_msg = download_video_from_url(video_url)
        if error_msg:
            error_html = f"<div class='video-summary-content-wrapper'><pre>{error_msg}</pre></div>"
            empty_object_details = {}
            return None, error_html, {"error": error_msg}, empty_object_details
    
    if not video_path:
        error_msg = "Please provide a video file or valid URL."
        error_html = f"<div class='video-summary-content-wrapper'><pre>{error_msg}</pre></div>"
        empty_object_details = {}
        return None, error_html, {"error": "No video input provided"}, empty_object_details

    print(f"Starting practical video analysis: model={model_name}, confidence={confidence_threshold}, interval={process_interval}")
    
    processing_start_time = time.time()
    
    try:
        output_video_path, analysis_results = video_processor.process_video(
            video_path=video_path,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            process_interval=int(process_interval)
        )
        
        print(f"Video processing function returned: path={output_video_path}")
        
        if output_video_path is None:
            error_msg = analysis_results.get("error", "Unknown error occurred during video processing")
            error_html = f"<div class='video-summary-content-wrapper'><pre>Processing failed: {error_msg}</pre></div>"
            empty_object_details = {}
            return None, error_html, analysis_results, empty_object_details
        
        # 生成摘要，直接用統計數據
        basic_summary = generate_basic_video_summary(analysis_results)
        
        # Final Result
        processing_time = time.time() - processing_start_time
        processing_info = analysis_results.get("processing_info", {})
        
        summary_lines = [
            f"Video processing completed in {processing_time:.2f} seconds.",
            f"Analyzed {processing_info.get('frames_analyzed', 0)} frames out of {processing_info.get('total_frames', 0)} total frames.",
            f"Processing interval: every {process_interval} frames",
            basic_summary
        ]

        summary_content = '\n'.join(summary_lines)
        summary_html = f"<div class='video-summary-content-wrapper'><pre>{summary_content}</pre></div>"
        
        # 提取物體詳情數據用於第四個輸出組件
        timeline_analysis = analysis_results.get("timeline_analysis", {})
        object_appearances = timeline_analysis.get("object_appearances", {})
        
        # 格式化物體詳情數據以便在JSON組件中顯示
        object_details_formatted = {}
        for obj_name, details in object_appearances.items():
            object_details_formatted[obj_name] = {
                "Estimated Count": details.get("estimated_count", 0),
                "First Appearance": details.get("first_appearance", "Unknown"),
                "Last Seen": details.get("last_seen", "Unknown"), 
                "Duration in Video": details.get("duration_in_video", "Unknown"),
                "Detection Confidence": details.get("detection_confidence", 0.0),
                "First Appearance (seconds)": details.get("first_appearance_seconds", 0),
                "Duration (seconds)": details.get("duration_seconds", 0)
            }
        
        print(f"Extracted object details for {len(object_details_formatted)} object types")
        
        return output_video_path, summary_html, analysis_results, object_details_formatted

    except Exception as e:
        print(f"Error in handle_video_upload: {e}")
        traceback.print_exc()
        error_msg = f"Video processing failed: {str(e)}"
        error_html = f"<div class='video-summary-content-wrapper'><pre>{error_msg}</pre></div>"
        empty_object_details = {}
        return None, error_html, {"error": str(e)}, empty_object_details

def main():
    """主函數，初始化並啟動Gradio"""
    global ui_manager
    
    print("=== VisionScout Application Starting ===")
    
    print("Initializing processors...")
    initialization_success = initialize_processors()
    if not initialization_success:
        print("ERROR: Failed to initialize processors. Application cannot start.")
        return

    print("Initializing UI manager...")
    ui_manager = initialize_ui_manager()

    print("Creating Gradio interface...")
    demo_interface = ui_manager.create_interface(
        handle_image_upload_fn=handle_image_upload,
        handle_video_upload_fn=handle_video_upload,
        download_video_from_url_fn=download_video_from_url
    )

    print("Launching application...")
    demo_interface.launch(debug=True)


if __name__ == "__main__":
    main()
