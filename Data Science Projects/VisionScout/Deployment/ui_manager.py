import gradio as gr
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt

from detection_model import DetectionModel
from style import Style

class UIManager:
    """
    Manages all UI-related functionality
    Handles Gradio interface creation, component definitions, and event binding.
    """

    def __init__(self):
        """Initialize the UI Manager."""
        self.available_models = None
        self.model_choices = []
        self.class_choices_formatted = []
        self._setup_model_choices()

    def _setup_model_choices(self):
        """Setup model choices for dropdowns."""
        try:
            self.available_models = DetectionModel.get_available_models()
            self.model_choices = [model["model_file"] for model in self.available_models]
        except ImportError:
            # Fallback model choices if DetectionModel is not available
            self.model_choices = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]

        # Setup class choices
        self.class_choices_formatted = [f"{id}: {name}" for id, name in self.get_all_classes()]

    def get_all_classes(self):
        """
        Gets all available COCO classes.

        Returns:
            List[Tuple[int, str]]: List of (class_id, class_name) tuples
        """
        # Try to get from a loaded model first
        try:
            # This will be injected by the main app when processors are available
            if hasattr(self, '_image_processor') and self._image_processor and self._image_processor.model_instances:
                for model_instance in self._image_processor.model_instances.values():
                    if model_instance and model_instance.is_model_loaded:
                        try:
                            # Ensure class_names is a dict {id: name}
                            if isinstance(model_instance.class_names, dict):
                                return sorted([(int(idx), name) for idx, name in model_instance.class_names.items()])
                        except Exception as e:
                            print(f"Error getting class names from model: {e}")
        except Exception:
            pass

        # COCO Classes
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

    def set_image_processor(self, image_processor):
        """
        Set the image processor reference for dynamic class retrieval.

        Args:
            image_processor: The ImageProcessor instance
        """
        self._image_processor = image_processor

    def get_css_styles(self):
        """
        Get CSS styles for the interface.

        Returns:
            str: CSS styles
        """
        try:
            return Style.get_css()
        except ImportError:
            # fallback defualt CSS style
            return """
            .app-header {
                text-align: center;
                padding: 2rem 0 3rem 0;
                background: linear-gradient(135deg, #f0f9ff, #e1f5fe);
            }
            .section-heading {
                font-size: 1.2rem;
                font-weight: bold;
                color: #2D3748;
                margin: 1rem 0 0.5rem 0;
            }
            .detect-btn {
                background: linear-gradient(90deg, #38b2ac, #4299e1) !important;
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
            }
            .video-summary-content-wrapper {
                max-height: 400px;
                overflow-y: auto;
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 15px;
                border: 1px solid #e2e8f0;
            }
            """

    def get_model_description(self, model_name):
        """
        Get model description for the given model name.

        Args:
            model_name: Name of the model

        Returns:
            str: Model description
        """
        try:
            return DetectionModel.get_model_description(model_name)
        except ImportError:
            return f"Model: {model_name}"

    def create_header(self):
        """
        Create the application header.

        Returns:
            gr.HTML: Header HTML component
        """
        return gr.HTML("""
            <div style="text-align: center; width: 100%; padding: 2rem 0 3rem 0; background: linear-gradient(135deg, #f0f9ff, #e1f5fe);">
                <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem; background: linear-gradient(90deg, #38b2ac, #4299e1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: bold; font-family: 'Arial', sans-serif;">VisionScout</h1>
                <h2 style="color: #4A5568; font-size: 1.2rem; font-weight: 400; margin-top: 0.5rem; margin-bottom: 1.5rem; font-family: 'Arial', sans-serif;">Object Detection and Scene Understanding</h2>
                <div style="display: flex; justify-content: center; gap: 10px; margin: 0.5rem 0;"><div style="height: 3px; width: 80px; background: linear-gradient(90deg, #38b2ac, #4299e1);"></div></div>
                <div style="display: flex; justify-content: center; gap: 25px; margin-top: 1.5rem;">
                    <div style="padding: 8px 15px; border-radius: 20px; background: rgba(66, 153, 225, 0.15); color: #2b6cb0; font-weight: 500; font-size: 0.9rem;"><span style="margin-right: 6px;">🖼️</span> Image Analysis</div>
                    <div style="padding: 8px 15px; border-radius: 20px; background: rgba(56, 178, 172, 0.15); color: #2b6cb0; font-weight: 500; font-size: 0.9rem;"><span style="margin-right: 6px;">🎬</span> Video Analysis with Temporal Tracking</div>
                </div>
                 <div style="margin-top: 20px; padding: 10px 15px; background-color: rgba(255, 248, 230, 0.9); border-left: 3px solid #f6ad55; border-radius: 6px; max-width: 600px; margin-left: auto; margin-right: auto; text-align: left;">
                     <p style="margin: 0; font-size: 0.9rem; color: #805ad5; font-weight: 500;">
                         <span style="margin-right: 5px;">📱</span> iPhone users: HEIC images may not be supported.
                         <a href="https://cloudconvert.com/heic-to-jpg" target="_blank" style="color: #3182ce; text-decoration: underline;">Convert HEIC to JPG</a> before uploading if needed.
                     </p>
                 </div>
            </div>
        """)

    def create_footer(self):
        """
        Create the application footer.

        Returns:
            gr.HTML: Footer HTML component
        """
        return gr.HTML("""
            <div class="footer" style="padding: 25px 0; text-align: center; background: linear-gradient(to right, #f5f9fc, #e1f5fe); border-top: 1px solid #e2e8f0; margin-top: 30px;">
                <div style="margin-bottom: 15px;">
                    <p style="font-size: 14px; color: #4A5568; margin: 5px 0;">Powered by YOLOv8, CLIP, Places365, Meta Llama3.2 and Ultralytics • Enhanced Video Processing with Temporal Analysis • Created with Gradio</p>
                </div>
                <div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-top: 15px;">
                    <p style="font-family: 'Arial', sans-serif; font-size: 14px; font-weight: 500; letter-spacing: 2px; background: linear-gradient(90deg, #38b2ac, #4299e1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; text-transform: uppercase; display: inline-block;">EXPLORE THE CODE →</p>
                    <a href="https://github.com/Eric-Chung-0511/Learning-Record/tree/main/Data%20Science%20Projects/VisionScout" target="_blank" style="text-decoration: none;">
                        <img src="https://img.shields.io/badge/GitHub-VisionScout-4299e1?logo=github&style=for-the-badge">
                    </a>
                </div>
            </div>
        """)

    def create_image_tab(self):
        """
        Create the image processing tab with all components.

        Returns:
            Dict: Dictionary containing all image tab components
        """
        components = {}

        with gr.Tab("Image Processing"):
            components['current_image_model'] = gr.State("yolov8m.pt")

            with gr.Row(equal_height=False):
                # Left Column: Image Input & Controls
                with gr.Column(scale=4, elem_classes="input-panel"):
                    with gr.Group():
                        gr.HTML('<div class="section-heading">Upload Image</div>')
                        components['image_input'] = gr.Image(
                            type="pil",
                            label="Upload an image",
                            elem_classes="upload-box"
                        )

                        with gr.Accordion("Image Analysis Settings", open=False):
                            components['image_model_dropdown'] = gr.Dropdown(
                                choices=self.model_choices,
                                value="yolov8m.pt",
                                label="Select Model",
                                info="Choose speed vs. accuracy (n=fast, m=balanced, x=accurate)"
                            )

                            components['image_model_info'] = gr.Markdown(
                                self.get_model_description("yolov8m.pt")
                            )

                            components['image_confidence'] = gr.Slider(
                                minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                                label="Confidence Threshold",
                                info="Minimum confidence for displaying a detected object"
                            )

                            components['use_llm'] = gr.Checkbox(
                                label="Use LLM for enhanced scene descriptions",
                                value=True,
                                info="Provides more detailed and natural language descriptions (may increase processing time)"
                            )

                            components['use_landmark_detection'] = gr.Checkbox(
                                label="Use CLIP for Landmark Detection",
                                value=False,
                                info="Detect famous landmarks, monuments, and tourist attractions that standard object detection cannot recognize (increases processing time)"
                            )

                            with gr.Accordion("Filter Classes", open=False):
                                gr.HTML('<div class="section-heading" style="font-size: 1rem;">Common Categories</div>')
                                with gr.Row():
                                    components['people_btn'] = gr.Button("People", size="sm")
                                    components['vehicles_btn'] = gr.Button("Vehicles", size="sm")
                                    components['animals_btn'] = gr.Button("Animals", size="sm")
                                    components['objects_btn'] = gr.Button("Common Objects", size="sm")

                                components['image_class_filter'] = gr.Dropdown(
                                    choices=self.class_choices_formatted,
                                    multiselect=True,
                                    label="Select Classes to Display",
                                    info="Leave empty to show all detected objects"
                                )

                    components['image_detect_btn'] = gr.Button(
                        "Analyze Image",
                        variant="primary",
                        elem_classes="detect-btn"
                    )

                    # How to use section
                    with gr.Group(elem_classes="how-to-use"):
                        gr.HTML('<div class="section-heading">How to Use (Image)</div>')
                        gr.Markdown("""
                            1. Upload an image or use the camera
                            2. *(Optional)* Adjust settings like confidence threshold or model size (n, m = balanced, x = accurate)
                            3. In **Analysis Settings**, you can:
                                * Uncheck **Use LLM** to skip enhanced descriptions (faster)
                                * Check **Use CLIP for Landmark Detection** to identify famous landmarks like museums, monuments, and tourist attractions *(may take longer)*
                                * Filter object classes to focus on specific types of objects *(optional)*
                            4. Click **Analyze Image** button

                            **💡 Tip:** For landmark recognition (e.g. Louvre Museum), make sure to enable **CLIP for Landmark Detection** in the settings above.
                        """)

                    # Image Examples
                    gr.Examples(
                        examples=[
                            "room_05.jpg",
                            "street_03.jpg",
                            "street_04.jpg",
                            "landmark_Louvre_01.jpg"
                        ],
                        inputs=components['image_input'],
                        label="Example Images"
                    )

                    gr.HTML("""
                        <div style="text-align: center; margin-top: 8px; padding: 6px; background-color: #f8f9fa; border-radius: 4px; border: 1px solid #e2e8f0;">
                            <p style="font-size: 12px; color: #718096; margin: 0;">
                                📷 Sample images sourced from <a href="https://unsplash.com" target="_blank" style="color: #3182ce; text-decoration: underline;">Unsplash</a>
                            </p>
                        </div>
                    """)

                # Right Column: Image Results
                with gr.Column(scale=6, elem_classes="output-panel"):
                    with gr.Tabs(elem_classes="tabs"):
                        # Detection Result Tab
                        with gr.Tab("Detection Result"):
                            components['image_result_image'] = gr.Image(
                                type="pil",
                                label="Detection Result"
                            )
                            gr.HTML('<div class="section-heading">Detection Details</div>')
                            components['image_result_text'] = gr.Textbox(
                                label=None,
                                lines=10,
                                elem_id="detection-details",
                                container=False
                            )

                        # Scene Understanding Tab
                        with gr.Tab("Scene Understanding"):
                            gr.HTML('<div class="section-heading">Scene Analysis</div>')

                            # Info details
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

                            components['image_scene_description_html'] = gr.HTML(
                                label=None,
                                elem_id="scene_analysis_description_text"
                            )

                            # Original Scene Analysis accordion
                            with gr.Accordion("Original Scene Analysis", open=False, elem_id="original_scene_analysis_accordion"):
                                components['image_llm_description'] = gr.HTML(
                                    label=None,
                                    elem_id="original_scene_description_text"
                                )

                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.HTML('<div class="section-heading" style="font-size:1rem; text-align:left;">Possible Activities</div>')
                                    components['image_activities_list'] = gr.Dataframe(
                                        headers=["Activity"],
                                        datatype=["str"],
                                        row_count=5,
                                        col_count=1,
                                        wrap=True
                                    )

                                with gr.Column(scale=1):
                                    gr.HTML('<div class="section-heading" style="font-size:1rem; text-align:left;">Safety Concerns</div>')
                                    components['image_safety_list'] = gr.Dataframe(
                                        headers=["Concern"],
                                        datatype=["str"],
                                        row_count=5,
                                        col_count=1,
                                        wrap=True
                                    )

                            gr.HTML('<div class="section-heading">Functional Zones</div>')
                            components['image_zones_json'] = gr.JSON(
                                label=None,
                                elem_classes="json-box"
                            )

                            gr.HTML('<div class="section-heading">Lighting Conditions</div>')
                            components['image_lighting_info'] = gr.JSON(
                                label=None,
                                elem_classes="json-box"
                            )

                        # Statistics Tab
                        with gr.Tab("Statistics"):
                            with gr.Row():
                                with gr.Column(scale=3, elem_classes="plot-column"):
                                    gr.HTML('<div class="section-heading">Object Distribution</div>')
                                    components['image_plot_output'] = gr.Plot(
                                        label=None,
                                        elem_classes="large-plot-container"
                                    )
                                with gr.Column(scale=2, elem_classes="stats-column"):
                                    gr.HTML('<div class="section-heading">Detection Statistics</div>')
                                    components['image_stats_json'] = gr.JSON(
                                        label=None,
                                        elem_classes="enhanced-json-display"
                                    )

        return components

    def create_video_tab(self):
        """
        Create the video processing tab with all components.
        注意：移除了複雜的時序分析控制項，簡化為基本的統計分析

        Returns:
            Dict: Dictionary containing all video tab components
        """
        components = {}

        with gr.Tab("Video Processing"):
            with gr.Row(equal_height=False):
                # Left Column: Video Input & Controls
                with gr.Column(scale=4, elem_classes="input-panel"):
                    with gr.Group():
                        gr.HTML('<div class="section-heading">Video Input</div>')

                        # Input type selection
                        components['video_input_type'] = gr.Radio(
                            ["upload", "url"],
                            label="Input Method",
                            value="upload",
                            info="Choose how to provide the video"
                        )

                        # File upload
                        with gr.Group(elem_id="upload-video-group"):
                            components['video_input'] = gr.Video(
                                label="Upload a video file (MP4, AVI, MOV)",
                                sources=["upload"],
                                visible=True
                            )

                        # URL input
                        with gr.Group(elem_id="url-video-group"):
                            components['video_url_input'] = gr.Textbox(
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
                            components['video_model_dropdown'] = gr.Dropdown(
                                choices=self.model_choices,
                                value="yolov8n.pt",
                                label="Select Model (Video)",
                                info="Faster models (like 'n') are recommended for video processing"
                            )
                            components['video_confidence'] = gr.Slider(
                                minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                                label="Confidence Threshold (Video)",
                                info="Higher threshold reduces false detections"
                            )
                            components['video_process_interval'] = gr.Slider(
                                minimum=1, maximum=60, value=10, step=1,
                                label="Processing Interval (Frames)",
                                info="Analyze every Nth frame (higher value = faster processing)"
                            )

                            # 簡化的分析說明
                            gr.HTML("""
                                <div style="padding: 8px; margin-top: 10px; background-color: #f0f7ff; border-radius: 4px; border-left: 3px solid #4299e1; font-size: 12px;">
                                    <p style="margin: 0; color: #4a5568;">
                                        <b>Analysis Features:</b><br>
                                        • Accurate object counting with duplicate detection removal<br>
                                        • Timeline analysis showing when objects first appear<br>
                                        • Duration tracking for object presence in video<br>
                                        • Simple, clear statistical summaries
                                    </p>
                                </div>
                            """)

                    components['video_process_btn'] = gr.Button(
                        "Analyze Video",
                        variant="primary",
                        elem_classes="detect-btn"
                    )

                    # How to use section
                    with gr.Group(elem_classes="how-to-use"):
                        gr.HTML('<div class="section-heading">How to Use (Video)</div>')
                        gr.Markdown("""
                        1. Choose your input method: Upload a file or enter a URL.
                        2. Adjust settings if needed:
                            * Use **faster models** (yolov8n) for quicker processing
                            * Set **larger intervals** (15+ frames) for longer videos
                            * Adjust **confidence threshold** to filter low-quality detections
                        3. Click "Analyze Video". **Processing time varies based on video length.**
                        4. Review the results: annotated video and statistical analysis.

                        **⚡ Performance Tips:**
                        * For videos longer than 2 minutes, use interval ≥ 15 frames
                        * YOLOv8n model provides best speed for video processing
                        * Higher confidence thresholds reduce processing noise
                        """)

                    # Video examples
                    gr.HTML('<div class="section-heading">Example Videos</div>')
                    gr.HTML("""
                        <div style="padding: 10px; background-color: #f0f7ff; border-radius: 6px; margin-bottom: 15px;">
                            <p style="font-size: 14px; color: #4A5568; margin: 0;">
                                Upload any video containing objects that YOLO can detect. For testing, find sample videos from
                                <a href="https://www.pexels.com/search/videos/street/" target="_blank" style="color: #3182ce; text-decoration: underline;">Pexels</a> or
                                <a href="https://www.youtube.com/results?search_query=traffic+camera+footage" target="_blank" style="color: #3182ce; text-decoration: underline;">YouTube traffic footage</a>.
                            </p>
                        </div>
                    """)

                # Right Column: Video Results
                with gr.Column(scale=6, elem_classes="output-panel video-result-panel"):
                    gr.HTML("""
                        <div class="section-heading">Video Analysis Results</div>
                        <details class="info-details" style="margin: 5px 0 15px 0;">
                            <summary style="padding: 8px; background-color: #f0f7ff; border-radius: 6px; border-left: 3px solid #4299e1; font-weight: bold; cursor: pointer; color: #2b6cb0;">
                                🎬 Simplified Video Analysis Features
                            </summary>
                            <div style="margin-top: 8px; padding: 10px; background-color: #f8f9fa; border-radius: 6px; border: 1px solid #e2e8f0;">
                                <p style="font-size: 13px; color: #718096; margin: 0;">
                                    <b>Focus on practical insights:</b> This analysis provides accurate object counts and timing information
                                    without complex tracking. The system uses spatial clustering to eliminate duplicate detections and
                                    provides clear timeline data showing when objects first appear and how long they remain visible.
                                    <br><br>
                                    <b>Key benefits:</b> Reliable object counting, clear timeline analysis, and easy-to-understand results
                                    that directly answer questions like "How many cars are in this video?" and "When do they appear?"
                                </p>
                            </div>
                        </details>
                    """)

                    components['video_output'] = gr.Video(
                        label="Analyzed Video with Object Detection",
                        elem_classes="video-output-container"
                    )

                    with gr.Tabs(elem_classes="video-results-tabs"):
                        # Analysis Summary Tab
                        with gr.Tab("Analysis Summary"):
                            gr.HTML('<div class="section-heading">Video Analysis Report</div>')
                            gr.HTML("""
                                <div style="margin-bottom: 10px; padding: 8px; background-color: #f0f9ff; border-radius: 4px; border-left: 3px solid #4299e1; font-size: 12px;">
                                    <p style="margin: 0; color: #4a5568;">
                                        This summary provides object counts, timeline information, and insights about what appears in your video.
                                        Results are based on spatial clustering analysis to ensure accurate counting.
                                    </p>
                                </div>
                            """)
                            components['video_summary_text'] = gr.HTML(
                                label=None,
                                elem_id="video-summary-html-output"
                            )

                        # Detailed Statistics Tab
                        with gr.Tab("Detailed Statistics"):
                            gr.HTML('<div class="section-heading">Complete Analysis Data</div>')

                            with gr.Accordion("Processing Information", open=True):
                                gr.HTML("""
                                    <div style="padding: 6px; background-color: #f8f9fa; border-radius: 4px; margin-bottom: 10px; font-size: 12px;">
                                        <p style="margin: 0; color: #4a5568;">
                                            Basic information about video processing parameters and performance.
                                        </p>
                                    </div>
                                """)
                                components['video_stats_json'] = gr.JSON(
                                    label=None,
                                    elem_classes="video-stats-display"
                                )

                            with gr.Accordion("Object Details", open=False):
                                gr.HTML("""
                                    <div style="padding: 6px; background-color: #f8f9fa; border-radius: 4px; margin-bottom: 10px; font-size: 12px;">
                                        <p style="margin: 0; color: #4a5568;">
                                            Detailed breakdown of each object type detected, including timing and confidence information.
                                        </p>
                                    </div>
                                """)
                                components['video_object_details'] = gr.JSON(
                                    label="Object-by-Object Analysis",
                                    elem_classes="object-details-display"
                                )

        return components

    def get_filter_button_mappings(self):
        """
        Get the class ID mappings for filter buttons.

        Returns:
            Dict: Dictionary containing class ID lists for different categories
        """
        available_classes_list = self.get_all_classes()

        return {
            'people_classes_ids': [0],
            'vehicles_classes_ids': [1, 2, 3, 4, 5, 6, 7, 8],
            'animals_classes_ids': list(range(14, 24)),
            'common_objects_ids': [39, 41, 42, 43, 44, 45, 56, 57, 60, 62, 63, 67, 73],
            'available_classes_list': available_classes_list
        }

    def create_interface(self,
                        handle_image_upload_fn,
                        handle_video_upload_fn,
                        download_video_from_url_fn):
        """
        Create the complete Gradio interface.

        Args:
            handle_image_upload_fn: Function to handle image upload
            handle_video_upload_fn: Function to handle video upload
            download_video_from_url_fn: Function to download video from URL

        Returns:
            gr.Blocks: Complete Gradio interface
        """
        css = self.get_css_styles()

        with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="teal", secondary_hue="blue")) as demo:

            # Header
            with gr.Group(elem_classes="app-header"):
                self.create_header()

            # Main Content with Tabs
            with gr.Tabs(elem_classes="tabs"):

                # Image Processing Tab
                image_components = self.create_image_tab()

                # Video Processing Tab
                video_components = self.create_video_tab()

            # Footer
            self.create_footer()

            # Setup Event Listeners
            self._setup_event_listeners(
                image_components,
                video_components,
                handle_image_upload_fn,
                handle_video_upload_fn
            )

        return demo

    def _setup_event_listeners(self,
                              image_components,
                              video_components,
                              handle_image_upload_fn,
                              handle_video_upload_fn):
        """
        Setup all event listeners for the interface.

        Args:
            image_components: Dictionary of image tab components
            video_components: Dictionary of video tab components
            handle_image_upload_fn: Function to handle image upload
            handle_video_upload_fn: Function to handle video upload
        """
        # Image Model Change Handler
        image_components['image_model_dropdown'].change(
            fn=lambda model: (model, self.get_model_description(model)),
            inputs=[image_components['image_model_dropdown']],
            outputs=[image_components['current_image_model'], image_components['image_model_info']]
        )

        # Image Filter Buttons
        filter_mappings = self.get_filter_button_mappings()
        available_classes_list = filter_mappings['available_classes_list']
        people_classes_ids = filter_mappings['people_classes_ids']
        vehicles_classes_ids = filter_mappings['vehicles_classes_ids']
        animals_classes_ids = filter_mappings['animals_classes_ids']
        common_objects_ids = filter_mappings['common_objects_ids']

        image_components['people_btn'].click(
            lambda: [f"{id}: {name}" for id, name in available_classes_list if id in people_classes_ids],
            outputs=image_components['image_class_filter']
        )
        image_components['vehicles_btn'].click(
            lambda: [f"{id}: {name}" for id, name in available_classes_list if id in vehicles_classes_ids],
            outputs=image_components['image_class_filter']
        )
        image_components['animals_btn'].click(
            lambda: [f"{id}: {name}" for id, name in available_classes_list if id in animals_classes_ids],
            outputs=image_components['image_class_filter']
        )
        image_components['objects_btn'].click(
            lambda: [f"{id}: {name}" for id, name in available_classes_list if id in common_objects_ids],
            outputs=image_components['image_class_filter']
        )

        # Video Input Type Change Handler
        video_components['video_input_type'].change(
        fn=lambda input_type: [
            # Show/hide file upload
            gr.update(visible=(input_type == "upload")),
            # Show/hide URL input
            gr.update(visible=(input_type == "url"))
        ],
        inputs=[video_components['video_input_type']],
        outputs=[video_components['video_input'], video_components['video_url_input']]
        )

        # Image Detect Button Click Handler
        image_components['image_detect_btn'].click(
            fn=handle_image_upload_fn,
            inputs=[
                image_components['image_input'],
                image_components['image_model_dropdown'],
                image_components['image_confidence'],
                image_components['image_class_filter'],
                image_components['use_llm'],
                image_components['use_landmark_detection']
            ],
            outputs=[
                image_components['image_result_image'],
                image_components['image_result_text'],
                image_components['image_stats_json'],
                image_components['image_plot_output'],
                image_components['image_scene_description_html'],
                image_components['image_llm_description'],
                image_components['image_activities_list'],
                image_components['image_safety_list'],
                image_components['image_zones_json'],
                image_components['image_lighting_info']
            ]
        )

        # Video Process Button Click Handler
        video_components['video_process_btn'].click(
        fn=handle_video_upload_fn,
        inputs=[
            video_components['video_input'],
            video_components['video_url_input'],
            video_components['video_input_type'],
            video_components['video_model_dropdown'],
            video_components['video_confidence'],
            video_components['video_process_interval']
            ],
        outputs=[
            video_components['video_output'],
            video_components['video_summary_text'],
            video_components['video_stats_json'],
            video_components['video_object_details']
            ]
        )
