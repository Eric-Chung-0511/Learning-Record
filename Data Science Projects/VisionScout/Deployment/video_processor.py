import cv2
import os
import tempfile
import uuid
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
from collections import defaultdict

from image_processor import ImageProcessor
from evaluation_metrics import EvaluationMetrics
from scene_analyzer import SceneAnalyzer
from detection_model import DetectionModel

class VideoProcessor:
    """
    Handles the processing of video files, including object detection
    and scene analysis on selected frames.
    """
    def __init__(self, image_processor: ImageProcessor):
        """
        Initializes the VideoProcessor.

        Args:
            image_processor (ImageProcessor): An initialized ImageProcessor instance.
        """
        self.image_processor = image_processor

    def process_video_file(self,
                           video_path: str,
                           model_name: str,
                           confidence_threshold: float,
                           process_interval: int = 5,
                           scene_desc_interval_sec: int = 3) -> Tuple[Optional[str], str, Dict]:
        """
        Processes an uploaded video file, performs detection and periodic scene analysis,
        and returns the path to the annotated output video file along with a summary.

        Args:
            video_path (str): Path to the input video file.
            model_name (str): Name of the YOLO model to use.
            confidence_threshold (float): Confidence threshold for object detection.
            process_interval (int): Process every Nth frame. Defaults to 5.
            scene_desc_interval_sec (int): Update scene description every N seconds. Defaults to 3.

        Returns:
            Tuple[Optional[str], str, Dict]: (Path to output video or None, Summary text, Statistics dictionary)
        """
        if not video_path or not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return None, "Error: Video file not found.", {}

        print(f"Starting video processing for: {video_path}")
        start_time = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None, "Error opening video file.", {}

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: # Handle case where fps is not available or invalid
             fps = 30 # Assume a default fps
             print(f"Warning: Could not get valid FPS for video. Assuming {fps} FPS.")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video properties: {width}x{height} @ {fps:.2f} FPS, Total Frames: {total_frames_video}")

        # Calculate description update interval in frames
        description_update_interval_frames = int(fps * scene_desc_interval_sec)
        if description_update_interval_frames < 1:
            description_update_interval_frames = int(fps) # Update at least once per second if interval is too short

        object_trackers = {}  # 儲存ID與物體的映射
        last_detected_objects = {}  # 儲存上一次檢測到的物體資訊
        next_object_id = 0  # 下一個可用的物體ID
        tracking_threshold = 0.6  # 相同物體的IoU
        object_colors = {}  # 每個被追蹤的物體分配固定顏色

        # Setup Output Video
        output_filename = f"processed_{uuid.uuid4().hex}_{os.path.basename(video_path)}"
        temp_dir = tempfile.gettempdir() # Use system's temp directory
        output_path = os.path.join(temp_dir, output_filename)
        # Ensure the output path has a compatible extension (like .mp4)
        if not output_path.lower().endswith(('.mp4', '.avi', '.mov')):
            output_path += ".mp4"

        # Use 'mp4v' for MP4, common and well-supported
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"Error: Could not open VideoWriter for path: {output_path}")
            cap.release()
            return None, f"Error creating output video file at {output_path}.", {}
        print(f"Output video will be saved to: {output_path}")

        frame_count = 0
        processed_frame_count = 0
        all_stats = [] # Store stats for each processed frame
        summary_lines = []
        last_description = "Analyzing scene..." # Initial description
        frame_since_last_desc = description_update_interval_frames # Trigger analysis on first processed frame

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break # End of video

                frame_count += 1
                frame_since_last_desc += 1
                current_frame_annotated = False # Flag if this frame was processed and annotated

                # Process frame based on interval
                if frame_count % process_interval == 0:
                    processed_frame_count += 1
                    print(f"Processing frame {frame_count}...")
                    current_frame_annotated = True

                    # Use ImageProcessor for single-frame tasks
                    # 1. Convert frame format BGR -> RGB -> PIL
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                    except Exception as e:
                        print(f"Error converting frame {frame_count}: {e}")
                        continue # Skip this frame

                    # 2. Get appropriate model instance
                    # Confidence is passed from UI, model_name too
                    model_instance = self.image_processor.get_model_instance(model_name, confidence_threshold)
                    if not model_instance or not model_instance.is_model_loaded:
                         print(f"Error: Model {model_name} not loaded. Skipping frame {frame_count}.")
                         # Draw basic frame without annotation
                         cv2.putText(frame, f"Scene: {last_description[:80]}...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                         cv2.putText(frame, f"Scene: {last_description[:80]}...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                         out.write(frame)
                         continue


                    # 3. Perform detection
                    detection_result = model_instance.detect(pil_image) # Use PIL image

                    current_description_for_frame = last_description # Default to last known description
                    scene_analysis_result = None
                    stats = {}

                    if detection_result and hasattr(detection_result, 'boxes') and len(detection_result.boxes) > 0:
                        # Ensure SceneAnalyzer is ready within ImageProcessor
                        if not hasattr(self.image_processor, 'scene_analyzer') or self.image_processor.scene_analyzer is None:
                             print("Initializing SceneAnalyzer...")
                             # Pass class names from the current detection result
                             self.image_processor.scene_analyzer = SceneAnalyzer(class_names=detection_result.names)
                        elif self.image_processor.scene_analyzer.class_names is None:
                             # Update class names if they were missing
                             self.image_processor.scene_analyzer.class_names = detection_result.names
                             if hasattr(self.image_processor.scene_analyzer, 'spatial_analyzer'):
                                 self.image_processor.scene_analyzer.spatial_analyzer.class_names = detection_result.names


                        # 4. Perform Scene Analysis (periodically)
                        if frame_since_last_desc >= description_update_interval_frames:
                            print(f"Analyzing scene at frame {frame_count} (threshold: {description_update_interval_frames} frames)...")
                            # Pass lighting_info=None for now, as it's disabled for performance
                            scene_analysis_result = self.image_processor.analyze_scene(detection_result, lighting_info=None)
                            current_description_for_frame = scene_analysis_result.get("description", last_description)
                            last_description = current_description_for_frame # Cache the new description
                            frame_since_last_desc = 0 # Reset counter

                        # 5. Calculate Statistics for this frame
                        stats = EvaluationMetrics.calculate_basic_stats(detection_result)
                        stats['frame_number'] = frame_count # Add frame number to stats
                        all_stats.append(stats)

                        # 6. Draw annotations
                        names = detection_result.names
                        boxes = detection_result.boxes.xyxy.cpu().numpy()
                        classes = detection_result.boxes.cls.cpu().numpy().astype(int)
                        confs = detection_result.boxes.conf.cpu().numpy()

                        def calculate_iou(box1, box2):
                            """Calculate Intersection IOU value"""
                            x1_1, y1_1, x2_1, y2_1 = box1
                            x1_2, y1_2, x2_2, y2_2 = box2

                            xi1 = max(x1_1, x1_2)
                            yi1 = max(y1_1, y1_2)
                            xi2 = min(x2_1, x2_2)
                            yi2 = min(y2_1, y2_2)

                            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                            box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
                            box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

                            union_area = box1_area + box2_area - inter_area

                            return inter_area / union_area if union_area > 0 else 0

                        # 處理當前幀中的所有檢測
                        current_detected_objects = {}

                        for box, cls_id, conf in zip(boxes, classes, confs):
                            x1, y1, x2, y2 = map(int, box)

                            # 查找最匹配的已追蹤物體
                            best_match_id = None
                            best_match_iou = 0

                            for obj_id, (old_box, old_cls_id, _) in last_detected_objects.items():
                                if old_cls_id == cls_id:  # 同一類別才比較
                                    iou = calculate_iou(box, old_box)
                                    if iou > tracking_threshold and iou > best_match_iou:
                                        best_match_id = obj_id
                                        best_match_iou = iou

                            # 如果找到匹配，使用現有ID；否則分配新ID
                            if best_match_id is not None:
                                obj_id = best_match_id
                            else:
                                obj_id = next_object_id
                                next_object_id += 1

                                # 使用更明顯的顏色
                                bright_colors = [
                                    (0, 0, 255),    # red
                                    (0, 255, 0),    # green
                                    (255, 0, 0),    # blue
                                    (0, 255, 255),  # yellow
                                    (255, 0, 255),  # purple
                                    (255, 128, 0),  # orange
                                    (128, 0, 255)   # purple
                                ]
                                object_colors[obj_id] = bright_colors[obj_id % len(bright_colors)]

                            # update tracking info
                            current_detected_objects[obj_id] = (box, cls_id, conf)

                            color = object_colors.get(obj_id, (0, 255, 0))  # default is green
                            label = f"{names.get(cls_id, 'Unknown')}-{obj_id}: {conf:.2f}"

                            # 平滑化邊界框：如果是已知物體，與上一幀位置平均
                            if obj_id in last_detected_objects:
                                old_box, _, _ = last_detected_objects[obj_id]
                                old_x1, old_y1, old_x2, old_y2 = map(int, old_box)
                                # 平滑係數
                                alpha = 0.7  # current weight
                                beta = 0.3   # history weight

                                x1 = int(alpha * x1 + beta * old_x1)
                                y1 = int(alpha * y1 + beta * old_y1)
                                x2 = int(alpha * x2 + beta * old_x2)
                                y2 = int(alpha * y2 + beta * old_y2)

                            # draw box and label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            # add text
                            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1 - 10), color, -1)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                        # update tracking info
                        last_detected_objects = current_detected_objects.copy()


                    # Draw the current scene description on the frame
                    cv2.putText(frame, f"Scene: {current_description_for_frame[:80]}...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA) # Black outline
                    cv2.putText(frame, f"Scene: {current_description_for_frame[:80]}...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA) # White text

                # Write the frame (annotated or original) to the output video
                # Draw last known description if this frame wasn't processed
                if not current_frame_annotated:
                    cv2.putText(frame, f"Scene: {last_description[:80]}...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, f"Scene: {last_description[:80]}...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                out.write(frame) # Write frame to output file

        except Exception as e:
            print(f"Error during video processing loop for {video_path}: {e}")
            import traceback
            traceback.print_exc()
            summary_lines.append(f"An error occurred during processing: {e}")
        finally:
            # Release resources
            cap.release()
            out.release()
            print(f"Video processing finished. Resources released. Output path: {output_path}")
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                print(f"Error: Output video file was not created or is empty at {output_path}")
                summary_lines.append("Error: Failed to create output video.")
                output_path = None

        end_time = time.time()
        processing_time = end_time - start_time
        summary_lines.insert(0, f"Finished processing in {processing_time:.2f} seconds.")
        summary_lines.insert(1, f"Processed {processed_frame_count} frames out of {frame_count} (interval: {process_interval} frames).")
        summary_lines.insert(2, f"Scene description updated approximately every {scene_desc_interval_sec} seconds.")

        # Generate Aggregate Statistics
        aggregated_stats = {
            "total_frames_read": frame_count,
            "total_frames_processed": processed_frame_count,
            "avg_objects_per_processed_frame": 0, # Calculate below
            "cumulative_detections": {}, # Total times each class was detected
            "max_concurrent_detections": {} # Max count of each class in a single processed frame
            }
        object_cumulative_counts = {}
        object_max_concurrent_counts = {} # Store the max count found for each object type
        total_detected_in_processed = 0

        # Iterate through stats collected from each processed frame
        for frame_stats in all_stats:
            total_objects_in_frame = frame_stats.get("total_objects", 0)
            total_detected_in_processed += total_objects_in_frame

            # Iterate through object classes detected in this frame
            for obj_name, obj_data in frame_stats.get("class_statistics", {}).items():
                count_in_frame = obj_data.get("count", 0)

                # Cumulative count
                if obj_name not in object_cumulative_counts:
                    object_cumulative_counts[obj_name] = 0
                object_cumulative_counts[obj_name] += count_in_frame

                # Max concurrent count
                if obj_name not in object_max_concurrent_counts:
                    object_max_concurrent_counts[obj_name] = 0
                # Update the max count if the current frame's count is higher
                object_max_concurrent_counts[obj_name] = max(object_max_concurrent_counts[obj_name], count_in_frame)

        # Add sorted results to the final dictionary
        aggregated_stats["cumulative_detections"] = dict(sorted(object_cumulative_counts.items(), key=lambda item: item[1], reverse=True))
        aggregated_stats["max_concurrent_detections"] = dict(sorted(object_max_concurrent_counts.items(), key=lambda item: item[1], reverse=True))

        # Calculate average objects per processed frame
        if processed_frame_count > 0:
             aggregated_stats["avg_objects_per_processed_frame"] = round(total_detected_in_processed / processed_frame_count, 2)

        summary_text = "\n".join(summary_lines)
        print("Generated Summary:\n", summary_text)
        print("Aggregated Stats (Revised):\n", aggregated_stats) # Print the revised stats

        # Return the potentially updated output_path
        return output_path, summary_text, aggregated_stats
