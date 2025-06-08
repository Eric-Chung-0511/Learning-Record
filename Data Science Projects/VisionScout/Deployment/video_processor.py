import cv2
import os
import tempfile
import uuid
import time
import traceback
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass
import math

from detection_model import DetectionModel
from evaluation_metrics import EvaluationMetrics

@dataclass
class ObjectRecord:
    """物體記錄數據結構"""
    class_name: str
    first_seen_time: float
    last_seen_time: float
    total_detections: int
    peak_count_in_frame: int
    confidence_avg: float
    
    def get_duration(self) -> float:
        """獲取物體在影片中的持續時間"""
        return self.last_seen_time - self.first_seen_time
    
    def format_time(self, seconds: float) -> str:
        """格式化時間顯示"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes > 0:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"

class VideoProcessor:
    """
    專注於實用統計分析的視頻處理器：
    - 準確的物體計數和識別
    - 物體出現時間分析
    - 檢測品質評估
    - 活動密度統計
    """
    
    def __init__(self):
        """初始化視頻處理器"""
        self.detection_models: Dict[str, DetectionModel] = {}
        
        # 分析參數
        self.spatial_cluster_threshold = 100  # 像素距離閾值，用於合併重複檢測
        self.confidence_filter_threshold = 0.1  # 最低信心度過濾
        
        # 統計數據收集
        self.frame_detections = []  # 每幀檢測結果
        self.object_timeline = defaultdict(list)  # 物體時間線記錄
        self.frame_timestamps = []  # 幀時間戳記錄
    
    def get_or_create_model(self, model_name: str, confidence_threshold: float) -> DetectionModel:
        """獲取或創建檢測模型實例"""
        model_key = f"{model_name}_{confidence_threshold}"
        
        if model_key not in self.detection_models:
            try:
                model = DetectionModel(model_name, confidence_threshold)
                self.detection_models[model_key] = model
                print(f"Loaded detection model: {model_name} with confidence {confidence_threshold}")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                raise
                
        return self.detection_models[model_key]
    
    def cluster_detections_by_position(self, detections: List[Dict], threshold: float = 100) -> List[Dict]:
        """根據位置聚類檢測結果，合併相近的重複檢測"""
        if not detections:
            return []
        
        # 按物體類別分組進行聚類處理
        class_groups = defaultdict(list)
        for det in detections:
            class_groups[det['class_name']].append(det)
        
        clustered_results = []
        
        for class_name, class_detections in class_groups.items():
            if len(class_detections) == 1:
                clustered_results.extend(class_detections)
                continue
            
            # 執行空間聚類算法
            clusters = []
            used = set()
            
            for i, det1 in enumerate(class_detections):
                if i in used:
                    continue
                    
                cluster = [det1]
                used.add(i)
                
                # 計算檢測框中心點
                x1_center = (det1['bbox'][0] + det1['bbox'][2]) / 2
                y1_center = (det1['bbox'][1] + det1['bbox'][3]) / 2
                
                # 查找相近的檢測結果
                for j, det2 in enumerate(class_detections):
                    if j in used:
                        continue
                    
                    x2_center = (det2['bbox'][0] + det2['bbox'][2]) / 2
                    y2_center = (det2['bbox'][1] + det2['bbox'][3]) / 2
                    
                    distance = math.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
                    
                    if distance < threshold:
                        cluster.append(det2)
                        used.add(j)
                
                clusters.append(cluster)
            
            # 為每個聚類生成代表性檢測結果
            for cluster in clusters:
                best_detection = max(cluster, key=lambda x: x['confidence'])
                avg_confidence = sum(det['confidence'] for det in cluster) / len(cluster)
                best_detection['confidence'] = avg_confidence
                best_detection['cluster_size'] = len(cluster)
                clustered_results.append(best_detection)
        
        return clustered_results
    
    def analyze_frame_detections(self, detections: Any, timestamp: float, class_names: Dict[int, str]):
        """分析單幀的檢測結果並更新統計記錄"""
        if not hasattr(detections, 'boxes') or len(detections.boxes) == 0:
            self.frame_detections.append([])
            self.frame_timestamps.append(timestamp)
            return
        
        # extract detected data
        boxes = detections.boxes.xyxy.cpu().numpy()
        classes = detections.boxes.cls.cpu().numpy().astype(int)
        confidences = detections.boxes.conf.cpu().numpy()
        
        # 轉換為統一的檢測格式
        frame_detections = []
        for box, cls_id, conf in zip(boxes, classes, confidences):
            if conf >= self.confidence_filter_threshold:
                frame_detections.append({
                    'bbox': tuple(box),
                    'class_id': cls_id,
                    'class_name': class_names.get(cls_id, f'class_{cls_id}'),
                    'confidence': conf,
                    'timestamp': timestamp
                })
        
        # 為了避免有重複偵測, 用空間聚類
        clustered_detections = self.cluster_detections_by_position(
            frame_detections, self.spatial_cluster_threshold
        )
        
        # record results
        self.frame_detections.append(clustered_detections)
        self.frame_timestamps.append(timestamp)
        
        # 更新物體時間線記錄
        for detection in clustered_detections:
            class_name = detection['class_name']
            self.object_timeline[class_name].append({
                'timestamp': timestamp,
                'confidence': detection['confidence'],
                'bbox': detection['bbox']
            })
    
    def generate_object_statistics(self, fps: float) -> Dict[str, ObjectRecord]:
        """生成物體統計數據"""
        object_stats = {}
        
        for class_name, timeline in self.object_timeline.items():
            if not timeline:
                continue
            
            # 計算基本時間統計
            timestamps = [entry['timestamp'] for entry in timeline]
            confidences = [entry['confidence'] for entry in timeline]
            
            first_seen = min(timestamps)
            last_seen = max(timestamps)
            total_detections = len(timeline)
            avg_confidence = sum(confidences) / len(confidences)
            
            # 計算每個時間點的物體數量以確定峰值
            frame_counts = defaultdict(int)
            for entry in timeline:
                frame_timestamp = entry['timestamp']
                frame_counts[frame_timestamp] += 1
            
            peak_count = max(frame_counts.values()) if frame_counts else 1
            
            # 創建物體記錄
            object_stats[class_name] = ObjectRecord(
                class_name=class_name,
                first_seen_time=first_seen,
                last_seen_time=last_seen,
                total_detections=total_detections,
                peak_count_in_frame=peak_count,
                confidence_avg=avg_confidence
            )
        
        return object_stats
    
    def analyze_object_density(self, object_stats: Dict[str, ObjectRecord], video_duration: float) -> Dict[str, Any]:
        """分析物體密度和活動模式"""
        total_objects = sum(record.peak_count_in_frame for record in object_stats.values())
        objects_per_minute = (total_objects / video_duration) * 60 if video_duration > 0 else 0
        
        # 分析每30秒時間段的活動分布
        time_segments = defaultdict(int)
        segment_duration = 30
        
        for detections, timestamp in zip(self.frame_detections, self.frame_timestamps):
            segment = int(timestamp // segment_duration) * segment_duration
            time_segments[segment] += len(detections)
        
        # 辨識活動高峰時段
        peak_segments = []
        if time_segments:
            max_activity = max(time_segments.values())
            threshold = max_activity * 0.8  # 80%活動量代表高度活躍
            
            for segment, activity in time_segments.items():
                if activity >= threshold:
                    peak_segments.append({
                        'start_time': segment,
                        'end_time': min(segment + segment_duration, video_duration),
                        'activity_count': activity,
                        'description': f"{segment}s-{min(segment + segment_duration, video_duration):.0f}s"
                    })
        
        return {
            'total_objects_detected': total_objects,
            'objects_per_minute': round(objects_per_minute, 2),
            'video_duration_seconds': video_duration,
            'peak_activity_periods': peak_segments,
            'activity_distribution': {str(k): v for k, v in time_segments.items()}
        }
    
    def analyze_quality_metrics(self, object_stats: Dict[str, ObjectRecord]) -> Dict[str, Any]:
        """分析檢測品質指標"""
        all_confidences = []
        class_confidence_stats = {}
        
        # 收集所有置信度數據進行分析
        for class_name, record in object_stats.items():
            class_confidences = []
            for detection_data in self.object_timeline[class_name]:
                conf = detection_data['confidence']
                all_confidences.append(conf)
                class_confidences.append(conf)
            
            # 計算各類別的置信度統計
            if class_confidences:
                class_confidence_stats[class_name] = {
                    'average_confidence': round(np.mean(class_confidences), 3),
                    'min_confidence': round(np.min(class_confidences), 3),
                    'max_confidence': round(np.max(class_confidences), 3),
                    'confidence_stability': round(1 - np.std(class_confidences), 3),
                    'detection_count': len(class_confidences)
                }
        
        # 計算整體品質指標
        if all_confidences:
            overall_confidence = np.mean(all_confidences)
            confidence_std = np.std(all_confidences)
            
            # 品質等級評估
            if overall_confidence > 0.8 and confidence_std < 0.1:
                quality_grade = "excellent"
            elif overall_confidence > 0.6 and confidence_std < 0.2:
                quality_grade = "good"
            elif overall_confidence > 0.4:
                quality_grade = "fair"
            else:
                quality_grade = "poor"
                
            quality_analysis = f"Detection quality: {quality_grade} (avg confidence: {overall_confidence:.3f})"
        else:
            overall_confidence = 0
            confidence_std = 0
            quality_grade = "no_data"
            quality_analysis = "No detection data available for quality analysis"
        
        return {
            'overall_confidence': round(overall_confidence, 3),
            'confidence_stability': round(1 - confidence_std, 3),
            'quality_grade': quality_grade,
            'class_confidence_breakdown': class_confidence_stats,
            'total_detections_analyzed': len(all_confidences),
            'quality_analysis': quality_analysis
        }
    
    def generate_timeline_analysis(self, object_stats: Dict[str, ObjectRecord], video_duration: float) -> Dict[str, Any]:
        """生成時間線分析報告"""
        timeline_analysis = {
            'video_duration_seconds': video_duration,
            'object_appearances': {},
            'timeline_summary': []
        }
        
        # 分析每個物體的出現的時序
        for class_name, record in object_stats.items():
            timeline_analysis['object_appearances'][class_name] = {
                'first_appearance': record.format_time(record.first_seen_time),
                'first_appearance_seconds': round(record.first_seen_time, 1),
                'last_seen': record.format_time(record.last_seen_time),
                'last_seen_seconds': round(record.last_seen_time, 1),
                'duration_in_video': record.format_time(record.get_duration()),
                'duration_seconds': round(record.get_duration(), 1),
                'estimated_count': record.peak_count_in_frame,
                'detection_confidence': round(record.confidence_avg, 3)
            }
        
        # timeline summary
        if object_stats:
            sorted_objects = sorted(object_stats.values(), key=lambda x: x.first_seen_time)
            
            for i, record in enumerate(sorted_objects):
                if record.first_seen_time < 2.0:
                    summary = f"{record.peak_count_in_frame} {record.class_name}(s) present from the beginning"
                else:
                    summary = f"{record.peak_count_in_frame} {record.class_name}(s) first appeared at {record.format_time(record.first_seen_time)}"
                
                timeline_analysis['timeline_summary'].append(summary)
        
        return timeline_analysis
    
    def draw_simple_annotations(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """在視頻幀上繪製檢測標註"""
        annotated_frame = frame.copy()
        
        # 不同物體類別分配顏色
        colors = {
            'person': (0, 255, 0),     # green
            'car': (255, 0, 0),        # blue
            'truck': (0, 0, 255),      # red
            'bus': (255, 255, 0),      # 青色
            'bicycle': (255, 0, 255),  # purple
            'motorcycle': (0, 255, 255) # yellow
        }
        
        # 繪製每個檢測結果
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            color = colors.get(class_name, (128, 128, 128))  # set gray to default color 
            
            # 繪製邊界框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # 準備標籤文字
            label = f"{class_name}: {confidence:.2f}"
            if 'cluster_size' in detection and detection['cluster_size'] > 1:
                label += f" (merged: {detection['cluster_size']})"
            
            # 繪製標籤背景和文字
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def _ensure_string_keys(self, data):
        """確保所有字典鍵值都轉換為字串格式以支援JSON序列化"""
        if isinstance(data, dict):
            return {str(key): self._ensure_string_keys(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._ensure_string_keys(item) for item in data]
        else:
            return data
    
    def process_video(self, 
                     video_path: str,
                     model_name: str,
                     confidence_threshold: float,
                     process_interval: int = 10) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        處理視頻文件，執行物體檢測和統計分析
        
        Args:
            video_path: 視頻文件路徑
            model_name: YOLO模型名稱
            confidence_threshold: 置信度閾值
            process_interval: 處理間隔（每N幀處理一次）
            
        Returns:
            Tuple[Optional[str], Dict[str, Any]]: (輸出視頻路徑, 分析結果)
        """
        if not video_path or not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return None, {"error": "Video file not found"}
        
        print(f"Starting focused video analysis: {video_path}")
        start_time = time.time()
        
        # 重置處理狀態
        self.frame_detections.clear()
        self.object_timeline.clear()
        self.frame_timestamps.clear()
        
        # 開啟視頻文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, {"error": "Could not open video file"}
        
        # 取得視頻基本屬性
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        print(f"Video properties: {width}x{height} @ {fps:.2f} FPS")
        print(f"Duration: {video_duration:.1f}s, Total frames: {total_frames}")
        print(f"Processing every {process_interval} frames")
        
        # 設定輸出視頻文件
        output_filename = f"analyzed_{uuid.uuid4().hex}_{os.path.basename(video_path)}"
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, output_filename)
        if not output_path.lower().endswith(('.mp4', '.avi', '.mov')):
            output_path += ".mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            return None, {"error": "Could not create output video file"}
        
        print(f"Output video will be saved to: {output_path}")
        
        # 載入檢測模型
        try:
            detection_model = self.get_or_create_model(model_name, confidence_threshold)
        except Exception as e:
            cap.release()
            out.release()
            return None, {"error": f"Failed to load detection model: {str(e)}"}
        
        # 主要視頻處理循環
        frame_count = 0
        processed_frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                timestamp = frame_count / fps
                
                # 根據處理間隔決定是否分析此幀
                if frame_count % process_interval == 0:
                    processed_frame_count += 1
                    
                    if processed_frame_count % 5 == 0:
                        print(f"Processing frame {frame_count}/{total_frames} ({timestamp:.1f}s)")
                    
                    try:
                        # 執行物體檢測
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        detections = detection_model.detect(pil_image)
                        
                        # 分析檢測結果
                        class_names = detections.names if hasattr(detections, 'names') else {}
                        self.analyze_frame_detections(detections, timestamp, class_names)
                        
                        # 繪製檢測標註
                        current_detections = self.frame_detections[-1] if self.frame_detections else []
                        frame = self.draw_simple_annotations(frame, current_detections)
                        
                    except Exception as e:
                        print(f"Error processing frame {frame_count}: {e}")
                        continue
                
                # 寫入處理後的幀到輸出視頻
                out.write(frame)
                
        except Exception as e:
            print(f"Error during video processing: {e}")
            traceback.print_exc()
        finally:
            cap.release()
            out.release()
        
        # 生成最終分析結果
        processing_time = time.time() - start_time
        
        # 執行各項統計分析
        object_stats = self.generate_object_statistics(fps)
        object_density = self.analyze_object_density(object_stats, video_duration)
        quality_metrics = self.analyze_quality_metrics(object_stats)
        timeline_analysis = self.generate_timeline_analysis(object_stats, video_duration)
        
        # 計算基本統計數據
        total_unique_objects = sum(record.peak_count_in_frame for record in object_stats.values())
        
        # 組織分析結果
        analysis_results = {
            "processing_info": {
                "processing_time_seconds": round(processing_time, 2),
                "total_frames": frame_count,
                "frames_analyzed": processed_frame_count,
                "processing_interval": process_interval,
                "video_duration_seconds": round(video_duration, 2),
                "fps": fps
            },
            "object_summary": {
                "total_unique_objects_detected": total_unique_objects,
                "object_types_found": len(object_stats),
                "detailed_counts": {
                    name: record.peak_count_in_frame 
                    for name, record in object_stats.items()
                }
            },
            "timeline_analysis": timeline_analysis,
            "analytics": {
                "object_density": object_density,
                "quality_metrics": quality_metrics
            }
        }
        
        # 確保所有字典鍵值都是字串格式
        analysis_results = self._ensure_string_keys(analysis_results)
        
        # 驗證輸出文件
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print(f"Warning: Output video file was not created properly")
            return None, analysis_results
        
        print(f"Video processing completed in {processing_time:.2f} seconds")
        print(f"Found {total_unique_objects} total objects across {len(object_stats)} categories")
        print(f"Quality grade: {quality_metrics['quality_grade']}")
        
        return output_path, analysis_results
