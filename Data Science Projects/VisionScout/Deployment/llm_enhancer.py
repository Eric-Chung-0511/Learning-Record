import logging
import traceback
import re
from typing import Dict, List, Any, Optional

from model_manager import ModelManager
from prompt_template_manager import PromptTemplateManager
from response_processor import ResponseProcessor
from text_quality_validator import TextQualityValidator
from landmark_data import ALL_LANDMARKS

class LLMEnhancer:
    """
    LLM增強器的主要窗口，協調模型管理、提示模板、回應處理和品質驗證等組件。
    提供統一的接口來處理場景描述增強、檢測結果驗證和無檢測情況處理。
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 tokenizer_path: Optional[str] = None,
                 device: Optional[str] = None,
                 max_length: int = 2048,
                 temperature: float = 0.3,
                 top_p: float = 0.85):
        """
        初始化LLM增強器門面

        Args:
            model_path: LLM模型的路徑或HuggingFace模型名稱，預設使用Llama 3.2
            tokenizer_path: tokenizer的路徑，通常與model_path相同
            device: 運行設備 ('cpu'或'cuda')，None時自動檢測
            max_length: 輸入文本的最大長度
            temperature: 生成文本的溫度參數
            top_p: 生成文本時的核心採樣機率閾值
        """
        # 設置專屬logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        try:
            # 初始化四個核心組件
            self.model_manager = ModelManager(
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                device=device,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )

            self.prompt_manager = PromptTemplateManager()
            self.response_processor = ResponseProcessor()
            self.quality_validator = TextQualityValidator()

            # 保存模型路徑以供後續使用
            self.model_path = model_path or "meta-llama/Llama-3.2-3B-Instruct"

            self.logger.info("LLMEnhancer facade initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize LLMEnhancer facade: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            raise Exception(error_msg) from e

    def enhance_description(self, scene_data: Dict[str, Any]) -> str:
        """
        場景描述增強器主要入口方法，整合所有組件來處理場景描述增強

        Args:
            scene_data: 包含場景資訊的字典，包括原始描述、檢測物件 (含 is_landmark)、
                        場景類型、時間/光線資訊等

        Returns:
            str: 增強後的場景描述
        """
        try:
            self.logger.info("Starting scene description enhancement")

            # 1. 重置模型上下文
            self.model_manager.reset_context()

            # 2. 取出原始描述
            original_desc = scene_data.get("original_description", "")
            if not original_desc:
                self.logger.warning("No original description provided")
                return "No original description provided."

            # 3. 準備物件統計資訊
            object_list = self._prepare_object_statistics(scene_data)
            if not object_list:
                object_keywords = self.quality_validator.extract_objects_from_description(original_desc)
                object_list = ", ".join(object_keywords) if object_keywords else "objects visible in the scene"

            # 4. 檢測地標並準備地標資訊
            landmark_info = self._extract_landmark_info(scene_data)

            # 5. 將地標資訊加入scene_data
            enhanced_scene_data = scene_data.copy()
            if landmark_info:
                enhanced_scene_data["landmark_location_info"] = landmark_info

            # 6. 生成 prompt
            prompt = self.prompt_manager.format_enhancement_prompt_with_landmark(
                scene_data=enhanced_scene_data,
                object_list=object_list,
                original_description=original_desc
            )

            # 7. 生成 LLM 回應
            self.logger.info("Generating LLM response")
            response = self.model_manager.generate_response(prompt)

            # 8. 處理不完整回應（重試機制）
            response = self._handle_incomplete_response(response, prompt, original_desc)

            # 9. 清理 LLM 回應
            model_type = self.model_path
            raw_cleaned = self.response_processor.clean_response(response, model_type)

            # 10. 移除解釋性注釋
            cleaned_response = self.response_processor.remove_explanatory_notes(raw_cleaned)

            # 11. 事實準確性驗證
            try:
                cleaned_response = self.quality_validator.verify_factual_accuracy(
                    original_desc, cleaned_response, object_list
                )
            except Exception:
                self.logger.warning("Fact verification failed; using response without verification")

            # 12. 場景類型一致性確保
            scene_type = scene_data.get("scene_type", "unknown scene")
            word_count = len(cleaned_response.split())
            if word_count >= 5 and scene_type.lower() not in cleaned_response.lower():
                cleaned_response = self.quality_validator.ensure_scene_type_consistency(
                    cleaned_response, scene_type, original_desc
                )

            # 13. 視角一致性處理
            perspective = self.quality_validator.extract_perspective_from_description(original_desc)
            if perspective and perspective.lower() not in cleaned_response.lower():
                cleaned_response = f"{perspective}, {cleaned_response[0].lower()}{cleaned_response[1:]}"

            # 13.5. 最終的 identical 詞彙清理（確保LLM輸出不包含重複性描述）
            identical_final_cleanup = [
                (r'\b(\d+)\s+identical\s+([a-zA-Z\s]+)', r'\1 \2'),
                (r'\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+identical\s+([a-zA-Z\s]+)', r'\1 \2'),
                (r'\bidentical\s+([a-zA-Z\s]+)', r'\1'),
                (r'\bcomprehensive arrangement of\b', 'arrangement of'),
            ]

            for pattern, replacement in identical_final_cleanup:
                cleaned_response = re.sub(pattern, replacement, cleaned_response, flags=re.IGNORECASE)

            # 14. 最終驗證：如果結果過短，嘗試fallback
            final_result = cleaned_response.strip()
            if not final_result or len(final_result) < 20:
                self.logger.warning("Enhanced description too short; attempting fallback")

                # Fallback prompt
                fallback_scene_data = enhanced_scene_data.copy()
                fallback_scene_data["is_fallback"] = True
                fallback_prompt = self.prompt_manager.format_enhancement_prompt_with_landmark(
                    scene_data=fallback_scene_data,
                    object_list=object_list,
                    original_description=original_desc
                )

                fallback_resp = self.model_manager.generate_response(fallback_prompt)
                fallback_cleaned = self.response_processor.clean_response(fallback_resp, model_type)
                fallback_cleaned = self.response_processor.remove_explanatory_notes(fallback_cleaned)

                final_result = fallback_cleaned.strip()
                if not final_result or len(final_result) < 20:
                    self.logger.warning("Fallback also insufficient; returning original")
                    return original_desc

            # 15. display enhanced description
            self.logger.info(f"Scene description enhancement completed successfully ({len(final_result)} chars)")
            return final_result

        except Exception as e:
            error_msg = f"Enhancement failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            return scene_data.get("original_description", "Unable to enhance description")

    def _extract_landmark_info(self, scene_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        提取地標資訊，但不構建prompt內容

        Args:
            scene_data: 場景資料字典

        Returns:
            Optional[Dict[str, str]]: 地標資訊字典，包含name和location，如果沒有地標則返回None
        """
        try:
            # 檢查是否有地標
            lm_id_in_data = scene_data.get("landmark_id")
            if not lm_id_in_data:
                # 從檢測物件中尋找地標
                for obj in scene_data.get("detected_objects", []):
                    if obj.get("is_landmark") and obj.get("landmark_id"):
                        lm_id_in_data = obj["landmark_id"]
                        break

            # 如果沒有檢測到地標，返回None
            if not lm_id_in_data:
                return None

            # 從landmark_data.py提取地標資訊
            if lm_id_in_data in ALL_LANDMARKS:
                lm_info = ALL_LANDMARKS[lm_id_in_data]
                landmark_name = scene_data.get("scene_name", lm_info.get("name", lm_id_in_data))
                landmark_location = lm_info.get("location", "")

                if landmark_location:
                    return {
                        "name": landmark_name,
                        "location": landmark_location,
                        "landmark_id": lm_id_in_data
                    }

            return None

        except Exception as e:
            self.logger.error(f"Error extracting landmark info: {str(e)}")
            return None


    def _prepare_object_statistics(self, scene_data: Dict[str, Any]) -> str:
        """
        準備物件統計資訊用於提示詞生成

        Args:
            scene_data: 場景資料字典

        Returns:
            str: 格式化的物件統計資訊
        """
        try:
            # 高信心度閾值
            high_confidence_threshold = 0.65

            # 優先使用預計算的統計資訊
            object_statistics = scene_data.get("object_statistics", {})
            object_counts = {}

            if object_statistics:
                for class_name, stats in object_statistics.items():
                    if stats.get("count", 0) > 0 and stats.get("avg_confidence", 0) >= high_confidence_threshold:
                        object_counts[class_name] = stats["count"]
            else:
                # 回退到原有的計算方式
                detected_objects = scene_data.get("detected_objects", [])
                filtered_objects = []

                for obj in detected_objects:
                    confidence = obj.get("confidence", 0)
                    class_name = obj.get("class_name", "")

                    # 為特殊類別設置更高閾值
                    special_classes = ["airplane", "helicopter", "boat"]
                    if class_name in special_classes:
                        if confidence < 0.75:
                            continue

                    if confidence >= high_confidence_threshold:
                        filtered_objects.append(obj)

                for obj in filtered_objects:
                    class_name = obj.get("class_name", "")
                    if class_name not in object_counts:
                        object_counts[class_name] = 0
                    object_counts[class_name] += 1

            # 格式化物件描述
            return ", ".join([
                f"{count} {obj}{'s' if count > 1 else ''}"
                for obj, count in object_counts.items()
            ])

        except Exception as e:
            self.logger.error(f"Object statistics preparation failed: {str(e)}")
            return "objects visible in the scene"

    def _handle_incomplete_response(self, response: str, prompt: str, original_desc: str) -> str:
        """
        處理不完整的回應，必要時重新生成

        Args:
            response: 原始回應
            prompt: 使用的提示詞
            original_desc: 原始描述

        Returns:
            str: 處理後的回應
        """
        try:
            # 檢查回應完整性
            is_complete, issue = self.quality_validator.validate_response_completeness(response)

            max_retries = 3
            attempts = 0

            while not is_complete and attempts < max_retries:
                self.logger.warning(f"Incomplete response detected ({issue}), retrying... Attempt {attempts+1}/{max_retries}")

                # 重新生成
                response = self.model_manager.generate_response(prompt)
                is_complete, issue = self.quality_validator.validate_response_completeness(response)
                attempts += 1

            if not response or len(response.strip()) < 10:
                self.logger.warning("Generated response was empty or too short, returning original description")
                return original_desc

            return response

        except Exception as e:
            self.logger.error(f"Incomplete response handling failed: {str(e)}")
            return response  # 返回原始回應

    def verify_detection(self,
                        detected_objects: List[Dict],
                        clip_analysis: Dict[str, Any],
                        scene_type: str,
                        scene_name: str,
                        confidence: float) -> Dict[str, Any]:
        """
        驗證並可能修正YOLO的檢測結果

        Args:
            detected_objects: YOLO檢測到的物體列表
            clip_analysis: CLIP分析結果
            scene_type: 識別的場景類型
            scene_name: 場景名稱
            confidence: 場景分類的信心度

        Returns:
            Dict: 包含驗證結果和建議的字典
        """
        try:
            self.logger.info("Starting detection verification")

            # 格式化驗證提示
            prompt = self.prompt_manager.format_verification_prompt(
                detected_objects=detected_objects,
                clip_analysis=clip_analysis,
                scene_type=scene_type,
                scene_name=scene_name,
                confidence=confidence
            )

            # 調用LLM進行驗證
            verification_result = self.model_manager.generate_response(prompt)

            # 清理回應
            cleaned_result = self.response_processor.clean_response(verification_result, self.model_path)

            # 解析驗證結果
            result = {
                "verification_text": cleaned_result,
                "has_errors": "appear accurate" not in cleaned_result.lower(),
                "corrected_objects": None
            }

            self.logger.info("Detection verification completed")
            return result

        except Exception as e:
            error_msg = f"Detection verification failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            return {
                "verification_text": "Verification failed",
                "has_errors": False,
                "corrected_objects": None
            }

    def handle_no_detection(self, clip_analysis: Dict[str, Any]) -> str:
        """
        處理YOLO未檢測到物體的情況

        Args:
            clip_analysis: CLIP分析結果

        Returns:
            str: 生成的場景描述
        """
        try:
            self.logger.info("Handling no detection scenario")

            # 格式化無檢測提示
            prompt = self.prompt_manager.format_no_detection_prompt(clip_analysis)

            # 調用LLM生成描述
            description = self.model_manager.generate_response(prompt)

            # 清理回應
            cleaned_description = self.response_processor.clean_response(description, self.model_path)

            self.logger.info("No detection handling completed")
            return cleaned_description

        except Exception as e:
            error_msg = f"No detection handling failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            return "Unable to generate scene description"

    def reset_context(self):
        """重置LLM模型上下文"""
        try:
            self.model_manager.reset_context()
            self.logger.info("LLM context reset completed")
        except Exception as e:
            self.logger.error(f"Context reset failed: {str(e)}")

    def get_call_count(self) -> int:
        """
        獲取模型調用次數

        Returns:
            int: 調用次數
        """
        return self.model_manager.get_call_count()

    def get_model_info(self) -> Dict[str, Any]:
        """
        獲取模型和組件資訊

        Returns:
            Dict[str, Any]: 包含所有組件狀態的綜合資訊
        """
        try:
            return {
                "model_manager": self.model_manager.get_model_info(),
                "prompt_manager": self.prompt_manager.get_template_info(),
                "response_processor": self.response_processor.get_processor_info(),
                "quality_validator": self.quality_validator.get_validator_info(),
                "facade_status": "initialized"
            }
        except Exception as e:
            self.logger.error(f"Failed to get component info: {str(e)}")
            return {"facade_status": "error", "error_message": str(e)}

    def is_model_loaded(self) -> bool:
        """
        檢查模型是否已載入

        Returns:
            bool: 模型載入狀態
        """
        return self.model_manager.is_model_loaded()

    def get_current_device(self) -> str:
        """
        獲取當前運行設備

        Returns:
            str: 當前設備名稱
        """
        return self.model_manager.get_current_device()

    def _detect_scene_type(self, detected_objects: List[Dict]) -> str:
        """
        基於物件分佈和模式檢測場景類型

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            str: 檢測到的場景類型
        """
        try:
            # 預設場景類型
            scene_type = "intersection"

            # 計算物件數量
            object_counts = {}
            for obj in detected_objects:
                class_name = obj.get("class_name", "")
                if class_name not in object_counts:
                    object_counts[class_name] = 0
                object_counts[class_name] += 1

            # 人數統計
            people_count = object_counts.get("person", 0)

            # 交通工具統計
            car_count = object_counts.get("car", 0)
            bus_count = object_counts.get("bus", 0)
            truck_count = object_counts.get("truck", 0)
            total_vehicles = car_count + bus_count + truck_count

            # 簡單的場景類型檢測邏輯
            if people_count > 8 and total_vehicles < 2:
                scene_type = "pedestrian_crossing"
            elif people_count > 5 and total_vehicles > 2:
                scene_type = "busy_intersection"
            elif people_count < 3 and total_vehicles > 3:
                scene_type = "traffic_junction"

            return scene_type

        except Exception as e:
            self.logger.error(f"Scene type detection failed: {str(e)}")
            return "intersection"