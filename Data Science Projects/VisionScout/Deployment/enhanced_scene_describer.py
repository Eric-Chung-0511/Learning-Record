import os
import re
import json
import logging
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from scene_type import SCENE_TYPES
from scene_detail_templates import SCENE_DETAIL_TEMPLATES
from object_template_fillers import OBJECT_TEMPLATE_FILLERS
from lighting_conditions import LIGHTING_CONDITIONS
from viewpoint_templates import VIEWPOINT_TEMPLATES
from cultural_templates import CULTURAL_TEMPLATES
from confidence_templates import CONFIDENCE_TEMPLATES
from landmark_data import ALL_LANDMARKS
from region_analyzer import RegionAnalyzer
from viewpoint_detector import ViewpointDetector, ViewpointDetectionError
from template_manager import TemplateManager, TemplateLoadingError, TemplateFillError
from object_description_generator import ObjectDescriptionGenerator, ObjectDescriptionError
from cultural_context_analyzer import CulturalContextAnalyzer, CulturalContextError
from text_formatter import TextFormatter, TextFormattingError

class EnhancedSceneDescriberError(Exception):
    """場景描述生成過程中的自定義異常"""
    pass

class EnhancedSceneDescriber:
    """
    增強場景描述器 - 提供詳細自然語言場景描述的主要窗口，其他相關class匯集於此

    此class會協調多個專門組件來生成高質量的場景描述，包括視角檢測、
    模板管理、物件描述、文化語境分析和文本格式化。
    """

    def __init__(self, templates_db: Optional[Dict] = None, scene_types: Optional[Dict] = None, spatial_analyzer_instance: Optional[Any] = None):
        """
        初始化增強場景描述器

        Args:
            templates_db: 可選的自定義模板數據庫
            scene_types: 場景類型定義字典
            spatial_analyzer_instance: 空間分析器實例（保持兼容性）
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # 如果沒有logger，就加一個
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        try:
            # 載入場景類型定義
            self.scene_types = scene_types or self._load_default_scene_types()

            # 初始化子組件
            self._initialize_components(templates_db)

            # 保存空間分析器實例以保持兼容性
            self.spatial_analyzer_instance = spatial_analyzer_instance

            self.logger.info("EnhancedSceneDescriber initialized successfully with %d scene types",
                           len(self.scene_types))

        except Exception as e:
            error_msg = f"Failed to initialize EnhancedSceneDescriber: {str(e)}"
            self.logger.error(f"{error_msg}\n{e.__class__.__name__}: {str(e)}")
            raise EnhancedSceneDescriberError(error_msg) from e

    def _load_default_scene_types(self) -> Dict:
        """
        載入默認場景類型

        Returns:
            Dict: 場景類型定義
        """
        try:
            return SCENE_TYPES
        except Exception as e:
            self.logger.error(f"Failed to import SCENE_TYPES: {str(e)}")
            return {}  # 返回空字典

    def _initialize_components(self, templates_db: Optional[Dict]):
        """
        初始化所有子組件

        Args:
            templates_db: 可選的模板數據庫
        """
        try:
            # 初始化視角檢測器
            self.viewpoint_detector = ViewpointDetector()

            # 初始化區域分析器
            self.region_analyzer = RegionAnalyzer()

            # 初始化模板管理器
            self.template_manager = TemplateManager(custom_templates_db=templates_db)

            # 初始化物件描述生成器，傳入區域分析器
            self.object_description_generator = ObjectDescriptionGenerator(
                region_analyzer=self.region_analyzer
            )

            # 初始化文化語境分析器
            self.cultural_context_analyzer = CulturalContextAnalyzer()

            # 初始化文本格式化器
            self.text_formatter = TextFormatter()

            self.logger.debug("All components initialized successfully")

        except Exception as e:
            error_msg = f"Component initialization failed: {str(e)}"
            self.logger.error(error_msg)
            # 初始化基本組件而不是拋出異常
            self._initialize_fallback_components()


    def generate_description(self, scene_type: str, detected_objects: List[Dict], confidence: float,
                           lighting_info: Dict, functional_zones: List[str], enable_landmark: bool = True,
                           scene_scores: Optional[Dict] = None, spatial_analysis: Optional[Dict] = None,
                           image_dimensions: Optional[Tuple[int, int]] = None, # 改為 Tuple
                           places365_info: Optional[Dict] = None,
                           object_statistics: Optional[Dict] = None) -> str:
        try:
            traffic_list = [obj for obj in detected_objects if obj.get("class_name", "") == "traffic light"]
            # print(f"[DEBUG] generate_description 一開始接收到的 traffic light 數量: {len(traffic_list)}") # 原始的 print
            self.logger.debug(f"Initial traffic light count in generate_description: {len(traffic_list)}") # 改用 logger
            # for idx, tl in enumerate(traffic_list): # 這部分 log 可能過於詳細，先註解
            #     self.logger.debug(f"    idx={idx}, confidence={tl.get('confidence', 0):.4f}, bbox={tl.get('bbox')}, region={tl.get('region')}")

            if scene_type == "unknown" or confidence < 0.4:
                generic_desc = self._generate_generic_description(detected_objects, lighting_info)
                return self.text_formatter.format_final_description(generic_desc)

            current_detected_objects = detected_objects
            if not enable_landmark:
                current_detected_objects = [obj for obj in detected_objects if not obj.get("is_landmark", False)]

            places365_context = ""
            if places365_info and places365_info.get('confidence', 0) > 0.3:
                scene_label = places365_info.get('scene_label', '')
                attributes = places365_info.get('attributes', [])
                is_indoor = places365_info.get('is_indoor', None)
                if scene_label:
                    places365_context = f"Scene context: {scene_label}"
                    if attributes:
                        places365_context += f" with characteristics: {', '.join(attributes[:3])}"
                    if is_indoor is not None:
                        indoor_outdoor = "indoor" if is_indoor else "outdoor"
                        places365_context += f" ({indoor_outdoor} environment)"
                self.logger.debug(f"Enhanced description incorporating Places365 context: {places365_context}")

            landmark_objects_in_scene = [obj for obj in current_detected_objects if obj.get("is_landmark", False)]
            has_landmark_in_scene = len(landmark_objects_in_scene) > 0

            if enable_landmark and (scene_type in ["tourist_landmark", "natural_landmark", "historical_monument"] or has_landmark_in_scene):
                landmark_desc = self._generate_landmark_description(
                    scene_type, current_detected_objects, confidence,
                    lighting_info, functional_zones, landmark_objects_in_scene
                )
                return self.text_formatter.format_final_description(landmark_desc)

            viewpoint = self.viewpoint_detector.detect_viewpoint(current_detected_objects)
            current_scene_type = scene_type

            if viewpoint == "aerial":
                if "intersection" in current_scene_type.lower() or self._is_intersection(current_detected_objects):
                    current_scene_type = "aerial_view_intersection"
                elif any(keyword in current_scene_type.lower() for keyword in ["commercial", "shopping", "retail"]):
                    current_scene_type = "aerial_view_commercial_area"
                elif any(keyword in current_scene_type.lower() for keyword in ["plaza", "square"]):
                    current_scene_type = "aerial_view_plaza"
                else:
                    current_scene_type = "aerial_view_general"

            current_scene_type = self._sanitize_scene_type_for_description(current_scene_type)

            # 偵測文化背景資訊
            cultural_context = None
            if viewpoint != "aerial":
                cultural_context = self.cultural_context_analyzer.detect_cultural_context(current_scene_type, current_detected_objects)

             # 設定基礎描述
            base_description = "A scene"
            if viewpoint == "aerial":
                if current_scene_type in self.scene_types: # 確保 self.scene_types 已有
                    base_description = self.scene_types.get(current_scene_type, {}).get("description", "An aerial view showing the layout and movement patterns from above")
                else:
                    base_description = "An aerial view showing the layout and movement patterns from above"
            elif current_scene_type in self.scene_types: # 確保 self.scene_types 已有
                 base_description = self.scene_types.get(current_scene_type, {}).get("description", "A scene")

            # 假設 template_manager 內部可以處理 List[str] 的 functional_zones
            selected_template = self.template_manager.get_template_by_scene_type(
                scene_type=current_scene_type,
                detected_objects=current_detected_objects,
                functional_zones=functional_zones or [] # 傳入 List[str]
            )

            # 用於 fill_template 中的某些佔位符
            processed_functional_zones = {}
            if functional_zones:
                if isinstance(functional_zones, dict): # 如果外部傳入的就是dict
                     processed_functional_zones = functional_zones
                elif isinstance(functional_zones, list): # 如果是 list of strings
                     processed_functional_zones = {f"zone_{i}": {"description": zone_desc} for i, zone_desc in enumerate(functional_zones)}


            # 組織場景資料
            scene_data = {
                "detected_objects": current_detected_objects,
                "functional_zones": processed_functional_zones, # 傳入處理過的字典
                "scene_type": current_scene_type,
                "object_statistics": object_statistics or {},
                "lighting_info": lighting_info,
                "spatial_analysis": spatial_analysis,
                "places365_info": places365_info
            }

            # 應用模板產生核心場景描述
            core_scene_details = self.template_manager.apply_template(selected_template, scene_data)

            # 組合基礎描述與核心場景細節
            description = base_description
            if core_scene_details and core_scene_details.strip():
                cleaned_scene_details = self._validate_and_clean_scene_details(core_scene_details)
                if base_description.lower() == "a scene" and len(cleaned_scene_details) > len(base_description):
                    description = cleaned_scene_details
                else:
                    description = self.text_formatter.smart_append(description, cleaned_scene_details)
            elif not core_scene_details and not description: # 如果兩者都為空
                description = self._generate_generic_description(current_detected_objects, lighting_info)

            # 添加次要描述資訊
            if current_scene_type in self.scene_types and "secondary_description" in self.scene_types[current_scene_type]:
                secondary_desc = self.scene_types[current_scene_type]["secondary_description"]
                if secondary_desc:
                    description = self.text_formatter.smart_append(description, secondary_desc)
                    
            # 處理人物相關的描述
            people_objs = [obj for obj in current_detected_objects if obj.get("class_id") == 0]
            if people_objs:
                people_count = len(people_objs)
                if people_count == 1: people_phrase = "a single person"
                elif 1 < people_count <= 3: people_phrase = f"{people_count} people"
                elif 3 < people_count <= 7: people_phrase = "several people"
                else: people_phrase = "multiple people"
                if not any(p_word in description.lower() for p_word in ["person", "people", "pedestrian"]):
                    description = self.text_formatter.smart_append(description, f"The scene includes {people_phrase}.")

            # 添加文化背景元素(非空中視角）
            if cultural_context and viewpoint != "aerial":
                cultural_elements = self.cultural_context_analyzer.generate_cultural_elements(cultural_context)
                if cultural_elements:
                    description = self.text_formatter.smart_append(description, cultural_elements)

            # 處理光照條件描述
            lighting_description_text = ""
            if lighting_info and "time_of_day" in lighting_info:
                lighting_type = lighting_info["time_of_day"]
                lighting_desc_template = self.template_manager.get_lighting_template(lighting_type)
                if lighting_desc_template: lighting_description_text = lighting_desc_template
            if lighting_description_text and lighting_description_text.lower() not in description.lower():
                description = self.text_formatter.smart_append(description, lighting_description_text)

             # 添加視角特定的觀察描述
            if viewpoint != "eye_level":
                viewpoint_template = self.template_manager.get_viewpoint_template(viewpoint)
                prefix = viewpoint_template.get('prefix', '')
                observation_template = viewpoint_template.get("observation", "")
                scene_elements_for_vp = "the overall layout and objects"
                if viewpoint == "aerial": scene_elements_for_vp = "crossing patterns and general layout"
                viewpoint_observation_text = observation_template.format(scene_elements=scene_elements_for_vp)
                full_viewpoint_text = ""
                if prefix:
                    full_viewpoint_text = prefix.strip() + " "
                    if viewpoint_observation_text and viewpoint_observation_text[0].islower():
                        full_viewpoint_text += viewpoint_observation_text
                    elif viewpoint_observation_text:
                        full_viewpoint_text = prefix + (viewpoint_observation_text[0].lower() + viewpoint_observation_text[1:] if description else viewpoint_observation_text)
                elif viewpoint_observation_text:
                    full_viewpoint_text = viewpoint_observation_text[0].upper() + viewpoint_observation_text[1:]
                if full_viewpoint_text and full_viewpoint_text.lower() not in description.lower():
                    description = self.text_formatter.smart_append(description, full_viewpoint_text)

            # 需要轉換或調整 describe_functional_zones
            if functional_zones and len(functional_zones) > 0:
                if isinstance(functional_zones, dict):
                     zones_desc_text = self.object_description_generator.describe_functional_zones(functional_zones)
                else: # 如果是 list of strings
                     temp_zones_dict = {f"area_{i}": {"description": desc} for i, desc in enumerate(functional_zones)}
                     zones_desc_text = self.object_description_generator.describe_functional_zones(temp_zones_dict)

                if zones_desc_text:
                    description = self.text_formatter.smart_append(description, zones_desc_text)

            # 避免重複提到
            if hasattr(self.text_formatter, 'deduplicate_sentences_in_description'):
                deduplicated_description = self.text_formatter.deduplicate_sentences_in_description(description)
                self.logger.info(f"Description before pre-LLM deduplication (len {len(description)}): '{description[:150]}...'")
                self.logger.info(f"Description after pre-LLM deduplication (len {len(deduplicated_description)}): '{deduplicated_description[:150]}...'")
                description = deduplicated_description # 更新 description 為去除重複後的版本
            else:
                self.logger.warning("TextFormatter does not have 'deduplicate_sentences_in_description'. Skipping pre-LLM deduplication of the internally generated description.")

            # 格式化最終描述
            final_formatted_description = self.text_formatter.format_final_description(description)

            # 如果禁用地標，過濾地標引用
            if not enable_landmark:
                final_formatted_description = self.text_formatter.filter_landmark_references(final_formatted_description, enable_landmark=False)

            # 如果描述為空，使用備用描述
            if not final_formatted_description.strip() or final_formatted_description.strip() == ".":
                self.logger.warning(f"Description for scene_type '{current_scene_type}' became empty after processing. Falling back.")
                final_formatted_description = self.text_formatter.format_final_description(
                    self._generate_generic_description(current_detected_objects, lighting_info)
                )

            return final_formatted_description

        except Exception as e:
            error_msg = f"Error generating scene description: {str(e)}"
            self.logger.error(f"{error_msg}\n{e.__class__.__name__}: {str(e)}")
            try:
                fallback_desc = self._generate_generic_description(detected_objects, lighting_info)
                return self.text_formatter.format_final_description(fallback_desc)
            except:
                return "A scene with various elements is visible."

    def deduplicate_sentences_in_description(self, description: str, similarity_threshold: float = 0.80) -> str:
        """
        從一段描述文本中移除重複或高度相似的句子。
        此方法會嘗試保留更長、資訊更豐富的句子版本。

        Args:
            description (str): 原始描述文本。
            similarity_threshold (float): 判斷句子是否相似的 Jaccard 相似度閾值 (0 到 1)。
                                         預設為 0.8，表示詞彙重疊度達到80%即視為相似。

        Returns:
            str: 移除了重複或高度相似句子後的文本。
        """
        try:
            if not description or not description.strip():
                self.logger.debug("deduplicate_sentences_in_description: Received empty or blank description.")
                return ""

            # 使用正則表達式分割句子，保留句尾標點符號
            sentences = re.split(r'(?<=[.!?])\s+', description.strip())

            if not sentences:
                self.logger.debug("deduplicate_sentences_in_description: No sentences found after splitting.")
                return ""

            unique_sentences_data = []  # 存儲 (原始句子文本, 該句子的詞彙集合)

            for current_sentence_text in sentences:
                current_sentence_text = current_sentence_text.strip()
                if not current_sentence_text:
                    continue

                # 預處理當前句子以進行比較：轉小寫、移除標點、分割成詞彙集合
                simplified_current_text = re.sub(r'[^\w\s\d]', '', current_sentence_text.lower()) # 保留數字
                current_sentence_words = set(simplified_current_text.split())

                if not current_sentence_words: # 如果處理後是空集合 (例如句子只包含標點)
                    # 如果原始句子有內容（例如只有一個標點），就保留它
                    if current_sentence_text and not unique_sentences_data: # 避免在開頭加入孤立標點
                         unique_sentences_data.append((current_sentence_text, current_sentence_words))
                    continue

                is_subsumed_or_highly_similar = False
                index_to_replace = -1

                for i, (kept_sentence_text, kept_sentence_words) in enumerate(unique_sentences_data):
                    if not kept_sentence_words: # 跳過已保留的空詞彙集合
                        continue

                    # 計算 Jaccard 相似度
                    intersection_len = len(current_sentence_words.intersection(kept_sentence_words))
                    union_len = len(current_sentence_words.union(kept_sentence_words))

                    jaccard_similarity = 0.0
                    if union_len > 0:
                        jaccard_similarity = intersection_len / union_len
                    elif not current_sentence_words and not kept_sentence_words: # 兩個都是空的
                        jaccard_similarity = 1.0


                    if jaccard_similarity >= similarity_threshold:
                        # 如果當前句子比已保留的句子長，則標記替換舊的
                        if len(current_sentence_words) > len(kept_sentence_words):
                            self.logger.debug(f"Deduplication: Replacing shorter \"{kept_sentence_text[:50]}...\" "
                                              f"with longer similar \"{current_sentence_text[:50]}...\" (Jaccard: {jaccard_similarity:.2f})")
                            index_to_replace = i
                            break # 找到一個可以被替換的，就跳出內層循環
                        # 如果當前句子比已保留的句子短，或者長度相近但內容高度相似，則標記當前句子為重複
                        else: # current_sentence_words is shorter or of similar length
                            is_subsumed_or_highly_similar = True
                            self.logger.debug(f"Deduplication: Current sentence \"{current_sentence_text[:50]}...\" "
                                              f"is subsumed by or highly similar to \"{kept_sentence_text[:50]}...\" (Jaccard: {jaccard_similarity:.2f}). Skipping.")
                            break

                if index_to_replace != -1:
                    unique_sentences_data[index_to_replace] = (current_sentence_text, current_sentence_words)
                elif not is_subsumed_or_highly_similar:
                    unique_sentences_data.append((current_sentence_text, current_sentence_words))

            # 從 unique_sentences_data 中提取最終的句子文本
            final_sentences = [s_data[0] for s_data in unique_sentences_data]

            # 重組句子，確保每個句子以標點符號結尾，並且句子間有空格
            reconstructed_response = ""
            for i, s_text in enumerate(final_sentences):
                s_text = s_text.strip()
                if not s_text:
                    continue
                # 確保句子以標點結尾
                if not re.search(r'[.!?]$', s_text):
                    s_text += "."

                reconstructed_response += s_text
                if i < len(final_sentences) - 1: # 如果不是最後一句，添加空格
                    reconstructed_response += " "

            self.logger.debug(f"Deduplicated description (len {len(reconstructed_response.strip())}): '{reconstructed_response.strip()[:150]}...'")
            return reconstructed_response.strip()

        except Exception as e:
            self.logger.error(f"Error in deduplicate_sentences_in_description: {str(e)}")
            self.logger.error(traceback.format_exc())
            return description # 發生錯誤時返回原始描述

    def _extract_placeholders(self, template: str) -> List[str]:
        """提取模板中的佔位符"""
        import re
        return re.findall(r'\{([^}]+)\}', template)

    def _generate_placeholder_content(self, placeholder: str, detected_objects: List[Dict],
                                    functional_zones: List, scene_type: str,
                                    object_statistics: Dict) -> str:
        """生成佔位符內容"""
        all_replacements = self._generate_default_replacements()
        return self._get_placeholder_replacement(
            placeholder, {}, all_replacements, detected_objects, scene_type
        )

    def _preprocess_functional_zones(self, functional_zones: List) -> Dict:
        """預處理功能區域數據"""
        if isinstance(functional_zones, list):
            # 將列表轉換為字典格式
            zones_dict = {}
            for i, zone in enumerate(functional_zones):
                if isinstance(zone, str):
                    zones_dict[f"area {i+1}"] = {"description": zone}
                elif isinstance(zone, dict):
                    zones_dict[f"area {i+1}"] = zone
            return zones_dict
        elif isinstance(functional_zones, dict):
            return functional_zones
        else:
            return {}

    def _standardize_placeholder_content(self, content: str, placeholder_type: str) -> str:
        """標準化佔位符內容"""
        if not content:
            return "various elements"
        return content.strip()

    def _finalize_description_output(self, description: str) -> str:
        """最終化描述輸出"""
        if not description:
            return "A scene featuring various elements and organized areas of activity."

        # 基本清理
        import re
        finalized = re.sub(r'\s+', ' ', description).strip()

        # 確保適當結尾
        if finalized and not finalized.endswith(('.', '!', '?')):
            finalized += '.'

        # 首字母大寫
        if finalized:
            finalized = finalized[0].upper() + finalized[1:] if len(finalized) > 1 else finalized.upper()

        return finalized

    def _sanitize_scene_type_for_description(self, scene_type: str) -> str:
        """
        清理場景類型名稱，確保不包含內部標識符格式

        Args:
            scene_type: 原始場景類型名稱

        Returns:
            str: 清理後的場景類型名稱
        """
        try:
            # 移除下劃線並轉換為空格分隔的自然語言
            cleaned_type = scene_type.replace('_', ' ')

            # 確保不直接在描述中使用技術性場景類型名稱
            return cleaned_type

        except Exception as e:
            self.logger.warning(f"Error sanitizing scene type '{scene_type}': {str(e)}")
            return "general scene"

    def _validate_and_clean_scene_details(self, scene_details: str) -> str:
        """
        驗證並清理場景詳細信息，移除可能的模板填充錯誤

        Args:
            scene_details: 原始場景詳細信息

        Returns:
            str: 清理後的場景詳細信息
        """
        try:
            if not scene_details or not scene_details.strip():
                return ""

            cleaned = scene_details.strip()

            # 移除常見的模板填充錯誤模式
            import re

            # 修復 "In ," 類型的錯誤
            cleaned = re.sub(r'\bIn\s*,\s*', 'In this scene, ', cleaned)
            cleaned = re.sub(r'\bAt\s*,\s*', 'At this location, ', cleaned)
            cleaned = re.sub(r'\bWithin\s*,\s*', 'Within this area, ', cleaned)

            # 移除內部標識符格式
            cleaned = re.sub(r'\b\w+_\w+(?:_\w+)*\b(?!\s+(area|zone|region))',
                            lambda m: m.group(0).replace('_', ' '), cleaned)

            # 確保句子完整性
            if cleaned and not cleaned.endswith(('.', '!', '?')):
                cleaned += '.'

            return cleaned

        except Exception as e:
            self.logger.warning(f"Error validating scene details: {str(e)}")
            return scene_details if scene_details else ""

    def _generate_landmark_description(self,
                                     scene_type: str,
                                     detected_objects: List[Dict],
                                     confidence: float,
                                     lighting_info: Optional[Dict] = None,
                                     functional_zones: Optional[Dict] = None,
                                     landmark_objects: Optional[List[Dict]] = None) -> str:
        """
        生成包含地標信息的場景描述

        Args:
            scene_type: 識別的場景類型
            detected_objects: 檢測到的物件列表
            confidence: 場景分類置信度
            lighting_info: 照明條件信息
            functional_zones: 功能區域信息
            landmark_objects: 識別為地標的物件列表

        Returns:
            str: 包含地標信息的自然語言場景描述
        """
        try:
            # 如果沒有提供地標物件，從檢測物件中篩選
            if landmark_objects is None:
                landmark_objects = [obj for obj in detected_objects if obj.get("is_landmark", False)]

            # 如果沒有地標，退回到標準描述
            if not landmark_objects:
                if scene_type in ["tourist_landmark", "natural_landmark", "historical_monument"]:
                    base_description = "A scenic area that appears to be a tourist destination, though specific landmarks are not clearly identifiable."
                else:
                    return self.text_formatter.format_final_description(self._generate_scene_details(
                        scene_type,
                        detected_objects,
                        lighting_info,
                        self.viewpoint_detector.detect_viewpoint(detected_objects)
                    ))
            else:
                # 獲取主要地標
                primary_landmark = max(landmark_objects, key=lambda x: x.get("confidence", 0))
                landmark_name = primary_landmark.get("class_name", "landmark")
                # 先取原生 location
                landmark_location = primary_landmark.get("location", "")
                # 如果 location 為空，就從全域 ALL_LANDMARKS 補上
                lm_id = primary_landmark.get("landmark_id")
                if not landmark_location and lm_id and lm_id in ALL_LANDMARKS:
                    landmark_location = ALL_LANDMARKS[lm_id].get("location", "")

                # 根據地標類型選擇適當的描述模板，並插入 location
                if scene_type == "natural_landmark" or primary_landmark.get("landmark_type") == "natural":
                    base_description = f"A natural landmark scene featuring {landmark_name} in {landmark_location}."
                elif scene_type == "historical_monument" or primary_landmark.get("landmark_type") == "monument":
                    base_description = f"A historical monument scene showcasing {landmark_name}, a significant landmark in {landmark_location}."
                else:
                    base_description = f"A tourist landmark scene centered around {landmark_name}, an iconic structure in {landmark_location}."

            # 添加地標的額外信息
            landmark_details = []
            for landmark in landmark_objects:
                details = []

                if "year_built" in landmark:
                    details.append(f"built in {landmark['year_built']}")

                if "architectural_style" in landmark:
                    details.append(f"featuring {landmark['architectural_style']} architectural style")

                if "significance" in landmark:
                    details.append(landmark["significance"])

                # 補 location（如果該物件沒有 location，就再從 ALL_LANDMARKS 撈一次）
                loc = landmark.get("location", "")
                lm_id_iter = landmark.get("landmark_id")
                if not loc and lm_id_iter and lm_id_iter in ALL_LANDMARKS:
                    loc = ALL_LANDMARKS[lm_id_iter].get("location", "")
                if loc:
                    details.append(f"located in {loc}")

                if details:
                    landmark_details.append(f"{landmark['class_name']} ({', '.join(details)})")

            # 將詳細信息添加到基本描述中
            if landmark_details:
                description = base_description + " The scene features " + ", ".join(landmark_details) + "."
            else:
                description = base_description

            # 獲取視角
            viewpoint = self.viewpoint_detector.detect_viewpoint(detected_objects)

            # 生成人員活動描述
            people_count = len([obj for obj in detected_objects if obj["class_id"] == 0])

            if people_count > 0:
                if people_count == 1:
                    people_description = "There is one person in the scene, likely a tourist or visitor."
                elif people_count < 5:
                    people_description = f"There are {people_count} people in the scene, possibly tourists visiting the landmark."
                else:
                    people_description = f"The scene includes a group of {people_count} people, indicating this is a popular tourist destination."

                description = self.text_formatter.smart_append(description, people_description)

            # 添加照明信息
            if lighting_info and "time_of_day" in lighting_info:
                lighting_type = lighting_info["time_of_day"]
                lighting_description = self.template_manager.get_lighting_template(lighting_type)
                description = self.text_formatter.smart_append(description, lighting_description)

            # 添加視角描述
            if viewpoint != "eye_level":
                viewpoint_template = self.template_manager.get_viewpoint_template(viewpoint)

                prefix = viewpoint_template.get('prefix', '')
                if prefix and not description.startswith(prefix):
                    if description and description[0].isupper():
                        description = prefix + description[0].lower() + description[1:]
                    else:
                        description = prefix + description

                viewpoint_desc = viewpoint_template.get("observation", "").format(
                    scene_elements="the landmark and surrounding area"
                )

                if viewpoint_desc and viewpoint_desc not in description:
                    description = self.text_formatter.smart_append(description, viewpoint_desc)

            # 添加功能區域描述
            if functional_zones and len(functional_zones) > 0:
                zones_desc = self.object_description_generator.describe_functional_zones(functional_zones)
                if zones_desc:
                    description = self.text_formatter.smart_append(description, zones_desc)

            # 描述可能的活動
            landmark_activities = []

            if scene_type == "natural_landmark" or any(obj.get("landmark_type") == "natural" for obj in landmark_objects):
                landmark_activities = [
                    "nature photography",
                    "scenic viewing",
                    "hiking or walking",
                    "guided nature tours",
                    "outdoor appreciation"
                ]
            elif scene_type == "historical_monument" or any(obj.get("landmark_type") == "monument" for obj in landmark_objects):
                landmark_activities = [
                    "historical sightseeing",
                    "educational tours",
                    "cultural appreciation",
                    "photography of historical architecture",
                    "learning about historical significance"
                ]
            else:
                landmark_activities = [
                    "sightseeing",
                    "taking photographs",
                    "guided tours",
                    "cultural tourism",
                    "souvenir shopping"
                ]

            # 添加活動描述
            if landmark_activities:
                activities_text = "Common activities at this location include " + ", ".join(landmark_activities[:3]) + "."
                description = self.text_formatter.smart_append(description, activities_text)

            return self.text_formatter.format_final_description(description)

        except Exception as e:
            self.logger.warning(f"Error generating landmark description: {str(e)}")
            # 備用處理
            return self.text_formatter.format_final_description(
                "A landmark scene with notable architectural or natural features."
            )


    def _is_intersection(self, detected_objects: List[Dict]) -> bool:
        """
        通過分析物件分布來判斷場景是否為十字路口

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            bool: 是否為十字路口
        """
        try:
            pedestrians = [obj for obj in detected_objects if obj.get("class_id") == 0]

            if len(pedestrians) >= 8:
                positions = [obj.get("normalized_center", (0, 0)) for obj in pedestrians]

                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]

                x_variance = np.var(x_coords) if len(x_coords) > 1 else 0
                y_variance = np.var(y_coords) if len(y_coords) > 1 else 0

                x_range = max(x_coords) - min(x_coords)
                y_range = max(y_coords) - min(y_coords)

                if x_range > 0.5 and y_range > 0.5 and 0.7 < (x_range / y_range) < 1.3:
                    return True

            return False

        except Exception as e:
            self.logger.warning(f"Error detecting intersection: {str(e)}")
            return False

    def _generate_generic_description(self, detected_objects: List[Dict], lighting_info: Optional[Dict] = None) -> str:
        """
        當場景類型未知或置信度極低時生成通用描述

        Args:
            detected_objects: 檢測到的物件列表
            lighting_info: 可選的照明條件信息

        Returns:
            str: 基於檢測物件的通用描述
        """
        try:
            obj_counts = {}
            for obj in detected_objects:
                class_name = obj.get("class_name", "unknown object")
                if class_name not in obj_counts:
                    obj_counts[class_name] = 0
                obj_counts[class_name] += 1

            top_objects = sorted(obj_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            if not top_objects:
                base_desc = "This scene displays various elements, though specific objects are not clearly identifiable."
            else:
                objects_text = []
                for name, count in top_objects:
                    # 確保物件名稱不包含技術性格式
                    clean_name = name.replace('_', ' ') if isinstance(name, str) else str(name)
                    if count > 1:
                        objects_text.append(f"{count} {clean_name}s")
                    else:
                        objects_text.append(f"a {clean_name}" if clean_name[0].lower() not in 'aeiou' else f"an {clean_name}")

                if len(objects_text) == 1:
                    objects_list = objects_text[0]
                elif len(objects_text) == 2:
                    objects_list = f"{objects_text[0]} and {objects_text[1]}"
                else:
                    objects_list = ", ".join(objects_text[:-1]) + f", and {objects_text[-1]}"

                base_desc = f"This scene features {objects_list}."

            # 添加照明信息
            if lighting_info and "time_of_day" in lighting_info:
                lighting_type = lighting_info["time_of_day"]
                lighting_desc = self.template_manager.get_lighting_template(lighting_type)
                base_desc += f" {lighting_desc}"

            return base_desc

        except Exception as e:
            self.logger.warning(f"Error generating generic description: {str(e)}")
            return "A general scene is visible with various elements."

    def _generate_scene_details(self,
                              scene_type: str,
                              detected_objects: List[Dict],
                              lighting_info: Optional[Dict] = None,
                              viewpoint: str = "eye_level",
                              spatial_analysis: Optional[Dict] = None,
                              image_dimensions: Optional[Tuple[int, int]] = None,
                              places365_info: Optional[Dict] = None,
                              object_statistics: Optional[Dict] = None) -> str:
        """
        基於場景類型和檢測物件生成詳細描述

        Args:
            scene_type: 識別的場景類型
            detected_objects: 檢測到的物件列表
            lighting_info: 可選的照明條件信息
            viewpoint: 檢測到的視角
            spatial_analysis: 可選的空間分析結果
            image_dimensions: 可選的圖像尺寸
            places365_info: 可選的 Places365 場景分類結果
            object_statistics: 可選的詳細物件統計信息

        Returns:
            str: 詳細場景描述
        """
        try:
            scene_details = ""

            # 日常場景類型列表
            everyday_scene_types = [
                "general_indoor_space", "generic_street_view",
                "desk_area_workspace", "outdoor_gathering_spot",
                "kitchen_counter_or_utility_area", "unknown"
            ]

            # 預處理場景類型以避免內部格式洩漏
            processed_scene_type = self._sanitize_scene_type_for_description(scene_type)

            # 確定場景描述方法
            is_confident_specific_scene = scene_type not in everyday_scene_types and scene_type in self.template_manager.get_scene_detail_templates(scene_type)
            treat_as_everyday = scene_type in everyday_scene_types

            if hasattr(self, 'enable_landmark') and not self.enable_landmark:
                if scene_type not in ["kitchen", "bedroom", "living_room", "office_workspace", "dining_area", "professional_kitchen"]:
                    treat_as_everyday = True

            if treat_as_everyday or not is_confident_specific_scene:
                self.logger.debug(f"Generating dynamic description for scene_type: {scene_type}")
                scene_details = self.object_description_generator.generate_dynamic_everyday_description(
                    detected_objects,
                    lighting_info,
                    viewpoint,
                    spatial_analysis,
                    image_dimensions,
                    places365_info,
                    object_statistics
                )
            else:
                self.logger.debug(f"Using template for scene_type: {scene_type}")
                templates_list = self.template_manager.get_scene_detail_templates(scene_type, viewpoint)

                if templates_list:
                    detail_template = random.choice(templates_list)
                    scene_details = self.template_manager.fill_template(
                        detail_template,
                        detected_objects,
                        scene_type,
                        places365_info,
                        object_statistics
                    )
                else:
                    scene_details = self.object_description_generator.generate_dynamic_everyday_description(
                        detected_objects, lighting_info, viewpoint, spatial_analysis,
                        image_dimensions, places365_info, object_statistics
                    )

            # 如果禁用地標檢測，過濾地標引用
            if hasattr(self, 'enable_landmark') and not self.enable_landmark:
                scene_details = self.text_formatter.filter_landmark_references(scene_details, enable_landmark=False)

            return scene_details if scene_details else "A scene with some visual elements."

        except Exception as e:
            self.logger.warning(f"Error generating scene details: {str(e)}")
            return "A scene with various elements."

    def filter_landmark_references(self, text, enable_landmark=True):
        """
        動態過濾文本中的地標引用

        Args:
            text: 需要過濾的文本
            enable_landmark: 是否啟用地標功能

        Returns:
            str: 過濾後的文本
        """
        return self.text_formatter.filter_landmark_references(text, enable_landmark)

    def get_prominent_objects(self, detected_objects: List[Dict],
                          min_prominence_score: float = 0.5,
                          max_categories_to_return: Optional[int] = None,
                          max_total_objects: Optional[int] = None) -> List[Dict]:
        """
        獲取最重要的物件

        Args:
            detected_objects: 檢測到的物件列表
            min_prominence_score: 最小重要性分數閾值，預設為0.5
            max_categories_to_return: 可選的最大返回類別數量限制
            max_total_objects: 可選的最大返回物件總數限制

        Returns:
            List[Dict]: 重要物件列表
        """
        try:
            # 傳遞所有參數
            prominent_objects = self.object_description_generator.get_prominent_objects(
                detected_objects,
                min_prominence_score,
                max_categories_to_return
            )

            # 如果指定了最大物件總數限制，進行額外過濾
            if max_total_objects is not None and max_total_objects > 0:
                # 限制總物件數量，保持重要性排序
                prominent_objects = prominent_objects[:max_total_objects]

            # 如果指定了最大類別數量限制，則進行額外過濾
            if max_categories_to_return is not None and max_categories_to_return > 0:
                # 按類別分組物件
                categories_seen = set()
                filtered_objects = []

                for obj in prominent_objects:
                    class_name = obj.get("class_name", "unknown")
                    if class_name not in categories_seen:
                        categories_seen.add(class_name)
                        filtered_objects.append(obj)

                        # 如果已達到最大類別數量，停止添加新類別
                        if len(categories_seen) >= max_categories_to_return:
                            break
                    elif class_name in categories_seen:
                        # 如果是已見過的類別，仍然添加該物件
                        filtered_objects.append(obj)

                return filtered_objects

            return prominent_objects

        except Exception as e:
            self.logger.warning(f"Error getting prominent objects: {str(e)}")
            return []

    def detect_viewpoint(self, detected_objects: List[Dict]) -> str:
        """
        檢測圖像視角類型

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            str: 檢測到的視角類型
        """
        try:
            return self.viewpoint_detector.detect_viewpoint(detected_objects)
        except Exception as e:
            self.logger.warning(f"Error detecting viewpoint: {str(e)}")
            return "eye_level"

    def detect_cultural_context(self, scene_type: str, detected_objects: List[Dict]) -> Optional[str]:
        """
        檢測場景的文化語境

        Args:
            scene_type: 識別的場景類型
            detected_objects: 檢測到的物件列表

        Returns:
            Optional[str]: 檢測到的文化語境或None
        """
        try:
            return self.cultural_context_analyzer.detect_cultural_context(scene_type, detected_objects)
        except CulturalContextError as e:
            self.logger.warning(f"Error detecting cultural context: {str(e)}")
            return None

    def generate_cultural_elements(self, cultural_context: str) -> str:
        """
        為檢測到的文化語境生成描述元素

        Args:
            cultural_context: 檢測到的文化語境

        Returns:
            str: 文化元素描述
        """
        try:
            return self.cultural_context_analyzer.generate_cultural_elements(cultural_context)
        except CulturalContextError as e:
            self.logger.warning(f"Error generating cultural elements: {str(e)}")
            return ""

    def format_object_list_for_description(self, objects: List[Dict],
                                         use_indefinite_article_for_one: bool = False,
                                         count_threshold_for_generalization: int = -1,
                                         max_types_to_list: int = 5) -> str:
        """
        將物件列表格式化為人類可讀的字符串

        Args:
            objects: 物件字典列表
            use_indefinite_article_for_one: 單個物件是否使用 "a/an"
            count_threshold_for_generalization: 計數閾值
            max_types_to_list: 最大物件類型數量

        Returns:
            str: 格式化的物件描述字符串
        """
        try:
            return self.object_description_generator.format_object_list_for_description(
                objects, use_indefinite_article_for_one, count_threshold_for_generalization, max_types_to_list
            )
        except ObjectDescriptionError as e:
            self.logger.warning(f"Error formatting object list: {str(e)}")
            return "various objects"

    def get_spatial_description(self, obj: Dict, image_width: Optional[int] = None,
                              image_height: Optional[int] = None) -> str:
        """
        為物件生成空間位置描述

        Args:
            obj: 物件字典
            image_width: 可選的圖像寬度
            image_height: 可選的圖像高度

        Returns:
            str: 空間描述字符串
        """
        try:
            return self.object_description_generator.get_spatial_description(obj, image_width, image_height)
        except ObjectDescriptionError as e:
            self.logger.warning(f"Error generating spatial description: {str(e)}")
            return "in the scene"

    def optimize_object_description(self, description: str) -> str:
        """
        優化物件描述，避免重複列舉相同物件

        Args:
            description: 原始描述文本

        Returns:
            str: 優化後的描述文本
        """
        try:
            return self.object_description_generator.optimize_object_description(description)
        except ObjectDescriptionError as e:
            self.logger.warning(f"Error optimizing object description: {str(e)}")
            return description

    def describe_functional_zones(self, functional_zones: Dict) -> str:
        """
        生成場景功能區域的描述

        Args:
            functional_zones: 識別出的功能區域字典

        Returns:
            str: 功能區域描述
        """
        try:
            return self.object_description_generator.describe_functional_zones(functional_zones)
        except ObjectDescriptionError as e:
            self.logger.warning(f"Error describing functional zones: {str(e)}")
            return ""

    def smart_append(self, current_text: str, new_fragment: str) -> str:
        """
        智能地將新文本片段附加到現有文本

        Args:
            current_text: 要附加到的現有文本
            new_fragment: 要附加的新文本片段

        Returns:
            str: 合併後的文本
        """
        try:
            return self.text_formatter.smart_append(current_text, new_fragment)
        except TextFormattingError as e:
            self.logger.warning(f"Error in smart append: {str(e)}")
            return f"{current_text} {new_fragment}" if current_text else new_fragment

    def format_final_description(self, text: str) -> str:
        """
        格式化最終描述文本

        Args:
            text: 要格式化的文本

        Returns:
            str: 格式化後的文本
        """
        try:
            return self.text_formatter.format_final_description(text)
        except TextFormattingError as e:
            self.logger.warning(f"Error formatting final description: {str(e)}")
            return text

    def get_template(self, category: str, key: Optional[str] = None):
        """
        獲取指定類別的模板

        Args:
            category: 模板類別名稱
            key: 可選的具體模板鍵值

        Returns:
            模板內容
        """
        try:
            return self.template_manager.get_template(category, key)
        except (TemplateLoadingError, TemplateFillError) as e:
            self.logger.warning(f"Error getting template: {str(e)}")
            return None

    def get_viewpoint_confidence(self, detected_objects: List[Dict]) -> Tuple[str, float]:
        """
        獲取視角檢測結果及其信心度

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            Tuple[str, float]: (視角類型, 信心度)
        """
        try:
            return self.viewpoint_detector.get_viewpoint_confidence(detected_objects)
        except ViewpointDetectionError as e:
            self.logger.warning(f"Error getting viewpoint confidence: {str(e)}")
            return "eye_level", 0.5

    def get_supported_cultures(self) -> List[str]:
        """
        獲取所有支援的文化語境列表

        Returns:
            List[str]: 支援的文化語境名稱列表
        """
        return self.cultural_context_analyzer.get_supported_cultures()

    def has_cultural_context(self, cultural_context: str) -> bool:
        """
        檢查是否支援指定的文化語境

        Args:
            cultural_context: 文化語境名稱

        Returns:
            bool: 是否支援該文化語境
        """
        return self.cultural_context_analyzer.has_cultural_context(cultural_context)

    def validate_text_quality(self, text: str) -> Dict[str, bool]:
        """
        驗證文本質量

        Args:
            text: 要驗證的文本

        Returns:
            Dict[str, bool]: 質量檢查結果
        """
        try:
            return self.text_formatter.validate_text_quality(text)
        except TextFormattingError as e:
            self.logger.warning(f"Error validating text quality: {str(e)}")
            return {"error": True}

    def get_text_statistics(self, text: str) -> Dict[str, int]:
        """
        獲取文本統計信息

        Args:
            text: 要分析的文本

        Returns:
            Dict[str, int]: 文本統計信息
        """
        try:
            return self.text_formatter.get_text_statistics(text)
        except TextFormattingError as e:
            self.logger.warning(f"Error getting text statistics: {str(e)}")
            return {"characters": 0, "words": 0, "sentences": 0}

    def reload_templates(self):
        """
        重新載入所有模板
        """
        try:
            self.template_manager.reload_templates()
            self.logger.info("Templates reloaded successfully")
        except (TemplateLoadingError, TemplateFillError) as e:
            self.logger.error(f"Error reloading templates: {str(e)}")
            raise EnhancedSceneDescriberError(f"Failed to reload templates: {str(e)}") from e

    def get_configuration(self) -> Dict[str, Any]:
        """
        獲取當前配置信息

        Returns:
            Dict[str, Any]: 配置信息字典
        """
        try:
            return {
                "scene_types_count": len(self.scene_types),
                "viewpoint_detector_config": self.viewpoint_detector.viewpoint_params,
                "object_generator_config": self.object_description_generator.get_configuration(),
                "supported_cultures": self.cultural_context_analyzer.get_supported_cultures(),
                "template_categories": self.template_manager.get_template_categories()
            }
        except Exception as e:
            self.logger.warning(f"Error getting configuration: {str(e)}")
            return {"error": str(e)}

    def _initialize_fallback_components(self):
        """備用組件初始化"""
        try:
            self.region_analyzer = RegionAnalyzer()
            self.object_description_generator = ObjectDescriptionGenerator(
                region_analyzer=self.region_analyzer
            )
        except Exception as e:
            self.logger.error(f"Fallback component initialization failed: {str(e)}")
