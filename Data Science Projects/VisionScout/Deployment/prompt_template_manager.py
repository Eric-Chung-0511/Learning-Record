import logging
import traceback
from typing import Dict, List, Any, Optional

class PromptTemplateError(Exception):
    """提示模板相關錯誤的自定義異常"""
    pass


class PromptTemplateManager:
    """
    負責管理和格式化各種LLM提示模板。
    包含場景描述增強、錯誤檢測、無檢測處理等不同場景的模板。
    """

    def __init__(self):
        """初始化提示模板管理器"""
        # set the logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # initialize all templates
        self._initialize_templates()
        self.logger.info("PromptTemplateManager initialized successfully")

    def _initialize_templates(self):
        """初始化所有提示模板"""
        try:
            self._setup_enhancement_template()
            self._setup_verification_template()
            self._setup_no_detection_template()
            self.logger.info("All prompt templates initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize templates: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise PromptTemplateError(f"Template initialization failed: {str(e)}") from e


    def format_enhancement_prompt_with_landmark(self, scene_data: Dict[str, Any], object_list: str, original_description: str) -> str:
        try:
            # 確保場景類型被正確清理
            scene_type = scene_data.get("scene_type", "unknown scene")
            cleaned_scene_type = self._clean_scene_type(scene_type)

            # 通用文本格式清理：處理底線和格式化問題
            cleaned_description = self._clean_text_formatting(original_description)

            # 額外清理場景類型底線格式
            cleaned_description = self._clean_scene_type_underscores(cleaned_description)

            # 強化輸入清理
            cleaned_description = self._enhance_input_cleaning(cleaned_description)

            # 在原始描述中替換未清理的場景類型
            if scene_type != cleaned_scene_type:
                cleaned_description = cleaned_description.replace(scene_type, cleaned_scene_type)

            # 檢查是否有地標資訊
            landmark_info = scene_data.get("landmark_location_info")
            is_fallback = scene_data.get("is_fallback", False)

            # 準備額外的地標指導內容
            additional_guidance = ""
            if landmark_info:
                landmark_name = landmark_info.get("name", "")
                landmark_location = landmark_info.get("location", "")
                additional_guidance = f"""
            LANDMARK LOCATION REQUIREMENT: This scene features {landmark_name} located in {landmark_location}.
            16. MANDATORY: Include the specific location "{landmark_location}" when first mentioning {landmark_name}. Use natural phrasing such as "Located in {landmark_location}, the {landmark_name}..." or "The {landmark_name} in {landmark_location}..." or "Standing majestically in {landmark_location}, {landmark_name}...".
            17. Avoid mechanical openings like "The tourist landmark is centered around" or "The scene is centered around". Instead, begin with the landmark itself as the subject.
            18. NEVER use terms with underscores like "tourist_landmark" or "historical_site" in your response. Use natural language: "tourist landmark", "historical site", "cultural attraction" etc.
            19. The geographical reference must appear naturally in the opening sentence, integrated as essential context rather than supplementary information."""
            elif is_fallback:
                additional_guidance = """
            FALLBACK MODE: The previous enhancement was insufficient. Provide a more detailed description focusing on key visual elements, human activities, atmospheric details, and architectural features."""

            # 建構完整的模板內容
            if additional_guidance:
                # 在CRITICAL RULES後添加地標相關指導
                enhanced_template = self.enhance_description_template.replace(
                    "15. When describing quantities or arrangements, use only information explicitly confirmed by the object detection system.",
                    f"15. When describing quantities or arrangements, use only information explicitly confirmed by the object detection system.{additional_guidance}"
                )
            else:
                enhanced_template = self.enhance_description_template

            formatted_prompt = enhanced_template.format(
                original_description=cleaned_description,
                object_list=object_list
            )

            return formatted_prompt

        except Exception as e:
            self.logger.error(f"Failed to format enhancement prompt: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise PromptTemplateError(f"Prompt formatting failed: {e}") from e

    def _clean_text_formatting(self, text: str) -> str:
        """
        通用文本格式清理方法，處理底線、格式化等問題

        Args:
            text: 需要清理的原始文本

        Returns:
            str: 清理後的文本
        """
        if not text:
            return text

        try:
            import re

            # 替換常見的技術性詞彙
            replacements = {
                'tourist_landmark': 'tourist landmark',
                'historical_site': 'historical site',
                'religious_building': 'religious building',
                'cultural_landmark': 'cultural landmark',
                'architectural_site': 'architectural site',
                'natural_landmark': 'natural landmark'
            }

            cleaned = text
            for old_term, new_term in replacements.items():
                cleaned = cleaned.replace(old_term, new_term)

            # 處理其他底線情況
            cleaned = re.sub(r'(\w+)_(\w+)', lambda m: f"{m.group(1)} {m.group(2)}", cleaned)

            # 處理多個連續底線
            cleaned = re.sub(r'_+', ' ', cleaned)

            # 清理多餘空格
            cleaned = re.sub(r'\s+', ' ', cleaned)

            return cleaned.strip()

        except Exception as e:
            self.logger.warning(f"Error in text formatting cleanup: {str(e)}")
            return text

    def _clean_scene_type_underscores(self, text: str) -> str:
        """
        專門清理場景類型中的底線格式

        Args:
            text: 需要清理的文本

        Returns:
            str: 清理後的文本
        """
        if not text:
            return text

        try:
            import re

            # 專門處理場景類型的底線格式
            scene_type_patterns = [
                'urban_intersection', 'city_street', 'downtown_area', 'business_district',
                'residential_area', 'commercial_zone', 'industrial_area', 'shopping_center',
                'traffic_intersection', 'pedestrian_crossing', 'public_square'
            ]

            for pattern in scene_type_patterns:
                if pattern in text:
                    replacement = pattern.replace('_', ' ')
                    text = text.replace(pattern, replacement)

            # 處理任何剩餘的場景類型底線模式
            text = re.sub(r'\b([a-z]+)_([a-z]+)(?=\s+(?:features|shows|displays|contains|is|area|zone|scene))',
                        r'\1 \2', text, flags=re.IGNORECASE)

            return text

        except Exception as e:
            self.logger.warning(f"Error in scene type underscore cleanup: {str(e)}")
            return text

    def _enhance_input_cleaning(self, description: str) -> str:
        """
        增強輸入描述的清理功能

        Args:
            description: 待清理的描述

        Returns:
            str: 清理後的描述
        """
        if not description:
            return description

        try:
            import re

            # 預防性清理底線格式
            description = re.sub(r'\b(\w+)_(\w+)\b', r'\1 \2', description)

            # 清理可能導致語法問題的模式
            problematic_patterns = [
                (r'\s+,\s+', ', '),  # 修正空格-逗號問題
                (r'\bIn\s*,', 'In the area,'),  # 預防性修正
                (r'\s+\.', '.'),  # 修正句號前空格
            ]

            for pattern, replacement in problematic_patterns:
                description = re.sub(pattern, replacement, description)

            return description.strip()

        except Exception as e:
            self.logger.warning(f"Error in enhanced input cleaning: {str(e)}")
            return description

    def _setup_enhancement_template(self):
        """設置場景描述增強模板"""
        self.enhance_description_template = """
            <|system|>
            You are an expert visual analyst. Your task is to improve the readability and fluency of scene descriptions using STRICT factual accuracy.
            Your **top priority is to avoid hallucination** or fabrication. You are working in a computer vision pipeline using object detection (YOLO) and image embeddings. You MUST treat the input object list as a whitelist. Do not speculate beyond this list.
            </|system|>
            <|user|>
            Rewrite the following scene description to be fluent and clear. DO NOT add any objects, events, or spatial relationships that are not explicitly present in the original or object list.
            ORIGINAL:
            {original_description}
            CRITICAL RULES:
            1. CRITICAL ADHERENCE TO INPUT: Strictly adhere to the information explicitly provided in the ORIGINAL description and the {object_list}.
               a. NEVER assume or infer room types, object functions, scene purposes, or abstract conceptual zones (e.g., 'personal items zone', 'activity area') unless such concepts, along with their specific constituent objects and locations, are explicitly detailed in the ORIGINAL description or clearly supported by multiple items in the {object_list}.
               b. Your role is to rephrase and enhance the provided factual data, not to introduce new conceptual layers or interpretations not directly supported by the input.
            2. OBJECT WHITELIST & DETAIL ACCURACY:
               a. The provided {object_list} is an exhaustive list of objects confirmed by the vision system. Mention ONLY objects from this list or objects explicitly detailed in the ORIGINAL description.
               b. DO NOT invent additional objects or infer the presence of 'various scattered objects' if only a single specific item (e.g., one 'handbag') is mentioned in relation to a category or area. Describe only what is explicitly listed.
            3. NEVER speculate on object quantity. If the description says "10 people" , DO NOT say "dozens" or "many". Maintain the original quantity unless specified.
            4. SPATIAL ACCURACY - STRICTLY FROM ORIGINAL:
               a. Base ALL descriptions of object locations (e.g., 'foreground', 'background', 'middle center') and spatial relationships STRICTLY on the information explicitly provided in the ORIGINAL description.
               b. If the ORIGINAL description states an object is 'in the background,' use that exact term. If it specifies 'in the foreground,' use that. If it describes an object as being 'carried by a person', reflect this precise relationship.
               c. If the ORIGINAL description is less specific about an object's location (e.g., 'a car is present'), then use general, non-committal terms like 'visible in the scene' or 'present in the image.'
               d. DO NOT re-interpret object positions from any perceived understanding of the raw image; your sole source for spatial information is the ORIGINAL description. Do not relocate objects (e.g., moving a carried handbag from the person to 'the background').
            5. You MAY describe confirmed materials, colors, and composition style if visually obvious and non-speculative, AND if such details are hinted at or present in the ORIGINAL description or {object_list}.
            6. Write 2–4 complete, well-structured sentences with punctuation.
            7. Final output MUST be a single fluent paragraph of 60–200 words (not longer). Within this concise format, every sentence should aim to introduce new information or build upon previous statements without significant overlap.
            8. Begin your response directly with the scene description. Do NOT include any introductory phrases, explanations, or formatting indicators.
            9. Ensure grammatical completeness in all sentences. Each sentence must have a complete subject and predicate structure.
               a. NEVER use underscore formatting (e.g., tourist_landmark, urban_intersection). Always use natural spacing (tourist landmark, urban intersection).
               b. NEVER begin sentences with incomplete phrases like "In ," or "Overall," without proper subjects. Always ensure complete sentence structure.
               c. AVOID redundant or circular phrasing such as "with lights turned illuminating" or "atmosphere of is one of."
               d. If you encounter incomplete spatial descriptions like "visible in ," or "positioned in the middle of.", complete them naturally by adding appropriate context such as "visible in the scene" or "positioned in the middle of the frame", ensuring these completions are consistent with the ORIGINAL description. Always ensure spatial descriptions have complete prepositional phrases.
               e. GRAMMAR AND FLUENCY CHECK: Ensure all sentences are grammatically flawless and flow naturally. Avoid awkward phrasing or dangling prepositions (e.g., 'glow over ,'). Mentally re-read your generated description to catch and correct such minor errors before finalizing.
            10. Vary sentence structures naturally while maintaining grammatical accuracy.
            11. CRITICAL: Avoid repeating the mention of specific objects, groups of objects, or their spatial arrangements. Once an object or layout aspect is described, only refer to it again if providing genuinely NEW and DISTINCT information or a significantly different perspective that adds substantial value. Strive for conciseness and information density.
            12. Create natural spatial flow by connecting object descriptions organically rather than listing positions mechanically.
            13. Use transitional phrases to connect ideas smoothly, varying expression patterns throughout the description.
            14. For the concluding sentence, focus on the overall atmosphere, style, perceived activity, or overarching impression of the scene. DO NOT simply restate the primary objects or their layout as a summary or 'backdrop' if they have already been clearly described earlier in the paragraph. The conclusion should offer a higher-level takeaway.
            15. When describing quantities or arrangements, use only information explicitly confirmed by the object detection system or ORIGINAL description.
            </|user|>
            <|assistant|>
            """

    def _setup_verification_template(self):
        """設置檢測結果驗證模板"""
        self.verify_detection_template = """
            Task: You are an advanced vision system that verifies computer vision detections for accuracy.
            Analyze the following detection results and identify any potential errors or inconsistencies:
            SCENE TYPE: {scene_type}
            SCENE NAME: {scene_name}
            CONFIDENCE: {confidence:.2f}
            DETECTED OBJECTS: {detected_objects}
            CLIP ANALYSIS RESULTS:
            {clip_analysis}
            Possible Errors to Check:
            1. Objects misidentified (e.g., architectural elements labeled as vehicles)
            2. Cultural elements misunderstood (e.g., Asian temple structures labeled as boats)
            3. Objects that seem out of place for this type of scene
            4. Inconsistencies between different detection systems
            If you find potential errors, list them clearly with explanations. If the detections seem reasonable, state that they appear accurate.
            Verification Results:
            """

    def _setup_no_detection_template(self):
        """設置無檢測結果處理模板"""
        self.no_detection_template = """
            Task: You are an advanced scene understanding system analyzing an image where standard object detection failed to identify specific objects.
            Based on advanced image embeddings (CLIP analysis), we have the following information:
            MOST LIKELY SCENE: {top_scene} (confidence: {top_confidence:.2f})
            VIEWPOINT: {viewpoint}
            LIGHTING: {lighting_condition}
            CULTURAL ANALYSIS: {cultural_analysis}
            Create a detailed description of what might be in this scene, considering:
            1. The most likely type of location or setting
            2. Possible architectural or natural elements present
            3. The lighting and atmosphere
            4. Potential cultural or regional characteristics
            Your description should be natural, flowing, and offer insights into what the image likely contains despite the lack of specific object detection.
            Scene Description:
            """

    def format_enhancement_prompt(self, scene_data: Dict[str, Any], object_list: str, original_description: str) -> str:
        try:
            # 確保場景類型被正確清理
            scene_type = scene_data.get("scene_type", "unknown scene")
            cleaned_scene_type = self._clean_scene_type(scene_type)

            # 在原始描述中替換未清理的場景類型
            if scene_type != cleaned_scene_type:
                original_description = original_description.replace(scene_type, cleaned_scene_type)

            formatted_prompt = self.enhance_description_template.format(
                original_description=original_description,
                object_list=object_list
            )

            return formatted_prompt

        except Exception as e:
            self.logger.error(f"Failed to format enhancement prompt: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise PromptTemplateError(f"Prompt formatting failed: {e}") from e


    def format_verification_prompt(self,
                                 detected_objects: List[Dict],
                                 clip_analysis: Dict[str, Any],
                                 scene_type: str,
                                 scene_name: str,
                                 confidence: float) -> str:
        """
        格式化檢測結果驗證提示

        Args:
            detected_objects: 檢測到的物件列表
            clip_analysis: CLIP分析結果
            scene_type: 場景類型
            scene_name: 場景名稱
            confidence: 場景分類信心度

        Returns:
            str: 格式化後的驗證提示字符串

        Raises:
            PromptTemplateError: 當模板格式化失敗時
        """
        try:
            self.logger.debug("Formatting verification prompt")

            # 格式化物件列表和CLIP分析結果
            objects_str = self._format_objects_for_prompt(detected_objects)
            clip_str = self._format_clip_results(clip_analysis)

            # 格式化提示
            formatted_prompt = self.verify_detection_template.format(
                scene_type=scene_type,
                scene_name=scene_name,
                confidence=confidence,
                detected_objects=objects_str,
                clip_analysis=clip_str
            )

            self.logger.debug(f"Verification prompt formatted successfully (length: {len(formatted_prompt)})")
            return formatted_prompt

        except Exception as e:
            error_msg = f"Failed to format verification prompt: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            raise PromptTemplateError(error_msg) from e

    def format_no_detection_prompt(self, clip_analysis: Dict[str, Any]) -> str:
        """
        格式化無檢測結果處理提示

        Args:
            clip_analysis: CLIP分析結果字典

        Returns:
            str: 格式化後的無檢測處理提示字符串

        Raises:
            PromptTemplateError: 當模板格式化失敗時
        """
        try:
            self.logger.debug("Formatting no-detection prompt")

            # 提取CLIP分析結果
            top_scene, top_confidence = clip_analysis.get("top_scene", ("unknown", 0))
            viewpoint = clip_analysis.get("viewpoint", ("standard", 0))[0]
            lighting = clip_analysis.get("lighting_condition", ("unknown", 0))[0]

            # 格式化文化分析
            cultural_str = self._format_cultural_analysis(clip_analysis.get("cultural_analysis", {}))

            # 格式化提示
            formatted_prompt = self.no_detection_template.format(
                top_scene=top_scene,
                top_confidence=top_confidence,
                viewpoint=viewpoint,
                lighting_condition=lighting,
                cultural_analysis=cultural_str
            )

            self.logger.debug(f"No-detection prompt formatted successfully (length: {len(formatted_prompt)})")
            return formatted_prompt

        except Exception as e:
            error_msg = f"Failed to format no-detection prompt: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            raise PromptTemplateError(error_msg) from e

    def _clean_scene_type(self, scene_type: str) -> str:
        """
        清理場景類型，使其更適合用於提示詞

        Args:
            scene_type: 原始場景類型

        Returns:
            str: 清理後的場景類型
        """
        if not scene_type:
            return "scene"

        # 將底線替換為空格並首字母大寫
        if '_' in scene_type:
            return ' '.join(word.capitalize() for word in scene_type.split('_'))

        return scene_type

    def _format_objects_for_prompt(self, objects: List[Dict]) -> str:
        """
        格式化物件列表以用於提示

        Args:
            objects: 檢測到的物件列表

        Returns:
            str: 格式化後的物件字符串
        """
        if not objects:
            return "No objects detected"

        try:
            formatted = []
            for obj in objects:
                class_name = obj.get("class_name", "unknown")
                confidence = obj.get("confidence", 0)
                formatted.append(f"{class_name} (confidence: {confidence:.2f})")

            return "\n- " + "\n- ".join(formatted)

        except Exception as e:
            self.logger.warning(f"Error formatting objects: {str(e)}")
            return "Object formatting error"

    def _format_clip_results(self, clip_analysis: Dict) -> str:
        """
        格式化CLIP分析結果以用於提示

        Args:
            clip_analysis: CLIP分析結果字典

        Returns:
            str: 格式化後的CLIP分析字符串
        """
        if not clip_analysis or "error" in clip_analysis:
            return "No CLIP analysis available"

        try:
            parts = ["CLIP Analysis Results:"]

            # 添加頂級場景
            top_scene, confidence = clip_analysis.get("top_scene", ("unknown", 0))
            parts.append(f"- Most likely scene: {top_scene} (confidence: {confidence:.2f})")

            # 添加視角
            viewpoint, vp_conf = clip_analysis.get("viewpoint", ("standard", 0))
            parts.append(f"- Camera viewpoint: {viewpoint} (confidence: {vp_conf:.2f})")

            # 添加物件組合
            if "object_combinations" in clip_analysis:
                combos = []
                for combo, score in clip_analysis["object_combinations"][:3]:
                    combos.append(f"{combo} ({score:.2f})")
                parts.append(f"- Object combinations: {', '.join(combos)}")

            # 添加文化分析
            if "cultural_analysis" in clip_analysis:
                parts.append("- Cultural analysis:")
                for culture_type, data in clip_analysis["cultural_analysis"].items():
                    best_desc = data.get("best_description", "")
                    desc_conf = data.get("confidence", 0)
                    parts.append(f"  * {culture_type}: {best_desc} ({desc_conf:.2f})")

            return "\n".join(parts)

        except Exception as e:
            self.logger.warning(f"Error formatting CLIP results: {str(e)}")
            return "CLIP analysis formatting error"

    def _format_cultural_analysis(self, cultural_analysis: Dict) -> str:
        """
        格式化文化分析結果

        Args:
            cultural_analysis: 文化分析結果字典

        Returns:
            str: 格式化後的文化分析字符串
        """
        if not cultural_analysis:
            return "No specific cultural elements detected"

        try:
            parts = []
            for culture_type, data in cultural_analysis.items():
                best_desc = data.get("best_description", "")
                desc_conf = data.get("confidence", 0)
                parts.append(f"{culture_type}: {best_desc} (confidence: {desc_conf:.2f})")

            return "\n".join(parts)

        except Exception as e:
            self.logger.warning(f"Error formatting cultural analysis: {str(e)}")
            return "Cultural analysis formatting error"

    def get_template_info(self) -> Dict[str, Any]:
        """
        獲取模板管理器的信息

        Returns:
            Dict[str, Any]: 包含模板數量和狀態的信息
        """
        return {
            "templates_count": 3,
            "available_templates": [
                "enhance_description_template",
                "verify_detection_template",
                "no_detection_template"
            ],
            "initialization_status": "success"
        }
