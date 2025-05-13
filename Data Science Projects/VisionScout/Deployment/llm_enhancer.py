import re
import os
import torch
from typing import Dict, List, Tuple, Any, Optional
import logging

class LLMEnhancer:
    """
    負責使用LLM (Large Language Model) 增強場景理解和描述。
    未來可以再整合Llama或其他LLM模型進行場景描述的生成和豐富化。
    """

    def __init__(self,
                model_path: Optional[str] = None,
                tokenizer_path: Optional[str] = None,
                device: Optional[str] = None,
                max_length: int = 2048,
                temperature: float = 0.3,
                top_p: float = 0.85):
        """
        初始化LLM增強器

        Args:
            model_path: LLM模型的路徑或HuggingFace log in，默認使用Llama 3.2
            tokenizer_path: token處理器的路徑，通常與model_path相同
            device: 設備檢查 ('cpu'或'cuda')
            max_length: 生成文本的最大長度
            temperature: 生成文本的溫度（較高比較有創意，較低會偏保守）
            top_p: 生成文本時的核心採樣機率閾值
        """
        self.logger = logging.getLogger("LLMEnhancer")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # 設置默認模型路徑就是用Llama3.2
        self.model_path = model_path or "meta-llama/Llama-3.2-3B-Instruct"
        self.tokenizer_path = tokenizer_path or self.model_path

        # 確定運行設備
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # create parameters
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p

        self.model = None
        self.tokenizer = None

        # 計數器，用來追蹤模型調用次數
        self.call_count = 0

        self._initialize_prompts()

        # 只在需要時加載模型
        self._model_loaded = False

        try:
            self.hf_token = os.environ.get("HF_TOKEN")
            if self.hf_token:
                self.logger.info("Logging in to Hugging Face with token")
                from huggingface_hub import login
                login(token=self.hf_token)
            else:
                self.logger.warning("HF_TOKEN not found in environment variables. Access to gated models may be limited.")
        except Exception as e:
            self.logger.error(f"Error during Hugging Face login: {e}")

    def _load_model(self):
        """懶加載模型 - 僅在首次需要時加載，使用 8 位量化以節省記憶體"""
        if self._model_loaded:
            return

        try:
            self.logger.info(f"Loading LLM model from {self.model_path} with 8-bit quantization")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            torch.cuda.empty_cache()

            # 打印可用 GPU 記憶體
            if torch.cuda.is_available():
                free_in_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"Total GPU memory: {free_in_GB:.2f} GB")

            # 設置 8 位元量化配置
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )

            # 加載詞元處理器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                padding_side="left",
                use_fast=False,
                token=self.hf_token
            )

            # 設置特殊標記
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # 加載 8 位量化模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",  
                low_cpu_mem_usage=True,
                token=self.hf_token
            )

            self.logger.info("Model loaded successfully with 8-bit quantization")
            self._model_loaded = True

        except Exception as e:
            self.logger.error(f"Error loading LLM model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _initialize_prompts(self):
        """Return an optimized prompt template specifically for Zephyr model"""
        # the prompt for the model
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
            1. NEVER assume room type, object function, or scene purpose unless directly stated.
            2. NEVER invent object types. You are limited to: {object_list}
            3. NEVER speculate on object quantity. If the description says "10 people" , DO NOT say "dozens" or "many". Maintain the original quantity unless specified.
            4. Use terms like "in the scene", "visible in the background", or "positioned in the lower left" instead of assuming direction or layout logic.
            5. You MAY describe confirmed materials, colors, and composition style if visually obvious and non-speculative.
            6. Write 2–4 complete, well-structured sentences with punctuation.
            7. Final output MUST be a single fluent paragraph of 60–200 words (not longer).
            8. NEVER include explanations, reasoning, or tags. ONLY provide the enhanced description.
            9. Do not repeat any sentence structure or phrase more than once.
            </|user|>

            <|assistant|>
            """


        # 錯誤檢測提示
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

        # 無檢測處理提示
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

    def _clean_llama_response(self, response: str) -> str:
        """處理 Llama 模型特有的輸出格式問題"""
        # 首先應用通用清理
        response = self._clean_model_response(response)

        # 移除 Llama 常見的前綴短語
        prefixes_to_remove = [
            "Here's the enhanced description:",
            "Enhanced description:",
            "Here is the enhanced scene description:",
            "I've enhanced the description while preserving all factual details:"
        ]

        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        # 移除可能的後綴說明
        suffixes_to_remove = [
            "I've maintained all the key factual elements",
            "I've preserved all the factual details",
            "All factual elements have been maintained"
        ]

        for suffix in suffixes_to_remove:
            if response.lower().endswith(suffix.lower()):
                response = response[:response.rfind(suffix)].strip()

        return response

    def _detect_scene_type(self, detected_objects: List[Dict]) -> str:
        """
        Detect scene type based on object distribution and patterns
        """
        # Default scene type
        scene_type = "intersection"

        # Count objects by class
        object_counts = {}
        for obj in detected_objects:
            class_name = obj.get("class_name", "")
            if class_name not in object_counts:
                object_counts[class_name] = 0
            object_counts[class_name] += 1

        # 辨識人
        people_count = object_counts.get("person", 0)

        # 交通工具的
        car_count = object_counts.get("car", 0)
        bus_count = object_counts.get("bus", 0)
        truck_count = object_counts.get("truck", 0)
        total_vehicles = car_count + bus_count + truck_count

        # Simple scene type detection logic
        if people_count > 8 and total_vehicles < 2:
            scene_type = "pedestrian_crossing"
        elif people_count > 5 and total_vehicles > 2:
            scene_type = "busy_intersection"
        elif people_count < 3 and total_vehicles > 3:
            scene_type = "traffic_junction"

        return scene_type

    def _clean_scene_type(self, scene_type: str) -> str:
        """清理場景類型，使其更適合用於提示詞"""
        if not scene_type:
            return "scene"

        # replace underline to space or sometime capital letter
        if '_' in scene_type:
            return ' '.join(word.capitalize() for word in scene_type.split('_'))

        return scene_type

    def _clean_model_response(self, response: str) -> str:
        """清理模型回應以移除常見的標記和前綴"""
        # 移除任何可能殘留的系統樣式標記
        response = re.sub(r'<\|.*?\|>', '', response)

        # 移除任何 "This european_plaza" 或類似前綴
        response = re.sub(r'^This [a-z_]+\s+', '', response)

        # 確保響應以大寫字母開頭
        if response and not response[0].isupper():
            response = response[0].upper() + response[1:]

        return response.strip()

    def _validate_scene_facts(self, enhanced_desc: str, original_desc: str, people_count: int) -> str:
        """Validate key facts in enhanced description"""
        # Check if people count is preserved
        if people_count > 0:
            people_pattern = re.compile(r'(\d+)\s+(?:people|persons|pedestrians|individuals)', re.IGNORECASE)
            people_match = people_pattern.search(enhanced_desc)

            if not people_match or int(people_match.group(1)) != people_count:
                # Replace incorrect count or add if missing
                if people_match:
                    enhanced_desc = people_pattern.sub(f"{people_count} people", enhanced_desc)
                else:
                    enhanced_desc = f"The scene shows {people_count} people. " + enhanced_desc

        # Ensure aerial perspective is mentioned
        if "aerial" in original_desc.lower() and "aerial" not in enhanced_desc.lower():
            enhanced_desc = "From an aerial perspective, " + enhanced_desc[0].lower() + enhanced_desc[1:]

        return enhanced_desc

    def reset_context(self):
        """在處理新圖像前重置模型上下文"""
        if self._model_loaded:
            # 清除 GPU 緩存
            torch.cuda.empty_cache()
            self.logger.info("Model context reset")
        else:
            self.logger.info("Model not loaded, no context to reset")

    def _remove_introduction_sentences(self, response: str) -> str:
        """移除生成文本中可能的介紹性句子"""
        # 識別常見的介紹性模式
        intro_patterns = [
            r'^Here is the (?:rewritten|enhanced) .*?description:',
            r'^The (?:rewritten|enhanced) description:',
            r'^Here\'s the (?:rewritten|enhanced) description of .*?:'
        ]

        for pattern in intro_patterns:
            if re.match(pattern, response, re.IGNORECASE):
                # 找到冒號後的內容
                parts = re.split(r':', response, 1)
                if len(parts) > 1:
                    return parts[1].strip()

        return response

    def enhance_description(self, scene_data: Dict[str, Any]) -> str:
        """改進的場景描述增強器，處理各種場景類型並保留視角與光照資訊，並作為總窗口可運用於其他class"""
        try:
            # 重置上下文
            self.reset_context()

            # 確保模型已加載
            if not self._model_loaded:
                self._load_model()

            # extract original description
            original_desc = scene_data.get("original_description", "")
            if not original_desc:
                return "No original description provided."

            # 獲取scene type 並標準化
            scene_type = scene_data.get("scene_type", "unknown scene")
            scene_type = self._clean_scene_type(scene_type)

            # 提取檢測到的物件並過濾低置信度物件
            detected_objects = scene_data.get("detected_objects", [])
            filtered_objects = []

            # 高置信度閾值，嚴格過濾物件
            high_confidence_threshold = 0.65

            for obj in detected_objects:
                confidence = obj.get("confidence", 0)
                class_name = obj.get("class_name", "")

                # 為特殊類別設置更高閾值
                special_classes = ["airplane", "helicopter", "boat"]
                if class_name in special_classes:
                    if confidence < 0.75:  # 為這些類別設置更高閾值
                        continue

                # 僅保留高置信度物件
                if confidence >= high_confidence_threshold:
                    filtered_objects.append(obj)

            # 計算物件列表和數量 - 僅使用過濾後的高置信度物件
            object_counts = {}
            for obj in filtered_objects:
                class_name = obj.get("class_name", "")
                if class_name not in object_counts:
                    object_counts[class_name] = 0
                object_counts[class_name] += 1

            # 將高置信度物件格式化為清單
            high_confidence_objects = ", ".join([f"{count} {obj}" for obj, count in object_counts.items()])

            # 如果沒有高置信度物件，回退到使用原始描述中的關鍵詞
            if not high_confidence_objects:
                # 從原始描述中提取物件提及
                object_keywords = self._extract_objects_from_description(original_desc)
                high_confidence_objects = ", ".join(object_keywords) if object_keywords else "objects visible in the scene"

            # 保留原始描述中的關鍵視角信息
            perspective = self._extract_perspective_from_description(original_desc)

            # 提取光照資訊
            lighting_description = "unknown lighting"
            if "lighting_info" in scene_data:
                lighting_info = scene_data.get("lighting_info", {})
                time_of_day = lighting_info.get("time_of_day", "unknown")
                is_indoor = lighting_info.get("is_indoor", False)
                lighting_description = f"{'indoor' if is_indoor else 'outdoor'} {time_of_day} lighting"

            # 構建提示詞，整合所有關鍵資訊
            prompt = self.enhance_description_template.format(
                scene_type=scene_type,
                object_list=high_confidence_objects,
                original_description=original_desc,
                perspective=perspective,
                lighting_description=lighting_description
            )

            # 生成增強描述
            self.logger.info("Generating LLM response...")
            response = self._generate_llm_response(prompt)

            # 檢查回應完整性的更嚴格標準
            is_incomplete = (
                len(response) < 100 or  # 太短
                (len(response) < 200 and "." not in response[-30:]) or  # 結尾沒有適當標點
                any(response.endswith(phrase) for phrase in ["in the", "with the", "and the"])  # 以不完整短語結尾
            )

            max_retries = 3
            attempts = 0
            while attempts < max_retries and is_incomplete:
                self.logger.warning(f"Generated incomplete response, retrying... Attempt {attempts+1}/{max_retries}")
                # 重新生成
                response = self._generate_llm_response(prompt)
                attempts += 1

                # 重新檢查完整性
                is_incomplete = (len(response) < 100 or
                                (len(response) < 200 and "." not in response[-30:]) or
                                any(response.endswith(phrase) for phrase in ["in the", "with the", "and the"]))

            # 確保響應不為空
            if not response or len(response.strip()) < 10:
                self.logger.warning("Generated response was empty or too short, returning original description")
                return original_desc

            # 清理響應 - 使用與模型相符的清理方法
            if "llama" in self.model_path.lower():
                result = self._clean_llama_response(response)
            else:
                result = self._clean_model_response(response)

            # 移除介紹性句子
            result = self._remove_introduction_sentences(result)

            # 移除解釋性注釋
            result = self._remove_explanatory_notes(result)

            # 進行事實準確性檢查
            result = self._verify_factual_accuracy(original_desc, result, high_confidence_objects)

            # 確保場景類型和視角一致性
            result = self._ensure_scene_type_consistency(result, scene_type, original_desc)
            if perspective and perspective.lower() not in result.lower():
                result = f"{perspective}, {result[0].lower()}{result[1:]}"

            return str(result)

        except Exception as e:
            self.logger.error(f"Enhancement failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return original_desc  # 發生任何錯誤時返回原始描述

    def _verify_factual_accuracy(self, original: str, generated: str, object_list: str) -> str:
        """驗證生成的描述不包含原始描述或物體列表中沒有的信息"""

        # 將原始描述和物體列表合併為授權詞彙源
        authorized_content = original.lower() + " " + object_list.lower()

        # 提取生成描述中具有實質意義的名詞
        # 創建常見地點、文化和地域詞彙的列表
        location_terms = ["plaza", "square", "market", "mall", "avenue", "boulevard"]
        cultural_terms = ["european", "asian", "american", "african", "western", "eastern"]

        # 檢查生成文本中的每個詞
        for term in location_terms + cultural_terms:
            # 僅當該詞出現在生成文本但不在授權內容中時進行替換
            if term in generated.lower() and term not in authorized_content:
                # 根據詞語類型選擇適當的替換詞
                if term in location_terms:
                    replacement = "area"
                else:
                    replacement = "scene"

                # 使用正則表達式進行完整詞匹配替換
                pattern = re.compile(r'\b' + term + r'\b', re.IGNORECASE)
                generated = pattern.sub(replacement, generated)

        return generated


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
        # 確保模型已加載
        self._load_model()

        # 格式化數據
        objects_str = self._format_objects_for_prompt(detected_objects)
        clip_str = self._format_clip_results(clip_analysis)

        # 構建提示
        prompt = self.verify_detection_template.format(
            scene_type=scene_type,
            scene_name=scene_name,
            confidence=confidence,
            detected_objects=objects_str,
            clip_analysis=clip_str
        )

        # 調用LLM進行驗證
        verification_result = self._generate_llm_response(prompt)

        # 解析驗證結果
        result = {
            "verification_text": verification_result,
            "has_errors": "appear accurate" not in verification_result.lower(),
            "corrected_objects": None  # 可能在未來版本實現詳細錯誤修正
        }

        return result

    def _validate_content_consistency(self, original_desc: str, enhanced_desc: str) -> str:
        """驗證增強描述的內容與原始描述一致"""
        # 提取原始描述中的關鍵數值
        people_count_match = re.search(r'(\d+)\s+people', original_desc, re.IGNORECASE)
        people_count = int(people_count_match.group(1)) if people_count_match else None

        # 驗證人數一致性
        if people_count:
            enhanced_count_match = re.search(r'(\d+)\s+people', enhanced_desc, re.IGNORECASE)
            if not enhanced_count_match or int(enhanced_count_match.group(1)) != people_count:
                # 保留原始人數
                if enhanced_count_match:
                    enhanced_desc = re.sub(r'\b\d+\s+people\b', f"{people_count} people", enhanced_desc, flags=re.IGNORECASE)
                elif "people" in enhanced_desc.lower():
                    enhanced_desc = re.sub(r'\bpeople\b', f"{people_count} people", enhanced_desc, flags=re.IGNORECASE)

        # 驗證視角/透視一致性
        perspective_terms = ["aerial", "bird's-eye", "overhead", "ground level", "eye level"]

        for term in perspective_terms:
            if term in original_desc.lower() and term not in enhanced_desc.lower():
                # 添加缺失的視角信息
                if enhanced_desc[0].isupper():
                    enhanced_desc = f"From {term} view, {enhanced_desc[0].lower()}{enhanced_desc[1:]}"
                else:
                    enhanced_desc = f"From {term} view, {enhanced_desc}"
                break

        return enhanced_desc

    def _remove_explanatory_notes(self, response: str) -> str:
        """移除解釋性注釋、說明和其他非描述性內容"""

        # 識別常見的注釋和解釋模式
        note_patterns = [
            r'(?:^|\n)Note:.*?(?:\n|$)',
            r'(?:^|\n)I have (?:followed|adhered to|ensured).*?(?:\n|$)',
            r'(?:^|\n)This description (?:follows|adheres to|maintains).*?(?:\n|$)',
            r'(?:^|\n)The enhanced description (?:maintains|preserves).*?(?:\n|$)'
        ]

        # 尋找第一段完整的描述內容
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]

        # 如果只有一個段落，檢查並清理它
        if len(paragraphs) == 1:
            for pattern in note_patterns:
                paragraphs[0] = re.sub(pattern, '', paragraphs[0], flags=re.IGNORECASE)
            return paragraphs[0].strip()

        # 如果有多個段落，識別並移除注釋段落
        content_paragraphs = []
        for paragraph in paragraphs:
            is_note = False
            for pattern in note_patterns:
                if re.search(pattern, paragraph, flags=re.IGNORECASE):
                    is_note = True
                    break

            # 檢查段落是否以常見的注釋詞開頭
            if paragraph.lower().startswith(('note:', 'please note:', 'remember:')):
                is_note = True

            if not is_note:
                content_paragraphs.append(paragraph)

        # 返回清理後的內容
        return '\n\n'.join(content_paragraphs).strip()

    def handle_no_detection(self, clip_analysis: Dict[str, Any]) -> str:
        """
        處理YOLO未檢測到物體的情況

        Args:
            clip_analysis: CLIP分析結果

        Returns:
            str: 生成的場景描述
        """
        # 確保模型已加載
        self._load_model()

        # 提取CLIP結果
        top_scene, top_confidence = clip_analysis.get("top_scene", ("unknown", 0))
        viewpoint = clip_analysis.get("viewpoint", ("standard", 0))[0]
        lighting = clip_analysis.get("lighting_condition", ("unknown", 0))[0]

        # 格式化文化分析
        cultural_str = self._format_cultural_analysis(clip_analysis.get("cultural_analysis", {}))

        # 構建提示
        prompt = self.no_detection_template.format(
            top_scene=top_scene,
            top_confidence=top_confidence,
            viewpoint=viewpoint,
            lighting_condition=lighting,
            cultural_analysis=cultural_str
        )

        # 調用LLM生成描述
        description = self._generate_llm_response(prompt)

        # 優化輸出
        return self._clean_llm_response(description)

    def _clean_input_text(self, text: str) -> str:
        """
        對輸入文本進行通用的格式清理，處理常見的格式問題。

        Args:
            text: 輸入文本

        Returns:
            清理後的文本
        """
        if not text:
            return ""

        # 清理格式的問題
        # 1. 處理連續標點符號問題
        text = re.sub(r'([.,;:!?])\1+', r'\1', text)

        # 2. 修復不完整句子的標點（如 "Something," 後沒有繼續句子）
        text = re.sub(r',\s*$', '.', text)

        # 3. 修復如 "word." 後未加空格即接下一句的問題
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

        # 4. 移除多餘空格
        text = re.sub(r'\s+', ' ', text).strip()

        # 5. 確保句子正確結束（句尾加句號）
        if text and not text[-1] in '.!?':
            text += '.'

        return text

    def _fact_check_description(self, original_desc: str, enhanced_desc: str, scene_type: str, detected_objects: List[str]) -> str:
        """
        驗證並可能修正增強後的描述，確保其保持事實準確性，針對普遍事實而非特定場景。

        Args:
            original_desc: 原始場景描述
            enhanced_desc: 增強後的描述待驗證
            scene_type: 場景類型
            detected_objects: 檢測到的物體名稱列表

        Returns:
            經過事實檢查的描述
        """
        # 如果增強描述為空或太短，返回原始描述
        if not enhanced_desc or len(enhanced_desc) < 30:
            return original_desc

        # 1. 檢查數值一致性（如人數、物體數量等）
        # 從原始描述中提取數字和相關名詞
        number_patterns = [
            (r'(\d+)\s+(people|person|pedestrians|individuals)', r'\1', r'\2'), # 人數
            (r'(\d+)\s+(cars|vehicles|automobiles)', r'\1', r'\2'),            # 車輛數
            (r'(\d+)\s+(buildings|structures)', r'\1', r'\2')                  # 建築數
        ]

        # 檢查原始描述中的每個數字
        for pattern, num_group, word_group in number_patterns:
            original_matches = re.finditer(pattern, original_desc, re.IGNORECASE)
            for match in original_matches:
                number = match.group(1)
                noun = match.group(2)

                # 檢查增強描述中是否保留了這個數字
                # 創建一個更通用的模式來檢查增強描述中是否包含此數字和對象類別
                enhanced_pattern = r'(\d+)\s+(' + re.escape(noun) + r'|' + re.escape(noun.rstrip('s')) + r'|' + re.escape(noun + 's') + r')'
                enhanced_matches = list(re.finditer(enhanced_pattern, enhanced_desc, re.IGNORECASE))

                if not enhanced_matches:
                    # 數字+名詞未在增強描述中找到
                    plural_form = noun if noun.endswith('s') or number == '1' else noun + 's'
                    if enhanced_desc.startswith("This") or enhanced_desc.startswith("The"):
                        enhanced_desc = enhanced_desc.replace("This ", f"This scene with {number} {plural_form} ", 1)
                        enhanced_desc = enhanced_desc.replace("The ", f"The scene with {number} {plural_form} ", 1)
                    else:
                        enhanced_desc = f"The scene includes {number} {plural_form}. " + enhanced_desc
                elif enhanced_matches and match.group(1) != number:
                    # 存在但數字不一致，就要更正數字
                    for ematch in enhanced_matches:
                        wrong_number = ematch.group(1)
                        enhanced_desc = enhanced_desc.replace(f"{wrong_number} {ematch.group(2)}", f"{number} {ematch.group(2)}")

        # 2. 檢查視角的一致性
        perspective_terms = {
            "aerial": ["aerial", "bird's-eye", "overhead", "top-down", "above", "looking down"],
            "ground": ["street-level", "ground level", "eye-level", "standing"],
            "indoor": ["inside", "interior", "indoor", "within"],
            "close-up": ["close-up", "detailed view", "close shot"]
        }

        # 確定原始視角
        original_perspective = None
        for persp, terms in perspective_terms.items():
            if any(term in original_desc.lower() for term in terms):
                original_perspective = persp
                break

        # 檢查是否保留了視角方面
        if original_perspective:
            enhanced_has_perspective = any(term in enhanced_desc.lower() for term in perspective_terms[original_perspective])

            if not enhanced_has_perspective:
                # 添加之前缺的視角方面
                perspective_prefixes = {
                    "aerial": "From an aerial perspective, ",
                    "ground": "From street level, ",
                    "indoor": "In this indoor setting, ",
                    "close-up": "In this close-up view, "
                }

                prefix = perspective_prefixes.get(original_perspective, "")
                if prefix:
                    if enhanced_desc[0].isupper():
                        enhanced_desc = prefix + enhanced_desc[0].lower() + enhanced_desc[1:]
                    else:
                        enhanced_desc = prefix + enhanced_desc

        # 3. 檢查場景類型一致性
        if scene_type and scene_type.lower() != "unknown" and scene_type.lower() not in enhanced_desc.lower():
            # 優雅地添加場景類型
            if enhanced_desc.startswith("This ") or enhanced_desc.startswith("The "):
                # 避免產生 "This scene" 和 "This intersection" 的重複
                if "scene" in enhanced_desc[:15].lower():
                    fixed_type = scene_type.lower()
                    enhanced_desc = enhanced_desc.replace("scene", fixed_type, 1)
                else:
                    enhanced_desc = enhanced_desc.replace("This ", f"This {scene_type} ", 1)
                    enhanced_desc = enhanced_desc.replace("The ", f"The {scene_type} ", 1)
            else:
                enhanced_desc = f"This {scene_type} " + enhanced_desc

        # 4. 確保文字長度適當，這邊的限制要與prompt相同,否則會產生矛盾
        words = enhanced_desc.split()
        if len(words) > 200:
            # 找尋接近字數限制的句子結束處
            truncated = ' '.join(words[:200])
            last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))

            if last_period > 0:
                enhanced_desc = truncated[:last_period+1]
            else:
                enhanced_desc = truncated + '.'

        return enhanced_desc

    def _extract_perspective_from_description(self, description: str) -> str:
        """從原始描述中提取視角/透視信息"""
        perspective_terms = {
            "aerial": ["aerial perspective", "aerial view", "bird's-eye view", "overhead view", "from above"],
            "ground": ["ground level", "eye level", "street level"],
            "indoor": ["indoor setting", "inside", "interior"]
        }

        for persp_type, terms in perspective_terms.items():
            for term in terms:
                if term.lower() in description.lower():
                    return term

        return ""

    def _extract_objects_from_description(self, description: str) -> List[str]:
        """從原始描述中提取物件提及"""
        # 常見物件正則表達式模式
        object_patterns = [
            r'(\d+)\s+(people|persons|pedestrians|individuals)',
            r'(\d+)\s+(cars|vehicles|automobiles)',
            r'(\d+)\s+(buildings|structures)',
            r'(\d+)\s+(plants|potted plants|flowers)',
            r'(\d+)\s+(beds|furniture|tables|chairs)'
        ]

        extracted_objects = []

        for pattern in object_patterns:
            matches = re.finditer(pattern, description, re.IGNORECASE)
            for match in matches:
                number = match.group(1)
                object_type = match.group(2)
                extracted_objects.append(f"{number} {object_type}")

        return extracted_objects

    def _ensure_scene_type_consistency(self, description: str, scene_type: str, original_desc: str) -> str:
        """確保描述中的場景類型與指定的場景類型一致"""
        # 禁止使用的錯誤場景詞列表
        prohibited_scene_words = ["plaza", "square", "european", "asian", "american"]

        # 檢查是否包含禁止的場景詞
        for word in prohibited_scene_words:
            if word in description.lower() and word not in original_desc.lower() and word not in scene_type.lower():
                # 替換錯誤場景詞為正確場景類型
                pattern = re.compile(r'\b' + word + r'\b', re.IGNORECASE)
                description = pattern.sub(scene_type, description)

        # 確保場景類型在描述中被提及
        if scene_type.lower() not in description.lower():
            # 尋找通用場景詞並替換
            for general_term in ["scene", "area", "place", "location"]:
                if general_term in description.lower():
                    pattern = re.compile(r'\b' + general_term + r'\b', re.IGNORECASE)
                    description = pattern.sub(scene_type, description, count=1)
                    break
            else:
                # 如果沒有找到通用詞，在開頭添加場景類型
                if description.startswith("The "):
                    description = description.replace("The ", f"The {scene_type} ", 1)
                elif description.startswith("This "):
                    description = description.replace("This ", f"This {scene_type} ", 1)
                else:
                    description = f"This {scene_type} " + description

        return description

    def _generate_llm_response(self, prompt: str) -> str:
        """生成 LLM 的回應"""
        self._load_model()

        try:
            self.call_count += 1
            self.logger.info(f"LLM call #{self.call_count}")

            # 清除 GPU 緩存
            torch.cuda.empty_cache()

            # 設置固定種子以提高一致性
            torch.manual_seed(42)

            # 準備輸入
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)

            # 根據模型類型調整參數
            generation_params = {
                "max_new_tokens": 120,
                "pad_token_id": self.tokenizer.eos_token_id,
                "attention_mask": inputs.attention_mask,
                "use_cache": True,
            }

            # 為 Llama 模型設置特定參數
            if "llama" in self.model_path.lower():
                generation_params.update({
                    "temperature": 0.4,        # 不要太高, 否則模型可能會太有主觀意見
                    "max_new_tokens": 600,      
                    "do_sample": True,          
                    "top_p": 0.8,              
                    "repetition_penalty": 1.2,  # 重複的懲罰權重,可避免掉重複字
                    "num_beams": 4 ,            
                    "length_penalty": 1.2,      
                })

            else:
                # 如果用其他模型的參數
                generation_params.update({
                    "temperature": 0.6,
                    "max_new_tokens": 300,
                    "top_p": 0.9,
                    "do_sample": True,
                    "num_beams": 1,
                    "repetition_penalty": 1.05
                })

            # 生成回應
            with torch.no_grad():
                outputs = self.model.generate(inputs.input_ids, **generation_params)

            # 解碼完整輸出
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 提取生成的響應部分
            assistant_tag = "<|assistant|>"
            if assistant_tag in full_response:
                response = full_response.split(assistant_tag)[-1].strip()

                # 檢查是否有未閉合的 <|assistant|> 
                user_tag = "<|user|>"
                if user_tag in response:
                    response = response.split(user_tag)[0].strip()
            else:
                # 移除輸入提示
                input_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
                response = full_response
                if response.startswith(input_text):
                    response = response[len(input_text):].strip()

            # 確保不返回空響應
            if not response or len(response.strip()) < 10:
                self.logger.warning("生成的回應為空的或太短，返回默認回應")
                return "No detailed description could be generated."

            return response

        except Exception as e:
            self.logger.error(f"生成 LLM 響應時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return "Unable to generate enhanced description."

    def _clean_llm_response(self, response: str) -> str:
        """
        Clean the LLM response to ensure the output contains only clean descriptive text.
        Sometimes it will not only display the description but display tags, notes...etc

        Args:
            response: Original response from the LLM

        Returns:
            Cleaned description text
        """
        if not response:
            return ""

        # Save original response as backup
        original_response = response

        # 1. Extract content between markers (if present)
        output_start = response.find("[OUTPUT_START]")
        output_end = response.find("[OUTPUT_END]")
        if output_start != -1 and output_end != -1 and output_end > output_start:
            response = response[output_start + len("[OUTPUT_START]"):output_end].strip()

        # 2. Remove all remaining section markers and instructions
        section_markers = [
            r'\[.*?\]',                      # [any text]
            r'OUTPUT_START\s*:|OUTPUT_END\s*:',  # OUTPUT_START: or OUTPUT_END:
            r'ENHANCED DESCRIPTION\s*:',      # ENHANCED DESCRIPTION:
            r'Scene Type\s*:.*?(?=\n|$)',    # Scene Type: text
            r'Original Description\s*:.*?(?=\n|$)', # Original Description: text
            r'GOOD\s*:|BAD\s*:',             # GOOD: or BAD:
            r'PROBLEM\s*:.*?(?=\n|$)',       # PROBLEM: text
            r'</?\|(?:assistant|system|user)\|>',  # Dialog markers
            r'\(Note:.*?\)',                 # Notes in parentheses
            r'\(.*?I\'ve.*?\)',              # Common explanatory content
            r'\(.*?as per your request.*?\)' # References to instructions
        ]

        for marker in section_markers:
            response = re.sub(marker, '', response, flags=re.IGNORECASE)

        # 3. Remove common prefixes and suffixes
        prefixes_to_remove = [
            "Enhanced Description:",
            "Scene Description:",
            "Description:",
            "Here is the enhanced description:",
            "Here's the enhanced description:"
        ]

        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        # 4. Remove any Context tags or text containing Context
        response = re.sub(r'<\s*Context:.*?>', '', response)
        response = re.sub(r'Context:.*?(?=\n|$)', '', response)
        response = re.sub(r'Note:.*?(?=\n|$)', '', response, flags=re.IGNORECASE)

        # 5. Clean improper scene type references
        scene_type_pattern = r'This ([a-zA-Z_]+) (features|shows|displays|contains)'
        match = re.search(scene_type_pattern, response)
        if match and '_' in match.group(1):
            fixed_text = f"This scene {match.group(2)}"
            response = re.sub(scene_type_pattern, fixed_text, response)

        # 6. Reduce dash usage for more natural punctuation
        response = re.sub(r'—', ', ', response)
        response = re.sub(r' - ', ', ', response)

        # 7. Remove excess whitespace and line breaks
        response = response.replace('\r', ' ')
        response = re.sub(r'\n+', ' ', response)  # 將所有換行符替換為空格
        response = re.sub(r'\s{2,}', ' ', response)  # 將多個空格替換為單個空格

        # 8. Remove Markdown formatting
        response = re.sub(r'\*\*|\*|__|\|', '', response)  # Remove Markdown indicators

        # 9. Detect and remove sentence duplicates
        sentences = re.split(r'(?<=[.!?])\s+', response)
        unique_sentences = []
        seen_content = set()

        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue

            # Create simplified version for comparison (lowercase, no punctuation)
            simplified = re.sub(r'[^\w\s]', '', sentence.lower())
            simplified = ' '.join(simplified.split())  # Standardize whitespace

            # Check if we've seen a similar sentence
            is_duplicate = False
            for existing in seen_content:
                if len(simplified) > 10 and (existing in simplified or simplified in existing):
                    is_duplicate = True
                    break

            if not is_duplicate and simplified:
                unique_sentences.append(sentence)
                seen_content.add(simplified)

        # Recombine unique sentences
        response = ' '.join(unique_sentences)

        # 10. Ensure word count is within limits (50-150 words)
        words = response.split()
        if len(words) > 200:
            # Find sentence ending near the word limit
            truncated = ' '.join(words[:200])
            last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))

            if last_period > 0:
                response = truncated[:last_period+1]
            else:
                response = truncated + "."

        # 11. Check sentence completeness
        if response and not response.strip()[-1] in ['.', '!', '?']:
            # Find the last preposition or conjunction
            common_prepositions = ["into", "onto", "about", "above", "across", "after", "along", "around", "at", "before", "behind", "below", "beneath", "beside", "between", "beyond", "by", "down", "during", "except", "for", "from", "in", "inside", "near", "of", "off", "on", "over", "through", "to", "toward", "under", "up", "upon", "with", "within"]

            # Check if ending with preposition or conjunction
            last_word = response.strip().split()[-1].lower() if response.strip().split() else ""
            if last_word in common_prepositions or last_word in ["and", "or", "but"]:
                # Find the last complete sentence
                last_period = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
                if last_period > 0:
                    response = response[:last_period+1]
                else:
                    # If no complete sentence found, modify the ending
                    words = response.strip().split()
                    if words:
                        # Remove the last preposition or conjunction
                        response = " ".join(words[:-1]) + "."

        # 12. Ensure haven't over-filtered
        if not response or len(response) < 40:
            # Try to get the first meaningful paragraph from the original response
            paragraphs = [p for p in original_response.split('\n\n') if p.strip()]
            if paragraphs:
                # Choose the longest paragraph as it's most likely the actual description
                best_para = max(paragraphs, key=len)
                # Clean it using a subset of the above rules
                best_para = re.sub(r'\[.*?\]', '', best_para)  # Remove [SECTION] markers
                best_para = re.sub(r'\s{2,}', ' ', best_para).strip()  # Clean whitespace

                if len(best_para) >= 40:
                    return best_para

            # If still no good content, return a simple message
            return "Unable to generate a valid enhanced description."

        # 13. Final cleaning - catch any missed special cases
        response = re.sub(r'</?\|.*?\|>', '', response)  # Any remaining tags
        response = re.sub(r'\(.*?\)', '', response)  # Any remaining parenthetical content
        response = re.sub(r'Note:.*?(?=\n|$)', '', response, flags=re.IGNORECASE)  # Any remaining notes

        # Ensure proper spacing after punctuation
        response = re.sub(r'([.!?])([A-Z])', r'\1 \2', response)

        # Ensure first letter is capitalized
        if response and response[0].islower():
            response = response[0].upper() + response[1:]

        # 14. 統一格式 - 確保輸出始終是單一段落
        response = re.sub(r'\s*\n\s*', ' ', response)  # 將所有換行符替換為空格
        response = ' '.join(response.split())  

        return response.strip()

    def _format_objects_for_prompt(self, objects: List[Dict]) -> str:
        """格式化物體列表以用於提示"""
        if not objects:
            return "No objects detected"

        formatted = []
        for obj in objects:
            formatted.append(f"{obj['class_name']} (confidence: {obj['confidence']:.2f})")

        return "\n- " + "\n- ".join(formatted)

    def _format_lighting(self, lighting_info: Dict) -> str:
        """格式化光照信息以用於提示"""
        if not lighting_info:
            return "Unknown lighting conditions"

        time = lighting_info.get("time_of_day", "unknown")
        conf = lighting_info.get("confidence", 0)
        is_indoor = lighting_info.get("is_indoor", False)

        base_info = f"{'Indoor' if is_indoor else 'Outdoor'} {time} (confidence: {conf:.2f})"

        # 添加更詳細的診斷信息
        diagnostics = lighting_info.get("diagnostics", {})
        if diagnostics:
            diag_str = "\nAdditional lighting diagnostics:"
            for key, value in diagnostics.items():
                diag_str += f"\n- {key}: {value}"
            base_info += diag_str

        return base_info

    def _format_zones(self, zones: Dict) -> str:
        """格式化功能區域以用於提示"""
        if not zones:
            return "No distinct functional zones identified"

        formatted = ["Identified functional zones:"]
        for zone_name, zone_data in zones.items():
            desc = zone_data.get("description", "")
            objects = zone_data.get("objects", [])

            zone_str = f"- {zone_name}: {desc}"
            if objects:
                zone_str += f" (Contains: {', '.join(objects)})"

            formatted.append(zone_str)

        return "\n".join(formatted)

    def _format_clip_results(self, clip_analysis: Dict) -> str:
        """格式化CLIP分析結果以用於提示"""
        if not clip_analysis or "error" in clip_analysis:
            return "No CLIP analysis available"

        parts = ["CLIP Analysis Results:"]

        # 加上頂級場景
        top_scene, confidence = clip_analysis.get("top_scene", ("unknown", 0))
        parts.append(f"- Most likely scene: {top_scene} (confidence: {confidence:.2f})")

        # 加上視角
        viewpoint, vp_conf = clip_analysis.get("viewpoint", ("standard", 0))
        parts.append(f"- Camera viewpoint: {viewpoint} (confidence: {vp_conf:.2f})")

        # 加上物體組合
        if "object_combinations" in clip_analysis:
            combos = []
            for combo, score in clip_analysis["object_combinations"][:3]:
                combos.append(f"{combo} ({score:.2f})")
            parts.append(f"- Object combinations: {', '.join(combos)}")

        # 加上文化分析
        if "cultural_analysis" in clip_analysis:
            parts.append("- Cultural analysis:")
            for culture_type, data in clip_analysis["cultural_analysis"].items():
                best_desc = data.get("best_description", "")
                desc_conf = data.get("confidence", 0)
                parts.append(f"  * {culture_type}: {best_desc} ({desc_conf:.2f})")

        return "\n".join(parts)

    def _format_cultural_analysis(self, cultural_analysis: Dict) -> str:
        """格式化文化分析結果"""
        if not cultural_analysis:
            return "No specific cultural elements detected"

        parts = []
        for culture_type, data in cultural_analysis.items():
            best_desc = data.get("best_description", "")
            desc_conf = data.get("confidence", 0)
            parts.append(f"{culture_type}: {best_desc} (confidence: {desc_conf:.2f})")

        return "\n".join(parts)
