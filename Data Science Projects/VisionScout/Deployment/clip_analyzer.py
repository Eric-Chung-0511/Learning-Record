import torch
import clip
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional, Union

from clip_prompts import (
    SCENE_TYPE_PROMPTS,
    CULTURAL_SCENE_PROMPTS,
    COMPARATIVE_PROMPTS,
    LIGHTING_CONDITION_PROMPTS,
    SPECIALIZED_SCENE_PROMPTS,
    VIEWPOINT_PROMPTS,
    OBJECT_COMBINATION_PROMPTS,
    ACTIVITY_PROMPTS
)

class CLIPAnalyzer:
    """
    Use Clip to intergrate scene understanding function
    """

    def __init__(self, model_name: str = "ViT-B/16", device: str = None):
        """
        初始化 CLIP 分析器。

        Args:
            model_name: CLIP Model name, 默認 "ViT-B/16"
            device: Use GPU if it can use
        """
        # 自動選擇設備
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading CLIP model {model_name} on {self.device}...")
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            print(f"CLIP model loaded successfully.")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise

        self.scene_type_prompts = SCENE_TYPE_PROMPTS
        self.cultural_scene_prompts = CULTURAL_SCENE_PROMPTS
        self.comparative_prompts = COMPARATIVE_PROMPTS
        self.lighting_condition_prompts = LIGHTING_CONDITION_PROMPTS
        self.specialized_scene_prompts = SPECIALIZED_SCENE_PROMPTS
        self.viewpoint_prompts = VIEWPOINT_PROMPTS
        self.object_combination_prompts = OBJECT_COMBINATION_PROMPTS
        self.activity_prompts = ACTIVITY_PROMPTS

        # turn to CLIP format
        self._prepare_text_prompts()

    def _prepare_text_prompts(self):
        """準備所有文本提示的 CLIP 特徵並存儲到 self.text_features_cache 中"""
        self.text_features_cache = {}

        # 處理基礎場景類型 (SCENE_TYPE_PROMPTS)
        if hasattr(self, 'scene_type_prompts') and self.scene_type_prompts:
            scene_texts = [prompt for scene_type, prompt in self.scene_type_prompts.items()]
            if scene_texts:
                self.text_features_cache["scene_type_keys"] = list(self.scene_type_prompts.keys())
                try:
                    self.text_features_cache["scene_type_tokens"] = clip.tokenize(scene_texts).to(self.device)
                except Exception as e:
                    print(f"Warning: Error tokenizing scene_type_prompts: {e}")
                    self.text_features_cache["scene_type_tokens"] = None # 標記錯誤或空
            else:
                self.text_features_cache["scene_type_keys"] = []
                self.text_features_cache["scene_type_tokens"] = None
        else:
            self.text_features_cache["scene_type_keys"] = []
            self.text_features_cache["scene_type_tokens"] = None

        # 處理文化場景 (CULTURAL_SCENE_PROMPTS)
        # cultural_tokens_dict 存儲的是 tokenized prompts
        cultural_tokens_dict_val = {}
        if hasattr(self, 'cultural_scene_prompts') and self.cultural_scene_prompts:
            for scene_type, prompts in self.cultural_scene_prompts.items():
                if prompts and isinstance(prompts, list) and all(isinstance(p, str) for p in prompts):
                    try:
                        cultural_tokens_dict_val[scene_type] = clip.tokenize(prompts).to(self.device)
                    except Exception as e:
                        print(f"Warning: Error tokenizing cultural_scene_prompts for {scene_type}: {e}")
                        cultural_tokens_dict_val[scene_type] = None # 標記錯誤或空
                else:
                    cultural_tokens_dict_val[scene_type] = None # prompts 不合規
        self.text_features_cache["cultural_tokens_dict"] = cultural_tokens_dict_val

        # 處理光照條件 (LIGHTING_CONDITION_PROMPTS)
        if hasattr(self, 'lighting_condition_prompts') and self.lighting_condition_prompts:
            lighting_texts = [prompt for cond, prompt in self.lighting_condition_prompts.items()]
            if lighting_texts:
                self.text_features_cache["lighting_condition_keys"] = list(self.lighting_condition_prompts.keys())
                try:
                    self.text_features_cache["lighting_tokens"] = clip.tokenize(lighting_texts).to(self.device)
                except Exception as e:
                    print(f"Warning: Error tokenizing lighting_condition_prompts: {e}")
                    self.text_features_cache["lighting_tokens"] = None
            else:
                self.text_features_cache["lighting_condition_keys"] = []
                self.text_features_cache["lighting_tokens"] = None
        else:
            self.text_features_cache["lighting_condition_keys"] = []
            self.text_features_cache["lighting_tokens"] = None

        # 處理特殊場景 (SPECIALIZED_SCENE_PROMPTS)
        specialized_tokens_dict_val = {}
        if hasattr(self, 'specialized_scene_prompts') and self.specialized_scene_prompts:
            for scene_type, prompts in self.specialized_scene_prompts.items():
                if prompts and isinstance(prompts, list) and all(isinstance(p, str) for p in prompts):
                    try:
                        specialized_tokens_dict_val[scene_type] = clip.tokenize(prompts).to(self.device)
                    except Exception as e:
                        print(f"Warning: Error tokenizing specialized_scene_prompts for {scene_type}: {e}")
                        specialized_tokens_dict_val[scene_type] = None
                else:
                    specialized_tokens_dict_val[scene_type] = None
        self.text_features_cache["specialized_tokens_dict"] = specialized_tokens_dict_val

        # 處理視角 (VIEWPOINT_PROMPTS)
        if hasattr(self, 'viewpoint_prompts') and self.viewpoint_prompts:
            viewpoint_texts = [prompt for viewpoint, prompt in self.viewpoint_prompts.items()]
            if viewpoint_texts:
                self.text_features_cache["viewpoint_keys"] = list(self.viewpoint_prompts.keys())
                try:
                    self.text_features_cache["viewpoint_tokens"] = clip.tokenize(viewpoint_texts).to(self.device)
                except Exception as e:
                    print(f"Warning: Error tokenizing viewpoint_prompts: {e}")
                    self.text_features_cache["viewpoint_tokens"] = None
            else:
                self.text_features_cache["viewpoint_keys"] = []
                self.text_features_cache["viewpoint_tokens"] = None
        else:
            self.text_features_cache["viewpoint_keys"] = []
            self.text_features_cache["viewpoint_tokens"] = None

        # 處理物件組合 (OBJECT_COMBINATION_PROMPTS)
        if hasattr(self, 'object_combination_prompts') and self.object_combination_prompts:
            object_combination_texts = [prompt for combo, prompt in self.object_combination_prompts.items()]
            if object_combination_texts:
                self.text_features_cache["object_combination_keys"] = list(self.object_combination_prompts.keys())
                try:
                    self.text_features_cache["object_combination_tokens"] = clip.tokenize(object_combination_texts).to(self.device)
                except Exception as e:
                    print(f"Warning: Error tokenizing object_combination_prompts: {e}")
                    self.text_features_cache["object_combination_tokens"] = None
            else:
                self.text_features_cache["object_combination_keys"] = []
                self.text_features_cache["object_combination_tokens"] = None
        else:
            self.text_features_cache["object_combination_keys"] = []
            self.text_features_cache["object_combination_tokens"] = None

        # 處理活動 (ACTIVITY_PROMPTS)
        if hasattr(self, 'activity_prompts') and self.activity_prompts:
            activity_texts = [prompt for activity, prompt in self.activity_prompts.items()]
            if activity_texts:
                self.text_features_cache["activity_keys"] = list(self.activity_prompts.keys())
                try:
                    self.text_features_cache["activity_tokens"] = clip.tokenize(activity_texts).to(self.device)
                except Exception as e:
                    print(f"Warning: Error tokenizing activity_prompts: {e}")
                    self.text_features_cache["activity_tokens"] = None
            else:
                self.text_features_cache["activity_keys"] = []
                self.text_features_cache["activity_tokens"] = None
        else:
            self.text_features_cache["activity_keys"] = []
            self.text_features_cache["activity_tokens"] = None

        self.scene_type_tokens = self.text_features_cache["scene_type_tokens"]
        self.lighting_tokens = self.text_features_cache["lighting_tokens"]
        self.viewpoint_tokens = self.text_features_cache["viewpoint_tokens"]
        self.object_combination_tokens = self.text_features_cache["object_combination_tokens"]
        self.activity_tokens = self.text_features_cache["activity_tokens"]
        self.cultural_tokens_dict = self.text_features_cache["cultural_tokens_dict"]
        self.specialized_tokens_dict = self.text_features_cache["specialized_tokens_dict"]

        print("CLIP text_features_cache prepared.")

    def analyze_image(self, image, include_cultural_analysis=True, exclude_categories=None, enable_landmark=True, places365_guidance=None):
        """
        分析圖像，預測場景類型和光照條件。

        Args:
            image: 輸入圖像 (PIL Image 或 numpy array)
            include_cultural_analysis: 是否包含文化場景的詳細分析
            exclude_categories: 要排除的類別列表
            enable_landmark: 是否啟用地標檢測功能
            places365_guidance: Places365 提供的場景指導信息 (可選)


        Returns:
            Dict: 包含場景類型預測和光照條件的分析結果
        """
        try:
            self.enable_landmark = enable_landmark # 更新實例的 enable_landmark 狀態
            # 確保圖像是 PIL 格式
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    raise ValueError("Unsupported image format. Expected PIL Image or numpy array.")

            # 預處理圖像
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            # 獲取圖像特徵
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            places365_focus_areas = []
            places365_scene_context = "" # 用於存儲 Places365 提供的場景描述

            if places365_guidance and isinstance(places365_guidance, dict) and places365_guidance.get('confidence', 0) > 0.4:
                mapped_scene = places365_guidance.get('mapped_scene_type', '')
                scene_label = places365_guidance.get('scene_label', '')
                # is_indoor = places365_guidance.get('is_indoor', None) # 未使用，可註釋
                attributes = places365_guidance.get('attributes', [])

                places365_scene_context = f"Scene identified by Places365 as {scene_label}" # 更新上下文描述

                # Adjust CLIP analysis focus based on Places365 scene type
                if mapped_scene in ['kitchen', 'dining_area', 'restaurant']:
                    places365_focus_areas.extend(['food preparation', 'dining setup', 'kitchen appliances'])
                elif mapped_scene in ['office_workspace', 'educational_setting', 'library', 'conference_room']:
                    places365_focus_areas.extend(['work environment', 'professional setting', 'learning space', 'study area'])
                elif mapped_scene in ['retail_store', 'shopping_mall', 'market', 'supermarket']: # 擴展匹配
                    places365_focus_areas.extend(['commercial space', 'shopping environment', 'retail display', 'goods for sale'])
                elif mapped_scene in ['park_area', 'beach', 'natural_outdoor_area', 'playground', 'sports_field']: # 擴展匹配
                    places365_focus_areas.extend(['outdoor recreation', 'natural environment', 'leisure activity', 'open space'])

                # 根據屬性添加更通用的 focus areas
                if isinstance(attributes, list): # 確保 attributes 是列表
                    if 'commercial' in attributes:
                        places365_focus_areas.append('business activity')
                    if 'recreational' in attributes:
                        places365_focus_areas.append('entertainment or leisure')
                    if 'residential' in attributes:
                        places365_focus_areas.append('living space')

                # 去重
                places365_focus_areas = list(set(places365_focus_areas))

                if places365_focus_areas: # 只有在確實有 focus areas 時才打印
                    print(f"CLIP analysis guided by Places365: {places365_scene_context}, focus areas: {places365_focus_areas}")

            # 分析場景類型，傳遞 enable_landmark 參數和 Places365 指導
            scene_scores = self._analyze_scene_type(image_features,
                                                  enable_landmark=self.enable_landmark, # 使用更新後的實例屬性
                                                  places365_focus=places365_focus_areas)

            # 如果禁用地標功能，確保排除地標相關類別
            current_exclude_categories = list(exclude_categories) if exclude_categories is not None else []
            if not self.enable_landmark: # 使用更新後的實例屬性
                landmark_related_terms = ["landmark", "monument", "tower", "tourist", "attraction", "historical", "famous", "iconic"]
                for term in landmark_related_terms:
                    if term not in current_exclude_categories:
                        current_exclude_categories.append(term)

            if current_exclude_categories:
                filtered_scores = {}
                for scene, score in scene_scores.items():
                    # 檢查 scene 的鍵名（通常是英文）是否包含任何排除詞彙
                    if not any(cat.lower() in scene.lower() for cat in current_exclude_categories):
                        filtered_scores[scene] = score

                if filtered_scores:
                    total_score = sum(filtered_scores.values())
                    if total_score > 1e-5: # 避免除以零或非常小的數
                        scene_scores = {k: v / total_score for k, v in filtered_scores.items()}
                    else: # 如果總分趨近於0，則保持原樣或設為0
                        scene_scores = {k: 0.0 for k in filtered_scores.keys()} # 或者 scene_scores = filtered_scores
                else: # 如果過濾後沒有場景了
                    scene_scores = {k: (0.0 if any(cat.lower() in k.lower() for cat in current_exclude_categories) else v) for k,v in scene_scores.items()}
                    if not any(s > 1e-5 for s in scene_scores.values()): # 如果還是全0
                         scene_scores = {"unknown": 1.0} # 給一個默認值避免空字典

            lighting_scores = self._analyze_lighting_condition(image_features)
            cultural_analysis = {}
            if include_cultural_analysis and self.enable_landmark: # 使用更新後的實例屬性
                for scene_type_cultural_key in self.text_features_cache.get("cultural_tokens_dict", {}).keys():
                     # 確保 scene_type_cultural_key 是 SCENE_TYPE_PROMPTS 中的鍵，或者有一個映射關係
                    if scene_type_cultural_key in scene_scores and scene_scores[scene_type_cultural_key] > 0.2:
                        cultural_analysis[scene_type_cultural_key] = self._analyze_cultural_scene(
                            image_features, scene_type_cultural_key
                        )

            specialized_analysis = {}
            for scene_type_specialized_key in self.text_features_cache.get("specialized_tokens_dict", {}).keys():
                if scene_type_specialized_key in scene_scores and scene_scores[scene_type_specialized_key] > 0.2:
                    specialized_analysis[scene_type_specialized_key] = self._analyze_specialized_scene(
                        image_features, scene_type_specialized_key
                    )

            viewpoint_scores = self._analyze_viewpoint(image_features)
            object_combination_scores = self._analyze_object_combinations(image_features)
            activity_scores = self._analyze_activities(image_features)

            if scene_scores: # 確保 scene_scores 不是空的
                top_scene = max(scene_scores.items(), key=lambda x: x[1])
                 # 如果禁用地標，再次確認 top_scene 不是地標相關
                if not self.enable_landmark and any(cat.lower() in top_scene[0].lower() for cat in current_exclude_categories):
                    non_excluded_scores = {k:v for k,v in scene_scores.items() if not any(cat.lower() in k.lower() for cat in current_exclude_categories)}
                    if non_excluded_scores:
                        top_scene = max(non_excluded_scores.items(), key=lambda x: x[1])
                    else:
                        top_scene = ("unknown", 0.0) # 或其他合適的默認值
            else:
                top_scene = ("unknown", 0.0)


            result = {
                "scene_scores": scene_scores,
                "top_scene": top_scene,
                "lighting_condition": max(lighting_scores.items(), key=lambda x: x[1]) if lighting_scores else ("unknown", 0.0),
                "embedding": image_features.cpu().numpy().tolist()[0], # 簡化
                "viewpoint": max(viewpoint_scores.items(), key=lambda x: x[1]) if viewpoint_scores else ("unknown", 0.0),
                "object_combinations": sorted(object_combination_scores.items(), key=lambda x: x[1], reverse=True)[:3] if object_combination_scores else [],
                "activities": sorted(activity_scores.items(), key=lambda x: x[1], reverse=True)[:3] if activity_scores else []
            }

            if places365_guidance and isinstance(places365_guidance, dict) and places365_focus_areas: # 檢查 places365_focus_areas 是否被填充
                result["places365_guidance"] = {
                    "scene_context": places365_scene_context,
                    "focus_areas": places365_focus_areas, # 現在這個會包含基於 guidance 的內容
                    "guided_analysis": True,
                    "original_places365_scene": places365_guidance.get('scene_label', 'N/A'),
                    "original_places365_confidence": places365_guidance.get('confidence', 0.0)
                }

            if cultural_analysis and self.enable_landmark:
                result["cultural_analysis"] = cultural_analysis

            if specialized_analysis:
                result["specialized_analysis"] = specialized_analysis

            return result

        except Exception as e:
            print(f"Error analyzing image with CLIP: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "scene_scores": {}, "top_scene": ("error", 0.0)}

    def _analyze_scene_type(self, image_features: torch.Tensor, enable_landmark: bool = True, places365_focus: List[str] = None) -> Dict[str, float]:
        """
        分析圖像特徵與各場景類型的相似度，並可選擇性地排除地標相關場景

        Args:
            image_features: 經過 CLIP 編碼的圖像特徵
            enable_landmark: 是否啟用地標識別功能

        Returns:
            Dict[str, float]: 各場景類型的相似度分數字典
        """
        with torch.no_grad():
            # 計算場景類型文本特徵
            text_features = self.model.encode_text(self.scene_type_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Apply Places365 guidance if available
            if places365_focus and len(places365_focus) > 0:
                # Create enhanced prompts that incorporate Places365 guidance
                enhanced_prompts = []
                for scene_type in self.scene_type_prompts.keys():
                    base_prompt = self.scene_type_prompts[scene_type]

                    # Check if this scene type should be emphasized based on Places365 guidance
                    scene_lower = scene_type.lower()
                    should_enhance = False

                    for focus_area in places365_focus:
                        if any(keyword in scene_lower for keyword in focus_area.split()):
                            should_enhance = True
                            enhanced_prompts.append(f"{base_prompt} with {focus_area}")
                            break

                    if not should_enhance:
                        enhanced_prompts.append(base_prompt)

                # Re-tokenize and encode enhanced prompts
                enhanced_tokens = clip.tokenize(enhanced_prompts).to(self.device)
                text_features = self.model.encode_text(enhanced_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 計算相似度分數
            similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
            similarity = similarity.cpu().numpy()[0] if self.device == "cuda" else similarity.numpy()[0]

            # 建立場景分數字典
            scene_scores = {}
            for i, scene_type in enumerate(self.scene_type_prompts.keys()):
                # 如果未啟用地標功能，則跳過地標相關場景類型
                if not enable_landmark and scene_type in ["tourist_landmark", "natural_landmark", "historical_monument"]:
                    scene_scores[scene_type] = 0.0  # 將地標場景分數設為零
                else:
                    base_score = float(similarity[i])

                    # Apply Places365 guidance boost if applicable
                    if places365_focus:
                        scene_lower = scene_type.lower()
                        boost_factor = 1.0

                        for focus_area in places365_focus:
                            if any(keyword in scene_lower for keyword in focus_area.split()):
                                boost_factor = 1.15  # 15% boost for matching scenes
                                break

                        scene_scores[scene_type] = base_score * boost_factor
                    else:
                        scene_scores[scene_type] = base_score

            # 如果禁用地標功能，確保重新歸一化剩餘場景分數
            if not enable_landmark:
                # 獲取所有非零分數
                non_zero_scores = {k: v for k, v in scene_scores.items() if v > 0}
                if non_zero_scores:
                    # 計算總和並歸一化
                    total_score = sum(non_zero_scores.values())
                    if total_score > 0:
                        for scene_type in non_zero_scores:
                            scene_scores[scene_type] = non_zero_scores[scene_type] / total_score

            return scene_scores

    def _analyze_lighting_condition(self, image_features: torch.Tensor) -> Dict[str, float]:
        """分析圖像的光照條件"""
        with torch.no_grad():
            # 計算光照條件文本特徵
            text_features = self.model.encode_text(self.lighting_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 計算相似度分數
            similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
            similarity = similarity.cpu().numpy()[0] if self.device == "cuda" else similarity.numpy()[0]

            # 建立光照條件分數字典
            lighting_scores = {}
            for i, lighting_type in enumerate(self.lighting_condition_prompts.keys()):
                lighting_scores[lighting_type] = float(similarity[i])

            return lighting_scores

    def _analyze_cultural_scene(self, image_features: torch.Tensor, scene_type: str) -> Dict[str, Any]:
        """針對特定文化場景進行深入分析"""
        if scene_type not in self.cultural_tokens_dict:
            return {"error": f"No cultural analysis available for {scene_type}"}

        with torch.no_grad():
            # 獲取特定文化場景的文本特徵
            cultural_tokens = self.cultural_tokens_dict[scene_type]
            text_features = self.model.encode_text(cultural_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 計算相似度分數
            similarity = (100 * image_features @ text_features.T)
            similarity = similarity.cpu().numpy()[0] if self.device == "cuda" else similarity.numpy()[0]

            # 找到最匹配的文化描述
            prompts = self.cultural_scene_prompts[scene_type]
            scores = [(prompts[i], float(similarity[i])) for i in range(len(prompts))]
            scores.sort(key=lambda x: x[1], reverse=True)

            return {
                "best_description": scores[0][0],
                "confidence": scores[0][1],
                "all_matches": scores
            }

    def _analyze_specialized_scene(self, image_features: torch.Tensor, scene_type: str) -> Dict[str, Any]:
        """針對特定專門場景進行深入分析"""
        if scene_type not in self.specialized_tokens_dict:
            return {"error": f"No specialized analysis available for {scene_type}"}

        with torch.no_grad():
            # 獲取特定專門場景的文本特徵
            specialized_tokens = self.specialized_tokens_dict[scene_type]
            text_features = self.model.encode_text(specialized_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 計算相似度分數
            similarity = (100 * image_features @ text_features.T)
            similarity = similarity.cpu().numpy()[0] if self.device == "cuda" else similarity.numpy()[0]

            # 找到最匹配的專門描述
            prompts = self.specialized_scene_prompts[scene_type]
            scores = [(prompts[i], float(similarity[i])) for i in range(len(prompts))]
            scores.sort(key=lambda x: x[1], reverse=True)

            return {
                "best_description": scores[0][0],
                "confidence": scores[0][1],
                "all_matches": scores
            }

    def _analyze_viewpoint(self, image_features: torch.Tensor) -> Dict[str, float]:
        """分析圖像的拍攝視角"""
        with torch.no_grad():
            # 計算視角文本特徵
            text_features = self.model.encode_text(self.viewpoint_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 計算相似度分數
            similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
            similarity = similarity.cpu().numpy()[0] if self.device == "cuda" else similarity.numpy()[0]

            # 建立視角分數字典
            viewpoint_scores = {}
            for i, viewpoint in enumerate(self.viewpoint_prompts.keys()):
                viewpoint_scores[viewpoint] = float(similarity[i])

            return viewpoint_scores

    def _analyze_object_combinations(self, image_features: torch.Tensor) -> Dict[str, float]:
        """分析圖像中的物體組合"""
        with torch.no_grad():
            # 計算物體組合文本特徵
            text_features = self.model.encode_text(self.object_combination_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 計算相似度分數
            similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
            similarity = similarity.cpu().numpy()[0] if self.device == "cuda" else similarity.numpy()[0]

            # 建立物體組合分數字典
            combination_scores = {}
            for i, combination in enumerate(self.object_combination_prompts.keys()):
                combination_scores[combination] = float(similarity[i])

            return combination_scores

    def _analyze_activities(self, image_features: torch.Tensor) -> Dict[str, float]:
        """分析圖像中的活動"""
        with torch.no_grad():
            # 計算活動文本特徵
            text_features = self.model.encode_text(self.activity_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 計算相似度分數
            similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
            similarity = similarity.cpu().numpy()[0] if self.device == "cuda" else similarity.numpy()[0]

            # 建立活動分數字典
            activity_scores = {}
            for i, activity in enumerate(self.activity_prompts.keys()):
                activity_scores[activity] = float(similarity[i])

            return activity_scores

    def get_image_embedding(self, image) -> np.ndarray:
        """
        獲取圖像的 CLIP 嵌入表示

        Args:
            image: PIL Image 或 numpy array

        Returns:
            np.ndarray: 圖像的 CLIP 特徵向量
        """
        # 確保圖像是 PIL 格式
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                raise ValueError("Unsupported image format. Expected PIL Image or numpy array.")

        # 預處理並編碼
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 轉換為 numpy 並返回
        return image_features.cpu().numpy()[0] if self.device == "cuda" else image_features.numpy()[0]

    def text_to_embedding(self, text: str) -> np.ndarray:
        """
        將文本轉換為 CLIP 嵌入表示

        Args:
            text: 輸入文本

        Returns:
            np.ndarray: 文本的 CLIP 特徵向量
        """
        text_token = clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_token)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy()[0] if self.device == "cuda" else text_features.numpy()[0]

    def calculate_similarity(self, image, text_queries: List[str]) -> Dict[str, float]:
        """
        計算圖像與多個文本查詢的相似度

        Args:
            image: PIL Image 或 numpy array
            text_queries: 文本查詢列表

        Returns:
            Dict: 每個查詢的相似度分數
        """
        # 獲取圖像嵌入
        if isinstance(image, np.ndarray) and len(image.shape) == 1:
            # 已經是嵌入向量
            image_features = torch.tensor(image).unsqueeze(0).to(self.device)
        else:
            # 是圖像，需要提取嵌入
            image_features = torch.tensor(self.get_image_embedding(image)).unsqueeze(0).to(self.device)

        # calulate similarity
        text_tokens = clip.tokenize(text_queries).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            similarity = similarity.cpu().numpy()[0] if self.device == "cuda" else similarity.numpy()[0]

        # display results
        result = {}
        for i, query in enumerate(text_queries):
            result[query] = float(similarity[i])

        return result

    def get_clip_instance(self):
        """
        獲取初始化好的CLIP模型實例，便於其他模組重用

        Returns:
            tuple: (模型實例, 預處理函數, 設備名稱)
        """
        return self.model, self.preprocess, self.device
