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

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        初始化 CLIP 分析器。

        Args:
            model_name: CLIP Model name,  "ViT-B/32"、"ViT-B/16"、"ViT-L/14"
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
        """準備所有文本提示的 CLIP 特徵"""
        # base prompt
        scene_texts = [self.scene_type_prompts[scene_type] for scene_type in self.scene_type_prompts]
        self.scene_type_tokens = clip.tokenize(scene_texts).to(self.device)

        # cultural
        self.cultural_tokens_dict = {}
        for scene_type, prompts in self.cultural_scene_prompts.items():
            self.cultural_tokens_dict[scene_type] = clip.tokenize(prompts).to(self.device)

        # Light
        lighting_texts = [self.lighting_condition_prompts[cond] for cond in self.lighting_condition_prompts]
        self.lighting_tokens = clip.tokenize(lighting_texts).to(self.device)

        # specializes_status
        self.specialized_tokens_dict = {}
        for scene_type, prompts in self.specialized_scene_prompts.items():
            self.specialized_tokens_dict[scene_type] = clip.tokenize(prompts).to(self.device)

        # view point
        viewpoint_texts = [self.viewpoint_prompts[viewpoint] for viewpoint in self.viewpoint_prompts]
        self.viewpoint_tokens = clip.tokenize(viewpoint_texts).to(self.device)

        # object combination
        object_combination_texts = [self.object_combination_prompts[combo] for combo in self.object_combination_prompts]
        self.object_combination_tokens = clip.tokenize(object_combination_texts).to(self.device)

        # activicty prompt
        activity_texts = [self.activity_prompts[activity] for activity in self.activity_prompts]
        self.activity_tokens = clip.tokenize(activity_texts).to(self.device)

    def analyze_image(self, image, include_cultural_analysis: bool = True) -> Dict[str, Any]:
        """
        分析圖像，預測場景類型和光照條件。

        Args:
            image: 輸入圖像 (PIL Image 或 numpy array)
            include_cultural_analysis: 是否包含文化場景的詳細分析

        Returns:
            Dict: 包含場景類型預測和光照條件的分析結果
        """
        try:
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

            # 分析場景類型
            scene_scores = self._analyze_scene_type(image_features)

            # 分析光照條件
            lighting_scores = self._analyze_lighting_condition(image_features)

            # 文化場景的增強分析
            cultural_analysis = {}
            if include_cultural_analysis:
                for scene_type in self.cultural_scene_prompts:
                    if scene_type in scene_scores and scene_scores[scene_type] > 0.2:
                        cultural_analysis[scene_type] = self._analyze_cultural_scene(
                            image_features, scene_type
                        )

            specialized_analysis = {}
            for scene_type in self.specialized_scene_prompts:
                if scene_type in scene_scores and scene_scores[scene_type] > 0.2:
                    specialized_analysis[scene_type] = self._analyze_specialized_scene(
                        image_features, scene_type
                    )

            viewpoint_scores = self._analyze_viewpoint(image_features)

            object_combination_scores = self._analyze_object_combinations(image_features)

            activity_scores = self._analyze_activities(image_features)

            # display results
            result = {
                "scene_scores": scene_scores,
                "top_scene": max(scene_scores.items(), key=lambda x: x[1]),
                "lighting_condition": max(lighting_scores.items(), key=lambda x: x[1]),
                "embedding": image_features.cpu().numpy().tolist()[0] if self.device == "cuda" else image_features.numpy().tolist()[0],
                "viewpoint": max(viewpoint_scores.items(), key=lambda x: x[1]),
                "object_combinations": sorted(object_combination_scores.items(), key=lambda x: x[1], reverse=True)[:3],
                "activities": sorted(activity_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            }

            if cultural_analysis:
                result["cultural_analysis"] = cultural_analysis

            if specialized_analysis:
                result["specialized_analysis"] = specialized_analysis

            return result

        except Exception as e:
            print(f"Error analyzing image with CLIP: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def _analyze_scene_type(self, image_features: torch.Tensor) -> Dict[str, float]:
        """分析圖像特徵與各場景類型的相似度"""
        with torch.no_grad():
            # 計算場景類型文本特徵
            text_features = self.model.encode_text(self.scene_type_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 計算相似度分數
            similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
            similarity = similarity.cpu().numpy()[0] if self.device == "cuda" else similarity.numpy()[0]

            # 建立場景分數字典
            scene_scores = {}
            for i, scene_type in enumerate(self.scene_type_prompts.keys()):
                scene_scores[scene_type] = float(similarity[i])

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
