
import numpy as np
import logging
import traceback
from typing import List, Dict, Tuple, Optional, Union, Any
from PIL import Image

class ImageAnalyzer:
    """
    專注於圖像分析和預處理，包括多尺度金字塔分析、視角分析、建築特徵識別和圖像增強等功能
    """

    def __init__(self):
        """
        初始化圖像分析器
        """
        self.logger = logging.getLogger(__name__)

    def get_image_hash(self, image: Union[Image.Image, np.ndarray]) -> int:
        """
        為圖像生成簡單的 hash 值用於快取

        Args:
            image: PIL Image 或 numpy 數組

        Returns:
            int: 圖像的 hash 值
        """
        try:
            if isinstance(image, np.ndarray):
                # 對於 numpy 數組，降採樣並計算簡單 hash
                small_img = image[::10, ::10] if image.ndim == 3 else image
                return hash(small_img.tobytes())
            else:
                # 對於 PIL 圖像，調整大小後轉換為 bytes
                small_img = image.resize((32, 32))
                return hash(small_img.tobytes())
        except Exception as e:
            self.logger.error(f"Error generating image hash: {e}")
            self.logger.error(traceback.format_exc())
            return 0

    def enhance_features(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        增強圖像特徵以改善地標檢測

        Args:
            image: 輸入圖像

        Returns:
            PIL.Image: 增強後的圖像
        """
        try:
            # ensure PIL format
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    raise ValueError("Unsupported image format. Expected PIL Image or numpy array.")

            # 轉換為numpy進行處理
            img_array = np.array(image)

            # 跳過灰度圖像的處理
            if len(img_array.shape) < 3:
                return image

            # 應用自適應對比度增強
            try:
                from skimage import color, exposure

                # 轉換到LAB色彩空間
                if img_array.shape[2] == 4:  # 處理RGBA
                    img_array = img_array[:,:,:3]

                lab = color.rgb2lab(img_array[:,:,:3] / 255.0)
                l_channel = lab[:,:,0]

                # 增強L通道的對比度
                p2, p98 = np.percentile(l_channel, (2, 98))
                l_channel_enhanced = exposure.rescale_intensity(l_channel, in_range=(p2, p98))

                # 替換L通道並轉換回RGB
                lab[:,:,0] = l_channel_enhanced
                enhanced_img = color.lab2rgb(lab) * 255.0
                enhanced_img = enhanced_img.astype(np.uint8)

                return Image.fromarray(enhanced_img)

            except ImportError:
                self.logger.warning("skimage not available for feature enhancement")
                return image

        except Exception as e:
            self.logger.error(f"Error in feature enhancement: {e}")
            self.logger.error(traceback.format_exc())
            return image

    def analyze_viewpoint(self, image: Union[Image.Image, np.ndarray],
                         clip_model_manager) -> Dict[str, Any]:
        """
        分析圖像視角以調整檢測參數

        Args:
            image: 輸入圖像
            clip_model_manager: CLIP模型管理器實例

        Returns:
            Dict: 視角分析結果
        """
        try:
            viewpoint_prompts = {
                "aerial_view": "an aerial view from above looking down",
                "street_level": "a street level view looking up at a tall structure",
                "eye_level": "an eye-level horizontal view of a landmark",
                "distant": "a distant view of a landmark on the horizon",
                "close_up": "a close-up detailed view of architectural features",
                "interior": "an interior view inside a structure",
                "angled_view": "an angled view of a structure",
                "low_angle": "a low angle view looking up at a building"
            }

            # 計算相似度分數
            viewpoint_scores = self.calculate_similarity_scores(image, viewpoint_prompts, clip_model_manager)

            # 找到主要視角
            dominant_viewpoint = max(viewpoint_scores.items(), key=lambda x: x[1])

            return {
                "viewpoint_scores": viewpoint_scores,
                "dominant_viewpoint": dominant_viewpoint[0],
                "confidence": dominant_viewpoint[1]
            }

        except Exception as e:
            self.logger.error(f"Error in viewpoint analysis: {e}")
            self.logger.error(traceback.format_exc())
            return {
                "viewpoint_scores": {},
                "dominant_viewpoint": "eye_level",
                "confidence": 0.0
            }

    def calculate_similarity_scores(self, image: Union[Image.Image, np.ndarray],
                                  prompts: Dict[str, str],
                                  clip_model_manager) -> Dict[str, float]:
        """
        計算圖像與一組特定提示之間的相似度分數

        Args:
            image: 輸入圖像
            prompts: 提示詞字典 {名稱: 提示文本}
            clip_model_manager: CLIP模型管理器實例

        Returns:
            Dict[str, float]: 每個提示的相似度分數
        """
        try:
            # ensure PIL format
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    raise ValueError("Unsupported image format. Expected PIL Image or numpy array.")

            # preprocess image
            image_input = clip_model_manager.preprocess_image(image)

            # get image features
            image_features = clip_model_manager.encode_image(image_input)

            # 計算與每個提示的similarity
            scores = {}
            prompt_texts = list(prompts.values())
            prompt_features = clip_model_manager.encode_single_text(prompt_texts)

            # 計算相似度
            similarity = clip_model_manager.calculate_similarity(image_features, prompt_features)

            # result
            for i, (name, _) in enumerate(prompts.items()):
                scores[name] = float(similarity[0][i])

            return scores

        except Exception as e:
            self.logger.error(f"Error calculating similarity scores: {e}")
            self.logger.error(traceback.format_exc())
            return {}

    def analyze_architectural_features(self, image: Union[Image.Image, np.ndarray],
                                     clip_model_manager) -> Dict[str, Any]:
        """
        分析圖像中結構的建築特徵，不硬編碼特定地標

        Args:
            image: 輸入圖像
            clip_model_manager: CLIP模型管理器實例

        Returns:
            Dict: 建築特徵分析結果
        """
        try:
            # 定義通用建築特徵提示，適用於所有類型的地標
            architecture_prompts = {
                "tall_structure": "a tall vertical structure standing alone",
                "tiered_building": "a building with multiple stacked tiers or segments",
                "historical_structure": "a building with historical architectural elements",
                "modern_design": "a modern structure with contemporary architectural design",
                "segmented_exterior": "a structure with visible segmented or sectioned exterior",
                "viewing_platform": "a tall structure with observation area at the top",
                "time_display": "a structure with timepiece features",
                "glass_facade": "a building with prominent glass exterior surfaces",
                "memorial_structure": "a monument or memorial structure",
                "ancient_construction": "ancient constructed elements or archaeological features",
                "natural_landmark": "a natural geographic formation or landmark",
                "slanted_design": "a structure with non-vertical or leaning profile"
            }

            # 計算與通用建築模式的相似度分數
            context_scores = self.calculate_similarity_scores(image, architecture_prompts, clip_model_manager)

            # 確定最相關的建築特徵
            top_features = sorted(context_scores.items(), key=lambda x: x[1], reverse=True)[:3]

            # 計算特徵置信度
            context_confidence = sum(score for _, score in top_features) / 3

            # 根據頂級特徵確定主要建築類別
            architectural_categories = {
                "tower": ["tall_structure", "viewing_platform", "time_display"],
                "skyscraper": ["tall_structure", "modern_design", "glass_facade"],
                "historical": ["historical_structure", "ancient_construction", "memorial_structure"],
                "natural": ["natural_landmark"],
                "distinctive": ["tiered_building", "segmented_exterior", "slanted_design"]
            }

            # 根據頂級特徵為每個類別評分
            category_scores = {}
            for category, features in architectural_categories.items():
                category_score = 0
                for feature, score in context_scores.items():
                    if feature in features:
                        category_score += score
                category_scores[category] = category_score

            primary_category = max(category_scores.items(), key=lambda x: x[1])[0]

            return {
                "architectural_features": top_features,
                "context_confidence": context_confidence,
                "primary_category": primary_category,
                "category_scores": category_scores
            }

        except Exception as e:
            self.logger.error(f"Error in architectural feature analysis: {e}")
            self.logger.error(traceback.format_exc())
            return {
                "architectural_features": [],
                "context_confidence": 0.0,
                "primary_category": "building",
                "category_scores": {}
            }

    def perform_pyramid_analysis(self, image: Union[Image.Image, np.ndarray],
                               clip_model_manager, landmark_data_manager,
                               levels: int = 4, base_threshold: float = 0.25,
                               aspect_ratios: List[float] = [1.0, 0.75, 1.5]) -> Dict[str, Any]:
        """
        對圖像執行多尺度金字塔分析以改善地標檢測

        Args:
            image: 輸入圖像
            clip_model_manager: CLIP模型管理器實例
            landmark_data_manager: 地標數據管理器實例
            levels: 金字塔層級數
            base_threshold: 基礎置信度閾值
            aspect_ratios: 不同縱橫比列表

        Returns:
            Dict: 金字塔分析結果
        """
        try:
            # 確保圖像是PIL格式
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    raise ValueError("Unsupported image format. Expected PIL Image or numpy array.")

            width, height = image.size
            pyramid_results = []

            # 獲取預計算的地標文本特徵
            landmark_prompts = landmark_data_manager.get_landmark_prompts()
            if not landmark_prompts:
                return {
                    "is_landmark": False,
                    "results": [],
                    "best_result": None
                }

            landmark_text_features = clip_model_manager.encode_text_batch(landmark_prompts)

            # 對每個縮放和縱橫比組合進行處理
            for level in range(levels):
                # 計算縮放因子
                scale_factor = 1.0 - (level * 0.2)

                for aspect_ratio in aspect_ratios:
                    # 計算新尺寸，保持面積近似不變
                    if aspect_ratio != 1.0:
                        # 保持面積近似不變的情況下調整縱橫比
                        new_width = int(width * scale_factor * (1/aspect_ratio)**0.5)
                        new_height = int(height * scale_factor * aspect_ratio**0.5)
                    else:
                        new_width = int(width * scale_factor)
                        new_height = int(height * scale_factor)

                    # 調整圖像大小
                    scaled_image = image.resize((new_width, new_height), Image.LANCZOS)

                    # 預處理圖像
                    image_input = clip_model_manager.preprocess_image(scaled_image)

                    # 獲取圖像特徵
                    image_features = clip_model_manager.encode_image(image_input)

                    # 計算相似度
                    similarity = clip_model_manager.calculate_similarity(image_features, landmark_text_features)

                    # 找到最佳匹配
                    best_idx = similarity[0].argmax().item()
                    best_score = similarity[0][best_idx]

                    if best_score >= base_threshold:
                        landmark_id, landmark_info = landmark_data_manager.get_landmark_by_index(best_idx)
                        if landmark_id:
                            pyramid_results.append({
                                "landmark_id": landmark_id,
                                "landmark_name": landmark_info.get("name", "Unknown"),
                                "confidence": float(best_score),
                                "scale_factor": scale_factor,
                                "aspect_ratio": aspect_ratio,
                                "location": landmark_info.get("location", "Unknown Location")
                            })

            # 按置信度排序
            pyramid_results.sort(key=lambda x: x["confidence"], reverse=True)

            return {
                "is_landmark": len(pyramid_results) > 0,
                "results": pyramid_results,
                "best_result": pyramid_results[0] if pyramid_results else None
            }

        except Exception as e:
            self.logger.error(f"Error in pyramid analysis: {e}")
            self.logger.error(traceback.format_exc())
            return {
                "is_landmark": False,
                "results": [],
                "best_result": None
            }
