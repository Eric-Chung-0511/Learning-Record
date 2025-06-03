
import torch
import clip
import numpy as np
import logging
import traceback
from typing import List, Dict, Tuple, Optional, Union, Any
from PIL import Image

class CLIPModelManager:
    """
    專門管理 CLIP 模型相關的操作，包括模型載入、設備管理、圖像和文本的特徵編碼等核心功能
    """

    def __init__(self, model_name: str = "ViT-B/16", device: str = None):
        """
        初始化 CLIP 模型管理器

        Args:
            model_name: CLIP模型名稱，默認為"ViT-B/16"
            device: 運行設備，None則自動選擇
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name

        # 設置運行設備
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.preprocess = None

        self._initialize_model()

    def _initialize_model(self):
        """
        初始化CLIP模型
        """
        try:
            self.logger.info(f"Initializing CLIP model ({self.model_name}) on {self.device}")
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.logger.info("Successfully loaded CLIP model")
        except Exception as e:
            self.logger.error(f"Error loading CLIP model: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def encode_image(self, image_input: torch.Tensor) -> torch.Tensor:
        """
        編碼圖像特徵

        Args:
            image_input: 預處理後的圖像張量

        Returns:
            torch.Tensor: 標準化後的圖像特徵
        """
        try:
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                return image_features
        except Exception as e:
            self.logger.error(f"Error encoding image features: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def encode_text_batch(self, text_prompts: List[str], batch_size: int = 128) -> torch.Tensor:
        """
        批量編碼文本特徵，避免CUDA內存問題

        Args:
            text_prompts: 文本提示列表
            batch_size: 批處理大小

        Returns:
            torch.Tensor: 標準化後的文本特徵
        """
        if not text_prompts:
            return None

        try:
            with torch.no_grad():
                features_list = []

                for i in range(0, len(text_prompts), batch_size):
                    batch_prompts = text_prompts[i:i+batch_size]
                    text_tokens = clip.tokenize(batch_prompts).to(self.device)
                    batch_features = self.model.encode_text(text_tokens)
                    batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                    features_list.append(batch_features)

                # 連接所有批次
                if len(features_list) > 1:
                    text_features = torch.cat(features_list, dim=0)
                else:
                    text_features = features_list[0]

                return text_features

        except Exception as e:
            self.logger.error(f"Error encoding text features: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def encode_single_text(self, text_prompts: List[str]) -> torch.Tensor:
        """
        編碼單個文本批次的特徵

        Args:
            text_prompts: 文本提示列表

        Returns:
            torch.Tensor: 標準化後的文本特徵
        """
        try:
            with torch.no_grad():
                text_tokens = clip.tokenize(text_prompts).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                return text_features
        except Exception as e:
            self.logger.error(f"Error encoding single text batch: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def calculate_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> np.ndarray:
        """
        計算圖像和文本特徵之間的相似度

        Args:
            image_features: 圖像特徵張量
            text_features: 文本特徵張量

        Returns:
            np.ndarray: 相似度分數數組
        """
        try:
            with torch.no_grad():
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                similarity = similarity.cpu().numpy() if self.device == "cuda" else similarity.numpy()
                return similarity
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        預處理圖像以供CLIP模型使用

        Args:
            image: PIL圖像或numpy數組

        Returns:
            torch.Tensor: 預處理後的圖像張量
        """
        try:
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    raise ValueError("Unsupported image format. Expected PIL Image or numpy array.")

            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            return image_input

        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def process_image_region(self, image: Union[Image.Image, np.ndarray], box: List[float]) -> torch.Tensor:
        """
        處理圖像的特定區域

        Args:
            image: 原始圖像
            box: 邊界框 [x1, y1, x2, y2]

        Returns:
            torch.Tensor: 區域圖像的特徵
        """
        try:
            # 確保圖像是PIL格式
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    raise ValueError("Unsupported image format. Expected PIL Image or numpy array.")

            # 裁剪區域
            x1, y1, x2, y2 = map(int, box)
            cropped_image = image.crop((x1, y1, x2, y2))

            # 預處理並編碼
            image_input = self.preprocess_image(cropped_image)
            image_features = self.encode_image(image_input)

            return image_features

        except Exception as e:
            self.logger.error(f"Error processing image region: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def batch_process_regions(self, image: Union[Image.Image, np.ndarray],
                             boxes: List[List[float]]) -> torch.Tensor:
        """
        批量處理多個圖像區域

        Args:
            image: 原始圖像
            boxes: 邊界框列表

        Returns:
            torch.Tensor: 所有區域的圖像特徵
        """
        try:
            # ensure PIL format
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    raise ValueError("Unsupported image format. Expected PIL Image or numpy array.")

            if not boxes:
                return torch.empty(0)

            # 裁剪並預處理所有區域
            cropped_inputs = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cropped_image = image.crop((x1, y1, x2, y2))
                processed_image = self.preprocess(cropped_image).unsqueeze(0)
                cropped_inputs.append(processed_image)

            # 批量處理
            batch_tensor = torch.cat(cropped_inputs).to(self.device)
            image_features = self.encode_image(batch_tensor)

            return image_features

        except Exception as e:
            self.logger.error(f"Error batch processing regions: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def is_model_loaded(self) -> bool:
        """
        檢查模型是否已成功載入

        Returns:
            bool: 模型載入狀態
        """
        return self.model is not None and self.preprocess is not None

    def get_device(self) -> str:
        """
        獲取當前設備

        Returns:
            str: 設備名稱
        """
        return self.device

    def get_model_name(self) -> str:
        """
        獲取模型名稱

        Returns:
            str: 模型名稱
        """
        return self.model_name
