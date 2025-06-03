
import logging
import traceback
from typing import Dict, Any, Tuple, Optional, Union
from PIL import Image
import numpy as np

class ResultCacheManager:
    """
    專門處理結果快取和性能優化，包括快取策略管理、快取大小控制和快取命中率優化
    """

    def __init__(self, cache_max_size: int = 100):
        """
        初始化結果快取管理器

        Args:
            cache_max_size: 最大快取項目數
        """
        self.logger = logging.getLogger(__name__)

        # 初始化結果快取
        self.results_cache = {}  # 使用圖像hash作為鍵
        self.cache_max_size = cache_max_size  # 最大快取項目數

    def generate_cache_key(self, image_hash: int, additional_params: Tuple) -> Tuple:
        """
        生成快取鍵

        Args:
            image_hash
            additional_params: 附加參數元組

        Returns:
            Tuple: 快取鍵
        """
        try:
            return (image_hash, additional_params)
        except Exception as e:
            self.logger.error(f"Error generating cache key: {e}")
            self.logger.error(traceback.format_exc())
            return (0, additional_params)

    def get_region_cache_key(self, image_hash: int, box: Tuple[float, ...],
                           detection_type: str) -> Tuple:
        """
        生成區域分析的快取鍵

        Args:
            image_hash
            box: 邊界框
            detection_type: 檢測類型

        Returns:
            Tuple: 區域快取鍵
        """
        try:
            return self.generate_cache_key(image_hash, (tuple(box), detection_type))
        except Exception as e:
            self.logger.error(f"Error generating region cache key: {e}")
            self.logger.error(traceback.format_exc())
            return (0, (tuple(box), detection_type))

    def get_image_cache_key(self, image_hash: int, analysis_type: str,
                          detailed_analysis: bool = False) -> Tuple:
        """
        生成整張圖像分析的快取鍵

        Args:
            image_hash: 圖像哈希值
            analysis_type: 分析類型
            detailed_analysis: 是否詳細分析

        Returns:
            Tuple: 圖像快取鍵
        """
        try:
            return self.generate_cache_key(image_hash, (analysis_type, detailed_analysis))
        except Exception as e:
            self.logger.error(f"Error generating image cache key: {e}")
            self.logger.error(traceback.format_exc())
            return (0, (analysis_type, detailed_analysis))

    def get_cached_result(self, cache_key: Tuple) -> Optional[Dict[str, Any]]:
        """
        獲取快取結果

        Args:
            cache_key: 快取鍵

        Returns:
            Optional[Dict[str, Any]]: 快取結果，如果不存在則返回None
        """
        try:
            return self.results_cache.get(cache_key)
        except Exception as e:
            self.logger.error(f"Error getting cached result: {e}")
            self.logger.error(traceback.format_exc())
            return None

    def set_cached_result(self, cache_key: Tuple, result: Dict[str, Any]):
        """
        設置快取結果

        Args:
            cache_key: 快取鍵
            result: 要快取的結果
        """
        try:
            self.results_cache[cache_key] = result
            self.manage_cache_size()
        except Exception as e:
            self.logger.error(f"Error setting cached result: {e}")
            self.logger.error(traceback.format_exc())

    def manage_cache_size(self):
        """
        管理結果快取大小
        """
        try:
            if len(self.results_cache) > self.cache_max_size:
                oldest_key = next(iter(self.results_cache))
                del self.results_cache[oldest_key]
        except Exception as e:
            self.logger.error(f"Error managing cache size: {e}")
            self.logger.error(traceback.format_exc())

    def clear_cache(self):
        """
        清空快取
        """
        try:
            self.results_cache.clear()
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            self.logger.error(traceback.format_exc())

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        獲取快取統計信息

        Returns:
            Dict[str, Any]: 快取統計信息
        """
        try:
            return {
                "cache_size": len(self.results_cache),
                "max_cache_size": self.cache_max_size,
                "cache_usage_ratio": len(self.results_cache) / self.cache_max_size if self.cache_max_size > 0 else 0
            }
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            self.logger.error(traceback.format_exc())
            return {
                "cache_size": 0,
                "max_cache_size": self.cache_max_size,
                "cache_usage_ratio": 0
            }

    def set_max_cache_size(self, max_size: int):
        """
        設置最大快取大小

        Args:
            max_size: 新的最大快取大小
        """
        try:
            self.cache_max_size = max(1, max_size)
            self.manage_cache_size()
            self.logger.info(f"Max cache size set to {self.cache_max_size}")
        except Exception as e:
            self.logger.error(f"Error setting max cache size: {e}")
            self.logger.error(traceback.format_exc())

    def remove_cached_result(self, cache_key: Tuple) -> bool:
        """
        移除特定的快取結果

        Args:
            cache_key: 快取鍵

        Returns:
            bool: 是否成功移除
        """
        try:
            if cache_key in self.results_cache:
                del self.results_cache[cache_key]
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error removing cached result: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def is_cache_enabled(self) -> bool:
        """
        檢查快取是否啟用

        Returns:
            bool: 快取啟用狀態
        """
        return self.cache_max_size > 0

    def get_cache_keys(self) -> list:
        """
        獲取所有快取鍵

        Returns:
            list: 快取鍵列表
        """
        try:
            return list(self.results_cache.keys())
        except Exception as e:
            self.logger.error(f"Error getting cache keys: {e}")
            self.logger.error(traceback.format_exc())
            return []

    def has_cached_result(self, cache_key: Tuple) -> bool:
        """
        檢查是否存在快取結果

        Args:
            cache_key: 快取鍵

        Returns:
            bool: 是否存在快取結果
        """
        try:
            return cache_key in self.results_cache
        except Exception as e:
            self.logger.error(f"Error checking cached result: {e}")
            self.logger.error(traceback.format_exc())
            return False
