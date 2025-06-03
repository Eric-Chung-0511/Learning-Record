
import logging
import traceback
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ZoneEvaluator:
    """
    負責功能區域辨識的可行性評估和物件關聯性計算
    評估是否應該進行區域劃分以及計算物件間的功能關聯性
    """

    def __init__(self):
        """初始化區域評估器"""
        try:
            # 定義物件間的功能關聯性評分表
            # 分數越高表示兩個物件在功能上越相關，更可能出現在同一功能區域
            self.relationship_pairs = {
                # 家具組合關係 - 這些組合通常出現在特定功能區域
                frozenset([56, 60]): 1.0,  # 椅子+桌子 (dining/work area)
                frozenset([57, 62]): 0.9,  # 沙發+電視 (living area)
                frozenset([59, 58]): 0.7,  # 床+植物 (bedroom decor)

                # 工作相關組合 - 工作環境的典型配置
                frozenset([63, 66]): 0.9,  # 筆電+鍵盤 (workspace)
                frozenset([63, 64]): 0.8,  # 筆電+滑鼠 (workspace)
                frozenset([60, 63]): 0.8,  # 桌子+筆電 (workspace)

                # 廚房相關組合 - 廚房設備的常見的物品
                frozenset([68, 72]): 0.9,  # 微波爐+冰箱 (kitchen)
                frozenset([69, 71]): 0.8,  # 烤箱+水槽 (kitchen)

                # 用餐相關組合 - 餐廳或用餐區域的典型物品
                frozenset([60, 40]): 0.8,  # 桌子+酒杯 (dining)
                frozenset([60, 41]): 0.8,  # 桌子+杯子 (dining)
                frozenset([56, 40]): 0.7,  # 椅子+酒杯 (dining)

                # 交通相關組合 - 城市交通的環境
                frozenset([2, 9]): 0.8,   # 汽車+交通燈 (traffic)
                frozenset([0, 9]): 0.7,   # 行人+交通燈 (crosswalk)
            }

            logger.info("ZoneEvaluator initialized with predefined relationship pairs")

        except Exception as e:
            logger.error(f"Failed to initialize ZoneEvaluator: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def evaluate_zone_identification_feasibility(self, detected_objects: List[Dict], scene_type: str) -> bool:
        """
        基於物件關聯性和分布特徵的彈性可行性評估
        決定是否應該進行功能區域劃分

        Args:
            detected_objects: 檢測到的物件列表
            scene_type: 場景類型

        Returns:
            是否適合進行區域識別
        """
        try:
            if len(detected_objects) < 2:
                logger.info("Insufficient objects for zone identification (minimum 2 required)")
                return False

            # 計算不同置信度層級的物件分布
            # 高信心度物件更可靠，用於核心區域判斷
            high_conf_objects = [obj for obj in detected_objects if obj.get("confidence", 0) >= 0.6]
            # 中等置信度物件提供補充資訊
            medium_conf_objects = [obj for obj in detected_objects if obj.get("confidence", 0) >= 0.4]

            # 基礎條件：至少需要一定數量的可信物件才值得進行區域分析
            if len(medium_conf_objects) < 2:
                logger.info("Insufficient medium confidence objects for zone identification")
                return False

            # 評估物件間的功能關聯性，關聯性高的物件更適合劃分功能區域
            functional_relationships = self.calculate_functional_relationships(detected_objects)

            # 評估空間分布多樣性 - 物件分散在多個區域才有劃分的意義
            spatial_diversity = self.calculate_spatial_diversity(detected_objects)

            # 綜合評分機制，用各項指標加權計算最終可行性評分
            feasibility_score = 0

            # 物件數量的貢獻（權重30%）- 更多物件提供更多劃分依據
            object_count_score = min(len(detected_objects) / 5.0, 1.0) * 0.3

            # 信心度質量貢獻（權重25%）- 高置信度物件比例影響可靠性
            confidence_score = len(high_conf_objects) / max(len(detected_objects), 1) * 0.25

            # 功能關聯性貢獻（權重25%）- 有功能關聯的物件更適合劃分區域
            relationship_score = functional_relationships * 0.25

            # 空間多樣性貢獻（權重20%）- 分散的物件才需要區域劃分
            diversity_score = spatial_diversity * 0.20

            feasibility_score = object_count_score + confidence_score + relationship_score + diversity_score

            # 動態閾值：根據場景複雜度調整可行性標準
            complexity_threshold = self.get_complexity_threshold(scene_type)

            is_feasible = feasibility_score >= complexity_threshold

            logger.info(f"Zone identification feasibility: {is_feasible} (score: {feasibility_score:.3f}, threshold: {complexity_threshold:.3f})")
            logger.debug(f"Score breakdown - objects: {object_count_score:.3f}, confidence: {confidence_score:.3f}, relationships: {relationship_score:.3f}, diversity: {diversity_score:.3f}")

            return is_feasible

        except Exception as e:
            logger.error(f"Error evaluating zone identification feasibility: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def calculate_functional_relationships(self, detected_objects: List[Dict]) -> float:
        """
        計算物件間的功能關聯性評分
        基於常見的物件組合模式評估功能相關性

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            功能關聯性評分 (0.0-1.0)
        """
        try:
            detected_class_ids = set(obj.get("class_id") for obj in detected_objects)
            max_possible_score = 0
            actual_score = 0

            # 遍歷所有預定義的關聯性組合，計算實際場景中的關聯性評分
            for pair, score in self.relationship_pairs.items():
                max_possible_score += score
                # 如果檢測到的物件中包含這個關聯組合，累加其評分
                if pair.issubset(detected_class_ids):
                    actual_score += score
                    logger.debug(f"Found functional relationship: {pair} with score {score}")

            # 標準化評分：實際評分除以最大可能評分
            relationship_score = actual_score / max_possible_score if max_possible_score > 0 else 0

            logger.info(f"Functional relationships calculated: {relationship_score:.3f} (found {actual_score:.1f}/{max_possible_score:.1f} possible relationships)")
            return relationship_score

        except Exception as e:
            logger.error(f"Error calculating functional relationships: {str(e)}")
            logger.error(traceback.format_exc())
            return 0

    def calculate_spatial_diversity(self, detected_objects: List[Dict]) -> float:
        """
        計算物件空間分布的多樣性
        評估物件是否分散在不同區域，避免所有物件集中在單一區域

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            空間多樣性評分 (0.0-1.0)
        """
        try:
            # 收集所有物件所在的不同區域
            regions = set(obj.get("region", "center") for obj in detected_objects)
            unique_regions = len(regions)

            # 標準化多樣性評分：假設理想情況是物件分散在2個以上區域
            # 更多區域意味著更高的空間多樣性，更適合進行區域劃分
            diversity_score = min(unique_regions / 2.0, 1.0)

            logger.info(f"Spatial diversity calculated: {diversity_score:.3f} (objects distributed across {unique_regions} regions)")
            return diversity_score

        except Exception as e:
            logger.error(f"Error calculating spatial diversity: {str(e)}")
            logger.error(traceback.format_exc())
            return 0

    def get_complexity_threshold(self, scene_type: str) -> float:
        """
        根據場景類型返回適當的複雜度閾值
        平衡不同場景的區域劃分需求

        Args:
            scene_type: 場景類型

        Returns:
            複雜度閾值 (0.0-1.0)
        """
        try:
            # 較簡單場景需要較高分數才進行區域劃分
            # 這些場景通常功能較為單純，不太需要細分
            simple_scenes = ["bedroom", "bathroom", "closet"]

            # 較複雜場景可以較低分數進行區域劃分
            # 這些場景通常有多種功能，適合劃分不同區域
            complex_scenes = ["living_room", "kitchen", "office_workspace", "dining_area"]

            if scene_type in simple_scenes:
                threshold = 0.65  # 較高閾值，避免過度細分
                logger.debug(f"Using high threshold {threshold} for simple scene: {scene_type}")
            elif scene_type in complex_scenes:
                threshold = 0.45  # 較低閾值，允許合理劃分
                logger.debug(f"Using low threshold {threshold} for complex scene: {scene_type}")
            else:
                threshold = 0.55  # 中等閾值，平衡策略
                logger.debug(f"Using medium threshold {threshold} for scene: {scene_type}")

            return threshold

        except Exception as e:
            logger.error(f"Error getting complexity threshold for scene '{scene_type}': {str(e)}")
            logger.error(traceback.format_exc())
            return 0.55  # 預設中等閾值

    def analyze_object_clustering(self, detected_objects: List[Dict]) -> Dict:
        """
        分析物件的聚集模式
        識別物件是否形成明顯的聚集群組，這有助於功能區域的劃分

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            包含聚集分析結果的字典
        """
        try:
            clustering_result = {
                "has_clusters": False,
                "cluster_count": 0,
                "cluster_regions": [],
                "clustering_score": 0.0
            }

            if len(detected_objects) < 3:
                logger.info("Insufficient objects for clustering analysis")
                return clustering_result

            # 統計每個區域的物件數量
            region_counts = {}
            for obj in detected_objects:
                region = obj.get("region", "unknown")
                region_counts[region] = region_counts.get(region, 0) + 1

            # 找出有顯著物件聚集的區域（物件數量 >= 2）
            significant_regions = [region for region, count in region_counts.items() if count >= 2]

            # 計算聚集：聚集區域數量與總區域數量的比例
            total_regions_with_objects = len([count for count in region_counts.values() if count > 0])
            clustering_score = len(significant_regions) / max(total_regions_with_objects, 1)

            clustering_result.update({
                "has_clusters": len(significant_regions) >= 2,
                "cluster_count": len(significant_regions),
                "cluster_regions": significant_regions,
                "clustering_score": clustering_score
            })

            logger.info(f"Object clustering analysis: {len(significant_regions)} clusters found in regions {significant_regions}")
            return clustering_result

        except Exception as e:
            logger.error(f"Error analyzing object clustering: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "has_clusters": False,
                "cluster_count": 0,
                "cluster_regions": [],
                "clustering_score": 0.0
            }
