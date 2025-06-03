
import logging
import traceback
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class RegionAnalyzer:
    """
    負責處理圖像區域劃分和基礎空間分析功能
    專注於3x3網格的區域劃分、物件分布分析和空間多樣性計算
    """

    def __init__(self):
        """初始化區域分析器，定義3x3網格區域"""
        try:
            # 定義圖像的3x3網格區域
            self.regions = {
                "top_left": (0, 0, 1/3, 1/3),
                "top_center": (1/3, 0, 2/3, 1/3),
                "top_right": (2/3, 0, 1, 1/3),
                "middle_left": (0, 1/3, 1/3, 2/3),
                "middle_center": (1/3, 1/3, 2/3, 2/3),
                "middle_right": (2/3, 1/3, 1, 2/3),
                "bottom_left": (0, 2/3, 1/3, 1),
                "bottom_center": (1/3, 2/3, 2/3, 1),
                "bottom_right": (2/3, 2/3, 1, 1)
            }
            logger.info("RegionAnalyzer initialized successfully with 3x3 grid regions")
        except Exception as e:
            logger.error(f"Failed to initialize RegionAnalyzer: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def determine_region(self, x: float, y: float) -> str:
        """
        判斷點位於哪個區域

        Args:
            x: 標準化x座標 (0-1)
            y: 標準化y座標 (0-1)

        Returns:
            區域名稱
        """
        try:
            for region_name, (x1, y1, x2, y2) in self.regions.items():
                if x1 <= x < x2 and y1 <= y < y2:
                    return region_name

            logger.warning(f"Point ({x}, {y}) does not fall into any defined region")
            return "unknown"

        except Exception as e:
            logger.error(f"Error determining region for point ({x}, {y}): {str(e)}")
            logger.error(traceback.format_exc())
            return "unknown"

    def get_spatial_description_phrase(self, region: str) -> str:
        """
        將region ID轉換為完整的空間描述短語，包含適當的介詞結構

        Args:
            region: 區域標識符（如 "middle_center", "top_left"）

        Returns:
            str: 完整的空間描述短語，空值時返回空字串
        """
        try:
            # 處理空值或無效輸入
            if not region or region.strip() == "" or region == "unknown":
                return "within the visible area"

            # 清理region格式，移除底線
            clean_region = region.replace('_', ' ').strip().lower()

            # 根據區域位置生成自然語言描述
            region_mappings = {
                "top left": "in the upper left area",
                "top center": "in the upper area",
                "top right": "in the upper right area",
                "middle left": "on the left side",
                "middle center": "in the center",
                "center": "in the center",
                "middle right": "on the right side",
                "bottom left": "in the lower left area",
                "bottom center": "in the lower area",
                "bottom right": "in the lower right area"
            }

            # 直接映射匹配
            if clean_region in region_mappings:
                return region_mappings[clean_region]

            # 模糊匹配方位的處理
            if "top" in clean_region and "left" in clean_region:
                return "in the upper left area"
            elif "top" in clean_region and "right" in clean_region:
                return "in the upper right area"
            elif "bottom" in clean_region and "left" in clean_region:
                return "in the lower left area"
            elif "bottom" in clean_region and "right" in clean_region:
                return "in the lower right area"
            elif "top" in clean_region:
                return "in the upper area"
            elif "bottom" in clean_region:
                return "in the lower area"
            elif "left" in clean_region:
                return "on the left side"
            elif "right" in clean_region:
                return "on the right side"
            elif "center" in clean_region or "middle" in clean_region:
                return "in the center"
            else:
                # 對於無法辨識的區域，返回通用描述
                return f"in the {clean_region} area"

        except Exception as e:
            logger.warning(f"Error generating spatial description for region '{region}': {str(e)}")
            return ""

    def get_contextual_spatial_description(self, region: str, object_type: str = "") -> str:
        """
        根據物件類型提供更具情境的空間描述

        Args:
            region: 區域標識符
            object_type: 物件類型，用於優化描述語境

        Returns:
            str: 情境化的空間描述短語
        """
        try:
            # 獲取基礎空間描述
            base_description = self.get_spatial_description_phrase(region)

            if not base_description:
                return ""

            # 根據物件類型調整描述語境
            if object_type:
                object_type_lower = object_type.lower()

                # 對於辨識到人相關，用更自然的位置描述
                if "person" in object_type_lower or "people" in object_type_lower:
                    if "center" in base_description:
                        return "in the central area"
                    elif "upper" in base_description:
                        return "in the background"
                    elif "lower" in base_description:
                        return "in the foreground"

                # 對於車輛，強調道路位置
                elif any(vehicle in object_type_lower for vehicle in ["car", "vehicle", "truck", "bus"]):
                    if "left" in base_description:
                        return "on the left side of the scene"
                    elif "right" in base_description:
                        return "on the right side of the scene"
                    elif "center" in base_description:
                        return "in the central area"

                # 對於交通設施，使用更具體的位置描述
                elif "traffic" in object_type_lower:
                    if "upper" in base_description:
                        return "positioned in the upper portion"
                    elif "center" in base_description:
                        return "centrally positioned"
                    else:
                        return base_description.replace("in the", "positioned in the")

            return base_description

        except Exception as e:
            logger.warning(f"Error generating contextual spatial description: {str(e)}")
            return self.get_spatial_description_phrase(region)


    def validate_region_input(self, region: str) -> bool:
        """
        驗證region輸入是否有效

        Args:
            region: 待驗證的區域標識符

        Returns:
            bool: 是否為有效的region
        """
        try:
            if not region or region.strip() == "":
                return False

            # 清理並檢查是否為已知區域
            clean_region = region.replace('_', ' ').strip().lower()

            known_regions = [
                "top left", "top center", "top right",
                "middle left", "middle center", "middle right",
                "bottom left", "bottom center", "bottom right",
                "center", "unknown"
            ]

            # 直接匹配或包含關鍵詞匹配
            if clean_region in known_regions:
                return True

            # 檢查是否包含有效的位置關鍵詞組合
            position_keywords = ["top", "bottom", "left", "right", "center", "middle"]
            has_valid_keyword = any(keyword in clean_region for keyword in position_keywords)

            return has_valid_keyword

        except Exception as e:
            logger.warning(f"Error validating region input '{region}': {str(e)}")
            return False

    def get_enhanced_directional_description(self, region: str) -> str:
        """
        增強版的方位描述生成，提供更豐富的方位資訊
        擴展原有的get_directional_description方法功能

        Args:
            region: 區域名稱

        Returns:
            str: 增強的方位描述字串
        """
        try:
            if not self.validate_region_input(region):
                return "central"

            region_lower = region.replace('_', ' ').strip().lower()

            # 用比較準確的方位映射
            direction_mappings = {
                "top left": "northwest",
                "top center": "north",
                "top right": "northeast",
                "middle left": "west",
                "middle center": "central",
                "center": "central",
                "middle right": "east",
                "bottom left": "southwest",
                "bottom center": "south",
                "bottom right": "southeast"
            }

            if region_lower in direction_mappings:
                return direction_mappings[region_lower]

            # 模糊匹配邏輯保持與原方法相同
            if "top" in region_lower and "left" in region_lower:
                return "northwest"
            elif "top" in region_lower and "right" in region_lower:
                return "northeast"
            elif "bottom" in region_lower and "left" in region_lower:
                return "southwest"
            elif "bottom" in region_lower and "right" in region_lower:
                return "southeast"
            elif "top" in region_lower:
                return "north"
            elif "bottom" in region_lower:
                return "south"
            elif "left" in region_lower:
                return "west"
            elif "right" in region_lower:
                return "east"
            else:
                return "central"

        except Exception as e:
            logger.error(f"Error getting enhanced directional description for region '{region}': {str(e)}")
            return "central"

    def analyze_regions(self, detected_objects: List[Dict]) -> Dict:
        """
        分析物件在各區域的分布情況

        Args:
            detected_objects: 包含位置資訊的檢測物件列表

        Returns:
            包含區域分析結果的字典
        """
        try:
            if not detected_objects:
                logger.warning("No detected objects provided for region analysis")
                return {
                    "counts": {region: 0 for region in self.regions.keys()},
                    "main_focus": [],
                    "objects_by_region": {region: [] for region in self.regions.keys()}
                }

            # 計算每個區域的物件數量
            region_counts = {region: 0 for region in self.regions.keys()}
            region_objects = {region: [] for region in self.regions.keys()}

            for obj in detected_objects:
                try:
                    region = obj.get("region", "unknown")
                    if region in region_counts:
                        region_counts[region] += 1
                        region_objects[region].append({
                            "class_id": obj.get("class_id"),
                            "class_name": obj.get("class_name")
                        })
                    else:
                        logger.warning(f"Unknown region '{region}' found in object")

                except Exception as e:
                    logger.error(f"Error processing object in region analysis: {str(e)}")
                    continue

            # 確定主要焦點區域（按物件數量排序的前1-2個區域）
            sorted_regions = sorted(region_counts.items(), key=lambda x: x[1], reverse=True)
            main_regions = [region for region, count in sorted_regions if count > 0][:2]

            result = {
                "counts": region_counts,
                "main_focus": main_regions,
                "objects_by_region": region_objects
            }

            logger.info(f"Region analysis completed. Main focus areas: {main_regions}")
            return result

        except Exception as e:
            logger.error(f"Error in region analysis: {str(e)}")
            logger.error(traceback.format_exc())
            # 返回空的結果結構而不是拋出異常
            return {
                "counts": {region: 0 for region in self.regions.keys()},
                "main_focus": [],
                "objects_by_region": {region: [] for region in self.regions.keys()}
            }

    def create_distribution_map(self, detected_objects: List[Dict]) -> Dict:
        """
        創建物件在各區域分布的詳細地圖，用於空間分析

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            包含各區域分布詳情的字典
        """
        try:
            if not detected_objects:
                logger.warning("No detected objects provided for distribution map creation")
                return self._get_empty_distribution_map()

            distribution = {}

            # 初始化所有區域
            for region in self.regions.keys():
                distribution[region] = {
                    "total": 0,
                    "objects": {},
                    "density": 0
                }

            # 填充分布資料
            for obj in detected_objects:
                try:
                    region = obj.get("region", "unknown")
                    class_id = obj.get("class_id")
                    class_name = obj.get("class_name", "unknown")

                    if region not in distribution:
                        logger.warning(f"Unknown region '{region}' found, skipping object")
                        continue

                    distribution[region]["total"] += 1

                    if class_id not in distribution[region]["objects"]:
                        distribution[region]["objects"][class_id] = {
                            "name": class_name,
                            "count": 0,
                            "positions": []
                        }

                    distribution[region]["objects"][class_id]["count"] += 1

                    # 儲存位置資訊用於空間關係分析
                    normalized_center = obj.get("normalized_center")
                    if normalized_center:
                        distribution[region]["objects"][class_id]["positions"].append(normalized_center)

                except Exception as e:
                    logger.error(f"Error processing object in distribution map: {str(e)}")
                    continue

            # 計算每個區域的物件密度
            for region, data in distribution.items():
                # 假設所有區域在網格中大小相等
                data["density"] = data["total"] / 1

            logger.info("Distribution map created successfully")
            return distribution

        except Exception as e:
            logger.error(f"Error creating distribution map: {str(e)}")
            logger.error(traceback.format_exc())
            return self._get_empty_distribution_map()

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
            if not detected_objects:
                logger.warning("No detected objects provided for spatial diversity calculation")
                return 0.0

            regions = set()
            for obj in detected_objects:
                region = obj.get("region", "center")
                regions.add(region)

            unique_regions = len(regions)
            diversity_score = min(unique_regions / 2.0, 1.0)

            logger.info(f"Spatial diversity calculated: {diversity_score:.3f} (regions: {unique_regions})")
            return diversity_score

        except Exception as e:
            logger.error(f"Error calculating spatial diversity: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0

    def get_directional_description(self, region: str) -> str:
        """
        將區域名稱轉換為方位描述（東西南北）

        Args:
            region: 區域名稱

        Returns:
            方位描述字串
        """
        try:
            region_lower = region.lower()

            if "top" in region_lower and "left" in region_lower:
                return "northwest"
            elif "top" in region_lower and "right" in region_lower:
                return "northeast"
            elif "bottom" in region_lower and "left" in region_lower:
                return "southwest"
            elif "bottom" in region_lower and "right" in region_lower:
                return "southeast"
            elif "top" in region_lower:
                return "north"
            elif "bottom" in region_lower:
                return "south"
            elif "left" in region_lower:
                return "west"
            elif "right" in region_lower:
                return "east"
            else:
                return "central"

        except Exception as e:
            logger.error(f"Error getting directional description for region '{region}': {str(e)}")
            return "central"

    def _get_empty_distribution_map(self) -> Dict:
        """
        返回空的分布地圖結構

        Returns:
            空的分布地圖字典
        """
        distribution = {}
        for region in self.regions.keys():
            distribution[region] = {
                "total": 0,
                "objects": {},
                "density": 0
            }
        return distribution
