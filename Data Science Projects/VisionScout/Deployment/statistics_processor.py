import logging
from typing import Dict, List, Optional, Any

class StatisticsProcessor:
    """
    統計分析處理器 - 負責複雜的物件統計分析和數據轉換

    此類別專門處理物件統計信息的深度分析、Places365信息處理，
    以及基於統計數據生成替換內容的複雜邏輯。
    """

    def __init__(self):
        """初始化統計分析處理器"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("StatisticsProcessor initialized successfully")

    def generate_statistics_replacements(self, object_statistics: Optional[Dict]) -> Dict[str, str]:
        """
        基於物體統計信息生成模板替換內容

        Args:
            object_statistics: 物體統計信息

        Returns:
            Dict[str, str]: 統計信息基礎的替換內容
        """
        replacements = {}

        if not object_statistics:
            return replacements

        try:
            # 處理植物元素
            if "potted plant" in object_statistics:
                count = object_statistics["potted plant"]["count"]
                if count == 1:
                    replacements["plant_elements"] = "a potted plant"
                elif count <= 3:
                    replacements["plant_elements"] = f"{count} potted plants"
                else:
                    replacements["plant_elements"] = f"multiple potted plants ({count} total)"

            # 處理座位(椅子)相關
            if "chair" in object_statistics:
                count = object_statistics["chair"]["count"]

                # 使用統一的數字轉換邏輯
                number_words = {
                    1: "one", 2: "two", 3: "three", 4: "four",
                    5: "five", 6: "six", 7: "seven", 8: "eight",
                    9: "nine", 10: "ten", 11: "eleven", 12: "twelve"
                }

                if count == 1:
                    replacements["seating"] = "a chair"
                    replacements["furniture"] = "a chair"
                elif count in number_words:
                    word_count = number_words[count]
                    replacements["seating"] = f"{word_count} chairs"
                    replacements["furniture"] = f"{word_count} chairs"
                elif count <= 20:
                    replacements["seating"] = f"several chairs"
                    replacements["furniture"] = f"several chairs"
                else:
                    replacements["seating"] = f"numerous chairs ({count} total)"
                    replacements["furniture"] = f"numerous chairs"

            # 處理混合家具情況（當存在多種家具類型時）
            furniture_items = []
            furniture_counts = []

            # 收集所有家具類型的統計
            for furniture_type in ["chair", "dining table", "couch", "bed"]:
                if furniture_type in object_statistics:
                    count = object_statistics[furniture_type]["count"]
                    if count > 0:
                        furniture_items.append(furniture_type)
                        furniture_counts.append(count)

            # 如果只有椅子,那就用上面的方式
            # 如果有多種家具類型，生成組合描述
            if len(furniture_items) > 1 and "furniture" not in replacements:
                main_furniture = furniture_items[0]  # 數量最多的家具類型
                main_count = furniture_counts[0]

                if main_furniture == "chair":
                    number_words = ["", "one", "two", "three", "four", "five", "six"]
                    if main_count <= 6:
                        replacements["furniture"] = f"{number_words[main_count]} chairs and other furniture"
                    else:
                        replacements["furniture"] = "multiple chairs and other furniture"

            # 處理人員
            if "person" in object_statistics:
                count = object_statistics["person"]["count"]
                if count == 1:
                    replacements["people_and_vehicles"] = "a person"
                    replacements["pedestrian_flow"] = "an individual walking"
                elif count <= 5:
                    replacements["people_and_vehicles"] = f"{count} people"
                    replacements["pedestrian_flow"] = f"{count} people walking"
                else:
                    replacements["people_and_vehicles"] = f"many people ({count} individuals)"
                    replacements["pedestrian_flow"] = f"a crowd of {count} people"

            # 處理桌子設置
            if "dining table" in object_statistics:
                count = object_statistics["dining table"]["count"]
                if count == 1:
                    replacements["table_setup"] = "a dining table"
                    replacements["table_description"] = "a dining surface"
                else:
                    replacements["table_setup"] = f"{count} dining tables"
                    replacements["table_description"] = f"{count} dining surfaces"

            self.logger.debug(f"Generated {len(replacements)} statistics-based replacements")

        except Exception as e:
            self.logger.warning(f"Error generating statistics replacements: {str(e)}")

        return replacements

    def generate_places365_replacements(self, places365_info: Optional[Dict]) -> Dict[str, str]:
        """
        基於Places365信息生成模板替換內容

        Args:
            places365_info: Places365場景分類信息

        Returns:
            Dict[str, str]: Places365基礎的替換內容
        """
        replacements = {}

        if not places365_info or places365_info.get('confidence', 0) <= 0.35:
            replacements["places365_context"] = ""
            replacements["places365_atmosphere"] = ""
            return replacements

        try:
            scene_label = places365_info.get('scene_label', '').replace('_', ' ')
            attributes = places365_info.get('attributes', [])

            # 生成場景上下文
            if scene_label:
                replacements["places365_context"] = f"characteristic of a {scene_label}"
            else:
                replacements["places365_context"] = ""

            # 生成氛圍描述
            if 'natural_lighting' in attributes:
                replacements["places365_atmosphere"] = "with natural illumination"
            elif 'artificial_lighting' in attributes:
                replacements["places365_atmosphere"] = "under artificial lighting"
            else:
                replacements["places365_atmosphere"] = ""

            self.logger.debug("Generated Places365-based replacements")

        except Exception as e:
            self.logger.warning(f"Error generating Places365 replacements: {str(e)}")
            replacements["places365_context"] = ""
            replacements["places365_atmosphere"] = ""

        return replacements

    def analyze_scene_composition(self, detected_objects: List[Dict]) -> Dict:
        """
        分析場景組成以確定模板複雜度

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            Dict: 場景組成統計信息
        """
        try:
            total_objects = len(detected_objects)

            # 統計不同類型的物件
            object_categories = {}
            for obj in detected_objects:
                class_name = obj.get("class_name", "unknown")
                object_categories[class_name] = object_categories.get(class_name, 0) + 1

            # 計算場景多樣性
            unique_categories = len(object_categories)

            return {
                "total_objects": total_objects,
                "unique_categories": unique_categories,
                "category_distribution": object_categories,
                "complexity_score": min(total_objects * 0.3 + unique_categories * 0.7, 10)
            }

        except Exception as e:
            self.logger.warning(f"Error analyzing scene composition: {str(e)}")
            return {"total_objects": 0, "unique_categories": 0, "complexity_score": 0}

    def generate_zone_descriptions(self, zone_data: Dict[str, Any], section: Dict[str, Any]) -> List[str]:
        """
        生成功能區域描述

        Args:
            zone_data: 區域數據字典
            section: 區域配置信息

        Returns:
            List[str]: 區域描述列表
        """
        try:
            descriptions = []

            if not zone_data:
                return descriptions

            # 直接處理區域資料（zone_data 本身就是區域字典）
            sorted_zones = sorted(zone_data.items(),
                                key=lambda x: len(x[1].get("objects", [])),
                                reverse=True)

            for zone_name, zone_info in sorted_zones:
                description = zone_info.get("description", "")
                objects = zone_info.get("objects", [])

                if objects:
                    # 使用現有描述或生成基於物件的描述
                    if description and not any(tech in description.lower() for tech in ['zone', 'area', 'region']):
                        zone_desc = description
                    else:
                        # 生成更自然的區域描述
                        clean_zone_name = zone_name.replace('_', ' ').replace(' area', '').replace(' zone', '')
                        object_list = ', '.join(objects[:3])

                        if 'crossing' in zone_name or 'pedestrian' in zone_name:
                            zone_desc = f"In the central crossing area, there are {object_list}."
                        elif 'vehicle' in zone_name or 'traffic' in zone_name:
                            zone_desc = f"The vehicle movement area includes {object_list}."
                        elif 'control' in zone_name:
                            zone_desc = f"Traffic control elements include {object_list}."
                        else:
                            zone_desc = f"The {clean_zone_name} contains {object_list}."

                        if len(objects) > 3:
                            zone_desc += f" Along with {len(objects) - 3} additional elements."

                    descriptions.append(zone_desc)

            return descriptions

        except Exception as e:
            self.logger.error(f"Error generating zone descriptions: {str(e)}")
            return []

    def generate_object_summary(self, object_data: List[Dict], section: Dict[str, Any]) -> str:
        """
        生成物件摘要描述

        Args:
            object_data: 物件數據列表
            section: 摘要配置信息

        Returns:
            str: 物件摘要描述
        """
        try:
            if not object_data:
                return ""

            # 統計物件類型並計算重要性
            object_stats = {}
            for obj in object_data:
                class_name = obj.get("class_name", "unknown")
                confidence = obj.get("confidence", 0.5)

                if class_name not in object_stats:
                    object_stats[class_name] = {"count": 0, "total_confidence": 0}

                object_stats[class_name]["count"] += 1
                object_stats[class_name]["total_confidence"] += confidence

            # 按重要性排序（結合數量和置信度）
            sorted_objects = []
            for class_name, stats in object_stats.items():
                count = stats["count"]
                avg_confidence = stats["total_confidence"] / count
                importance = count * 0.6 + avg_confidence * 0.4
                sorted_objects.append((class_name, count, importance))

            sorted_objects.sort(key=lambda x: x[2], reverse=True)

            # 生成自然語言描述
            descriptions = []
            for class_name, count, _ in sorted_objects[:5]:
                clean_name = class_name.replace('_', ' ')
                if count == 1:
                    article = "an" if clean_name[0].lower() in 'aeiou' else "a"
                    descriptions.append(f"{article} {clean_name}")
                else:
                    descriptions.append(f"{count} {clean_name}s")

            if len(descriptions) == 1:
                return f"The scene features {descriptions[0]}."
            elif len(descriptions) == 2:
                return f"The scene features {descriptions[0]} and {descriptions[1]}."
            else:
                main_items = ", ".join(descriptions[:-1])
                return f"The scene features {main_items}, and {descriptions[-1]}."

        except Exception as e:
            self.logger.error(f"Error generating object summary: {str(e)}")
            return ""

    def generate_conclusion(self, template: Dict[str, Any], zone_data: Dict[str, Any],
                           object_data: List[Dict]) -> str:
        """
        生成結論描述

        Args:
            template: 模板配置信息
            zone_data: 區域數據
            object_data: 物件數據

        Returns:
            str: 結論描述
        """
        try:
            scene_type = template.get("scene_type", "general")
            zones_count = len(zone_data)
            objects_count = len(object_data)

            if scene_type == "indoor":
                conclusion = f"This indoor environment demonstrates clear functional organization with {zones_count} distinct areas and {objects_count} identified objects."
            elif scene_type == "outdoor":
                conclusion = f"This outdoor scene shows dynamic activity patterns across {zones_count} functional zones with {objects_count} detected elements."
            else:
                conclusion = f"The scene analysis reveals {zones_count} functional areas containing {objects_count} identifiable objects."

            return conclusion

        except Exception as e:
            self.logger.error(f"Error generating conclusion: {str(e)}")
            return ""
