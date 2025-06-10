import logging
import traceback
from typing import Dict, List, Tuple, Optional, Any

from scene_type import SCENE_TYPES

class SceneScoringEngine:
    """
    負責場景評分相關的所有計算邏輯，包括基於 YOLO 檢測的場景評分、
    多種場景分數融合，以及最終場景類型的確定。
    這邊會有YOLO, CLIP, Places365混合運用的分數計算
    """

    # 日常場景，用於特殊評分
    EVERYDAY_SCENE_TYPE_KEYS = [
        "general_indoor_space", "generic_street_view",
        "desk_area_workspace", "outdoor_gathering_spot",
        "kitchen_counter_or_utility_area"
    ]

    def __init__(self, scene_types: Dict[str, Any], enable_landmark: bool = True):
        """
        初始化場景評分引擎。

        Args:
            scene_types: 場景類型定義字典
            enable_landmark: 是否啟用地標檢測功能
        """
        self.logger = logging.getLogger(__name__)
        self.scene_types = scene_types
        self.enable_landmark = enable_landmark

    def compute_scene_scores(self, detected_objects: List[Dict],
                           spatial_analysis_results: Optional[Dict] = None) -> Dict[str, float]:
        """
        基於檢測到的物體計算各場景類型的置信度分數。
        增強了對日常場景的評分能力，並考慮物體豐富度和空間聚合性。

        Args:
            detected_objects: 檢測到的物體列表，包含物體詳細資訊
            spatial_analysis_results: 空間分析器的輸出結果，特別是 'objects_by_region' 部分

        Returns:
            場景類型到置信度分數的映射字典
        """
        scene_scores = {}
        if not detected_objects:
            for scene_type_key in self.scene_types:
                scene_scores[scene_type_key] = 0.0
            return scene_scores

        # 準備檢測物體的數據
        detected_class_ids_all = [obj["class_id"] for obj in detected_objects]
        detected_classes_set_all = set(detected_class_ids_all)
        class_counts_all = {}
        for obj in detected_objects:
            class_id = obj["class_id"]
            class_counts_all[class_id] = class_counts_all.get(class_id, 0) + 1

        # 評估 scene_types 中定義的每個場景類型
        for scene_type, scene_def in self.scene_types.items():
            required_obj_ids_defined = set(scene_def.get("required_objects", []))
            optional_obj_ids_defined = set(scene_def.get("optional_objects", []))
            min_required_matches_needed = scene_def.get("minimum_required", 0)

            # 確定哪些實際檢測到的物體與此場景類型相關
            # 這些列表將存儲實際檢測到的物體字典，而不僅僅是 class_ids
            actual_required_objects_found_list = []
            for req_id in required_obj_ids_defined:
                if req_id in detected_classes_set_all:
                    # 找到此必需物體的第一個實例添加到列表中（用於後續的聚合性檢查）
                    for dobj in detected_objects:
                        if dobj['class_id'] == req_id:
                            actual_required_objects_found_list.append(dobj)
                            break

            num_required_matches_found = len(actual_required_objects_found_list)

            actual_optional_objects_found_list = []
            for opt_id in optional_obj_ids_defined:
                if opt_id in detected_classes_set_all:
                    for dobj in detected_objects:
                        if dobj['class_id'] == opt_id:
                            actual_optional_objects_found_list.append(dobj)
                            break

            num_optional_matches_found = len(actual_optional_objects_found_list)

            # 初始分數計算權重
            # 基礎分數：55% 來自必需物體，25% 來自可選物體，10% 豐富度，10% 聚合性（最大值）
            required_weight = 0.55
            optional_weight = 0.25
            richness_bonus_max = 0.10
            cohesion_bonus_max = 0.10  # _get_object_spatial_cohesion_score 的最大獎勵是 0.1

            current_scene_score = 0.0
            objects_to_check_for_cohesion = []  # 用於空間聚合性評分

            # 檢查 minimum_required 條件並計算基礎分數
            if num_required_matches_found >= min_required_matches_needed:
                if len(required_obj_ids_defined) > 0:
                    required_ratio = num_required_matches_found / len(required_obj_ids_defined)
                else:  # 沒有定義必需物體，但 min_required_matches_needed 可能為 0
                    required_ratio = 1.0 if min_required_matches_needed == 0 else 0.0

                current_scene_score = required_ratio * required_weight
                objects_to_check_for_cohesion.extend(actual_required_objects_found_list)

                # 從可選物體添加分數
                if len(optional_obj_ids_defined) > 0:
                    optional_ratio = num_optional_matches_found / len(optional_obj_ids_defined)
                    current_scene_score += optional_ratio * optional_weight
                objects_to_check_for_cohesion.extend(actual_optional_objects_found_list)

            # 日常場景的靈活處理，如果嚴格的 minimum_required（基於 'required_objects'）未滿足
            elif scene_type in self.EVERYDAY_SCENE_TYPE_KEYS:
                # 如果日常場景有許多可選項目，它仍可能是一個弱候選
                # 檢查是否存在相當比例的 'optional_objects'
                if (len(optional_obj_ids_defined) > 0 and
                    (num_optional_matches_found / len(optional_obj_ids_defined)) >= 0.25):  # 例如，至少 25% 的典型可選項目
                    # 對這些類型的基礎分數更多地基於可選物體的滿足度
                    current_scene_score = (num_optional_matches_found / len(optional_obj_ids_defined)) * (required_weight + optional_weight * 0.5)  # 給予一些基礎分數
                    objects_to_check_for_cohesion.extend(actual_optional_objects_found_list)
                else:
                    scene_scores[scene_type] = 0.0
                    continue  # 跳過此場景類型
            else:  # 對於非日常場景，如果未滿足 minimum_required，分數為 0
                scene_scores[scene_type] = 0.0
                continue

            # 物體豐富度/多樣性的獎勵
            # 考慮找到的與場景定義相關的唯一物體類別
            relevant_defined_class_ids = required_obj_ids_defined.union(optional_obj_ids_defined)
            unique_relevant_detected_classes = relevant_defined_class_ids.intersection(detected_classes_set_all)

            object_richness_score = 0.0
            if len(relevant_defined_class_ids) > 0:
                richness_ratio = len(unique_relevant_detected_classes) / len(relevant_defined_class_ids)
                object_richness_score = min(richness_bonus_max, richness_ratio * 0.15)  # 豐富度最大 10% 獎勵
            current_scene_score += object_richness_score

            # 空間聚合性的獎勵（如果提供了 spatial_analysis_results）
            spatial_cohesion_bonus = 0.0
            if spatial_analysis_results and objects_to_check_for_cohesion:
                spatial_cohesion_bonus = self._get_object_spatial_cohesion_score(
                    objects_to_check_for_cohesion,  # 傳遞實際檢測到的物體字典列表
                    spatial_analysis_results
                )
            current_scene_score += spatial_cohesion_bonus  # 此獎勵最大 0.1

            # 關鍵物體多個實例的獎勵（原始邏輯的精煉版）
            multiple_instance_bonus = 0.0
            # 對於多實例獎勵，專注於場景定義中心的物體
            key_objects_for_multi_instance_check = required_obj_ids_defined
            if scene_type in self.EVERYDAY_SCENE_TYPE_KEYS and len(optional_obj_ids_defined) > 0:
                # 對於日常場景，如果某些可選物體多次出現，也可以是關鍵的
                # 例如，"general_indoor_space" 中的多把椅子
                key_objects_for_multi_instance_check = key_objects_for_multi_instance_check.union(
                    set(list(optional_obj_ids_defined)[:max(1, len(optional_obj_ids_defined)//2)])  # 考慮前半部分的可選物體
                )

            for class_id_check in key_objects_for_multi_instance_check:
                if class_id_check in detected_classes_set_all and class_counts_all.get(class_id_check, 0) > 1:
                    multiple_instance_bonus += 0.025  # 每種類型稍微小一點的獎勵
            current_scene_score += min(0.075, multiple_instance_bonus)  # 最大 7.5% 獎勵

            # 應用 SCENE_TYPES 中定義的場景特定優先級
            if "priority" in scene_def:
                current_scene_score *= scene_def["priority"]

            scene_scores[scene_type] = min(1.0, max(0.0, current_scene_score))

        # 如果通過實例屬性 self.enable_landmark 禁用地標檢測，
        # 確保地標特定場景類型的分數被歸零。
        if not self.enable_landmark:
            landmark_scene_types = ["tourist_landmark", "natural_landmark", "historical_monument"]
            for lm_scene_type in landmark_scene_types:
                if lm_scene_type in scene_scores:
                    scene_scores[lm_scene_type] = 0.0

        return scene_scores

    def _get_object_spatial_cohesion_score(self, objects_for_scene: List[Dict],
                                         spatial_analysis_results: Optional[Dict]) -> float:
        """
        基於場景關鍵物體的空間聚合程度計算分數。
        較高的分數意味著物體在較少的區域中更加集中。
        這是一個啟發式方法，可以進一步精煉。

        Args:
            objects_for_scene: 與當前評估場景類型相關的檢測物體列表（至少包含 'class_id' 的字典）
            spatial_analysis_results: SpatialAnalyzer._analyze_regions 的輸出
                                    預期格式：{'objects_by_region': {'region_name': [{'class_id': id, ...}, ...]}}

        Returns:
            float: 聚合性分數，通常是小額獎勵（例如，0.0 到 0.1）
        """
        if (not objects_for_scene or not spatial_analysis_results or
            "objects_by_region" not in spatial_analysis_results or
            not spatial_analysis_results["objects_by_region"]):
            return 0.0

        # 獲取定義當前場景類型的關鍵物體的 class_ids 集合
        key_object_class_ids = {obj.get('class_id') for obj in objects_for_scene if obj.get('class_id') is not None}
        if not key_object_class_ids:
            return 0.0

        # 找出這些關鍵物體出現在哪些區域
        regions_containing_key_objects = set()
        # 計算找到的關鍵物體實例數量
        # 這有助於區分 1 個區域中的 1 把椅子與分佈在 5 個區域中的 5 把椅子
        total_key_object_instances_found = 0

        for region_name, objects_in_region_list in spatial_analysis_results["objects_by_region"].items():
            region_has_key_object = False
            for obj_in_region in objects_in_region_list:
                if obj_in_region.get('class_id') in key_object_class_ids:
                    region_has_key_object = True
                    total_key_object_instances_found += 1  # 計算每個實例
            if region_has_key_object:
                regions_containing_key_objects.add(region_name)

        num_distinct_key_objects_in_scene = len(key_object_class_ids)  # 關鍵物體的類型數量
        num_instances_of_key_objects_passed = len(objects_for_scene)  # 傳遞的實例數量

        if not regions_containing_key_objects or num_instances_of_key_objects_passed == 0:
            return 0.0

        # 簡單的啟發式方法：
        if (len(regions_containing_key_objects) == 1 and
            total_key_object_instances_found >= num_instances_of_key_objects_passed * 0.75):
            return 0.10  # 最強聚合性：大部分/所有關鍵物體實例在單個區域中
        elif (len(regions_containing_key_objects) <= 2 and
              total_key_object_instances_found >= num_instances_of_key_objects_passed * 0.60):
            return 0.05  # 中等聚合性：大部分/所有關鍵物體實例在最多兩個區域中
        elif (len(regions_containing_key_objects) <= 3 and
              total_key_object_instances_found >= num_instances_of_key_objects_passed * 0.50):
            return 0.02  # 較弱聚合性

        return 0.0

    def determine_scene_type(self, scene_scores: Dict[str, float]) -> Tuple[str, float]:
        """
        基於分數確定最可能的場景類型。如果偵測到地標分數夠高，則優先回傳 "tourist_landmark"。

        Args:
            scene_scores: 場景類型到置信度分數的映射字典

        Returns:
            (最佳場景類型, 置信度) 的元組
        """
        print(f"DEBUG: determine_scene_type input scores: {scene_scores}")
        if not scene_scores:
            return "unknown", 0.0

        # 檢查地標相關分數是否達到門檻，如果是，直接回傳 "tourist_landmark"
        # 假設場景分數 dictionary 中，"tourist_landmark"、"historical_monument"、"natural_landmark" 三個 key
        # 分別代表不同類型地標。將它們加總，若總分超過 0.3，就認定為地標場景。
        landmark_score = (
            scene_scores.get("tourist_landmark", 0.0) +
            scene_scores.get("historical_monument", 0.0) +
            scene_scores.get("natural_landmark", 0.0)
        )
        if landmark_score >= 0.3:
            # 回傳地標場景類型，以及該分數總和
            return "tourist_landmark", float(landmark_score)

        # 找分數最高的那個場景
        best_scene = max(scene_scores, key=scene_scores.get)
        best_score = scene_scores[best_scene]
        print(f"DEBUG: determine_scene_type result: scene={best_scene}, score={best_score}")
        return best_scene, float(best_score)

    def fuse_scene_scores(self, yolo_scene_scores: Dict[str, float],
                         clip_scene_scores: Dict[str, float],
                         num_yolo_detections: int = 0,
                         avg_yolo_confidence: float = 0.0,
                         lighting_info: Optional[Dict] = None,
                         places365_info: Optional[Dict] = None) -> Dict[str, float]:
        """
        融合來自 YOLO 物體檢測、CLIP 分析和 Places365 場景分類的場景分數。
        根據場景類型、YOLO 檢測的豐富度、照明資訊和 Places365 置信度調整權重。

        Args:
            yolo_scene_scores: 基於 YOLO 物體檢測的場景分數
            clip_scene_scores: 基於 CLIP 分析的場景分數
            num_yolo_detections: YOLO 檢測到的置信度足夠的非地標物體總數
            avg_yolo_confidence: YOLO 檢測到的非地標物體的平均置信度
            lighting_info: 可選的照明條件分析結果，預期包含 'is_indoor' (bool) 和 'confidence' (float)
            places365_info: 可選的 Places365 場景分類結果，預期包含 'mapped_scene_type'、'confidence' 和 'is_indoor'

        Returns:
            Dict: 融合了所有三個分析來源的場景分數
        """
        # 處理其中一個分數字典可能為空或所有分數實際上為零的情況
        # 提取和處理 Places365 場景分數
        # print(f"DEBUG: fuse_scene_scores input - yolo_scores: {yolo_scene_scores}")
        # print(f"DEBUG: fuse_scene_scores input - clip_scores: {clip_scene_scores}")
        # print(f"DEBUG: fuse_scene_scores input - num_yolo_detections: {num_yolo_detections}")
        # print(f"DEBUG: fuse_scene_scores input - avg_yolo_confidence: {avg_yolo_confidence}")
        # print(f"DEBUG: fuse_scene_scores input - lighting_info: {lighting_info}")
        # print(f"DEBUG: fuse_scene_scores input - places365_info: {places365_info}")

        places365_scene_scores_map = {}  # 修改變數名稱以避免與傳入的字典衝突
        if places365_info and places365_info.get('confidence', 0) > 0.1:
            mapped_scene_type = places365_info.get('mapped_scene_type', 'unknown')
            places365_confidence = places365_info.get('confidence', 0.0)

            if mapped_scene_type in self.scene_types.keys():
                places365_scene_scores_map[mapped_scene_type] = places365_confidence  # 使用新的字典
                self.logger.info(f"Places365 contributing: {mapped_scene_type} with confidence {places365_confidence:.3f}")

        # 檢查各個數據來源是否具有有意義的分數
        yolo_has_meaningful_scores = bool(yolo_scene_scores and any(s > 1e-5 for s in yolo_scene_scores.values()))  # 確保是布林值
        clip_has_meaningful_scores = bool(clip_scene_scores and any(s > 1e-5 for s in clip_scene_scores.values()))  # 確保是布林值
        places365_has_meaningful_scores = bool(places365_scene_scores_map and any(s > 1e-5 for s in places365_scene_scores_map.values()))

        # 計算有意義的數據來源數量
        meaningful_sources_count = sum([
            yolo_has_meaningful_scores,
            clip_has_meaningful_scores,
            places365_has_meaningful_scores
        ])

        # 處理特殊情況：無有效數據源或僅有單一數據源
        if meaningful_sources_count == 0:
            return {st: 0.0 for st in self.scene_types.keys()}
        elif meaningful_sources_count == 1:
            if yolo_has_meaningful_scores:
                return {st: yolo_scene_scores.get(st, 0.0) for st in self.scene_types.keys()}
            elif clip_has_meaningful_scores:
                return {st: clip_scene_scores.get(st, 0.0) for st in self.scene_types.keys()}
            elif places365_has_meaningful_scores:
                return {st: places365_scene_scores_map.get(st, 0.0) for st in self.scene_types.keys()}

        # 初始化融合分數結果字典
        fused_scores = {}
        all_relevant_scene_types = set(self.scene_types.keys())
        all_possible_scene_types = all_relevant_scene_types.union(
            set(yolo_scene_scores.keys()),
            set(clip_scene_scores.keys()),
            set(places365_scene_scores_map.keys())
        )

        # 基礎權重 - 調整以適應三個來源
        default_yolo_weight = 0.5
        default_clip_weight = 0.3
        default_places365_weight = 0.2

        is_lighting_indoor = None
        lighting_analysis_confidence = 0.0
        if lighting_info and isinstance(lighting_info, dict):
            is_lighting_indoor = lighting_info.get("is_indoor")
            lighting_analysis_confidence = lighting_info.get("confidence", 0.0)

        for scene_type in all_possible_scene_types:
            yolo_score = yolo_scene_scores.get(scene_type, 0.0)
            clip_score = clip_scene_scores.get(scene_type, 0.0)
            places365_score = places365_scene_scores_map.get(scene_type, 0.0)

            current_yolo_weight = default_yolo_weight
            current_clip_weight = default_clip_weight
            current_places365_weight = default_places365_weight
            print(f"DEBUG: Scene {scene_type} - yolo_score: {yolo_score}, clip_score: {clip_score}, places365_score: {places365_score}")
            print(f"DEBUG: Scene {scene_type} - weights: yolo={current_yolo_weight:.3f}, clip={current_clip_weight:.3f}, places365={current_places365_weight:.3f}")


            scene_definition = self.scene_types.get(scene_type, {})

            # 基於場景類型性質和 YOLO 豐富度的權重調整
            if scene_type in self.EVERYDAY_SCENE_TYPE_KEYS:
                # Places365 在日常場景分類方面表現出色
                if num_yolo_detections >= 5 and avg_yolo_confidence >= 0.45:  # 豐富的 YOLO 用於日常場景
                    current_yolo_weight = 0.60
                    current_clip_weight = 0.15
                    current_places365_weight = 0.25
                elif num_yolo_detections >= 3:  # 中等 YOLO 用於日常場景
                    current_yolo_weight = 0.50
                    current_clip_weight = 0.20
                    current_places365_weight = 0.30
                else:  # 降低 YOLO 用於日常場景，更多依賴 Places365
                    current_yolo_weight = 0.35
                    current_clip_weight = 0.25
                    current_places365_weight = 0.40

            # 對於 CLIP 的全域理解或特定訓練通常更有價值的場景
            elif any(keyword in scene_type.lower() for keyword in ["asian", "cultural", "aerial", "landmark", "monument", "tourist", "natural_landmark", "historical_monument"]):
                current_yolo_weight = 0.25
                current_clip_weight = 0.65
                current_places365_weight = 0.10  # 地標場景的較低權重

            # 對於特定室內常見場景（非地標），物體檢測是關鍵，但 Places365 提供強大的場景上下文
            elif any(keyword in scene_type.lower() for keyword in
                    ["room", "kitchen", "office", "bedroom", "desk_area", "indoor_space",
                     "professional_kitchen", "cafe", "library", "gym", "retail_store",
                     "supermarket", "classroom", "conference_room", "medical_facility",
                     "educational_setting", "dining_area"]):
                current_yolo_weight = 0.50
                current_clip_weight = 0.25
                current_places365_weight = 0.25

            # 對於特定室外常見場景（非地標），物體仍然重要
            elif any(keyword in scene_type.lower() for keyword in
                    ["parking_lot", "park_area", "beach", "harbor", "playground", "sports_field", "bus_stop", "train_station", "airport"]):
                current_yolo_weight = 0.50
                current_clip_weight = 0.25
                current_places365_weight = 0.25

            # 如果為此次運行全域禁用地標檢測
            if not self.enable_landmark:
                if any(keyword in scene_type.lower() for keyword in ["landmark", "monument", "tourist"]):
                    yolo_score = 0.0  # 應該已經從 compute_scene_scores 中為 0
                    clip_score *= 0.05  # 重度懲罰
                    places365_score *= 0.8 if scene_type not in self.EVERYDAY_SCENE_TYPE_KEYS else 1.0  # 地標場景的輕微懲罰
                elif (scene_type not in self.EVERYDAY_SCENE_TYPE_KEYS and
                      not any(keyword in scene_type.lower() for keyword in ["asian", "cultural", "aerial"])):
                    # 將權重從 CLIP 重新分配給 YOLO 和 Places365
                    weight_boost = 0.05
                    current_yolo_weight = min(0.9, current_yolo_weight + weight_boost)
                    current_places365_weight = min(0.9, current_places365_weight + weight_boost)
                    current_clip_weight = max(0.1, current_clip_weight - weight_boost * 2)

            # 如果 Places365 對此特定場景類型有高置信度，則提升其權重
            if places365_score > 0.0 and places365_info:  # 這裡的 places365_score 已經是從 map 中獲取
                places365_original_confidence = places365_info.get('confidence', 0.0)  # 獲取原始的 Places365 信心度
                if places365_original_confidence > 0.7:
                    boost_factor = min(0.2, (places365_original_confidence - 0.7) * 0.4)
                    current_places365_weight += boost_factor
                    total_other_weight = current_yolo_weight + current_clip_weight
                    if total_other_weight > 0:
                        reduction_factor = boost_factor / total_other_weight
                        current_yolo_weight *= (1 - reduction_factor)
                        current_clip_weight *= (1 - reduction_factor)

            # 權重標準化處理
            total_weight = current_yolo_weight + current_clip_weight + current_places365_weight
            if total_weight > 0:  # 避免除以零
                current_yolo_weight /= total_weight
                current_clip_weight /= total_weight
                current_places365_weight /= total_weight
            else:
                current_yolo_weight = 1/3
                current_clip_weight = 1/3
                current_places365_weight = 1/3

             # 計算融合score
            fused_score = (yolo_score * current_yolo_weight) + (clip_score * current_clip_weight) + (places365_score * current_places365_weight)

            # 處理室內外判斷的衝突分析
            places365_is_indoor = None
            places365_confidence_for_indoor = 0.0
            effective_is_indoor = is_lighting_indoor
            effective_confidence = lighting_analysis_confidence

            if places365_info and isinstance(places365_info, dict):
                places365_is_indoor = places365_info.get('is_indoor')
                places365_confidence_for_indoor = places365_info.get('confidence', 0.0)

                # Places365 在置信度高時覆蓋照明分析
                if places365_confidence_for_indoor >= 0.8 and places365_is_indoor is not None:
                    effective_is_indoor = places365_is_indoor
                    effective_confidence = places365_confidence_for_indoor

                    # 只在特定場景類型首次處理時輸出調試資訊
                    if (scene_type == "intersection" or
                        (scene_type in ["urban_intersection", "street_view"] and
                         scene_type == sorted(all_possible_scene_types)[0])):
                        self.logger.debug(f"Using Places365 indoor/outdoor decision: {places365_is_indoor} (confidence: {places365_confidence_for_indoor:.3f}) over lighting analysis")

            if effective_is_indoor is not None and effective_confidence >= 0.65:
                # 基於其定義確定場景類型本質上是室內還是室外
                is_defined_as_indoor = ("indoor" in scene_definition.get("description", "").lower() or
                                       any(kw in scene_type.lower() for kw in ["room", "kitchen", "office", "indoor", "library", "cafe", "gym"]))
                is_defined_as_outdoor = ("outdoor" in scene_definition.get("description", "").lower() or
                                        any(kw in scene_type.lower() for kw in ["street", "park", "aerial", "beach", "harbor", "intersection", "crosswalk"]))

                lighting_adjustment_strength = 0.20  # 最大調整因子（例如，20%）
                # 根據分析在閾值以上的置信度來縮放調整
                adjustment_scale = (effective_confidence - 0.65) / (1.0 - 0.65)  # 從 0 到 1 縮放
                adjustment = lighting_adjustment_strength * adjustment_scale
                adjustment = min(lighting_adjustment_strength, max(0, adjustment))  # 限制調整

                if effective_is_indoor and is_defined_as_outdoor:
                    fused_score *= (1.0 - adjustment)
                elif not effective_is_indoor and is_defined_as_indoor:
                    fused_score *= (1.0 - adjustment)
                elif effective_is_indoor and is_defined_as_indoor:
                    fused_score = min(1.0, fused_score * (1.0 + adjustment * 0.5))
                elif not effective_is_indoor and is_defined_as_outdoor:
                    fused_score = min(1.0, fused_score * (1.0 + adjustment * 0.5))

            fused_scores[scene_type] = min(1.0, max(0.0, fused_score))

        return fused_scores
        print(f"DEBUG: fuse_scene_scores final result: {fused_scores}")

    def update_enable_landmark_status(self, enable_landmark: bool):
        """
        更新地標檢測的啟用狀態。

        Args:
            enable_landmark: 是否啟用地標檢測
        """
        self.enable_landmark = enable_landmark
