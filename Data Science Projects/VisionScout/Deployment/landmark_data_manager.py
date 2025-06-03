
import logging
import traceback
from typing import List, Dict, Tuple, Optional, Union, Any

from landmark_data import ALL_LANDMARKS, get_all_landmark_prompts
from landmark_activities import LANDMARK_ACTIVITIES

class LandmarkDataManager:
    """
    專門處理地標數據的載入、管理和查詢功能，包括地標信息、提示詞和活動建議
    """

    def __init__(self):
        """
        initialize landmark related
        """
        self.logger = logging.getLogger(__name__)
        self.landmark_data = {}
        self.landmark_prompts = []
        self.landmark_id_to_index = {}
        self.is_enabled = False

        self._load_landmark_data()

    def _load_landmark_data(self):
        """
        載入地標數據和相關資訊
        """
        try:
            self.landmark_data = ALL_LANDMARKS
            self.landmark_prompts = get_all_landmark_prompts()
            self.logger.info(f"Loaded {len(self.landmark_prompts)} landmark prompts for classification")

            # 創建地標ID到索引的映射，可快速查找
            self.landmark_id_to_index = {landmark_id: i for i, landmark_id in enumerate(ALL_LANDMARKS.keys())}

            self.is_enabled = True
            self.logger.info(f"Successfully loaded landmark data with {len(self.landmark_data)} landmarks")

        except ImportError:
            self.logger.warning("landmark_data.py not found. Landmark classification will be limited")
            self.landmark_data = {}
            self.landmark_prompts = []
            self.landmark_id_to_index = {}
            self.is_enabled = False
        except Exception as e:
            self.logger.error(f"Error loading landmark data: {e}")
            self.logger.error(traceback.format_exc())
            self.landmark_data = {}
            self.landmark_prompts = []
            self.landmark_id_to_index = {}
            self.is_enabled = False

    def get_landmark_prompts(self) -> List[str]:
        """
        獲取所有地標提示詞

        Returns:
            List[str]: 地標提示詞列表
        """
        return self.landmark_prompts

    def get_landmark_by_id(self, landmark_id: str) -> Dict[str, Any]:
        """
        根據地標ID獲取地標信息

        Args:
            landmark_id: Landmark ID

        Returns:
            Dict[str, Any]: 地標詳細信息
        """
        return self.landmark_data.get(landmark_id, {})

    def get_landmark_by_index(self, index: int) -> Tuple[str, Dict[str, Any]]:
        """
        根據索引獲取地標信息

        Args:
            index: 地標在列表中的索引

        Returns:
            Tuple[str, Dict[str, Any]]: (地標ID, 地標info)
        """
        try:
            landmark_ids = list(self.landmark_data.keys())
            if 0 <= index < len(landmark_ids):
                landmark_id = landmark_ids[index]
                return landmark_id, self.landmark_data[landmark_id]
            else:
                self.logger.warning(f"Index {index} out of range for landmark data")
                return None, {}
        except Exception as e:
            self.logger.error(f"Error getting landmark by index {index}: {e}")
            self.logger.error(traceback.format_exc())
            return None, {}

    def get_landmark_index(self, landmark_id: str) -> Optional[int]:
        """
        獲取地標ID對應的index

        Args:
            landmark_id: 地標ID

        Returns:
            Optional[int]: 索引，如果不存在則返回None
        """
        return self.landmark_id_to_index.get(landmark_id)

    def determine_landmark_type(self, landmark_id: str) -> str:
        """
        自動判斷地標類型，基於地標數據和命名

        Args:
            landmark_id: 地標ID

        Returns:
            str: 地標類型，用於調整閾值
        """
        if not landmark_id:
            return "building"  # 預設類型

        try:
            # 獲取地標詳細數據
            landmark_info = self.landmark_data.get(landmark_id, {})

            # 獲取地標相關文本
            landmark_id_lower = landmark_id.lower()
            landmark_name = landmark_info.get("name", "").lower()
            landmark_location = landmark_info.get("location", "").lower()
            landmark_aliases = [alias.lower() for alias in landmark_info.get("aliases", [])]

            # 合併所有文本數據用於特徵判斷
            combined_text = " ".join([landmark_id_lower, landmark_name] + landmark_aliases)

            # 地標類型的特色特徵
            type_features = {
                "skyscraper": ["skyscraper", "tall", "tower", "高樓", "摩天", "大厦", "タワー"],
                "tower": ["tower", "bell", "clock", "塔", "鐘樓", "タワー", "campanile"],
                "monument": ["monument", "memorial", "statue", "紀念", "雕像", "像", "memorial"],
                "natural": ["mountain", "lake", "canyon", "falls", "beach", "山", "湖", "峽谷", "瀑布", "海灘"],
                "temple": ["temple", "shrine", "寺", "神社", "廟"],
                "palace": ["palace", "castle", "宮", "城", "皇宮", "宫殿"],
                "distinctive": ["unique", "leaning", "slanted", "傾斜", "斜", "獨特", "傾く"]
            }

            # 檢查是否位於亞洲地區
            asian_regions = ["china", "japan", "korea", "taiwan", "singapore", "vietnam", "thailand",
                            "hong kong", "中國", "日本", "韓國", "台灣", "新加坡", "越南", "泰國", "香港"]
            is_asian = any(region in landmark_location for region in asian_regions)

            # 判斷地標類型
            best_type = None
            max_matches = 0

            for type_name, features in type_features.items():
                # 計算特徵詞匹配數量
                matches = sum(1 for feature in features if feature in combined_text)
                if matches > max_matches:
                    max_matches = matches
                    best_type = type_name

            # 處理亞洲地區特例
            if is_asian and best_type == "tower":
                best_type = "skyscraper"  # 亞洲地區的塔型建築閾值較低

            # 特例處理：檢測傾斜建築
            if any(term in combined_text for term in ["leaning", "slanted", "tilt", "inclined", "斜", "傾斜"]):
                return "distinctive"  # 傾斜建築需要特殊處理

            return best_type if best_type and max_matches > 0 else "building"  # 預設為一般建築

        except Exception as e:
            self.logger.error(f"Error determining landmark type for {landmark_id}: {e}")
            self.logger.error(traceback.format_exc())
            return "building"

    def extract_landmark_specific_info(self, landmark_id: str) -> Dict[str, Any]:
        """
        提取特定地標的詳細信息，包括特色模板和活動建議

        Args:
            landmark_id: 地標ID

        Returns:
            Dict[str, Any]: 地標特定信息
        """
        if not landmark_id or landmark_id == "unknown":
            return {"has_specific_activities": False}

        specific_info = {"has_specific_activities": False}

        try:
            # 從 landmark_data 中提取基本信息
            landmark_data_source = self.landmark_data.get(landmark_id)

            # 處理地標基本數據
            if landmark_data_source:
                # 提取正確的地標名稱
                if "name" in landmark_data_source:
                    specific_info["landmark_name"] = landmark_data_source["name"]

                # 提取所有可用的 prompts 作為特色模板
                if "prompts" in landmark_data_source:
                    specific_info["feature_templates"] = landmark_data_source["prompts"][:5]
                    specific_info["primary_template"] = landmark_data_source["prompts"][0]

                # 提取別名info
                if "aliases" in landmark_data_source:
                    specific_info["aliases"] = landmark_data_source["aliases"]

                # 提取位置信息
                if "location" in landmark_data_source:
                    specific_info["location"] = landmark_data_source["location"]

                # 提取其他相關信息
                for key in ["year_built", "architectural_style", "significance", "description"]:
                    if key in landmark_data_source:
                        specific_info[key] = landmark_data_source[key]

            # 嘗試從 LANDMARK_ACTIVITIES 中提取活動建議
            try:
                if landmark_id in LANDMARK_ACTIVITIES:
                    activities = LANDMARK_ACTIVITIES[landmark_id]
                    specific_info["landmark_specific_activities"] = activities
                    specific_info["has_specific_activities"] = True
                    self.logger.info(f"Found {len(activities)} specific activities for landmark {landmark_id}")
                else:
                    self.logger.info(f"No specific activities found for landmark {landmark_id} in LANDMARK_ACTIVITIES")
                    specific_info["has_specific_activities"] = False
            except ImportError:
                self.logger.warning("Could not import LANDMARK_ACTIVITIES from landmark_activities")
                specific_info["has_specific_activities"] = False
            except Exception as e:
                self.logger.error(f"Error loading landmark activities for {landmark_id}: {e}")
                self.logger.error(traceback.format_exc())
                specific_info["has_specific_activities"] = False

        except Exception as e:
            self.logger.error(f"Error extracting landmark specific info for {landmark_id}: {e}")
            self.logger.error(traceback.format_exc())

        return specific_info

    def get_landmark_count(self) -> int:
        """
        獲取地標總數

        Returns:
            int: 地標數量
        """
        return len(self.landmark_data)

    def is_landmark_enabled(self) -> bool:
        """
        檢查地標功能是否啟用

        Returns:
            bool: 地標功能狀態
        """
        return self.is_enabled

    def get_all_landmark_ids(self) -> List[str]:
        """
        獲取所有地標ID列表

        Returns:
            List[str]: 地標ID列表
        """
        return list(self.landmark_data.keys())

    def validate_landmark_id(self, landmark_id: str) -> bool:
        """
        驗證地標ID是否有效

        Args:
            landmark_id: 要驗證的地標ID

        Returns:
            bool: ID是否有效
        """
        return landmark_id in self.landmark_data
