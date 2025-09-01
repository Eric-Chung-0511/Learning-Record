import json
import sqlite3
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import os
import traceback
from dog_database import get_dog_description
from breed_health_info import breed_health_info
from breed_noise_info import breed_noise_info

class DataQuality(Enum):
    """資料品質等級"""
    HIGH = "high"           # 完整且可靠的資料
    MEDIUM = "medium"       # 部分資料或推斷資料
    LOW = "low"             # 不完整或不確定的資料
    UNKNOWN = "unknown"     # 未知或缺失資料

@dataclass
class BreedStandardization:
    """品種標準化資料結構"""
    canonical_name: str
    display_name: str
    aliases: List[str] = field(default_factory=list)
    size_category: int = 1          # 1=tiny, 2=small, 3=medium, 4=large, 5=giant
    exercise_level: int = 2         # 1=low, 2=moderate, 3=high, 4=very_high
    noise_level: int = 2           # 1=low, 2=moderate, 3=high
    care_complexity: int = 2       # 1=low, 2=moderate, 3=high
    child_compatibility: float = 0.5  # 0=no, 0.5=unknown, 1=yes
    data_quality_scores: Dict[str, DataQuality] = field(default_factory=dict)
    confidence_flags: Dict[str, float] = field(default_factory=dict)

@dataclass
class ConfigurationSettings:
    """配置設定結構"""
    scoring_weights: Dict[str, float] = field(default_factory=dict)
    calibration_settings: Dict[str, Any] = field(default_factory=dict)
    constraint_thresholds: Dict[str, float] = field(default_factory=dict)
    semantic_model_config: Dict[str, Any] = field(default_factory=dict)
    data_imputation_rules: Dict[str, Any] = field(default_factory=dict)
    debug_mode: bool = False
    version: str = "1.0.0"

class ConfigManager:
    """
    中央化配置和資料標準化管理系統
    處理品種資料標準化、配置管理和資料品質評估
    """

    def __init__(self, config_file: Optional[str] = None):
        """初始化配置管理器"""
        self.config_file = config_file or "config.json"
        self.breed_standardization = {}
        self.configuration = ConfigurationSettings()
        self.breed_aliases = {}
        self._load_default_configuration()
        self._initialize_breed_standardization()

        # 嘗試載入自定義配置
        if os.path.exists(self.config_file):
            self._load_configuration()

    def _load_default_configuration(self):
        """載入預設配置"""
        self.configuration = ConfigurationSettings(
            scoring_weights={
                'activity_compatibility': 0.35,
                'noise_compatibility': 0.25,
                'spatial_compatibility': 0.15,
                'family_compatibility': 0.10,
                'maintenance_compatibility': 0.10,
                'size_compatibility': 0.05
            },
            calibration_settings={
                'target_range_min': 0.45,
                'target_range_max': 0.95,
                'min_effective_range': 0.3,
                'auto_calibration': True,
                'tie_breaking_enabled': True
            },
            constraint_thresholds={
                'apartment_size_limit': 3,      # 最大允許尺寸 (medium)
                'high_exercise_threshold': 3,   # 高運動需求閾值
                'quiet_noise_limit': 2,         # 安靜環境噪音限制
                'child_safety_threshold': 0.8   # 兒童安全最低分數
            },
            semantic_model_config={
                'model_name': 'all-MiniLM-L6-v2',
                'fallback_models': ['all-mpnet-base-v2', 'all-MiniLM-L12-v2'],
                'similarity_threshold': 0.5,
                'cache_embeddings': True
            },
            data_imputation_rules={
                'noise_level_defaults': {
                    'terrier': 'high',
                    'hound': 'high',
                    'herding': 'moderate',
                    'toy': 'moderate',
                    'working': 'moderate',
                    'sporting': 'moderate',
                    'non_sporting': 'low',
                    'unknown': 'moderate'
                },
                'exercise_level_defaults': {
                    'working': 'high',
                    'sporting': 'high',
                    'herding': 'high',
                    'terrier': 'moderate',
                    'hound': 'moderate',
                    'toy': 'low',
                    'non_sporting': 'moderate',
                    'unknown': 'moderate'
                }
            },
            debug_mode=False,
            version="1.0.0"
        )

    def _initialize_breed_standardization(self):
        """初始化品種標準化"""
        try:
            # 獲取所有品種
            breeds = self._get_all_breeds()

            for breed in breeds:
                standardized = self._standardize_breed_data(breed)
                self.breed_standardization[breed] = standardized

                # 建立別名映射
                for alias in standardized.aliases:
                    self.breed_aliases[alias.lower()] = breed

            print(f"Initialized standardization for {len(self.breed_standardization)} breeds")

        except Exception as e:
            print(f"Error initializing breed standardization: {str(e)}")
            print(traceback.format_exc())

    def _get_all_breeds(self) -> List[str]:
        """獲取所有品種清單"""
        try:
            conn = sqlite3.connect('animal_detector.db')
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT Breed FROM AnimalCatalog")
            breeds = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return breeds
        except Exception as e:
            print(f"Error getting breed list: {str(e)}")
            return []

    def _standardize_breed_data(self, breed: str) -> BreedStandardization:
        """標準化品種資料"""
        try:
            # 基本資訊
            breed_info = get_dog_description(breed) or {}
            health_info = breed_health_info.get(breed, {})
            noise_info = breed_noise_info.get(breed, {})

            # 建立標準化結構
            canonical_name = breed
            display_name = breed.replace('_', ' ')
            aliases = self._generate_breed_aliases(breed)

            # 標準化分類數據
            size_category = self._standardize_size(breed_info.get('Size', ''))
            exercise_level = self._standardize_exercise_needs(breed_info.get('Exercise Needs', ''))
            noise_level = self._standardize_noise_level(noise_info.get('noise_level', ''))
            care_complexity = self._standardize_care_level(breed_info.get('Care Level', ''))
            child_compatibility = self._standardize_child_compatibility(
                breed_info.get('Good with Children', '')
            )

            # 評估資料品質
            data_quality_scores = self._assess_data_quality(breed_info, health_info, noise_info)
            confidence_flags = self._calculate_confidence_flags(breed_info, health_info, noise_info)

            return BreedStandardization(
                canonical_name=canonical_name,
                display_name=display_name,
                aliases=aliases,
                size_category=size_category,
                exercise_level=exercise_level,
                noise_level=noise_level,
                care_complexity=care_complexity,
                child_compatibility=child_compatibility,
                data_quality_scores=data_quality_scores,
                confidence_flags=confidence_flags
            )

        except Exception as e:
            print(f"Error standardizing breed {breed}: {str(e)}")
            return BreedStandardization(
                canonical_name=breed,
                display_name=breed.replace('_', ' '),
                aliases=self._generate_breed_aliases(breed)
            )

    def _generate_breed_aliases(self, breed: str) -> List[str]:
        """生成品種別名"""
        aliases = []
        display_name = breed.replace('_', ' ')

        # 基本別名
        aliases.append(display_name.lower())
        aliases.append(breed.lower())

        # 常見縮寫和變體
        breed_aliases_map = {
            'German_Shepherd': ['gsd', 'german shepherd dog', 'alsatian'],
            'Labrador_Retriever': ['lab', 'labrador', 'retriever'],
            'Golden_Retriever': ['golden', 'goldie'],
            'Border_Collie': ['border', 'collie'],
            'Yorkshire_Terrier': ['yorkie', 'york', 'yorkshire'],
            'French_Bulldog': ['frenchie', 'french bull', 'bouledogue français'],
            'Boston_Terrier': ['boston bull', 'american gentleman'],
            'Cavalier_King_Charles_Spaniel': ['cavalier', 'ckcs', 'king charles'],
            'American_Staffordshire_Terrier': ['amstaff', 'american staff'],
            'Jack_Russell_Terrier': ['jrt', 'jack russell', 'parson russell'],
            'Shih_Tzu': ['shih tzu', 'lion dog'],
            'Bichon_Frise': ['bichon', 'powder puff'],
            'Cocker_Spaniel': ['cocker', 'english cocker', 'american cocker']
        }

        if breed in breed_aliases_map:
            aliases.extend(breed_aliases_map[breed])

        # 移除重複
        return list(set(aliases))

    def _standardize_size(self, size_str: str) -> int:
        """標準化體型分類"""
        size_mapping = {
            'tiny': 1, 'toy': 1,
            'small': 2, 'little': 2, 'compact': 2,
            'medium': 3, 'moderate': 3, 'average': 3,
            'large': 4, 'big': 4,
            'giant': 5, 'huge': 5, 'extra large': 5
        }

        size_lower = size_str.lower()
        for key, value in size_mapping.items():
            if key in size_lower:
                return value

        return 3  # 預設為 medium

    def _standardize_exercise_needs(self, exercise_str: str) -> int:
        """標準化運動需求"""
        exercise_mapping = {
            'low': 1, 'minimal': 1, 'light': 1,
            'moderate': 2, 'average': 2, 'medium': 2, 'regular': 2,
            'high': 3, 'active': 3, 'vigorous': 3,
            'very high': 4, 'extreme': 4, 'intense': 4
        }

        exercise_lower = exercise_str.lower()
        for key, value in exercise_mapping.items():
            if key in exercise_lower:
                return value

        return 2  # 預設為 moderate

    def _standardize_noise_level(self, noise_str: str) -> int:
        """標準化噪音水平"""
        noise_mapping = {
            'low': 1, 'quiet': 1, 'silent': 1, 'minimal': 1,
            'moderate': 2, 'average': 2, 'medium': 2, 'occasional': 2,
            'high': 3, 'loud': 3, 'vocal': 3, 'frequent': 3
        }

        noise_lower = noise_str.lower()
        for key, value in noise_mapping.items():
            if key in noise_lower:
                return value

        return 2  # 預設為 moderate

    def _standardize_care_level(self, care_str: str) -> int:
        """標準化護理複雜度"""
        care_mapping = {
            'low': 1, 'easy': 1, 'simple': 1, 'minimal': 1,
            'moderate': 2, 'average': 2, 'medium': 2, 'regular': 2,
            'high': 3, 'complex': 3, 'intensive': 3, 'demanding': 3
        }

        care_lower = care_str.lower()
        for key, value in care_mapping.items():
            if key in care_lower:
                return value

        return 2  # 預設為 moderate

    def _standardize_child_compatibility(self, child_str: str) -> float:
        """標準化兒童相容性"""
        if child_str.lower() == 'yes':
            return 1.0
        elif child_str.lower() == 'no':
            return 0.0
        else:
            return 0.5  # 未知或不確定

    def _assess_data_quality(self, breed_info: Dict, health_info: Dict,
                           noise_info: Dict) -> Dict[str, DataQuality]:
        """評估資料品質"""
        quality_scores = {}

        # 基本資訊品質
        if breed_info:
            required_fields = ['Size', 'Exercise Needs', 'Temperament', 'Good with Children']
            complete_fields = sum(1 for field in required_fields if breed_info.get(field))

            if complete_fields >= 4:
                quality_scores['basic_info'] = DataQuality.HIGH
            elif complete_fields >= 2:
                quality_scores['basic_info'] = DataQuality.MEDIUM
            else:
                quality_scores['basic_info'] = DataQuality.LOW
        else:
            quality_scores['basic_info'] = DataQuality.UNKNOWN

        # 健康資訊品質
        if health_info and health_info.get('health_notes'):
            quality_scores['health_info'] = DataQuality.HIGH
        elif health_info:
            quality_scores['health_info'] = DataQuality.MEDIUM
        else:
            quality_scores['health_info'] = DataQuality.UNKNOWN

        # 噪音資訊品質
        if noise_info and noise_info.get('noise_level'):
            quality_scores['noise_info'] = DataQuality.HIGH
        else:
            quality_scores['noise_info'] = DataQuality.LOW

        return quality_scores

    def _calculate_confidence_flags(self, breed_info: Dict, health_info: Dict,
                                  noise_info: Dict) -> Dict[str, float]:
        """計算信心度標記"""
        confidence_flags = {}

        # 基本資訊信心度
        basic_confidence = 0.8 if breed_info else 0.2
        if breed_info and breed_info.get('Description'):
            basic_confidence += 0.1
        confidence_flags['basic_info'] = min(1.0, basic_confidence)

        # 健康資訊信心度
        health_confidence = 0.7 if health_info else 0.3
        confidence_flags['health_info'] = health_confidence

        # 噪音資訊信心度
        noise_confidence = 0.8 if noise_info else 0.4
        confidence_flags['noise_info'] = noise_confidence

        # 整體信心度
        confidence_flags['overall'] = np.mean(list(confidence_flags.values()))

        return confidence_flags

    def get_standardized_breed_data(self, breed: str) -> Optional[BreedStandardization]:
        """獲取標準化品種資料"""
        # 嘗試直接匹配
        if breed in self.breed_standardization:
            return self.breed_standardization[breed]

        # 嘗試別名匹配
        breed_lower = breed.lower()
        if breed_lower in self.breed_aliases:
            canonical_breed = self.breed_aliases[breed_lower]
            return self.breed_standardization.get(canonical_breed)

        # 模糊匹配
        for alias, canonical_breed in self.breed_aliases.items():
            if breed_lower in alias or alias in breed_lower:
                return self.breed_standardization.get(canonical_breed)

        return None

    def apply_data_imputation(self, breed: str) -> BreedStandardization:
        """應用資料插補規則"""
        try:
            standardized = self.get_standardized_breed_data(breed)
            if not standardized:
                return BreedStandardization(canonical_name=breed, display_name=breed.replace('_', ' '))

            imputation_rules = self.configuration.data_imputation_rules

            # 噪音水平插補
            if standardized.noise_level == 2:  # moderate (可能是預設值)
                breed_group = self._determine_breed_group(breed)
                noise_defaults = imputation_rules.get('noise_level_defaults', {})
                if breed_group in noise_defaults:
                    imputed_noise = self._standardize_noise_level(noise_defaults[breed_group])
                    standardized.noise_level = imputed_noise
                    standardized.confidence_flags['noise_info'] *= 0.7  # 降低信心度

            # 運動需求插補
            if standardized.exercise_level == 2:  # moderate (可能是預設值)
                breed_group = self._determine_breed_group(breed)
                exercise_defaults = imputation_rules.get('exercise_level_defaults', {})
                if breed_group in exercise_defaults:
                    imputed_exercise = self._standardize_exercise_needs(exercise_defaults[breed_group])
                    standardized.exercise_level = imputed_exercise
                    standardized.confidence_flags['basic_info'] *= 0.8  # 降低信心度

            return standardized

        except Exception as e:
            print(f"Error applying data imputation for {breed}: {str(e)}")
            return self.get_standardized_breed_data(breed) or BreedStandardization(
                canonical_name=breed, display_name=breed.replace('_', ' ')
            )

    def _determine_breed_group(self, breed: str) -> str:
        """確定品種群組"""
        breed_lower = breed.lower()

        if 'terrier' in breed_lower:
            return 'terrier'
        elif 'hound' in breed_lower:
            return 'hound'
        elif any(word in breed_lower for word in ['shepherd', 'collie', 'cattle', 'sheepdog']):
            return 'herding'
        elif any(word in breed_lower for word in ['retriever', 'pointer', 'setter', 'spaniel']):
            return 'sporting'
        elif any(word in breed_lower for word in ['mastiff', 'great', 'rottweiler', 'akita']):
            return 'working'
        elif any(word in breed_lower for word in ['toy', 'pug', 'chihuahua', 'papillon']):
            return 'toy'
        else:
            return 'unknown'

    def _load_configuration(self):
        """載入配置檔案"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # 更新配置
            if 'scoring_weights' in config_data:
                self.configuration.scoring_weights.update(config_data['scoring_weights'])
            if 'calibration_settings' in config_data:
                self.configuration.calibration_settings.update(config_data['calibration_settings'])
            if 'constraint_thresholds' in config_data:
                self.configuration.constraint_thresholds.update(config_data['constraint_thresholds'])
            if 'semantic_model_config' in config_data:
                self.configuration.semantic_model_config.update(config_data['semantic_model_config'])
            if 'data_imputation_rules' in config_data:
                self.configuration.data_imputation_rules.update(config_data['data_imputation_rules'])
            if 'debug_mode' in config_data:
                self.configuration.debug_mode = config_data['debug_mode']

            print(f"Configuration loaded from {self.config_file}")

        except Exception as e:
            print(f"Error loading configuration: {str(e)}")

    def save_configuration(self):
        """儲存配置檔案"""
        try:
            config_data = asdict(self.configuration)

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            print(f"Configuration saved to {self.config_file}")

        except Exception as e:
            print(f"Error saving configuration: {str(e)}")

    def get_configuration(self) -> ConfigurationSettings:
        """獲取當前配置"""
        return self.configuration

    def update_configuration(self, updates: Dict[str, Any]):
        """更新配置"""
        try:
            for key, value in updates.items():
                if hasattr(self.configuration, key):
                    setattr(self.configuration, key, value)

            print(f"Configuration updated: {list(updates.keys())}")

        except Exception as e:
            print(f"Error updating configuration: {str(e)}")

    def get_breed_mapping_summary(self) -> Dict[str, Any]:
        """獲取品種映射摘要"""
        try:
            total_breeds = len(self.breed_standardization)
            total_aliases = len(self.breed_aliases)

            # 資料品質分布
            quality_distribution = {}
            for breed_data in self.breed_standardization.values():
                for category, quality in breed_data.data_quality_scores.items():
                    if category not in quality_distribution:
                        quality_distribution[category] = {}
                    quality_name = quality.value
                    quality_distribution[category][quality_name] = (
                        quality_distribution[category].get(quality_name, 0) + 1
                    )

            # 信心度統計
            confidence_stats = {}
            for breed_data in self.breed_standardization.values():
                for category, confidence in breed_data.confidence_flags.items():
                    if category not in confidence_stats:
                        confidence_stats[category] = []
                    confidence_stats[category].append(confidence)

            confidence_averages = {
                category: np.mean(values) for category, values in confidence_stats.items()
            }

            return {
                'total_breeds': total_breeds,
                'total_aliases': total_aliases,
                'quality_distribution': quality_distribution,
                'confidence_averages': confidence_averages,
                'configuration_version': self.configuration.version
            }

        except Exception as e:
            print(f"Error generating breed mapping summary: {str(e)}")
            return {'error': str(e)}

_config_manager = None

def get_config_manager() -> ConfigManager:
    """獲取全局配置管理器"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_standardized_breed_data(breed: str) -> Optional[BreedStandardization]:
    """獲取標準化品種資料"""
    manager = get_config_manager()
    return manager.get_standardized_breed_data(breed)

def get_breed_with_imputation(breed: str) -> BreedStandardization:
    """獲取應用補進後的品種資料"""
    manager = get_config_manager()
    return manager.apply_data_imputation(breed)

def get_system_configuration() -> ConfigurationSettings:
    """系統配置"""
    manager = get_config_manager()
    return manager.get_configuration()
