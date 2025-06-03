import os
import traceback
import logging
from typing import Dict, Optional, Any, Tuple

from spatial_analyzer import SpatialAnalyzer
from scene_description import SceneDescriptor
from enhanced_scene_describer import EnhancedSceneDescriber
from clip_analyzer import CLIPAnalyzer
from clip_zero_shot_classifier import CLIPZeroShotClassifier
from llm_enhancer import LLMEnhancer
from landmark_activities import LANDMARK_ACTIVITIES
from scene_type import SCENE_TYPES
from object_categories import OBJECT_CATEGORIES


class ComponentInitializer:
    """
    負責初始化和管理 SceneAnalyzer 的所有子組件。
    處理組件初始化失敗的情況並提供優雅的降級機制。
    """

    def __init__(self, class_names: Dict[int, str] = None, use_llm: bool = True,
                 use_clip: bool = True, enable_landmark: bool = True,
                 llm_model_path: str = None):
        """
        初始化組件管理器。

        Args:
            class_names: YOLO 類別 ID 到名稱的映射字典
            use_llm: 是否啟用 LLM 增強功能
            use_clip: 是否啟用 CLIP 分析功能
            enable_landmark: 是否啟用地標檢測功能
            llm_model_path: LLM 模型路徑（可選）
        """
        self.logger = logging.getLogger(__name__)

        # 存儲初始化參數
        self.class_names = class_names
        self.use_llm = use_llm
        self.use_clip = use_clip
        self.enable_landmark = enable_landmark
        self.llm_model_path = llm_model_path

        # 初始化組件容器
        self.components = {}
        self.data_structures = {}
        self.initialization_status = {}

        # 初始化所有組件
        self._initialize_all_components()

    def _initialize_all_components(self):
        """初始化所有必要的組件和數據結構。"""
        try:
            # 1. 首先載入數據
            self._load_data_structures()

            # 2. 初始化核心分析組件
            self._initialize_core_analyzers()

            # 3. 初始化 CLIP 相關內容
            if self.use_clip:
                self._initialize_clip_components()

            # 4. 初始化 LLM 組件
            if self.use_llm:
                self._initialize_llm_components()

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Error during component initialization: {e}")
            traceback.print_exc()
            raise

    def _load_data_structures(self):
        """載入必要的數據結構。"""
        data_loaders = {
            'LANDMARK_ACTIVITIES': self._load_landmark_activities,
            'SCENE_TYPES': self._load_scene_types,
            'OBJECT_CATEGORIES': self._load_object_categories
        }

        for data_name, loader_func in data_loaders.items():
            try:
                self.data_structures[data_name] = loader_func()
                self.initialization_status[data_name] = True
                self.logger.info(f"Loaded {data_name} successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load {data_name}: {e}")
                self.data_structures[data_name] = {}
                self.initialization_status[data_name] = False

    def _load_landmark_activities(self) -> Dict:
        """載入地標活動數據。"""
        try:
            return LANDMARK_ACTIVITIES
        except ImportError as e:
            self.logger.warning(f"Could not import LANDMARK_ACTIVITIES: {e}")
            return {}

    def _load_scene_types(self) -> Dict:
        """載入場景類型數據。"""
        try:
            return SCENE_TYPES
        except ImportError as e:
            self.logger.warning(f"Could not import SCENE_TYPES: {e}")
            return {}

    def _load_object_categories(self) -> Dict:
        """載入物體類別數據。"""
        try:
            return OBJECT_CATEGORIES
        except ImportError as e:
            self.logger.warning(f"Could not import OBJECT_CATEGORIES: {e}")
            return {}

    def _initialize_core_analyzers(self):
        """初始化核心分析組件。"""
        # 初始化 SpatialAnalyzer
        try:
            self.components['spatial_analyzer'] = SpatialAnalyzer(
                class_names=self.class_names,
                object_categories=self.data_structures.get('OBJECT_CATEGORIES', {})
            )
            self.initialization_status['spatial_analyzer'] = True
            self.logger.info("Initialized SpatialAnalyzer successfully")
        except Exception as e:
            self.logger.error(f"Error initializing SpatialAnalyzer: {e}")
            traceback.print_exc()
            self.initialization_status['spatial_analyzer'] = False
            self.components['spatial_analyzer'] = None

        # 初始化 SceneDescriptor
        try:
            self.components['descriptor'] = SceneDescriptor(
                scene_types=self.data_structures.get('SCENE_TYPES', {}),
                object_categories=self.data_structures.get('OBJECT_CATEGORIES', {})
            )
            self.initialization_status['descriptor'] = True
            self.logger.info("Initialized SceneDescriptor successfully")
        except Exception as e:
            self.logger.error(f"Error initializing SceneDescriptor: {e}")
            traceback.print_exc()
            self.initialization_status['descriptor'] = False
            self.components['descriptor'] = None

        # 初始化 EnhancedSceneDescriber
        try:
            if self.components.get('spatial_analyzer'):
                self.components['scene_describer'] = EnhancedSceneDescriber(
                    scene_types=self.data_structures.get('SCENE_TYPES', {}),
                    spatial_analyzer_instance=self.components['spatial_analyzer']
                )
                self.initialization_status['scene_describer'] = True
                self.logger.info("Initialized EnhancedSceneDescriber successfully")
            else:
                self.logger.warning("Cannot initialize EnhancedSceneDescriber without SpatialAnalyzer")
                self.initialization_status['scene_describer'] = False
                self.components['scene_describer'] = None
        except Exception as e:
            self.logger.error(f"Error initializing EnhancedSceneDescriber: {e}")
            traceback.print_exc()
            self.initialization_status['scene_describer'] = False
            self.components['scene_describer'] = None

    def _initialize_clip_components(self):
        """初始化 CLIP 相關組件。"""
        # 初始化 CLIPAnalyzer
        try:
            self.components['clip_analyzer'] = CLIPAnalyzer()
            self.initialization_status['clip_analyzer'] = True
            self.logger.info("Initialized CLIPAnalyzer successfully")

            # 如果啟用地標檢測，初始化 CLIPZeroShotClassifier
            if self.enable_landmark:
                self._initialize_landmark_classifier()

        except Exception as e:
            self.logger.warning(f"Could not initialize CLIP analyzer: {e}")
            self.logger.info("Scene analysis will proceed without CLIP. Install CLIP with 'pip install clip' for enhanced scene understanding.")
            self.use_clip = False
            self.initialization_status['clip_analyzer'] = False
            self.components['clip_analyzer'] = None

    def _initialize_landmark_classifier(self):
        """初始化地標分類器。"""
        try:
            # 嘗試使用已載入的 CLIP 模型實例
            if (self.components.get('clip_analyzer') and
                hasattr(self.components['clip_analyzer'], 'get_clip_instance')):
                model, preprocess, device = self.components['clip_analyzer'].get_clip_instance()
                self.components['landmark_classifier'] = CLIPZeroShotClassifier(device=device)
                self.logger.info("Initialized landmark classifier with shared CLIP model")
            else:
                self.components['landmark_classifier'] = CLIPZeroShotClassifier()
                self.logger.info("Initialized landmark classifier with independent CLIP model")

            # 配置地標檢測器參數
            self._configure_landmark_classifier()
            self.initialization_status['landmark_classifier'] = True

        except (ImportError, Exception) as e:
            self.logger.warning(f"Could not initialize landmark classifier: {e}")
            self.initialization_status['landmark_classifier'] = False
            self.components['landmark_classifier'] = None
            # 不完全禁用地標檢測，允許運行時重新嘗試

    def _configure_landmark_classifier(self):
        """配置地標分類器的參數。"""
        if self.components.get('landmark_classifier'):
            try:
                classifier = self.components['landmark_classifier']
                classifier.set_batch_size(8)
                classifier.adjust_confidence_threshold("full_image", 0.8)
                classifier.adjust_confidence_threshold("distant", 0.65)
                self.logger.info("Landmark detection enabled with optimized settings")
            except Exception as e:
                self.logger.warning(f"Error configuring landmark classifier: {e}")

    def _initialize_llm_components(self):
        """初始化 LLM 組件。"""
        try:
            self.components['llm_enhancer'] = LLMEnhancer(model_path=self.llm_model_path)
            self.initialization_status['llm_enhancer'] = True
            self.logger.info("LLM enhancer initialized successfully")
        except Exception as e:
            self.logger.warning(f"Could not initialize LLM enhancer: {e}")
            self.logger.info("Scene analysis will proceed without LLM. Make sure required packages are installed.")
            self.use_llm = False
            self.initialization_status['llm_enhancer'] = False
            self.components['llm_enhancer'] = None

    def get_component(self, component_name: str) -> Optional[Any]:
        """
        獲取指定的組件實例。

        Args:
            component_name: 組件名稱

        Returns:
            組件實例或 None（如果未初始化成功）
        """
        return self.components.get(component_name)

    def get_data_structure(self, data_name: str) -> Dict:
        """
        獲取指定的數據結構。

        Args:
            data_name: 數據結構名稱

        Returns:
            數據結構字典
        """
        return self.data_structures.get(data_name, {})

    def is_component_available(self, component_name: str) -> bool:
        """
        檢查指定組件是否可用。

        Args:
            component_name: 組件名稱

        Returns:
            組件是否可用
        """
        return self.initialization_status.get(component_name, False)

    def get_initialization_summary(self) -> Dict[str, bool]:
        """
        獲取所有組件的初始化狀態摘要。

        Returns:
            組件名稱到初始化狀態的映射
        """
        return self.initialization_status.copy()

    def reinitialize_component(self, component_name: str) -> bool:
        """
        重新初始化指定的組件。

        Args:
            component_name: 要重新初始化的組件名稱

        Returns:
            重新初始化是否成功
        """
        try:
            if component_name == 'landmark_classifier' and self.use_clip and self.enable_landmark:
                self._initialize_landmark_classifier()
                return self.initialization_status.get('landmark_classifier', False)
            else:
                self.logger.warning(f"Reinitializing {component_name} is not supported")
                return False
        except Exception as e:
            self.logger.error(f"Error reinitializing {component_name}: {e}")
            return False

    def update_landmark_enable_status(self, enable_landmark: bool):
        """
        更新地標檢測的啟用狀態。

        Args:
            enable_landmark: 是否啟用地標檢測
        """
        self.enable_landmark = enable_landmark

        # 如果啟用地標檢測但分類器不可用，嘗試重新初始化
        if enable_landmark and not self.is_component_available('landmark_classifier'):
            if self.use_clip:
                self.reinitialize_component('landmark_classifier')

        # 更新相關組件的狀態
        for component_name in ['scene_describer', 'clip_analyzer', 'landmark_classifier']:
            component = self.get_component(component_name)
            if component and hasattr(component, 'enable_landmark'):
                component.enable_landmark = enable_landmark
