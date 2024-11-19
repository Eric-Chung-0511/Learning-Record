import os
import json
import re
import random
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report


class DataManager:
    def __init__(self, data_dir, json_dir, data_type, dataset_cls, train_split=0.8):
        """
        通用化的數據管理器，支持不同類型的數據和自定義的 Dataset 類別。
        Args:
            data_dir (str): 數據目錄，包含分類數據的文件夾。
            json_dir (str): JSON 文件存儲目錄，用於保存類別信息。
            data_type (str): 數據類型（例如 'dog', 'cat'）。
            dataset_cls (class): PyTorch Dataset 類別，用於處理數據加載。
            train_split (float): 訓練數據分割比例，默認為 0.8。
        """
        self.data_dir = data_dir
        self.json_dir = json_dir
        self.data_type = data_type.lower()
        self.dataset_cls = dataset_cls  # 支持自定義的 Dataset 類
        self.train_split = train_split  # 訓練/測試分割比例
        self.classes = []  # 類別名稱列表
        self.dataset = None  # Dataset 物件
        self.train_indices = []  # 訓練數據索引
        self.test_indices = []  # 測試數據索引
        self.version = 0  # 用於追踪 JSON 文件的版本號

        self._load_json()  # 加載類別信息

    def _load_json(self):
        """
        加載 JSON 文件中的類別信息。
        如果文件不存在，則初始化空類別列表。
        """
        json_file = os.path.join(self.json_dir, f"{self.data_type}_data.json")
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.classes = data.get('classes', []) # classes 不存在就返回空列表([])
                self.version = data.get('version', 0)
        else:
            print(f"JSON file not found: {json_file}")
            self.classes = []

    def _save_json(self):
        """
        保存類別信息到 JSON 文件，確保目錄存在。
        """
        json_file = os.path.join(self.json_dir, f"{self.data_type}_data.json")
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({'classes': self.classes, 'version': self.version}, f, indent=4, ensure_ascii=False)

    def add_class(self, class_name):
        """
        添加新類別到類別列表中，避免重複。
        Args:
            class_name (str): 類別名稱。
        Returns:
            bool: 如果成功添加，返回 True；否則返回 False。
        """
        clean_name = re.sub(r'[^\w\s-]', '', class_name).strip()  # 清理非法字符
        clean_name = re.sub(r'\s+', '_', clean_name)  # 將空格替換為下劃線
        if clean_name.lower() not in [c.lower() for c in self.classes]:
            self.classes.append(clean_name)
            self.version += 1  # 增加版本號
            self._save_json()
            return True
        return False

    def load_classes_from_directory(self):
        """
        從資料夾結構中讀取類別名稱並更新類別列表。
        """
        if not os.path.exists(self.data_dir):
            print(f"Directory not found: {self.data_dir}")
            return

        new_classes = set()
        for folder_name in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, folder_name)):
                # 清理類別名稱, 過濾掉除單詞字符（字母、數字、下劃線）、空白字符和連接符（-）以外的所有特殊符號, strip()可清理字句前後多於空格
                clean_name = re.sub(r'[^\w\s-]', '', folder_name).strip()
                clean_name = re.sub(r'\s+', '_', clean_name)
                new_classes.add(clean_name)

        # 更新類別列表並保存
        self.classes = list(new_classes)
        self.version += 1
        self._save_json()
        print(f"Updated {self.data_type} classes from directory. Total classes: {len(self.classes)}")

    def create_or_update_dataset(self):
        """
        創建或更新 Dataset，並分割成訓練集和測試集。
        """
        self.dataset = self.dataset_cls(self.data_dir, is_train=True)  # 初始化 Dataset
        self.classes = self.dataset.classes  # 同步類別信息
        self._save_json()

        # 分割數據集
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(self.train_split * dataset_size)
        random.shuffle(indices)
        self.train_indices = indices[:split]
        self.test_indices = indices[split:]

        print(f"Dataset updated: {dataset_size} samples, {len(self.classes)} classes")
        print(f"Train: {len(self.train_indices)}, Test: {len(self.test_indices)}")

    def get_dataloader(self, batch_size=32, shuffle=True, train=True):
        """
        獲取 DataLoader，支持訓練或測試數據。
        Args:
            batch_size (int): 批量大小，默認為 32。
            shuffle (bool): 是否打亂數據，默認為 True。
            train (bool): 是否返回訓練數據，默認為 True。
        Returns:
            DataLoader: PyTorch 的 DataLoader。
        """
        if self.dataset is None:
            raise ValueError(f"{self.data_type} dataset has not been initialized.")

        # 根據是 train_indices or test_indices後, 這些索引從原始資料集中生成 Subset, 再傳給 DataLoader 加載
        indices = self.train_indices if train else self.test_indices
        subset = Subset(self.dataset, indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

    def generate_classification_report(self, y_true, y_pred):
        """
        生成分類報告。
        Args:
            y_true (list): 真實標籤。
            y_pred (list): 預測標籤。
        Returns:
            str: 分類報告。
        """
        return classification_report(y_true, y_pred, target_names=self.classes)

    def get_num_classes(self):
        """
        返回數據集中的類別數。
        """
        return len(self.classes)
