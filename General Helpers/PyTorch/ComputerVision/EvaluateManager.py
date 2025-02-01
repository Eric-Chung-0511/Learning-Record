import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging
import pandas as pd

class EvaluateManager:
    """
    評估管理器：負責模型評估和性能報告生成
    
    主要功能：
    1. 生成詳細的分類報告，包括精確率、召回率、F1分數等
    2. 計算模型在測試集上的整體表現
    3. 生成混淆矩陣視覺化
    4. 分析預測結果，找出模型的優勢和劣勢
    """
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        class_names: List[str],
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        """
        初始化評估管理器
        
        Args:
            model: 要評估的模型
            test_loader: 測試數據加載器
            class_names: 類別名稱列表
            device: 運算設備
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # 初始化評估結果存儲
        self.predictions = []
        self.true_labels = []
        self.eval_results = {}

    def evaluate(self) -> Dict:
        """
        執行完整的評估流程
        
        Returns:
            Dict: 包含所有評估指標的字典
        """
        self.model.eval()
        self.predictions = []
        self.true_labels = []
        
        # 收集預測結果
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                self.predictions.extend(pred.cpu().numpy())
                self.true_labels.extend(target.cpu().numpy())
        
        # 生成評估報告
        self.eval_results = {
            'classification_report': self._generate_classification_report(),
            'confusion_matrix': self._generate_confusion_matrix(),
            'overall_accuracy': self._calculate_overall_accuracy(),
            'class_wise_accuracy': self._calculate_class_wise_accuracy()
        }
        
        return self.eval_results

    def _generate_classification_report(self) -> Dict:
        """生成詳細的分類報告"""
        report = classification_report(
            self.true_labels,
            self.predictions,
            target_names=self.class_names,
            output_dict=True
        )
        
        # 轉換為更易讀的格式
        formatted_report = pd.DataFrame(report).round(4)
        self.logger.info("\nClassification Report:")
        self.logger.info("\n" + str(formatted_report))
        
        return report

    def _generate_confusion_matrix(self) -> np.ndarray:
        """生成混淆矩陣"""
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        # 繪製混淆矩陣熱圖
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        return cm

    def _calculate_overall_accuracy(self) -> float:
        """計算整體準確率"""
        correct = sum(p == t for p, t in zip(self.predictions, self.true_labels))
        accuracy = correct / len(self.true_labels)
        self.logger.info(f"\nOverall Accuracy: {accuracy:.4f}")
        return accuracy

    def _calculate_class_wise_accuracy(self) -> Dict[str, float]:
        """計算每個類別的準確率"""
        class_correct = {}
        class_total = {}
        
        for pred, true in zip(self.predictions, self.true_labels):
            class_name = self.class_names[true]
            class_total[class_name] = class_total.get(class_name, 0) + 1
            if pred == true:
                class_correct[class_name] = class_correct.get(class_name, 0) + 1
        
        class_accuracy = {
            name: class_correct.get(name, 0) / total 
            for name, total in class_total.items()
        }
        
        self.logger.info("\nClass-wise Accuracy:")
        for name, acc in class_accuracy.items():
            self.logger.info(f"{name}: {acc:.4f}")
        
        return class_accuracy

    def get_summary(self) -> Dict:
        """
        獲取評估結果的摘要
        
        Returns:
            Dict: 包含主要評估指標的摘要
        """
        if not self.eval_results:
            self.evaluate()
        
        report = self.eval_results['classification_report']
        summary = {
            'accuracy': report['accuracy'],
            'macro_avg_f1': report['macro avg']['f1-score'],
            'weighted_avg_f1': report['weighted avg']['f1-score'],
            'class_wise_f1': {
                class_name: report[class_name]['f1-score']
                for class_name in self.class_names
            }
        }
        
        return summary

# # Usage
# 假設已經有了 dataset 實例
# dataset = FlexibleImageDataset(...)

# # 創建數據管理器
# data_manager = DataManager(dataset, ...)

# # 1. 創建評估管理器
# evaluator = EvaluateManager(
#     model=model,
#     test_loader=test_loader,
#     class_names=dataset.classes  # 直接使用 dataset 的 classes 屬性
# )

# # 2. 執行評估
# eval_results = evaluator.evaluate()

# # 3. 獲取評估摘要
# summary = evaluator.get_summary()

# # 4. 查看具體結果
# print(f"Overall Accuracy: {summary['accuracy']:.4f}")
# print(f"Macro Average F1: {summary['macro_avg_f1']:.4f}")
# print("\nClass-wise F1 Scores:")
# for class_name, f1 in summary['class_wise_f1'].items():
#     print(f"{class_name}: {f1:.4f}")
