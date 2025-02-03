import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
import math
from tqdm import tqdm

class LRFinder:
    """
    學習率尋找器：使用進階的學習率範圍測試方法
    結合 Savitzky-Golay 濾波器和二階導數分析
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda'
    ):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        # 儲存學習率和損失
        self.lr_history = []
        self.loss_history = []
        
        # 最佳學習率
        self.best_lr = None
        
    def range_test(
        self,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
        smooth_window: int = 21,
        diverge_threshold: float = 4.0
    ):
        """
        執行學習率範圍測試
        
        Args:
            start_lr: 起始學習率
            end_lr: 結束學習率
            num_iter: 迭代次數
            smooth_window: Savitzky-Golay 濾波器窗口大小
            diverge_threshold: 發散閾值
        """
        # 儲存原始學習率
        original_lr = self.optimizer.param_groups[0]['lr']
        
        # 計算每次迭代的學習率增長
        mult = (end_lr / start_lr) ** (1 / num_iter)
        self.best_loss = float('inf')
        
        # 設置初始學習率
        self.optimizer.param_groups[0]['lr'] = start_lr
        
        # 迭代訓練
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader)):
            if batch_idx >= num_iter:
                break
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 前向傳播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向傳播
            loss.backward()
            self.optimizer.step()
            
            # 記錄當前學習率和損失
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_history.append(current_lr)
            self.loss_history.append(loss.item())
            
            # 檢查是否發散
            if loss.item() > diverge_threshold * self.best_loss:
                break
                
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
            
            # 更新學習率
            self.optimizer.param_groups[0]['lr'] *= mult
            
        # 使用 Savitzky-Golay 濾波器平滑損失曲線
        if len(self.loss_history) >= smooth_window:
            smooth_loss = savgol_filter(self.loss_history, smooth_window, 3)
            
            # 計算二階導數
            gradients = np.gradient(smooth_loss)
            curvature = np.gradient(gradients)
            
            # 找到最佳學習率（曲率最大的點）
            max_curvature_idx = np.argmin(curvature[:-1])  # 避免最後幾個點
            self.best_lr = self.lr_history[max_curvature_idx]
        else:
            # 如果數據點太少，使用簡單的最小損失點
            min_loss_idx = np.argmin(self.loss_history)
            self.best_lr = self.lr_history[min_loss_idx]
        
        # 恢復原始學習率
        self.optimizer.param_groups[0]['lr'] = original_lr
        
    def plot(self, skip_begin: int = 10, skip_end: int = 2):
        """
        繪製學習率vs損失圖
        
        Args:
            skip_begin: 跳過前幾個點（通常波動較大）
            skip_end: 跳過最後幾個點（可能發散）
        """
        plt.figure(figsize=(10, 6))
        plt.semilogx(
            self.lr_history[skip_begin:-skip_end], 
            self.loss_history[skip_begin:-skip_end]
        )
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        
        # 標註最佳學習率點
        if self.best_lr is not None:
            plt.plot(self.best_lr, self.loss_history[self.lr_history.index(self.best_lr)], 
                    'ro', label=f'Best LR: {self.best_lr:.2e}')
            plt.legend()
        
        plt.grid(True)
        plt.show()
