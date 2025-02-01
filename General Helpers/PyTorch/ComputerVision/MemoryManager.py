import os
import torch
import logging
from typing import Optional, Dict, Any

class MemoryManager:
    """
    記憶管理器：負責模型狀態的保存、加載和訓練過程的監控
    
    主要功能：
    1. 自動保存最佳模型狀態
    2. 提供模型斷點續訓功能
    3. 實現早期停止機制
    4. 維護訓練歷史記錄
    """
    def __init__(
        self,
        model: torch.nn.Module,
        save_dir: str,
        patience: int = 20,
        model_name: str = 'best_model.pth'
    ):
        """
        初始化記憶管理器
        
        Args:
            model: 需要管理的模型實例
            save_dir: 模型保存的目錄路徑
            patience: 早期停止的等待期（默認20個epoch）
            model_name: 保存模型的文件名
        """
        self.model = model
        self.save_dir = save_dir
        self.patience = patience
        self.model_name = model_name
        
        # 確保保存目錄存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化早期停止相關的變量
        self.counter = 0  # 計數器：記錄連續沒有改善的次數
        self.best_loss = float('inf')  # 記錄最佳損失值
        self.early_stop = False  # 早期停止的標誌
        
        # 設置日誌
        self.logger = logging.getLogger(__name__)
        
        # 訓練歷史記錄
        self.history = {
            'epochs': [],
            'train_losses': [],
            'val_losses': [],
            'learning_rates': []
        }

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        val_loss: float
    ) -> None:
        """
        保存模型檢查點，包含所有訓練狀態
        
        Args:
            model: 當前模型狀態
            optimizer: 優化器狀態
            scheduler: 學習率調度器狀態
            epoch: 當前訓練輪數
            val_loss: 當前驗證損失
        """
        checkpoint_path = os.path.join(self.save_dir, self.model_name)
        
        # 準備要保存的數據
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        
        # 如果有學習率調度器，也保存其狀態
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # 保存檢查點(.pth檔案)
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, Any]:
        """
        加載之前保存的檢查點
        
        Args:
            model: 要載入權重的模型
            optimizer: 要載入狀態的優化器（可選）
            scheduler: 要載入狀態的學習率調度器（可選）
            
        Returns:
            Dict: 包含載入的訓練歷史等信息
        """
        checkpoint_path = os.path.join(self.save_dir, self.model_name)
        
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"No checkpoint found at {checkpoint_path}")
            return {}
        
        # 載入檢查點
        checkpoint = torch.load(checkpoint_path)
        
        # 恢復模型權重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 如果提供了優化器，恢復其狀態
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # 如果提供了調度器，恢復其狀態
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # 恢復訓練歷史
        self.history = checkpoint.get('history', self.history)
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint

    def should_stop(self, val_loss: float) -> bool:
        """
        檢查是否應該觸發早期停止
        
        Args:
            val_loss: 當前的驗證損失值
            
        Returns:
            bool: 是否應該停止訓練
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False

    def update_history(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float
    ) -> None:
        """
        更新訓練歷史記錄
        
        Args:
            epoch: 當前訓練輪數
            train_loss: 訓練損失
            val_loss: 驗證損失
            learning_rate: 當前學習率
        """
        self.history['epochs'].append(epoch)
        self.history['train_losses'].append(train_loss)
        self.history['val_losses'].append(val_loss)
        self.history['learning_rates'].append(learning_rate)
