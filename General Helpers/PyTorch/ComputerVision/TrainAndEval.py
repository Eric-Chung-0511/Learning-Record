import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time
import logging
from typing import Optional, Dict, Any, Union

class TrainAndEval:
    """
    整合的訓練與評估管理器
    提供完整的模型訓練流程，包含：
    - 預設使用 CrossEntropyLoss、AdamW優化器和OneCycleLR調度器
    - 支援混合精度訓練
    - 自動保存最佳模型
    - 提供詳細的訓練統計
    - 整合早期停止機制
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        memory_manager: Any,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        use_amp: bool = False,
        learning_rate: float = 1e-3,
        total_epochs: int = 100
    ):
        """
        初始化訓練管理器
        
        Args:
            model: 待訓練的模型
            train_loader: 訓練數據加載器
            val_loader: 驗證數據加載器
            memory_manager: 記憶體管理器實例
            criterion: 損失函數（默認使用CrossEntropyLoss）
            optimizer: 優化器（默認使用AdamW）
            scheduler: 學習率調度器（默認使用OneCycleLR）
            device: 運算設備（默認使用GPU如果可用）
            use_amp: 是否使用混合精度訓練
            learning_rate: 初始學習率（用於默認優化器）
            total_epochs: 總訓練輪數（用於默認調度器）
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp
        
        # 設置默認的損失函數
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        
        # 設置默認的優化器
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate
            )
        else:
            self.optimizer = optimizer
            
        # 設置默認的學習率調度器
        if scheduler is None:
            steps_per_epoch = len(train_loader)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=learning_rate * 4,
                epochs=total_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.35,
                anneal_strategy='cos',
                div_factor=10,
                final_div_factor=150
            )
        else:
            self.scheduler = scheduler

        self.memory_manager = memory_manager
        self.scaler = GradScaler() if use_amp else None
        
        # 初始化訓練統計
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
        # 設定日誌
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def find_best_lr(self, **kwargs):
        """
        尋找最佳學習率
        
        Returns:
            float: 建議的最佳學習率
        """
        lr_finder = LRFinder(
            model=self.model,
            train_loader=self.train_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device
        )
        
        # 執行學習率範圍測試
        lr_finder.range_test(**kwargs)
        
        # 繪製結果
        lr_finder.plot()
        
        return lr_finder.best_lr

    def train_epoch(self) -> tuple[float, float]:
        """執行一個訓練epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} Training')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()
            total_correct += correct
            total_samples += target.size(0)
            
            total_loss += loss.item()
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100. * total_correct / total_samples
            
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
            
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * total_correct / total_samples
        
        return epoch_loss, epoch_acc

    def validate_epoch(self) -> tuple[float, float]:
        """執行一個驗證epoch"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validating'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                pred = output.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()
                total_samples += target.size(0)
                total_loss += loss.item()
        
        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = 100. * total_correct / total_samples
        
        return epoch_loss, epoch_acc

    def train(self, epochs: int) -> Dict[str, list]:
        """
        執行完整的訓練流程
        
        Args:
            epochs: 訓練的總輪數
            
        Returns:
            Dict: 包含訓練過程中的各種指標
        """
        self.logger.info(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            self.logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            
            val_loss, val_acc = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            if (epoch + 1) % 5 == 0:
                self.train_accuracies.append(train_acc)
                self.val_accuracies.append(val_acc)
                self.logger.info(
                    f"Train Accuracy: {train_acc:.2f}% | "
                    f"Val Accuracy: {val_acc:.2f}%"
                )
            
            self.logger.info(
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.memory_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    val_loss=val_loss
                )
                self.logger.info(f"New best model saved! (Val Loss: {val_loss:.4f})")
            
            if self.memory_manager.should_stop(val_loss):
                self.logger.info("Early stopping triggered!")
                break
        
        training_time = time.time() - start_time
        self.logger.info(f"\nTraining completed in {training_time/60:.2f} minutes")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time
        }


# Usage
# # 1. 創建模型和數據加載器
# model = Model() # The model itself
# train_loader = ...  
# val_loader = ...    

# # 2. 創建記憶體管理器
# memory_manager = MemoryManager(
#     model=model,
#     save_dir='checkpoints',
#     patience=20  # early stopping 的等待期
# )

# # 3. 使用默認配置創建訓練器
# trainer = TrainAndEval(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     memory_manager=memory_manager,
#     use_amp=True  # 使用混合精度訓練
# )

# # 4. 開始訓練
# training_history = trainer.train(epochs=100)

# # 5. 查看訓練結果
# print("最佳驗證損失:", training_history['best_val_loss'])
# print("總訓練時間:", training_history['training_time'] / 60, "分鐘")
