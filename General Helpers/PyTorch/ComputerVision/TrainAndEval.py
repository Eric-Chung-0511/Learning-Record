import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.cuda.amp import GradScaler, autocast
import logging
import time
from tqdm import tqdm
from typing import Any, Dict, Optional, Union

class TrainAndEval:
    """
    整合的訓練與評估管理器：提供進階的訓練控制和監控功能
    
    主要特色：
    1. 支援自動和手動的漸進式解凍策略
    2. 整合了學習率查找器功能
    3. 提供混合精度訓練支援
    4. 包含完整的訓練監控和評估機制
    """
    def __init__(
        self,
        model: ModelManager,                    # 模型管理器實例
        train_loader: torch.utils.data.DataLoader,  # 訓練數據加載器
        val_loader: torch.utils.data.DataLoader,    # 驗證數據加載器
        memory_manager: Any,                    # 記憶體管理器實例
        use_amp: bool = True,                   # 是否使用混合精度訓練
        use_lr_finder: bool = False,            # 是否使用學習率查找器
        learning_rate: Optional[float] = None,  # 手動指定的學習率
        lr_finder_config: Optional[Dict] = None,  # 學習率查找器的配置
        unfreeze_method: str = 'auto',          # 解凍方式：'auto' 或 'manual'
        unfreeze_strategy: Optional[Union[Dict, str]] = None,  # 解凍策略
        criterion: Optional[nn.Module] = None,   # 損失函數
        optimizer: Optional[torch.optim.Optimizer] = None,  # 優化器
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,  # 學習率調度器
        total_epochs: int = 100,                # 總訓練輪數
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        """初始化訓練管理器"""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp
        
        # 設置損失函數
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        
        # 設置解凍策略
        self.unfreeze_method = unfreeze_method
        self.unfreeze_strategy = self._setup_unfreeze_strategy(unfreeze_strategy)
        
        # 設置學習率
        self.learning_rate = self._setup_learning_rate(
            learning_rate,
            use_lr_finder,
            lr_finder_config or {}
        )
        
        # 設置優化器
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.get_trainable_params(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # 設置學習率調度器
        if scheduler is None:
            steps_per_epoch = len(train_loader)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[group['lr'] * 4 for group in self.optimizer.param_groups],
                epochs=total_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
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

    def _setup_learning_rate(
        self, 
        learning_rate: Optional[float],
        use_lr_finder: bool,
        lr_finder_config: Dict
    ) -> float:
        """
        設置學習率，根據配置決定是否使用 LRFinder
        """
        if learning_rate is not None and use_lr_finder:
            self.logger.warning(
                "Both learning_rate and use_lr_finder are specified. "
                "Will ignore learning_rate and use LRFinder."
            )
        
        if use_lr_finder:
            self.logger.info("Using LR Finder to determine optimal learning rate...")
            lr_finder = LRFinder(
                model=self.model,
                train_loader=self.train_loader,
                criterion=self.criterion,
                optimizer=torch.optim.AdamW(self.model.get_trainable_params())
            )
            
            # 使用配置或默認值
            lr_finder.range_test(
                start_lr=lr_finder_config.get('start_lr', 1e-7),
                end_lr=lr_finder_config.get('end_lr', 10),
                num_iter=lr_finder_config.get('num_iter', 100),
                smooth_window=lr_finder_config.get('smooth_window', 21),
                diverge_threshold=lr_finder_config.get('diverge_threshold', 4.0)
            )
            
            lr_finder.plot()  # 顯示學習率曲線
            learning_rate = lr_finder.best_lr
            self.logger.info(f"Found optimal learning rate: {learning_rate:.2e}")
            
        elif learning_rate is None:
            raise ValueError(
                "Must either provide a learning_rate or set use_lr_finder=True"
            )
        
        return learning_rate

    def _check_unfreeze_schedule(self, progress: float):
        """
        檢查並執行解凍策略
        
        此方法在每個 epoch 開始時被調用，用於檢查是否需要根據當前訓練進度解凍某些層。
        無論是自動還是手動策略，都會通過這個方法來執行實際的解凍操作。
        
        Args:
            progress (float): 當前訓練進度 (0~1)，例如：epoch 50/100 = 0.5
        """
        # 對策略字典按進度排序，確保按順序解凍
        for threshold, num_layers in sorted(self.unfreeze_strategy.items()):
            # 使用小數值比較，允許有少量誤差
            if abs(progress - threshold) < 1e-6:  
                if num_layers is None:
                    # 解凍所有層
                    self.logger.info(
                        f"Training Progress {progress*100:.1f}%: "
                        f"Unfreezing all layers for final fine-tuning"
                    )
                    self.model.unfreeze_layers(None)
                else:
                    # 解凍指定數量的層
                    self.logger.info(
                        f"Training Progress {progress*100:.1f}%: "
                        f"Unfreezing last {num_layers} layers"
                    )
                    self.model.unfreeze_layers(num_layers)

    def _setup_unfreeze_strategy(self, strategy: Optional[Union[Dict, str]]) -> Dict[float, Optional[int]]:
        """
        設置解凍策略，支援自動和手動兩種模式
        
        Args:
            strategy: 
                - 如果是字典：直接作為手動策略使用
                - 如果是字符串：用作自動策略的類型（'conservative'/'balanced'/'aggressive'）
                - 如果是 None：使用默認的平衡策略
        
        Returns:
            Dict[float, Optional[int]]: 解凍策略字典，
                key 為訓練進度（0-1），
                value 為要解凍的層數（None 表示解凍所有層）
        """
        if self.unfreeze_method == 'manual':
            if not isinstance(strategy, dict):
                raise ValueError("Manual unfreeze method requires a dictionary strategy")
            return strategy
        
        # 自動模式
        return self._generate_unfreeze_strategy(
            total_layers=len(self.model.layers),
            strategy_type=strategy or 'balanced'
        )

    def _generate_unfreeze_strategy(self, total_layers: int, strategy_type: str = 'balanced') -> Dict[float, Optional[int]]:
        """
        自動生成解凍策略
        
        解凍策略的設計原則：
        - conservative: 保守策略，較晚開始解凍，間隔較大，適合小數據集或遷移學習差異大的情況
        - balanced: 平衡策略，適中的解凍時間和間隔，適合一般情況
        - aggressive: 激進策略，較早開始解凍，間隔較小，適合大數據集或遷移學習差異小的情況
        
        Args:
            total_layers: 模型的總層數
            strategy_type: 策略類型（'conservative'/'balanced'/'aggressive'）
        
        Returns:
            Dict[float, Optional[int]]: 自動生成的解凍策略
        """
        # 定義不同策略的參數
        strategies = {
            'conservative': {
                'start': 0.15,      # 從 15% 的訓練進度開始解凍
                'interval': 0.15,   # 每 15% 解凍一次
                'stages': 5         # 分 5 個階段解凍
            },
            'balanced': {
                'start': 0.10,      # 從 10% 的訓練進度開始解凍
                'interval': 0.125,  # 每 12.5% 解凍一次
                'stages': 5         # 分 5 個階段解凍
            },
            'aggressive': {
                'start': 0.05,      # 從 5% 的訓練進度開始解凍
                'interval': 0.10,   # 每 10% 解凍一次
                'stages': 5         # 分 5 個階段解凍
            }
        }
        
        # 獲取選擇的策略參數，默認使用 balanced
        config = strategies.get(strategy_type, strategies['balanced'])
        strategy = {}
        
        # 計算每個階段要解凍的層數
        layers_per_stage = total_layers // (config['stages'] - 1)
        
        # 生成各個階段的解凍策略
        for i in range(config['stages'] - 1):
            progress = config['start'] + (i * config['interval'])
            layers = layers_per_stage * (i + 1)
            # 確保不會超過總層數
            strategy[progress] = min(layers, total_layers)
        
        # 最後一個階段解凍所有層
        final_progress = config['start'] + ((config['stages']-1) * config['interval'])
        strategy[final_progress] = None
        
        return strategy

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
                    features, output = self.model(data)  # 注意這裡現在有兩個返回值
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                features, output = self.model(data)
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
        """
        在訓練過程中進行驗證
        Returns:
            tuple[float, float]: (validation_loss, validation_accuracy)
        """
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                features, output = self.model(data)  # 這裡收到兩個返回值
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
        執行完整的訓練流程，包含漸進式解凍
        """
        self.logger.info(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            progress = epoch / epochs
            
            # 檢查是否需要解凍層
            self._check_unfreeze_schedule(progress)
            
            self.logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            
            val_loss, val_acc = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            # 紀錄和顯示訓練狀態
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
            
            # 儲存最佳模型
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
            
            # 檢查是否需要早期停止
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
