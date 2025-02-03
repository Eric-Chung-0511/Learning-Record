class TrainAndEval:
    """
    進階的訓練與評估管理器：整合漸進式訓練策略和完整的評估機制
    
    主要特色：
    1. 支援漸進式解凍訓練策略
    2. 自動學習率調整
    3. 完整的訓練監控
    4. 混合精度訓練支援
    5. 早期停止機制
    """
    def __init__(
        self,
        model: nn.Module,  # 現在接收的是 ModelManager 實例
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        memory_manager: Any,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        use_amp: bool = True,
        learning_rate: float = 1e-3,
        total_epochs: int = 100,
        unfreeze_strategy: Optional[Dict] = None  # 新增：解凍策略配置
    ):
        """
        初始化訓練管理器
        Args:
            ...原有參數...
            unfreeze_strategy: 解凍策略配置，例如：
                {
                    0.1: 2,    # 在 10% 的訓練進度時解凍後 2 層
                    0.3: 4,    # 在 30% 的訓練進度時解凍後 4 層
                    0.5: None  # 在 50% 的訓練進度時解凍所有層
                }
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp
        self.unfreeze_strategy = unfreeze_strategy or {
            0.2: 2,    # 默認在 20% 時解凍後 2 層
            0.4: 4,    # 40% 時解凍後 4 層
            0.6: None  # 60% 時解凍所有層
        }
        
        # 設置損失函數
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        
        # 設置優化器：使用 ModelManager 的參數分組
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.get_trainable_params(),  # 使用 ModelManager 的參數分組
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
            
        # 設置學習率調度器
        if scheduler is None:
            steps_per_epoch = len(train_loader)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[group['lr'] * 4 for group in self.optimizer.param_groups],  # 為每個參數組設置學習率
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

    def _check_unfreeze_schedule(self, progress: float):
        """
        檢查並執行解凍策略
        Args:
            progress: 當前訓練進度 (0~1)
        """
        for threshold, num_layers in sorted(self.unfreeze_strategy.items()):
            if abs(progress - threshold) < 1e-6:  # 達到設定的進度點
                if num_layers is None:
                    self.logger.info(f"Progress {progress*100:.1f}%: Unfreezing all layers")
                    self.model.unfreeze_layers(None)
                else:
                    self.logger.info(f"Progress {progress*100:.1f}%: Unfreezing last {num_layers} layers")
                    self.model.unfreeze_layers(num_layers)

    def train_epoch(self) -> tuple[float, float]:
        """執行一個訓練 epoch，現在包含特徵輸出"""
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
