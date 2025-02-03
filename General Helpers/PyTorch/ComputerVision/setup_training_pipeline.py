import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def setup_training_pipeline():
    """
    設置完整的訓練管道，展示所有可能的配置選項
    """
    
    # 第一步：準備數據集
    # FlexibleImageDataset 提供了靈活的數據處理和增強功能
    dataset = FlexibleImageDataset(
        root_dir='path/to/your/data',
        transform=None,          # 使用預設的數據增強
        is_train=True,
        image_size=224
    )
    
    # 使用 DataManager 進行數據集分割，確保訓練集和驗證集的分布一致
    data_manager = DataManager(
        dataset=dataset,
        train_ratio=0.7,        # 70% 用於訓練
        val_ratio=0.15,         # 15% 用於驗證
        seed=42                 # 確保可重現性
    )
    
    # 獲取數據加載器，設置適當的批次大小和工作線程數
    train_loader, val_loader, test_loader = data_manager.get_loaders(
        batch_size=32,
        num_workers=4,
        pin_memory=True         # 使用 GPU 時可加速數據傳輸
    )
    
    # 第二步：創建模型
    # ModelManager 自動處理模型的初始化和配置
    model = ModelManager(
        model_name='convnextv2_base',  # 可以是任何支援的模型名稱
        num_classes=len(dataset.classes),
        use_attention=True,            # 可選：使用 MultiHeadAttention
        num_heads=8
    )
    
    # 第三步：設置記憶體管理器，用於模型保存和加載
    memory_manager = MemoryManager(
        model=model,
        save_dir='./checkpoints',
        patience=20             # early stopping 的等待期
    )
    
    # 第四步：設置訓練器
    # 方式一：使用自動學習率查找和自動解凍策略
    trainer_auto = TrainAndEval(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        memory_manager=memory_manager,
        use_amp=True,           # 使用混合精度訓練
        use_lr_finder=True,     # 自動尋找最佳學習率
        lr_finder_config={      # 學習率查找器的配置
            'start_lr': 1e-7,
            'end_lr': 10,
            'num_iter': 100
        },
        unfreeze_method='auto',
        unfreeze_strategy='balanced'  # 或 'conservative' 或 'aggressive'
    )
    
    # 方式二：手動指定學習率和解凍策略
    manual_strategy = {
        0.1: 2,    # 10% 時解凍最後 2 層
        0.2: 4,    # 20% 時解凍最後 4 層
        0.3: 6,    # 30% 時解凍最後 6 層
        0.4: 8,    # 40% 時解凍最後 8 層
        0.5: None  # 50% 時解凍所有層
    }
    
    trainer_manual = TrainAndEval(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        memory_manager=memory_manager,
        use_amp=True,
        learning_rate=1e-5,     # 手動指定學習率，就不用使用 use_lr_finder
        unfreeze_method='manual',
        unfreeze_strategy=manual_strategy
    )
    
    # 第五步：開始訓練
    # 選擇使用自動或手動配置的訓練器
    trainer = trainer_auto  # 或 trainer_manual
    training_results = trainer.train(epochs=100)
    
    # 第六步：評估模型
    evaluator = EvaluateManager(
        model=model,
        test_loader=test_loader,
        class_names=dataset.classes,
        device=model.device
    )
    
    # 執行完整評估
    eval_results = evaluator.evaluate()
    
    # 輸出評估結果
    print("\nTraining Summary:")
    print(f"Best validation loss: {training_results['best_val_loss']:.4f}")
    print(f"Training time: {training_results['training_time']/60:.2f} minutes")
    
    print("\nFinal Evaluation Results:")
    print(f"Overall Accuracy: {eval_results['overall_accuracy']:.2f}%")
    print("\nClass-wise Performance:")
    for class_name, acc in eval_results['class_wise_accuracy'].items():
        print(f"{class_name}: {acc:.2f}%")
    
    return model, training_results, eval_results

# 執行完整的訓練流程
if __name__ == "__main__":
    model, training_results, eval_results = setup_training_pipeline()
