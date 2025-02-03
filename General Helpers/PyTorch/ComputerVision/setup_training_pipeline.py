import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def setup_training_pipeline():
    """設置完整的訓練流程"""
    
    # 第一步：數據準備
    # 首先創建數據集，這裡我們使用自定義的 FlexibleImageDataset
    dataset = FlexibleImageDataset(
        root_dir='path/to/your/data',
        transform=None,  # 使用預設的數據增強
        is_train=True,
        image_size=224
    )
    
    # 使用 DataManager 進行數據集分割
    data_manager = DataManager(
        dataset=dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42  # 確保可重現性
    )
    
    # 獲取數據加載器
    train_loader, val_loader, test_loader = data_manager.get_loaders(
        batch_size=32,
        num_workers=4,
        pin_memory=True
    )
    
    # 第二步：模型創建
    # ModelManager 會自動處理模型的載入和配置
    model = ModelManager(
        model_name='convnextv2_base',  # 可以是任何支援的模型
        num_classes=len(dataset.classes),
        use_attention=True,  # 可選：使用 MultiHeadAttention
        num_heads=8
    )
    
    # 第三步：設置記憶體管理器
    memory_manager = MemoryManager(
        model=model,
        save_dir='./checkpoints',
        patience=20
    )
    
    # 第四步：創建訓練器
    # 方式一：使用 LRFinder 自動尋找學習率
    trainer = TrainAndEval(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        memory_manager=memory_manager,
        use_amp=True,
        use_lr_finder=True,  # 啟用 LRFinder
        lr_finder_config={
            'start_lr': 1e-7,
            'end_lr': 10,
            'num_iter': 100
        },
        unfreeze_strategy={
            0.2: 2,    # 20% 時解凍最後 2 層
            0.4: 4,    # 40% 時解凍最後 4 層
            0.6: None  # 60% 時解凍所有層
        }
    )
    
    # 或者方式二：手動指定學習率
    trainer = TrainAndEval(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        memory_manager=memory_manager,
        use_amp=True,
        learning_rate=1e-3,  # 直接指定學習率
        unfreeze_strategy={
            0.2: 2,
            0.4: 4,
            0.6: None
        }
    )
    
    # 第五步：開始訓練
    training_results = trainer.train(epochs=100)
    
    # 第六步：最終評估
    evaluator = EvaluateManager(
        model=model,
        test_loader=test_loader,
        class_names=dataset.classes,
        device=model.device
    )
    
    # 執行評估並獲取結果
    eval_results = evaluator.evaluate()
    
    # 輸出評估結果
    print("\nTraining Summary:")
    print(f"Best validation loss: {training_results['best_val_loss']:.4f}")
    print(f"Training time: {training_results['training_time']/60:.2f} minutes")
    print(f"\nFinal Evaluation Results:")
    print(f"Overall Accuracy: {eval_results['overall_accuracy']:.2f}%")
    
    return model, training_results, eval_results

# 執行完整的訓練流程
if __name__ == "__main__":
    model, training_results, eval_results = setup_training_pipeline()
