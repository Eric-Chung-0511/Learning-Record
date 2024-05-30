import os
import random
import shutil
import zipfile
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# 下載資料集 Download the dataset
!wget https://example.com/dataset.zip

# 解壓資料集 Extract the dataset
with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('dataset')

# 設置資料夾路徑 Set directory paths
data_dir = 'dataset/images'
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# 創建訓練和測試資料夾 Create train and test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 創建類別資料夾 Create class directories
for class_name in os.listdir(data_dir):
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

# 設置訓練和測試比例 Set train and test split ratio
train_ratio = 0.8

# 隨機分割資料集並移動圖像 Randomly split dataset and move images
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    images = os.listdir(class_path)
    random.shuffle(images)
    
    train_size = int(len(images) * train_ratio)
    train_images = images[:train_size]
    test_images = images[train_size:]
    
    for img in train_images:
        src_path = os.path.join(class_path, img)  # 設定源圖像檔案的完整路徑 Set the full path of the source image file
        dst_path = os.path.join(train_dir, class_name, img)  # 設定目標訓練資料夾中的完整路徑 Set the full path in the target train directory
        shutil.move(src_path, dst_path)  # 將圖像從源路徑移動到目標路徑 Move the image from the source path to the target path
    
    for img in test_images:
        src_path = os.path.join(class_path, img)  # 設定源圖像檔案的完整路徑 Set the full path of the source image file
        dst_path = os.path.join(test_dir, class_name, img)  # 設定目標測試資料夾中的完整路徑 Set the full path in the target test directory
        shutil.move(src_path, dst_path)  # 將圖像從源路徑移動到目標路徑 Move the image from the source path to the target path

# 設置圖像大小和批次大小 Set image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 載入訓練資料 Load train data
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 載入測試資料 Load test data
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# 資料增強層 Data augmentation layer
data_augmentation = Sequential([
  layers.RandomFlip("horizontal"),  # 隨機水平翻轉 Random horizontal flip
  layers.RandomRotation(0.2),  # 隨機旋轉 Random rotation
  layers.RandomZoom(0.2),  # 隨機縮放 Random zoom
  layers.RandomHeight(0.2),  # 隨機高度調整 Random height adjustment
  layers.RandomWidth(0.2),  # 隨機寬度調整 Random width adjustment
], name="data_augmentation")

# 載入預訓練模型 Load pre-trained model
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False  # 冻結預訓練模型 Freeze the pre-trained model

# Build the model
inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(train_data.num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

# Set up checkpoint callback
checkpoint_path = "101_classes_10_percent_data_model_checkpoint"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,  # 只保存權重 Save weights only
    monitor="val_accuracy",  # 監控驗證準確率 Monitor validation accuracy
    save_best_only=True  # 只保存最佳模型 Save only the best model
)

# Set up Early Stopping callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # 監控驗證損失 Monitor validation loss
    patience=3  # 在停止訓練之前允許的無改進訓練輪次 Number of epochs with no improvement after which training will be stopped
)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
initial_epochs = 10
history = model.fit(
    train_data,
    epochs=initial_epochs,
    validation_data=test_data,
    callbacks=[checkpoint_callback, early_stopping_callback]  # 使用檢查點和早停回調 Use checkpoint and early stopping callbacks
)

# 解除預訓練模型的所有層的凍結 Unfreeze the entire pre-trained model
base_model.trainable = True 

# 模型微調 Fine-tuning the model, 解除最後十層的凍結 Unfreeze the last ten layers
# for layer in base_model.layers[-10:]:
    # layer.trainable = True

# Re-compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Use a lower learning rate
    loss="categorical_crossentropy",
    metrics=["accuracy"])

# Continue training the model
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs
history_fine = model.fit(
    train_data,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],  # 從上次訓練的最後一個epoch開始 Start from the last epoch of the previous training
    validation_data=test_data,
    callbacks=[checkpoint_callback, early_stopping_callback])
