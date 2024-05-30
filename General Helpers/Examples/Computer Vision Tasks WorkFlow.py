import os  
import random  
import shutil  # Load the shutil module for file operations such as moving files
import zipfile  # Load the zipfile module for extracting files
import tensorflow as tf  
from tensorflow.keras import layers  
from tensorflow.keras.models import Sequential 

# 下載資料集 Download the dataset
!wget https://example.com/dataset.zip

# 解壓縮資料集 Extract the dataset
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
for class_name in os.listdir(data_dir): # 遍歷每個類別資料夾 Iterate over each class directory
    class_path = os.path.join(data_dir, class_name) # 獲取類別資料夾的完整路徑 Get the full path of the class directory
    images = os.listdir(class_path) # 列出該類別資料夾中的所有圖像檔案 List all image files in the class directory
    random.shuffle(images) # 隨機打亂圖像檔案列表 Shuffle the list of image files
    
    train_size = int(len(images) * train_ratio) # 計算訓練集的大小 Calculate the size of the training set
    train_images = images[:train_size] # 選取前80%的圖像作為訓練集 Select the first 80% of images as the training set
    test_images = images[train_size:] # 剩下的20%作為測試集 The remaining 20% as the test set
    
    for img in train_images:
        src_path = os.path.join(class_path, img)  # 設定源圖像檔案的完整路徑 Set the full path of the source image file
        dst_path = os.path.join(train_dir, class_name, img)  # 設定目標訓練資料夾中的完整路徑 Set the full path in the target train directory
        shutil.move(src_path, dst_path)  # 將圖像從源路徑移動到目標路徑 Move the image from the source path to the target path
    
    for img in test_images:
        src_path = os.path.join(class_path, img)  # 設定源圖像檔案的完整路徑 Set the full path of the source image file
        dst_path = os.path.join(test_dir, class_name, img)  # 設定目標測試資料夾中的完整路徑 Set the full path in the target test directory
        shutil.move(src_path, dst_path)  # 將圖像從源路徑移動到目標路徑 Move the image from the source path to the target path

# Set image size and batch size
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
base_model = tf.keras.applications.EfficientNetB0(include_top=False)  # 載入EfficientNetB0模型且不包含頂層 Load EfficientNetB0 model without top layer
base_model.trainable = False  # 冻結預訓練模型 Freeze the pre-trained model

# 建立模型 Build the model
inputs = layers.Input(shape=(224, 224, 3))  # 定義輸入層 Define the input layer
x = data_augmentation(inputs)  # 應用資料增強 Apply data augmentation
x = base_model(x, training=False)  # 應用預訓練模型 Apply the pre-trained model
x = layers.GlobalAveragePooling2D()(x)  # 添加全局平均池化層 Add global average pooling layer
outputs = layers.Dense(train_data.num_classes, activation="softmax")(x)  # 添加輸出層 Add output layer
model = tf.keras.Model(inputs, outputs)  # 定義模型 Define the model

# 設置檢查點回調 Set up checkpoint callback
checkpoint_path = "101_classes_10_percent_data_model_checkpoint"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,  # 只保存權重 Save weights only
    monitor="val_accuracy",  # 監控驗證準確率 Monitor validation accuracy
    save_best_only=True  # 只保存最佳模型 Save only the best model
)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),  # Use Adam optimizer
    loss="categorical_crossentropy",  # Use categorical crossentropy loss
    metrics=["accuracy"]  # Evaluation metric is accuracy
)

# Train the model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data,
    callbacks=[checkpoint_callback]  # Use checkpoint callback
)
