# 使用 Python 3.11 作為基礎映像檔，選擇 slim 版本以減少映像檔大小
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴項，這些是 OpenCV 和其他圖像處理套件所需的
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 建立必要的目錄結構
RUN mkdir -p /app/models /app/data /app/assets/example_images

# 複製需求檔案並安裝 Python 依賴項
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 下載 YOLO 模型檔案
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8x.pt')" && \
    mkdir -p /app/yolo_models && \
    find /home/appuser/.cache -name "yolov8x.pt" -exec cp {} /app/yolo_models/ \; 2>/dev/null || \
    find /root/.cache -name "yolov8x.pt" -exec cp {} /app/yolo_models/ \; 2>/dev/null || \
    wget -O /app/yolo_models/yolov8x.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt

# 複製應用程式原始碼
COPY *.py ./

# 複製資源檔案
COPY models/ ./models/
COPY data/ ./data/
COPY assets/ ./assets/

# 設定環境變數
ENV MODEL_PATH=/app/models/ConvNextV2Base_best_model.pth
ENV ASSETS_PATH=/app/assets/example_images/
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV YOLO_MODEL_PATH=/app/yolo_models/yolov8x.pt

# 暴露應用程式埠號
EXPOSE 7860

# 建立非 root 使用者以提高安全性
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 啟動應用程式
CMD ["gradio", "app.py"]