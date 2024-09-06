## 1. 沒有使用 unsqueeze()（傳遞圖像給模型時）：
錯誤信息：

* RuntimeError: Expected 4-dimensional input for 4-dimensional weight, but got 3-dimensional input of size [channels, height, width] instead
* 原因：模型期望輸入的張量具有 4 個維度（包括批次大小），但你提供的張量只有 3 個維度。
解決方法：

```python
image_tensor = image_tensor.unsqueeze(0)  # 添加批次維度，形狀變為 [1, channels, height, width]
```

## 2. 沒有使用 permute(1, 2, 0)（使用 plt.imshow 顯示圖像時）：
錯誤信息：

* TypeError: Invalid shape (channels, height, width) for image data
* 原因：plt.imshow 期望輸入圖像數據的格式為 [height, width, channels]，但你傳入的張量格式為 [channels, height, width]，通道維度在前，導致顯示出錯。
解決方法：

```python
image_tensor = image_tensor.permute(1, 2, 0)  # 調整維度順序為 [height, width, channels]
```

## 3. 沒有使用 squeeze()（在某些情況下顯示圖像或處理輸出時）：
錯誤信息：

* ValueError: Expected input batch_size (1) to match target batch_size (N)
* 原因：如果你的張量有一個不必要的維度（例如 [1, channels, height, width]），而實際處理時不需要這個維度，可能會導致維度不匹配的錯誤。
解決方法：

```python
image_tensor = image_tensor.squeeze(0)  # 移除批次維度，形狀變為 [channels, height, width]
```

## 4. 沒有使用 unsqueeze() 在推理時（如果沒有批次維度傳遞圖像給模型）：
錯誤信息：

* RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
* 原因：雖然這不直接與維度相關，但如果你在沒有使用 unsqueeze() 的情況下將圖像傳遞給模型，且模型已經在 GPU 上，輸入的張量可能不會自動移動到 GPU，導致類型不匹配。
解決方法：

```python
image_tensor = image_tensor.unsqueeze(0)  # 添加批次維度，形狀變為 [1, channels, height, width]
image_tensor = image_tensor.to(device)   # 確保圖像數據與模型在相同設備上
```

## 5. 使用 squeeze() 時移除了不該移除的維度：
錯誤信息：

* RuntimeError: Given groups=1, weight of size [64, 3, 7, 7], expected input[batch_size, 64, height, width] to have 3 channels, but got 1 channels instead
* 原因：如果錯誤地使用 squeeze() 移除了顏色通道維度，可能導致模型輸入通道數不匹配，進而導致此類錯誤。
解決方法：

不要錯誤地移除通道維度，只移除你確定是冗余的維度（如批次維度）。在這種情況下，通常不需要 squeeze 操作，或者可以只移除指定的維度：
```python
image_tensor = image_tensor.squeeze(0)  # 只移除批次維度，而非通道維度
```

### 6. 沒有使用 permute(1, 2, 0) 在顯示 RGB 圖像時：
錯誤信息：

* ValueError: Invalid shape (3, height, width) for RGB image
* 原因：imshow 需要的 RGB 圖像格式為 [height, width, channels]，而不是 [channels, height, width]。
解決方法：

```python
image_tensor = image_tensor.permute(1, 2, 0)  # 調整維度順序為 [height, width, channels]
plt.imshow(image_tensor)
plt.show()
```
## 總結：
* **PyTorch**的形式為[channels, height, width], **TensorFlow**的形式為[height, width, channels]
* **unsqueeze()**：用於增加批次維度（從 [channels, height, width] 到 [1, channels, height, width]）。
* **permute(1, 2, 0)**：用於調整圖像通道順序（從 [channels, height, width] 到 [height, width, channels]），使其符合 imshow 的要求。
* **squeeze()**：用於移除多余的維度，通常是從 [1, channels, height, width] 到 [channels, height, width]。
