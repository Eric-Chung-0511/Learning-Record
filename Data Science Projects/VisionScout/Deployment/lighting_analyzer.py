import numpy as np
import cv2
from typing import Dict, Any, Optional

class LightingAnalyzer:
    """
    分析圖像的光照條件，提供增強的室內or室外判斷和光照類型分類，並專注於光照分析。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化光照分析器。

        Args:
            config: 可選的配置字典，用於自定義分析參數
        """
        self.config = config or self._get_default_config()

    def analyze(self, image):
        """
        分析圖像的光照條件。

        主要分析入口點，計算基本特徵，判斷室內/室外，確定光照條件。

        Args:
            image: 輸入圖像 (numpy array 或 PIL Image)

        Returns:
            Dict: 包含光照分析結果的字典
        """
        try:
            # 轉換圖像格式
            if not isinstance(image, np.ndarray):
                image_np = np.array(image)
            else:
                image_np = image.copy()

            # 確保 RGB 格式
            if image_np.shape[2] == 3 and isinstance(image_np, np.ndarray):
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_np

            # 計算基本特徵
            features = self._compute_basic_features(image_rgb)

            # 分析室內or室外
            indoor_result = self._analyze_indoor_outdoor(features)
            is_indoor = indoor_result["is_indoor"]
            indoor_probability = indoor_result["indoor_probability"]

            # 確定光照條件
            lighting_conditions = self._determine_lighting_conditions(features, is_indoor)

            # 整合結果
            result = {
                "time_of_day": lighting_conditions["time_of_day"],
                "confidence": float(lighting_conditions["confidence"]),
                "is_indoor": is_indoor,
                "indoor_probability": float(indoor_probability),
                "brightness": {
                    "average": float(features["avg_brightness"]),
                    "std_dev": float(features["brightness_std"]),
                    "dark_ratio": float(features["dark_pixel_ratio"])
                },
                "color_info": {
                    "blue_ratio": float(features["blue_ratio"]),
                    "yellow_orange_ratio": float(features["yellow_orange_ratio"]),
                    "gray_ratio": float(features["gray_ratio"]),
                    "avg_saturation": float(features["avg_saturation"]),
                    "sky_brightness": float(features["sky_brightness"]),
                    "color_atmosphere": features["color_atmosphere"],
                    "warm_ratio": float(features["warm_ratio"]),
                    "cool_ratio": float(features["cool_ratio"])
                }
            }

            # 添加診斷信息
            if self.config["include_diagnostics"]:
                result["diagnostics"] = {
                    "feature_contributions": indoor_result.get("feature_contributions", {}),
                    "lighting_diagnostics": lighting_conditions.get("diagnostics", {})
                }

            return result

        except Exception as e:
            print(f"Error in lighting analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "time_of_day": "unknown",
                "confidence": 0,
                "error": str(e)
            }

    def _compute_basic_features(self, image_rgb):
        """
        計算圖像的基本光照特徵（徹底優化版本）。

        Args:
            image_rgb: RGB 格式的圖像 (numpy array)

        Returns:
            Dict: 包含計算出的特徵值
        """
        # 獲取圖像尺寸
        height, width = image_rgb.shape[:2]

        # 根據圖像大小自適應縮放因子
        base_scale = 4
        scale_factor = base_scale + min(8, max(0, int((height * width) / (1000 * 1000))))

        # 創建縮小的圖像以加速處理
        small_rgb = cv2.resize(image_rgb, (width//scale_factor, height//scale_factor))

        # 一次性轉換所有顏色空間，避免重複計算
        hsv_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        gray_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        small_gray = cv2.resize(gray_img, (width//scale_factor, height//scale_factor))

        # 分離HSV通道
        h_channel = hsv_img[:,:,0]
        s_channel = hsv_img[:,:,1]
        v_channel = hsv_img[:,:,2]

        # 基本亮度特徵
        avg_brightness = np.mean(v_channel)
        brightness_std = np.std(v_channel)
        dark_pixel_ratio = np.sum(v_channel < 50) / (height * width)

        # 顏色特徵
        yellow_orange_mask = ((h_channel >= 15) & (h_channel <= 40))
        yellow_orange_ratio = np.sum(yellow_orange_mask) / (height * width)

        blue_mask = ((h_channel >= 90) & (h_channel <= 130))
        blue_ratio = np.sum(blue_mask) / (height * width)

        # 特別檢查圖像上部區域，尋找藍天特徵
        upper_region_h = h_channel[:height//4, :]
        upper_region_s = s_channel[:height//4, :]
        upper_region_v = v_channel[:height//4, :]

        # 藍天通常具有高飽和度的藍色
        sky_blue_mask = ((upper_region_h >= 90) & (upper_region_h <= 130) &
                        (upper_region_s > 70) & (upper_region_v > 150))
        sky_blue_ratio = np.sum(sky_blue_mask) / max(1, upper_region_h.size)

        gray_mask = (s_channel < 50) & (v_channel > 100)
        gray_ratio = np.sum(gray_mask) / (height * width)

        avg_saturation = np.mean(s_channel)

        # 天空亮度
        upper_half = v_channel[:height//2, :]
        sky_brightness = np.mean(upper_half)

        # 色調分析
        warm_colors = ((h_channel >= 0) & (h_channel <= 60)) | (h_channel >= 300)
        warm_ratio = np.sum(warm_colors) / (height * width)

        cool_colors = (h_channel >= 180) & (h_channel <= 270)
        cool_ratio = np.sum(cool_colors) / (height * width)

        # 確定色彩氛圍
        if warm_ratio > 0.4:
            color_atmosphere = "warm"
        elif cool_ratio > 0.4:
            color_atmosphere = "cool"
        else:
            color_atmosphere = "neutral"

        # 只在縮小的圖像上計算梯度，大幅提高效能
        gx = cv2.Sobel(small_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(small_gray, cv2.CV_32F, 0, 1, ksize=3)

        vertical_strength = np.mean(np.abs(gy))
        horizontal_strength = np.mean(np.abs(gx))
        gradient_ratio = vertical_strength / max(horizontal_strength, 1e-5)

        # -- 亮度均勻性 --
        brightness_uniformity = 1 - min(1, brightness_std / max(avg_brightness, 1e-5))

        # -- 高效的天花板分析 --
        # 使用更大的下採樣率分析頂部區域
        top_scale = scale_factor * 2  # 更積極的下採樣
        top_region = v_channel[:height//4:top_scale, ::top_scale]
        top_region_std = np.std(top_region)
        ceiling_uniformity = 1.0 - min(1, top_region_std / max(np.mean(top_region), 1e-5))

        # 使用更簡單的方法檢測上部水平線
        top_gradients = np.abs(gy[:small_gray.shape[0]//4, :])
        horizontal_lines_strength = np.mean(top_gradients)
        # 標準化
        horizontal_line_ratio = min(1, horizontal_lines_strength / 40)

        # 極簡的亮點檢測
        sampled_v = v_channel[::scale_factor*2, ::scale_factor*2]
        light_threshold = min(220, avg_brightness + 2*brightness_std)
        is_bright = sampled_v > light_threshold
        bright_spot_count = np.sum(is_bright)

        # 圓形光源分析的簡化替代方法
        circular_light_score = 0
        indoor_light_score = 0
        light_distribution_uniformity = 0.5

        # 只有當檢測到亮點，且不是大量亮點時（可能是室外光反射）才進行光源分析
        if 1 < bright_spot_count < 20:
            # 簡單統計亮點分布
            bright_y, bright_x = np.where(is_bright)
            if len(bright_y) > 1:
                # 檢查亮點是否成組出現 - 室內照明常見模式
                mean_x = np.mean(bright_x)
                mean_y = np.mean(bright_y)
                dist_from_center = np.sqrt((bright_x - mean_x)**2 + (bright_y - mean_y)**2)

                # 如果亮點分布較集中，可能是燈具
                if np.std(dist_from_center) < np.mean(dist_from_center):
                    circular_light_score = min(3, len(bright_y) // 2)
                    light_distribution_uniformity = 0.7

                # 評估亮點是否位於上部區域，常見於室內頂燈
                if np.mean(bright_y) < sampled_v.shape[0] / 2:
                    indoor_light_score = 0.6
                else:
                    indoor_light_score = 0.3

        # 使用邊緣區域梯度來快速估計邊界
        edge_scale = scale_factor * 2

        # 只採樣圖像邊緣部分進行分析
        left_edge = small_gray[:, :small_gray.shape[1]//6]
        right_edge = small_gray[:, 5*small_gray.shape[1]//6:]
        top_edge = small_gray[:small_gray.shape[0]//6, :]

        # 計算每個邊緣區域的梯度強度
        left_gradient = np.mean(np.abs(cv2.Sobel(left_edge, cv2.CV_32F, 1, 0, ksize=3)))
        right_gradient = np.mean(np.abs(cv2.Sobel(right_edge, cv2.CV_32F, 1, 0, ksize=3)))
        top_gradient = np.mean(np.abs(cv2.Sobel(top_edge, cv2.CV_32F, 0, 1, ksize=3)))

        # 標準化
        left_edge_density = min(1.0, left_gradient / 50)
        right_edge_density = min(1.0, right_gradient / 50)
        top_edge_density = min(1.0, top_gradient / 50)

        # 封閉環境通常在圖像邊緣有較強的梯度
        boundary_edge_score = (left_edge_density + right_edge_density + top_edge_density) / 3

        # 簡單估計整體邊緣密度
        edges_density = min(1, (np.mean(np.abs(gx)) + np.mean(np.abs(gy))) / 100)

        street_line_score = 0

        # 檢查下半部分是否有強烈的垂直線條
        bottom_half = small_gray[small_gray.shape[0]//2:, :]
        bottom_vert_gradient = cv2.Sobel(bottom_half, cv2.CV_32F, 0, 1, ksize=3)
        strong_vert_lines = np.abs(bottom_vert_gradient) > 50
        if np.sum(strong_vert_lines) > (bottom_half.size * 0.05):  # 如果超過5%的像素是強垂直線
            street_line_score = 0.7

        # 整合所有特徵
        features = {
            # 基本亮度和顏色特徵
            "avg_brightness": avg_brightness,
            "brightness_std": brightness_std,
            "dark_pixel_ratio": dark_pixel_ratio,
            "yellow_orange_ratio": yellow_orange_ratio,
            "blue_ratio": blue_ratio,
            "sky_blue_ratio": sky_blue_ratio,
            "gray_ratio": gray_ratio,
            "avg_saturation": avg_saturation,
            "sky_brightness": sky_brightness,
            "color_atmosphere": color_atmosphere,
            "warm_ratio": warm_ratio,
            "cool_ratio": cool_ratio,

            # 結構特徵
            "gradient_ratio": gradient_ratio,
            "brightness_uniformity": brightness_uniformity,
            "bright_spot_count": bright_spot_count,
            "vertical_strength": vertical_strength,
            "horizontal_strength": horizontal_strength,

            # 室內/室外判斷特徵
            "ceiling_uniformity": ceiling_uniformity,
            "horizontal_line_ratio": horizontal_line_ratio,
            "indoor_light_score": indoor_light_score,
            "circular_light_count": circular_light_score,
            "light_distribution_uniformity": light_distribution_uniformity,
            "boundary_edge_score": boundary_edge_score,
            "top_region_std": top_region_std,
            "edges_density": edges_density,

            # 新增：室外特定特徵
            "street_line_score": street_line_score
        }

        return features

    def _analyze_indoor_outdoor(self, features):
        """
        使用多特徵融合進行室內/室外判斷

        Args:
            features: 特徵字典

        Returns:
            Dict: 室內/室外判斷結果
        """
        # 獲取配置中的特徵權重
        weights = self.config["indoor_outdoor_weights"]

        # 初始概率值 - 開始時中性評估
        indoor_score = 0
        feature_contributions = {}
        diagnostics = {}

        # 1. 藍色區域（天空）特徵 - 藍色區域多通常表示室外
        if features.get("blue_ratio", 0) > 0.2:
            # 檢查是否有室內指標，如果有明顯的室內特徵，則減少藍色的負面影響
            if (features.get("ceiling_uniformity", 0) > 0.5 or
                features.get("boundary_edge_score", 0) > 0.3 or
                features.get("indoor_light_score", 0) > 0.2 or
                features.get("bright_spot_count", 0) > 0):
                blue_score = -weights["blue_ratio"] * features["blue_ratio"] * 8
            else:
                blue_score = -weights["blue_ratio"] * features["blue_ratio"] * 15
        else:
            blue_score = -weights["blue_ratio"] * features["blue_ratio"] * 15

        indoor_score += blue_score
        feature_contributions["blue_ratio"] = blue_score

        # 判斷視角 - 如果上部有藍天而上下亮度差異大，可能是仰視室外建築
        if (features.get("sky_blue_ratio", 0) > 0.01 and
            features["sky_brightness"] > features["avg_brightness"] * 1.1):
            viewpoint_outdoor_score = -1.8  # 強烈的室外指標
            indoor_score += viewpoint_outdoor_score
            feature_contributions["outdoor_viewpoint"] = viewpoint_outdoor_score

        # 2. 亮度均勻性特徵 - 室內通常光照更均勻
        uniformity_score = weights["brightness_uniformity"] * features["brightness_uniformity"]
        indoor_score += uniformity_score
        feature_contributions["brightness_uniformity"] = uniformity_score

        # 3. 天花板特徵 - 強化天花板檢測的權重
        ceiling_contribution = 0
        if "ceiling_uniformity" in features:
            ceiling_uniformity = features["ceiling_uniformity"]
            horizontal_line_ratio = features.get("horizontal_line_ratio", 0)

            # 增強天花板檢測的影響
            if ceiling_uniformity > 0.5:
                ceiling_weight = 3
                ceiling_contribution = weights.get("ceiling_features", 1.5) * ceiling_weight
                if horizontal_line_ratio > 0.2:  # 如果有水平線條，進一步增強
                    ceiling_contribution *= 1.5
            elif ceiling_uniformity > 0.4:
                ceiling_contribution = weights.get("ceiling_features", 1.5) * 1.2

            indoor_score += ceiling_contribution
            feature_contributions["ceiling_features"] = ceiling_contribution

        # 4. 強化吊燈的檢測
        light_contribution = 0
        if "indoor_light_score" in features:
            indoor_light_score = features["indoor_light_score"]
            circular_light_count = features.get("circular_light_count", 0)

            # 加強對特定類型光源的檢測
            if circular_light_count >= 1:  # 即便只有一個圓形光源也很可能是室內
                light_contribution = weights.get("light_features", 1.2) * 2.0
            elif indoor_light_score > 0.3:
                light_contribution = weights.get("light_features", 1.2) * 1.0

            indoor_score += light_contribution
            feature_contributions["light_features"] = light_contribution

        # 5. 環境封閉度特徵
        boundary_contribution = 0
        if "boundary_edge_score" in features:
            boundary_edge_score = features["boundary_edge_score"]
            edges_density = features.get("edges_density", 0)

            # 高邊界評分暗示封閉環境（室內）
            if boundary_edge_score > 0.3:
                boundary_contribution = weights.get("boundary_features", 1.2) * 2
            elif boundary_edge_score > 0.2:
                boundary_contribution = weights.get("boundary_features", 1.2) * 1.2

            indoor_score += boundary_contribution
            feature_contributions["boundary_features"] = boundary_contribution

        if (features.get("edges_density", 0) > 0.2 and
            features.get("bright_spot_count", 0) > 5 and
            features.get("vertical_strength", 0) > features.get("horizontal_strength", 0) * 1.5):
            # 商業街道特徵：高邊緣密度 + 多亮點 + 強垂直特徵
            street_feature_score = -weights.get("street_features", 1.2) * 1.5
            indoor_score += street_feature_score
            feature_contributions["street_features"] = street_feature_score

        # 添加對亞洲商業街道的專門檢測
        if (features.get("edges_density", 0) > 0.25 and  # 高邊緣密度
            features.get("vertical_strength", 0) > features.get("horizontal_strength", 0) * 1.8 and  # 更強的垂直結構
            features.get("brightness_uniformity", 0) < 0.6):  # 較低的亮度均勻性（招牌、燈光等造成）
            asian_street_score = -2.2  # 非常強的室外代表性特徵
            indoor_score += asian_street_score
            feature_contributions["asian_commercial_street"] = asian_street_score


        # 6. 垂直/水平梯度比率
        gradient_contribution = 0
        if features["gradient_ratio"] > 2.0:
            combined_uniformity = (features["brightness_uniformity"] +
                                features.get("ceiling_uniformity", 0)) / 2

            if combined_uniformity > 0.5:
                gradient_contribution = weights["gradient_ratio"] * 0.7
            else:
                gradient_contribution = -weights["gradient_ratio"] * 0.3

            indoor_score += gradient_contribution
            feature_contributions["gradient_ratio"] = gradient_contribution

        # 7. 亮點檢測（光源）
        bright_spot_contribution = 0
        bright_spot_count = features["bright_spot_count"]
        circular_light_count = features.get("circular_light_count", 0)

        # 調整亮點分析邏輯
        if circular_light_count >= 1:  # 即使只有一個圓形光源
            bright_spot_contribution = weights["bright_spots"] * 1.5
        elif bright_spot_count < 5:  # 適當放寬閾值
            bright_spot_contribution = weights["bright_spots"] * 0.5
        elif bright_spot_count > 15:  # 大量亮點比較有可能為室外
            bright_spot_contribution = -weights["bright_spots"] * 0.4

        indoor_score += bright_spot_contribution
        feature_contributions["bright_spots"] = bright_spot_contribution

        # 8. 色調分析
        yellow_contribution = 0
        if features["avg_brightness"] < 150 and features["yellow_orange_ratio"] > 0.15:
            if features.get("indoor_light_score", 0) > 0.2:
                yellow_contribution = weights["color_tone"] * 0.8
            else:
                yellow_contribution = weights["color_tone"] * 0.5

            indoor_score += yellow_contribution
            feature_contributions["yellow_tone"] = yellow_contribution

        if features.get("blue_ratio", 0) > 0.7:
            # 檢查是否有室內指標，如果有明顯的室內特徵，則減少藍色的負面影響
            if (features.get("ceiling_uniformity", 0) > 0.6 or
                features.get("boundary_edge_score", 0) > 0.3 or
                features.get("indoor_light_score", 0) > 0):
                blue_score = -weights["blue_ratio"] * features["blue_ratio"] * 10
            else:
                blue_score = -weights["blue_ratio"] * features["blue_ratio"] * 18
        else:
            blue_score = -weights["blue_ratio"] * features["blue_ratio"] * 18
        # 9. 上半部與下半部亮度對比
        sky_contribution = 0
        if features["sky_brightness"] > features["avg_brightness"] * 1.3:
            if features["blue_ratio"] > 0.15:
                sky_contribution = -weights["sky_brightness"] * 0.9
            else:
                sky_contribution = -weights["sky_brightness"] * 0.6

            indoor_score += sky_contribution
            feature_contributions["sky_brightness"] = sky_contribution

        # 加入額外的餐廳特徵檢測邏輯
        dining_feature_contribution = 0

        # 檢測中央懸掛式燈具，有懸掛燈代表有天花板，就代表是室內
        if circular_light_count >= 1 and features.get("light_distribution_uniformity", 0) > 0.4:
            dining_feature_contribution = 1.5
            indoor_score += dining_feature_contribution
            feature_contributions["dining_features"] = dining_feature_contribution

        # 10. 增強的藍天的檢測，即便是小面積的藍天也是很強的室外指標
        sky_contribution = 0
        if "sky_blue_ratio" in features:
            # 只有當藍色區域集中在上部且亮度高時，才認為是藍天
            if features["sky_blue_ratio"] > 0.01 and features["sky_brightness"] > features.get("avg_brightness", 0) * 1.2:
                sky_outdoor_score = -2.5 * features["sky_blue_ratio"] * weights.get("blue_ratio", 1.2)
                indoor_score += sky_outdoor_score
                feature_contributions["sky_blue_detection"] = sky_outdoor_score

        asian_street_indicators = 0

        # 1: 高垂直結構強度
        vertical_ratio = features.get("vertical_strength", 0) / max(features.get("horizontal_strength", 1e-5), 1e-5)
        if vertical_ratio > 1.8:
            asian_street_indicators += 1

        # 2: 高邊緣密度 + 路面標記特徵
        if features.get("edges_density", 0) > 0.25 and features.get("street_line_score", 0) > 0.2:
            asian_street_indicators += 2

        # 3: 多個亮點 + 亮度不均勻
        if features.get("bright_spot_count", 0) > 5 and features.get("brightness_uniformity", 0) < 0.6:
            asian_street_indicators += 1

        # 4: 藍色區域小（天空被高樓遮擋）但亮度高
        if features.get("blue_ratio", 0) < 0.1 and features.get("sky_brightness", 0) > features.get("avg_brightness", 0) * 1.1:
            asian_street_indicators += 1

        # 如果滿足至少 3 個指標，調整權重變成偏向室外的判斷
        if asian_street_indicators >= 3:
            # 記錄檢測到的模式
            feature_contributions["asian_street_pattern"] = -2.5
            indoor_score += -2.5  # 明顯向室外傾斜

            # 降低室內指標的權重
            if "boundary_features" in feature_contributions:
                adjusted_contribution = feature_contributions["boundary_features"] * 0.4
                indoor_score -= (feature_contributions["boundary_features"] - adjusted_contribution)
                feature_contributions["boundary_features"] = adjusted_contribution

            if "ceiling_features" in feature_contributions:
                adjusted_contribution = feature_contributions["ceiling_features"] * 0.3
                indoor_score -= (feature_contributions["ceiling_features"] - adjusted_contribution)
                feature_contributions["ceiling_features"] = adjusted_contribution

            # 添加信息到診斷數據
            diagnostics["asian_street_detected"] = True
            diagnostics["asian_street_indicators"] = asian_street_indicators

        bedroom_indicators = 0

        # 1: 窗戶和牆壁形成的直角
        if features.get("brightness_uniformity", 0) > 0.6 and features.get("boundary_edge_score", 0) > 0.3:
            bedroom_indicators += 1.5  # 增加權重

        # 2: 天花板和光源
        if features.get("ceiling_uniformity", 0) > 0.5 and features.get("bright_spot_count", 0) > 0:
            bedroom_indicators += 2.5

        # 3: 良好對比度的牆壁顏色，適合臥房還有客廳
        if features.get("brightness_uniformity", 0) > 0.6 and features.get("avg_saturation", 0) < 100:
            bedroom_indicators += 1.5

        # 特殊的檢測 4: 檢測窗戶
        if features.get("boundary_edge_score", 0) > 0.25 and features.get("brightness_std", 0) > 40:
            bedroom_indicators += 1.5

        # 如果滿足足夠的家居指標，提高多點室內判斷分數
        if bedroom_indicators >= 3:
            # 增加家居環境評分
            home_env_score = 3
            indoor_score += home_env_score
            feature_contributions["home_environment_pattern"] = home_env_score
        elif bedroom_indicators >= 2:
            # 適度增加家居環境評分
            home_env_score = 2
            indoor_score += home_env_score
            feature_contributions["home_environment_pattern"] = home_env_score

        # 根據總分轉換為概率（使用sigmoid函數）
        indoor_probability = 1 / (1 + np.exp(-indoor_score * 0.22))

        # 判斷結果
        is_indoor = indoor_probability > 0.5

        return {
            "is_indoor": is_indoor,
            "indoor_probability": indoor_probability,
            "indoor_score": indoor_score,
            "feature_contributions": feature_contributions,
            "diagnostics": diagnostics
        }

    def _determine_lighting_conditions(self, features, is_indoor):
        """
        基於特徵和室內/室外判斷確定光照條件。

        Args:
            features: 特徵字典
            is_indoor: 是否是室內環境

        Returns:
            Dict: 光照條件分析結果
        """
        # 初始化
        time_of_day = "unknown"
        confidence = 0.5
        diagnostics = {}

        avg_brightness = features["avg_brightness"]
        dark_pixel_ratio = features["dark_pixel_ratio"]
        yellow_orange_ratio = features["yellow_orange_ratio"]
        blue_ratio = features["blue_ratio"]
        gray_ratio = features["gray_ratio"]

        # 基於室內/室外分別判斷
        if is_indoor:
            # 計算室內住宅自然光指標
            natural_window_light = 0

            # 檢查窗戶特徵和光線特性
            if (features.get("blue_ratio", 0) > 0.1 and
                features.get("sky_brightness", 0) > avg_brightness * 1.1):
                natural_window_light += 1

            # 檢查均勻柔和的光線分布
            if (features.get("brightness_uniformity", 0) > 0.65 and
                features.get("brightness_std", 0) < 70):
                natural_window_light += 1

            # 檢查暖色調比例
            if features.get("warm_ratio", 0) > 0.2:
                natural_window_light += 1

            # 家居環境指標
            home_env_score = features.get("home_environment_pattern", 0)
            if home_env_score > 1.5:
                natural_window_light += 1

            # 1. 室內明亮環境，可能有窗戶自然光
            if avg_brightness > 130:
                # 檢測自然光住宅空間 - 新增類型!
                if natural_window_light >= 2 and home_env_score > 1.5:
                    time_of_day = "indoor_residential_natural"  # 家裡的自然光類型
                    confidence = 0.8
                    diagnostics["reason"] = "Bright residential space with natural window lighting"
                # 檢查窗戶特徵 - 如果有明亮的窗戶且色調為藍
                elif features.get("blue_ratio", 0) > 0.1 and features.get("sky_brightness", 0) > 150:
                    time_of_day = "indoor_bright"
                    confidence = 0.8
                    diagnostics["reason"] = "Bright indoor scene with window light"
                else:
                    time_of_day = "indoor_bright"
                    confidence = 0.75
                    diagnostics["reason"] = "High brightness in indoor environment"
            # 2. 室內中等亮度環境
            elif avg_brightness > 100:
                time_of_day = "indoor_moderate"
                confidence = 0.7
                diagnostics["reason"] = "Moderate brightness in indoor environment"
            # 3. 室內低光照環境
            else:
                time_of_day = "indoor_dim"
                confidence = 0.65 + dark_pixel_ratio / 3
                diagnostics["reason"] = "Low brightness in indoor environment"

            # 1. 檢測設計師風格住宅，可以偵測到比較多種類的狀況
            designer_residential_score = 0
            # 檢測特色燈具
            if (features.get("circular_light_count", 0) > 0 or features.get("bright_spot_count", 0) > 2):
                designer_residential_score += 1
            # 檢測高品質均勻照明
            if features.get("brightness_uniformity", 0) > 0.7:
                designer_residential_score += 1
            # 檢測溫暖色調
            if features.get("warm_ratio", 0) > 0.3:
                designer_residential_score += 1
            # 檢測家居環境特徵
            if home_env_score > 1.5:
                designer_residential_score += 1

            if designer_residential_score >= 3 and home_env_score > 1.5:
                time_of_day = "indoor_designer_residential"
                confidence = 0.85
                diagnostics["special_case"] = "Designer residential lighting with decorative elements"

            # 2. 檢測餐廳/酒吧場景
            elif avg_brightness < 150 and yellow_orange_ratio > 0.2:
                if features["warm_ratio"] > 0.4:
                    time_of_day = "indoor_restaurant"
                    confidence = 0.65 + yellow_orange_ratio / 4
                    diagnostics["special_case"] = "Warm, yellow-orange lighting suggests restaurant/bar setting"

            # 3. 檢測商業照明空間
            elif avg_brightness > 120 and features["bright_spot_count"] > 4:
                # 增加商業照明判別的精確度
                commercial_score = 0
                # 多個亮點
                commercial_score += min(1.0, features["bright_spot_count"] * 0.05)
                # 不太可能是住宅的指標
                if features.get("home_environment_pattern", 0) < 1.5:
                    commercial_score += 0.5
                # 整體照明結構化布局
                if features.get("light_distribution_uniformity", 0) > 0.6:
                    commercial_score += 0.5

                if commercial_score > 0.6 and designer_residential_score < 3:
                    time_of_day = "indoor_commercial"
                    confidence = 0.7 + commercial_score / 5
                    diagnostics["special_case"] = "Multiple structured light sources suggest commercial lighting"
        else:
            # 室外場景判斷保持不變
            if avg_brightness < 90:  # 降低夜間判斷的亮度閾值
                # 檢測是否有車燈/街燈
                has_lights = features["bright_spot_count"] > 3

                if has_lights:
                    time_of_day = "night"
                    confidence = 0.8 + dark_pixel_ratio / 5
                    diagnostics["reason"] = "Low brightness with light sources detected"

                    # 檢查是否是霓虹燈場景
                    if yellow_orange_ratio > 0.15 and features["bright_spot_count"] > 5:
                        time_of_day = "neon_night"
                        confidence = 0.75 + yellow_orange_ratio / 3
                        diagnostics["special_case"] = "Multiple colorful light sources suggest neon lighting"
                else:
                    time_of_day = "night"
                    confidence = 0.7 + dark_pixel_ratio / 3
                    diagnostics["reason"] = "Low brightness outdoor scene"
            elif avg_brightness < 130 and yellow_orange_ratio > 0.2:
                time_of_day = "sunset/sunrise"
                confidence = 0.7 + yellow_orange_ratio / 3
                diagnostics["reason"] = "Moderate brightness with yellow-orange tones"
            elif avg_brightness > 150 and blue_ratio > 0.15:
                time_of_day = "day_clear"
                confidence = 0.7 + blue_ratio / 3
                diagnostics["reason"] = "High brightness with blue tones (likely sky)"
            elif avg_brightness > 130:
                time_of_day = "day_cloudy"
                confidence = 0.7 + gray_ratio / 3
                diagnostics["reason"] = "Good brightness with higher gray tones"
            else:
                # 默認判斷
                if yellow_orange_ratio > gray_ratio:
                    time_of_day = "sunset/sunrise"
                    confidence = 0.6 + yellow_orange_ratio / 3
                    diagnostics["reason"] = "Yellow-orange tones dominant"
                else:
                    time_of_day = "day_cloudy"
                    confidence = 0.6 + gray_ratio / 3
                    diagnostics["reason"] = "Gray tones dominant"

            # 檢查是否是特殊室外場景（如體育場）
            if avg_brightness > 120 and features["brightness_uniformity"] > 0.8:
                # 高亮度且非常均勻的光照可能是體育場燈光
                time_of_day = "stadium_lighting"
                confidence = 0.7
                diagnostics["special_case"] = "Uniform bright lighting suggests stadium/sports lighting"

            # 檢查是否是混合光照（如室內/室外過渡區）
            if 100 < avg_brightness < 150 and 0.1 < blue_ratio < 0.2:
                if features["gradient_ratio"] > 1.5:
                    time_of_day = "mixed_lighting"
                    confidence = 0.65
                    diagnostics["special_case"] = "Features suggest indoor-outdoor transition area"

        # 確保信心值在 0-1 範圍內
        confidence = min(0.95, max(0.5, confidence))

        if time_of_day in ["indoor_residential_natural", "indoor_designer_residential"] and hasattr(self, "config"):
            # 確保 LIGHTING_CONDITIONS 中有這些新類型的描述
            if time_of_day == "indoor_residential_natural":
                lightingType = {
                    "template_modifiers": {
                        "indoor_residential_natural": "naturally-lit residential"
                    },
                    "time_descriptions": {
                        "indoor_residential_natural": {
                            "general": "The scene is captured in a residential space with ample natural light from windows.",
                            "bright": "The residential space is brightly lit with natural daylight streaming through windows.",
                            "medium": "The home environment has good natural lighting providing a warm, inviting atmosphere.",
                            "dim": "The living space has soft natural light filtering through windows or openings."
                        }
                    }
                }
            elif time_of_day == "indoor_designer_residential":
                lightingType = {
                    "template_modifiers": {
                        "indoor_designer_residential": "designer-lit residential"
                    },
                    "time_descriptions": {
                        "indoor_designer_residential": {
                            "general": "The scene is captured in a residential space with carefully designed lighting elements.",
                            "bright": "The home features professionally designed lighting with decorative fixtures creating a bright atmosphere.",
                            "medium": "The residential interior showcases curated lighting design balancing form and function.",
                            "dim": "The living space has thoughtfully placed designer lighting creating an intimate ambiance."
                        }
                    }
                }

        return {
            "time_of_day": time_of_day,
            "confidence": confidence,
            "diagnostics": diagnostics
        }


    def _get_default_config(self):
        """
        返回優化版本的默認配置參數。
        """
        return {
            "indoor_outdoor_weights": {
                "blue_ratio": 0.6,
                "brightness_uniformity": 1.2,
                "gradient_ratio": 0.7,
                "bright_spots": 0.8,
                "color_tone": 0.5,
                "sky_brightness": 0.9,
                "brightness_variation": 0.7,
                "ceiling_features": 1.5,
                "light_features": 1.1,
                "boundary_features": 2.8,
                "street_features": 2,
                "building_features": 1.6
            },
            "include_diagnostics": True
        }
