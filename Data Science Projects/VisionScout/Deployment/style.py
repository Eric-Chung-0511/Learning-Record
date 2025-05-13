
class Style:

    @staticmethod
    def get_css():

        css = """
        /* Base styles and typography */
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #f0f9ff, #e1f5fe);
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            min-height: 100vh;
        }

        /* Typography improvements */
        h1, h2, h3, h4, h5, h6, p, span, div, label, button {
            font-family: Arial, sans-serif;
        }

        /* Container styling */
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
            padding: 1rem;
            width: 100%;
        }

        /* Header area styling with gradient background */
        .app-header {
            text-align: center;
            margin-bottom: 2rem;
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            width: 100%;
        }

        .app-title {
            color: #2D3748;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #38b2ac, #4299e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }

        .app-subtitle {
            color: #4A5568;
            font-size: 1.2rem;
            font-weight: normal;
            margin-top: 0.25rem;
        }

        .app-divider {
            width: 80px;
            height: 3px;
            background: linear-gradient(90deg, #38b2ac, #4299e1);
            margin: 1rem auto;
        }

        /* Panel styling - gradient background */
        .input-panel, .output-panel {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            margin: 0 auto 1rem auto;
        }

        /* 修改輸出面板確保內容能夠完整顯示 */
        .output-panel {
            display: flex;
            flex-direction: column;
            width: 100%;
            padding: 0 !important;
        }

        /* 確保輸出面板內的元素寬度可以適應面板 */
        .output-panel > * {
            width: 100%;
        }

        /* How-to-use section with gradient background */
        .how-to-use {
            background: linear-gradient(135deg, #f8fafc, #e8f4fd);
            border-radius: 10px;
            padding: 1.5rem;
            margin-top: 1rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            color: #2d3748;
        }

        /* Detection button styling */
        .detect-btn {
            background: linear-gradient(90deg, #38b2ac, #4299e1) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            transition: transform 0.3s, box-shadow 0.3s !important;
            font-weight: bold !important;
            letter-spacing: 0.5px !important;
            padding: 0.75rem 1.5rem !important;
            width: 100%;
            margin: 1rem auto !important;
            font-family: Arial, sans-serif !important;
        }

        .detect-btn:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
        }

        .detect-btn:active {
            transform: translateY(1px) !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        }

        /* JSON display improvements */
        .json-display {
            width: 98% !important;
            margin: 0.5rem auto 1.5rem auto !important;
            padding: 1rem !important;
            border-radius: 8px !important;
            background-color: white !important;
            border: 1px solid #E2E8F0 !important;
            box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        }

        .json-key {
            color: #e53e3e;
        }

        .json-value {
            color: #2b6cb0;
        }

        .json-string {
            color: #38a169;
        }

        /* Chart/plot styling improvements */
        .plot-container {
            background: white;
            border-radius: 8px;
            padding: 0.5rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        }

        /* Larger font for plots */
        .plot-container text {
            font-family: Arial, sans-serif !important;
            font-size: 14px !important;
        }

        /* Title styling for charts */
        .plot-title {
            font-family: Arial, sans-serif !important;
            font-size: 16px !important;
            font-weight: bold !important;
        }

        /* Tab styling with subtle gradient */
        .tabs {
            width: 100%;
            display: flex;
            justify-content: center;
        }

        .tabs > div:first-child {
            background: linear-gradient(to right, #f8fafc, #e8f4fd) !important;
            border-radius: 8px 8px 0 0;
        }

        /* Tab content styling - 確保內容區域有足夠寬度 */
        .tab-content {
            width: 100% !important;
            box-sizing: border-box !important;
            padding: 0 !important;
        }

        /* Footer styling with gradient background */
        .footer {
            text-align: center;
            margin-top: 2rem;
            font-size: 0.9rem;
            color: #4A5568;
            padding: 1rem;
            background: linear-gradient(135deg, #f8f9fa, #e1effe);
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            width: 100%;
        }

        /* Ensure centering works for all elements */
        .container, .gr-container, .gr-row, .gr-col {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 100%;
        }

        /* 統一文本框樣式，確保寬度一致 */
        .gr-textbox, .gr-textarea, .gr-text-input {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 100% !important;
            box-sizing: border-box !important;
        }

        /* 確保文本區域可以適應容器寬度 */
        textarea.gr-textarea, .gr-textbox textarea, .gr-text-input textarea {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 100% !important;
            box-sizing: border-box !important;
            padding: 16px !important;
            font-family: 'Arial', sans-serif !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            word-break: normal !important;
        }

        /* 特別針對場景描述文本框樣式增強 */
        #scene-description-text, #detection-details {
            width: 100% !important;
            min-width: 100% !important;
            box-sizing: border-box !important;
            padding: 16px !important;
            line-height: 1.8 !important;
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            border-radius: 8px !important;
            min-height: 250px !important;
            overflow-y: auto !important;
            border: 1px solid #e2e8f0 !important;
            background-color: white !important;
            display: block !important;
            font-family: 'Arial', sans-serif !important;
            font-size: 14px !important;
            margin: 0 !important;
        }

        /* 針對場景描述容器的樣式 */
        .scene-description-container {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        /* Scene Understanding Tab 特定樣式 */
        .scene-understanding-tab .result-details-box {
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
            width: 100% !important;
            box-sizing: border-box !important;
            padding: 0 !important;
        }

        /* 場景分析描述區域樣式 */
        .scene-description-box {
            background-color: #f8f9fa !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            padding: 15px !important;
            margin: 10px 0 20px 0 !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
            font-family: Arial, sans-serif !important;
            line-height: 1.7 !important;
            color: #2D3748 !important;
            font-size: 16px !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }

        #scene_analysis_description_text {
            background-color: #f0f0f0 !important; /* 淺灰色背景 */
            padding: 15px !important;             /* 內邊距，讓文字和邊框有點空間 */
            border-radius: 8px !important;        /* 圓角 */
            margin: 10px 0 20px 0 !important;     /* 其他元素的間距，特別是上下的part */
            display: block !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }

        #scene_analysis_description_text p {
            margin: 0 !important;
            color: #2D3748 !important; /* 確保文字顏色 */
            font-family: Arial, sans-serif !important;
            font-size: 16px !important; /* 你可以調整文字大小 */
            line-height: 1.7 !important;
        }

        /* 結果容器樣式 */
        .result-container {
            width: 100% !important;
            padding: 1rem !important;
            border-radius: 8px !important;
            border: 1px solid #E2E8F0 !important;
            margin-bottom: 1.5rem !important;
            background-color: #F8FAFC !important;
            box-sizing: border-box !important;
        }

        /* 結果文本框的樣式 */
        .wide-result-text {
            width: 100% !important;
            min-width: 100% !important;
            box-sizing: border-box !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        /* 片段標題樣式 */
        .section-heading {
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            color: #2D3748 !important;
            margin: 1rem auto !important;
            padding: 0.75rem 1rem !important;
            background: linear-gradient(to right, #e6f3fc, #f0f9ff) !important;
            border-radius: 8px !important;
            width: 98% !important;
            display: inline-block !important;
            box-sizing: border-box !important;
            text-align: center !important;
            overflow: visible !important;
            line-height: 1.5 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        }

        /* JSON 顯示區域樣式 */
        .json-box {
            width: 100% !important;
            min-height: 200px !important;
            overflow-y: auto !important;
            background: white !important;
            padding: 1rem !important;
            border-radius: 8px !important;
            box-shadow: inset 0 0 6px rgba(0, 0, 0, 0.1) !important;
            font-family: monospace !important;
            box-sizing: border-box !important;
        }

        /* 欄佈局調整 */
        .plot-column, .stats-column {
            display: flex;
            flex-direction: column;
            padding: 1rem;
            box-sizing: border-box !important;
            width: 100% !important;
        }

        /* statistics plot */
        .large-plot-container {
            width: 100% !important;
            min-height: 400px !important;
            box-sizing: border-box !important;
        }

        /* 增強 JSON 顯示 */
        .enhanced-json-display {
            background: white !important;
            border-radius: 8px !important;
            padding: 1rem !important;
            box-shadow: inset 0 0 6px rgba(0, 0, 0, 0.1) !important;
            width: 100% !important;
            min-height: 300px !important;
            max-height: 500px !important;
            overflow-y: auto !important;
            font-family: monospace !important;
            box-sizing: border-box !important;
        }

        /* 確保全寬元素真正占滿整個寬度 */
        .full-width-element {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }

        /* Video summary HTML 容器與內容樣式 */
        #video-summary-html-output {
            width: 100% !important;
            box-sizing: border-box !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        .video-summary-content-wrapper {
            width: 100% !important;
            padding: 16px !important;
            line-height: 1.8 !important;
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            border-radius: 8px !important;
            min-height: 250px !important;
            max-height: 600px !important;
            overflow-y: auto !important;
            border: 1px solid #e2e8f0 !important;
            background-color: white !important;
            display: block !important;
            font-family: 'Arial', sans-serif !important;
            font-size: 14px !important;
            margin: 0 !important;
        }

        .video-summary-content-wrapper pre {
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            margin: 0 !important;
            padding: 0 !important;
            font-family: 'Arial', sans-serif !important;
            font-size: 14px !important;
            line-height: 1.8 !important;
            color: #2D3748 !important;
        }

        /* 視頻結果面板相關樣式 */
        .video-result-panel {
            padding: 1rem !important;
            background: white !important;
            border-radius: 10px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        }

        .video-output-container {
            width: 100% !important;
            margin-bottom: 1.5rem !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        }

        /* 視頻統計資料顯示增強 */
        .video-stats-display {
            background: white !important;
            border-radius: 8px !important;
            padding: 1rem !important;
            box-shadow: inset 0 0 6px rgba(0, 0, 0, 0.1) !important;
            width: 100% !important;
            min-height: 200px !important;
            max-height: 400px !important;
            overflow-y: auto !important;
            font-family: monospace !important;
            box-sizing: border-box !important;
            color: #2D3748 !important;
        }

        .custom-video-url-input {
            width: 100% !important;
        }

        .custom-video-url-input textarea {
            width: 100% !important;
            min-height: 120px !important;
            padding: 15px !important;
            font-size: 16px !important;
            line-height: 1.6 !important;
            background-color: #F7FAFC !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
            border: 2px solid #CBD5E0 !important;
            border-radius: 8px !important;
        }

        .custom-video-url-input textarea:focus {
            border-color: #4299E1 !important;
            box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.2) !important;
        }

        /* 輸入框容器100%寬度 */
        .custom-video-url-input > div {
            width: 100% !important;
            max-width: 100% !important;
        }

        /* LLM 增強描述樣式 */
        #llm_enhanced_description_text {
            padding: 15px !important;
            background-color: #ffffff !important;
            border-radius: 8px !important;
            border: 1px solid #e2e8f0 !important;
            margin-bottom: 20px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
            font-family: Arial, sans-serif !important;
            line-height: 1.7 !important;
            color: #2D3748 !important;
            font-size: 16px !important;
            width: 100% !important;
            box-sizing: border-box !important;
            min-height: 200px !important;
        }

        /* 原始描述折疊區域樣式 */
        #original_scene_analysis_accordion {
            margin-top: 10px !important;
            margin-bottom: 20px !important;
            background-color: #f8f9fa !important;
            border-radius: 8px !important;
            border: 1px solid #e2e8f0 !important;
        }

        /* 確保折疊區域內容與頁面樣式協調 */
        #original_scene_analysis_accordion > div:nth-child(2) {
            padding: 15px !important;
        }

        /* 動畫效果, 增加互動感 */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .video-result-panel > * {
            animation: fadeIn 0.5s ease-in-out;
        }

        /* 響應式調整 */
        @media (max-width: 768px) {
            .app-title {
                font-size: 2rem;
            }

            .app-subtitle {
                font-size: 1rem;
            }

            .gradio-container {
                padding: 0.5rem;
            }

            /* 在小螢幕上調整文本區域的高度 */
            #scene-description-text, #detection-details {
                min-height: 150px !important;
            }
        }

        """
        return css
