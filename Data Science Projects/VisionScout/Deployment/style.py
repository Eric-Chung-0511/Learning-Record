class Style:
    @staticmethod
    def get_css():
        """Return the application's CSS styles with improved aesthetics"""
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
            margin: 0 auto;
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
        
        /* Section heading styling with gradient background */
        .section-heading {
            font-size: 1.25rem;
            font-weight: 600;
            color: #2D3748;
            margin-bottom: 1rem;
            margin-top: 0.5rem;
            text-align: center;
            padding: 0.8rem;
            background: linear-gradient(to right, #e6f3fc, #f0f9ff);
            border-radius: 8px;
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
        .json-display pre {
            background: #f8fafc;
            border-radius: 6px;
            padding: 1rem;
            font-family: 'Consolas', 'Monaco', monospace;
            white-space: pre-wrap;
            max-height: 500px;
            overflow-y: auto;
            box-shadow: inset 0 0 4px rgba(0, 0, 0, 0.1);
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
        
        /* 結果文本框的改進樣式 */
        #detection-details, .wide-result-text {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }
        
        .wide-result-text textarea {
            width: 100% !important;
            min-width: 600px !important;
            font-family: 'Arial', sans-serif !important;
            font-size: 14px !important;
            line-height: 1.5 !important;  /* 減少行間距 */
            padding: 16px !important;
            white-space: pre-wrap !important;
            background-color: #f8f9fa !important;
            border-radius: 8px !important;
            min-height: 300px !important;
            resize: none !important;
            overflow-y: auto !important;
            border: 1px solid #e2e8f0 !important;
            display: block !important;
        }
        
        /* 結果詳情面板樣式 - 加入漸層背景 */
        .result-details-box {
            width: 100% !important;
            margin-top: 1.5rem;
            background: linear-gradient(135deg, #f8fafc, #e8f4fd);
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        /* 確保結果詳情面板內的元素寬度可以適應面板 */
        .result-details-box > * {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        /* 確保文本區域不會被限制寬度 */
        .result-details-box .gr-text-input {
            width: 100% !important;
            max-width: none !important;
        }
        
        /* 輸出面板內容的布局調整 */
        .output-panel {
            display: flex;
            flex-direction: column;
            width: 100%;
            padding: 0 !important;
        }
        
        /* 確保結果面板內的元素寬度可以適應面板 */
        .output-panel > * {
            width: 100%;
        }
        
        /* 改善統計面板列佈局 */
        .plot-column, .stats-column {
            display: flex;
            flex-direction: column;
            padding: 1rem;
        }
        
        /* Responsive adjustments */
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
        }
        """
        return css
