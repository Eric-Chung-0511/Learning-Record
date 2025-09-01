
def get_css_styles():
    return """
        /* SBERT Natural Language Recommendation Styles */
        button#find-match-btn {
            background: linear-gradient(90deg, #ff5f6d 0%, #ffc371 100%) !important;
            border: none !important;
            border-radius: 30px !important;
            padding: 12px 24px !important;
            color: white !important;
            font-weight: bold !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            width: 100% !important;
            margin: 20px 0 !important;
            font-size: 1.1em !important;
        }
        button#find-match-btn:hover {
            background: linear-gradient(90deg, #ff4f5d 0%, #ffb361 100%) !important;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2) !important;
            transform: translateY(-2px) !important;
        }
        button#find-match-btn:active {
            transform: translateY(1px) !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        }
        #search-status {
            text-align: center;
            padding: 15px;
            font-size: 1.1em;
            color: #666;
            margin: 10px 0;
            border-radius: 8px;
            background: rgba(200, 200, 200, 0.1);
            transition: opacity 0.3s ease;
        }

        /* Natural Language Search Button Styles */
        button#find-by-description-btn {
            background: linear-gradient(90deg, #4299e1 0%, #48bb78 100%) !important;
            border: none !important;
            border-radius: 30px !important;
            padding: 12px 24px !important;
            color: white !important;
            font-weight: bold !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            width: 100% !important;
            margin: 20px 0 !important;
            font-size: 1.1em !important;
        }
        button#find-by-description-btn:hover {
            background: linear-gradient(90deg, #3182ce 0%, #38a169 100%) !important;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2) !important;
            transform: translateY(-2px) !important;
        }
        button#find-by-description-btn:active {
            background: linear-gradient(90deg, #2c5aa0 0%, #2f7d32 100%) !important;
            transform: translateY(0px) scale(0.98) !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        }

        /* Description Input Styles */
        .description-input textarea {
            border-radius: 10px !important;
            border: 2px solid #e2e8f0 !important;
            transition: all 0.3s ease !important;
        }
        .description-input textarea:focus {
            border-color: #4299e1 !important;
            box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.1) !important;
        }

        /* Force override any other styles */
        .gradio-button {
            position: relative !important;
            overflow: visible !important;
        }

        /* Progress bars for semantic recommendations */
        .progress {
            transition: all 0.3s ease-in-out;
            border-radius: 4px;
            height: 12px;
        }
        /* Ensure 100% progress bars fill completely */
        .progress[style*="100%"] {
            width: 100% !important;
        }
        .progress-bar {
            background-color: #f5f5f5;
            border-radius: 4px;
            overflow: hidden;
            position: relative;
            width: 100%;
            height: 12px;
        }
        .score-item {
            margin: 10px 0;
        }
        .percentage {
            margin-left: 8px;
            font-weight: 500;
        }

        /* History display with colored tags */
        .history-tag-criteria {
            background: rgba(72, 187, 120, 0.1);
            color: #48bb78;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .history-tag-description {
            background: rgba(66, 153, 225, 0.1);
            color: #4299e1;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }

        .dog-info-card {
            margin: 0 0 20px 0;
            padding: 0;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            overflow: hidden;
            transition: all 0.3s ease;
            background: white;
            border: 1px solid #e1e4e8;
            position: relative;
        }
        .dog-info-card:hover {
            box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        }
        .dog-info-card:before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 8px;
            background-color: inherit;
        }
        .dog-info-header {
            padding: 24px 28px;  /* 增加內距 */
            margin: 0;
            font-size: 22px;
            font-weight: bold;
            border-bottom: 1px solid #e1e4e8;
        }
        .dog-info-header {
            background-color: transparent;
        }
        .colored-border {
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 8px;
        }
        .dog-info-header {
            border-left-width: 8px;
            border-left-style: solid;
        }
        .breed-info {
            padding: 28px;  /* 增加整體內距 */
            line-height: 1.6;
            font-size: 1rem;
            border: none;
        }
        .section-title {
            font-size: 1.2em !important;
            font-weight: 700;
            color: #2c3e50;
            margin: 32px 0 20px 0;
            padding: 12px 0;
            border-bottom: 2px solid #e1e4e8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 8px;
            position: relative;
        }
        .section-header {
            color: #2c3e50;
            font-size: 1.15rem;
            font-weight: 600;
            margin: 20px 0 12px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .icon {
            font-size: 1.2em;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        .info-section, .care-section, .family-section {
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            margin-bottom: 28px;  /* 增加底部間距 */
            padding: 20px;  /* 增加內距 */
            background: #f8f9fa;
            border-radius: 12px;
            border: 1px solid #e1e4e8;  /* 添加邊框 */
        }
        .info-item {
            background: white;  /* 改為白色背景 */
            padding: 14px 18px;  /* 增加內距 */
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #e1e4e8;
            flex: 1 1 auto;
            min-width: 200px;
        }
        .label {
            color: #666;
            font-weight: 600;
            font-size: 1.1rem;
        }
        .value {
            color: #2c3e50;
            font-weight: 500;
            font-size: 1.1rem;
        }
        .temperament-section {
            background: #f8f9fa;
            padding: 20px;  /* 增加內距 */
            border-radius: 12px;
            margin-bottom: 28px;  /* 增加間距 */
            color: #444;
            border: 1px solid #e1e4e8;  /* 添加邊框 */
        }
        .description-section {
            background: #f8f9fa;
            padding: 24px;  /* 增加內距 */
            border-radius: 12px;
            margin: 28px 0;  /* 增加上下間距 */
            line-height: 1.8;
            color: #444;
            border: 1px solid #e1e4e8;  /* 添加邊框 */
            fontsize: 1.1rem;
        }
        .description-section p {
            margin: 0;
            padding: 0;
            text-align: justify;  /* 文字兩端對齊 */
            word-wrap: break-word;  /* 確保長單字會換行 */
            white-space: pre-line;  /* 保留換行但合併空白 */
            max-width: 100%;  /* 確保不會超出容器 */
        }
        .action-section {
            margin-top: 24px;
            text-align: center;
        }
        .akc-button,
        .breed-section .akc-link,
        .breed-option .akc-link {
            display: inline-flex;
            align-items: center;
            padding: 14px 28px;
            background: linear-gradient(145deg, #00509E, #003F7F);
            color: white;
            border-radius: 12px;  /* 增加圓角 */
            text-decoration: none;
            gap: 12px;  /* 增加圖標和文字間距 */
            transition: all 0.3s ease;
            font-weight: 600;
            font-size: 1.1em;
            box-shadow:
                0 2px 4px rgba(0,0,0,0.1),
                inset 0 1px 1px rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .akc-button:hover,
        .breed-section .akc-link:hover,
        .breed-option .akc-link:hover {
            background: linear-gradient(145deg, #003F7F, #00509E);
            transform: translateY(-2px);
            color: white;
            box-shadow:
                0 6px 12px rgba(0,0,0,0.2),
                inset 0 1px 1px rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .icon {
            font-size: 1.3em;
            filter: drop-shadow(0 1px 1px rgba(0,0,0,0.2));
        }
        .warning-message {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #ff3b30;
            font-weight: 500;
            margin: 0;
            padding: 16px;
            background: #fff5f5;
            border-radius: 8px;
        }
        .model-uncertainty-note {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 16px;
            background-color: #f8f9fa;
            margin-bottom: 20px;
            color: #495057;
            border-radius: 4px;
        }
        .breeds-list {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .breed-option {
            background: white;
            border: 1px solid #e1e4e8;
            border-radius: 8px;
            overflow: hidden;
        }
        .breed-header {
            display: flex;
            align-items: center;
            padding: 16px;
            background: #f8f9fa;
            gap: 12px;
            border-bottom: 1px solid #e1e4e8;
        }
        .option-number {
            font-weight: 600;
            color: #666;
            padding: 4px 8px;
            background: #e1e4e8;
            border-radius: 4px;
        }

        .option-item {
            padding: 15px;
            margin-bottom: 10px;
            border-bottom: 1px solid #ddd;
        }

        .option-item:last-child {
            border-bottom: none; /* 最後一個選項去除邊線 */
        }

        .breed-name {
            font-size: 1.2em !important;
            font-weight: bold;
            color: #2c3e50;
            flex-grow: 1;
        }
        .confidence-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 500;
        }
        .breed-content {
            padding: 20px;
        }
        .breed-content li {
            margin-bottom: 8px;
            display: flex;
            align-items: flex-start;  /* 改為頂部對齊 */
            gap: 8px;
            flex-wrap: wrap;  /* 允許內容換行 */
        }
        .breed-content li strong {
            flex: 0 0 auto;  /* 不讓標題縮放 */
            min-width: 100px;  /* 給標題一個固定最小寬度 */
        }
        ul {
            padding-left: 0;
            margin: 0;
            list-style-type: none;
        }
        li {
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .action-section {
            margin-top: 20px;
            padding: 15px;
            text-align: center;
            border-top: 1px solid #dee2e6;
        }
        .akc-button {
            display: inline-block;
            padding: 12px 24px;
            background-color: #007bff;
            color: white !important;
            text-decoration: none;
            border-radius: 5px;
            font-weight: 500;
            transition: background-color 0.3s;
        }
        .akc-button:hover {
            background-color: #0056b3;
            text-decoration: none;
        }
        .akc-button .icon {
            margin-right: 8px;
        }
        .akc-link {
            color: white;
            text-decoration: none;
            font-weight: 600;
            font-size: 1.1em;
            transition: all 0.3s ease;
        }
        .akc-link:hover {
            text-decoration: underline;
            color: #D3E3F0;
        }
        .tooltip {
            position: relative;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            cursor: help;
        }
        .tooltip .tooltip-icon {
            display: inline-block;
            width: 18px;
            height: 18px;
            background: linear-gradient(135deg, #64748b, #475569);
            color: white;
            border-radius: 50%;
            text-align: center;
            line-height: 18px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 8px;
            cursor: help;
            box-shadow: 0 2px 4px rgba(100, 116, 139, 0.3);
            transition: all 0.2s ease;
        }
        .tooltip .tooltip-icon:hover {
            background: linear-gradient(135deg, #475569, #334155);
            transform: scale(1.1);
            box-shadow: 0 3px 6px rgba(100, 116, 139, 0.4);
        }
        .tooltip .tooltip-text {
            visibility: hidden;
            width: 280px;
            background-color: rgba(44, 62, 80, 0.95);
            color: white;
            text-align: left;
            border-radius: 8px;
            padding: 12px 15px;
            position: absolute;
            z-index: 1000;
            bottom: calc(100% + 15px);
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: all 0.3s ease;
            font-size: 14px;
            line-height: 1.4;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
            white-space: normal;
            margin-bottom: 10px;
        }
        .tooltip.tooltip-left .tooltip-text {
            left: 0;
            transform: translateX(0);
        }
        .tooltip.tooltip-right .tooltip-text {
            left: auto;
            right: 0;
            transform: translateX(0);
        }
        .tooltip-text strong {
            color: white !important;
            background-color: transparent !important;
            display: block;  /* 讓標題獨立一行 */
            margin-bottom: 2px;  /* 增加標題下方間距 */
            padding-bottom: 2px; /* 加入小間距 */
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        .tooltip-text {
            font-size: 13px;  /* 稍微縮小字體 */
        }
        /* 調整列表符號和文字的間距 */
        .tooltip-text ul {
            margin: 0;
            padding-left: 15px;  /* 減少列表符號的縮進 */
        }
        .tooltip-text li {
            margin-bottom: 1px;  /* 減少列表項目間的間距 */
        }
        .tooltip-text br {
            line-height: 1.2;  /* 減少行距 */
        }
        .tooltip .tooltip-text::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            border-width: 8px;
            border-style: solid;
            border-color: rgba(44, 62, 80, 0.95) transparent transparent transparent;
        }
        .tooltip-left .tooltip-text::after {
            left: 20%;
        }
        /* 右側箭頭 */
        .tooltip-right .tooltip-text::after {
            left: 80%;
        }
        .tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
        .uncertainty-mode .tooltip .tooltip-text {
            position: absolute;
            left: 100%;
            bottom: auto;
            top: 50%;
            transform: translateY(-50%);
            margin-left: 10px;
            z-index: 1000;  /* 確保提示框在最上層 */
        }
        .uncertainty-mode .tooltip .tooltip-text::after {
            content: "";
            position: absolute;
            top: 50%;
            right: 100%;
            transform: translateY(-50%);
            border-width: 5px;
            border-style: solid;
            border-color: transparent rgba(44, 62, 80, 0.95) transparent transparent;
        }
        .uncertainty-mode .breed-content {
            font-size: 1rem !important;  /* 增加字體大小 */
        }
        .description-section,
        .description-section p,
        .temperament-section,
        .temperament-section .value,
        .info-item,
        .info-item .value,
        .breed-content {
            font-size: 1rem !important;  /* 使用 !important 確保覆蓋其他樣式 */
        }
        .recommendation-card {
            margin-bottom: 40px;
        }
        .compatibility-scores {
            background: #f8f9fa;
            padding: 24px;
            border-radius: 12px;
            margin: 20px 0;
        }
        .score-item {
            margin: 15px 0;
        }
        .progress-bar {
            height: 12px;
            background-color: #e9ecef;
            border-radius: 6px;
            overflow: hidden;
            margin: 8px 0;
            width: 100%;
            position: relative;
        }
        .progress {
            height: 100%;
            border-radius: 6px;
            transition: width 0.6s ease;
            min-width: 0;
        }
        .progress[style*="width: 100%"] {
            width: 100% !important;
            border-radius: 6px;
        }
        .percentage {
            float: right;
            color: #34C759;
            font-weight: 600;
        }
        .breed-details-section {
            margin: 30px 0;
        }
        .subsection-title {
            font-size: 1.2em;
            color: #2c3e50;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .details-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #e1e4e8;
        }
        .detail-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e1e4e8;
        }
        .description-text {
            line-height: 1.8;
            color: #444;
            margin: 0;
            padding: 24px 30px;  /* 調整內部間距，從 20px 改為 24px 30px */
            background: #f8f9fa;
            border-radius: 12px;
            border: 1px solid #e1e4e8;
            text-align: justify;  /* 添加文字對齊 */
            word-wrap: break-word;  /* 確保長文字會換行 */
            word-spacing: 1px;  /* 加入字間距 */
        }
        /* 工具提示改進 */
        .tooltip {
            position: relative;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            cursor: help;
            padding: 5px 0;
        }
        .tooltip .tooltip-text {
            visibility: hidden;
            width: 280px;
            background-color: rgba(44, 62, 80, 0.95);
            color: white;
            text-align: left;
            border-radius: 8px;
            padding: 12px 15px;
            position: absolute;
            z-index: 1000;
            bottom: calc(100% + 15px);
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: all 0.3s ease;
            font-size: 14px;
            line-height: 1.4;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            white-space: normal;
        }
        .tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
        .score-badge {
            background-color: #34C759;
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            margin-left: 10px;
            font-weight: 500;
            box-shadow: 0 2px 4px rgba(52, 199, 89, 0.2);
        }
        .bonus-score .tooltip-text {
            width: 250px;
            line-height: 1.4;
            padding: 10px;
        }
        .bonus-score .progress {
            background: linear-gradient(90deg, #48bb78, #68d391);
        }
        .health-section {
            margin: 25px 0;
            padding: 24px;
            background-color: #f8f9fb;
            border-radius: 12px;
            border: 1px solid #e1e4e8;
        }
        .health-section .subsection-title {
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #2c3e50;
        }
        .health-info {
            background-color: white;
            padding: 24px;
            border-radius: 8px;
            margin: 15px 0;
            border: 1px solid #e1e4e8;
        }
        .health-details {
            font-size: 1.1rem;
            line-height: 1.6;
        }
        .health-details h4 {
            color: #2c3e50;
            font-size: 1.15rem;
            font-weight: 600;
            margin: 20px 0 15px 0;
        }
        .health-details h4:first-child {
            margin-top: 0;
        }
        .health-details ul {
            list-style-type: none;
            padding-left: 0;
            margin: 0 0 25px 0;
        }
        .health-details ul li {
            margin-bottom: 12px;
            padding-left: 20px;
            position: relative;
        }
        .health-details ul li:before {
            content: "•";
            position: absolute;
            left: 0;
            color: #2c3e50;
        }
        .health-item:before {
            content: "•";
            color: #dc3545;
            font-weight: bold;
        }
        .health-item.screening:before {
            color: #28a745;
        }
        /* 區塊間距 */
        .health-block, .noise-block {
            margin-bottom: 24px;
        }
        .health-disclaimer {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #e1e4e8;
        }
        .health-disclaimer p {
            margin: 6px 0;
            padding-left: 20px;
            position: relative;
            color: #888;  /* 統一設定灰色 */
            font-size: 0.95rem;
            line-height: 1.5;
            font-style: italic;
        }
        .health-disclaimer p:before {
            content: "›";
            position: absolute;
            left: 0;
            color: #999;
            font-style: normal;
            font-weight: 500;
        }
        .health-disclaimer p:first-child {
            font-style: normal;  /* 取消斜體 */
            font-weight: 500;    /* 加粗 */
            color: #666;         /* 稍深的灰色 */
        }
        .health-disclaimer p span,
        .health-disclaimer p strong,
        .health-disclaimer p em {
            color: inherit;
        }
        .health-list li:before {
            content: "•";
            color: #dc3545;
        }
        .history-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .history-entry {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .history-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        .timestamp {
            color: #666;
            font-size: 0.9em;
        }
        .delete-btn {
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1.2em;
            padding: 5px;
        }
        .delete-btn:hover {
            color: #dc3545;
        }
        .search-params ul {
            list-style: none;
            padding-left: 20px;
        }
        .search-params li {
            margin: 5px 0;
            color: #555;
        }
        .top-results ol {
            padding-left: 25px;
        }
        .top-results li {
            margin: 5px 0;
            color: #333;
        }
        .breed-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            margin: 8px 0;
            background-color: white;
            border-radius: 6px;
            border: 1px solid #e1e4e8;
            transition: all 0.2s ease;
        }
        .breed-item:hover {
            transform: translateX(5px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .breed-rank {
            font-weight: 600;
            color: #666;
            margin-right: 12px;
            min-width: 30px;
        }
        .breed-name {
            flex: 1;
            font-weight: 500;
            color: #2c3e50;
            padding: 0 12px;
        }
        .breed-score {
            font-weight: 600;
            color: #34C759;
            padding: 4px 8px;
            border-radius: 20px;
            background-color: rgba(52, 199, 89, 0.1);
            min-width: 70px;
            text-align: center;
        }
        .history-entry {
            background-color: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border: 1px solid #e1e4e8;
        }
        .history-header {
            margin-bottom: 15px;
            padding-bottom: 12px;
            border-bottom: 1px solid #e1e4e8;
        }
        .history-header .timestamp {
            color: #666;
            font-size: 0.9em;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        h4 {
            color: #2c3e50;
            font-size: 1.15rem;
            font-weight: 600;
            margin: 20px 0 12px 0;
        }
        .params-list ul {
            list-style: none;
            padding-left: 0;
            margin: 10px 0;
        }
        .params-list li {
            margin: 8px 0;
            color: #4a5568;
            display: flex;
            align-items: center;
        }
        .empty-history {
            text-align: center;
            padding: 40px 20px;
            color: #666;
            font-size: 1.1em;
            background-color: #f8f9fa;
            border-radius: 12px;
            border: 1px dashed #e1e4e8;
            margin: 20px 0;
        }
        .noise-section {
            margin: 25px 0;
            padding: 24px;
            background-color: #f8f9fb;
            border-radius: 12px;
            border: 1px solid #e1e4e8;
        }
        .noise-info {
            background-color: white;
            padding: 24px;
            border-radius: 8px;
            margin: 15px 0;
            border: 1px solid #e1e4e8;
        }
        .noise-details {
            font-size: 1.1rem;
            line-height: 1.6;
        }
        .noise-level {
            margin-bottom: 20px;
            padding: 10px 15px;
            background: #f8f9fa;
            border-radius: 6px;
            font-weight: 500;
        }
        .noise-level-block {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .noise-level-display {
            background: #f8f9fa;
            padding: 16px;
            border-radius: 8px;
            margin: 16px 0;
        }
        .level-indicator {
            background: white;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid #e1e4e8;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .level-text {
            font-weight: 500;
            color: #2c3e50;
        }
        .level-bars {
            display: flex;
            gap: 4px;
        }
        .level-bars .bar {
            width: 4px;
            height: 16px;
            background: #e9ecef;
            border-radius: 2px;
        }
        .level-indicator.low .bar:nth-child(1) {
            background: #4CAF50;
        }
        .level-indicator.medium .bar:nth-child(1),
        .level-indicator.medium .bar:nth-child(2) {
            background: #FFA726;
        }
        .level-indicator.high .bar {
            background: #EF5350;
        }
        .feature-list, .health-list, .screening-list {
            list-style: none;
            padding: 0;
            margin: 16px 0;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 12px;
        }
        .feature-list li, .health-list li, .screening-list li {
            background: white;
            padding: 12px 16px;
            border-radius: 6px;
            border: 1px solid #e1e4e8;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.95rem;
        }
        .feature-list li:before {
            content: "•";
            color: #2c3e50;
        }
        .noise-notes {
            font-family: inherit;
            white-space: pre-wrap;
            margin: 15px 0;
            padding: 0;
            background: transparent;
            border: none;
            font-size: 1.1rem;
            line-height: 1.6;
            color: #333;
        }
        .characteristics-block, .health-considerations, .health-screenings {
            margin-bottom: 24px;
        }
        .characteristics-block h4, .health-considerations h4, .health-screenings h4 {
            color: #2c3e50;
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 12px;
        }
        .characteristics-list,
        .triggers-list,
        .health-considerations-list,
        .health-screenings-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin: 16px 0;
        }
        .noise-details, .health-details {
            font-size: 1.1rem;
            line-height: 1.6;
        }
        .noise-details ul, .health-details ul {
            list-style: none;
            padding-left: 0;
            margin: 0 0 20px 0;
        }
        .noise-details li, .health-details li {
            padding-left: 20px;
            position: relative;
            margin-bottom: 10px;
            line-height: 1.5;
        }
        .noise-details li:before, .health-details li:before {
            content: "•";
            position: absolute;
            left: 0;
            color: #666;
        }
        .noise-section, .health-section {
            margin: 25px 0;
            padding: 24px;
            background-color: #f8f9fb;
            border-radius: 12px;
            border: 1px solid #e1e4e8;
        }
        .noise-info, .health-info {
            background-color: white;
            padding: 24px;
            border-radius: 8px;
            margin: 15px 0;
            border: 1px solid #e1e4e8;
        }
        .breed-info .description-tooltip {
            position: relative;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            cursor: help;
        }
        .description-tooltip .tooltip-icon {
            font-size: 14px;
            color: #666;
            margin-left: 4px;
            cursor: help;
        }
        .description-tooltip .tooltip-text {
            visibility: hidden;
            width: 280px;
            background-color: rgba(44, 62, 80, 0.95);
            color: white;
            text-align: left;
            border-radius: 8px;
            padding: 12px 15px;
            position: absolute;
            z-index: 1000;
            bottom: calc(100% + 15px);
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: all 0.3s ease;
            font-size: 14px;
            line-height: 1.4;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            white-space: normal;
        }
        .description-tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
        .description-tooltip .tooltip-text::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            border-width: 8px;
            border-style: solid;
            border-color: rgba(44, 62, 80, 0.95) transparent transparent transparent;
        }
        .description-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
        }
        .description-header h3 {
            margin: 0;
            font-size: 1.2em;
            color: #2c3e50;
        }
        .screening-list li:before {
            content: "•";
            color: #28a745;
        }
        .noise-disclaimer, .health-disclaimer {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #e1e4e8;
            color: #666;
        }
        .noise-disclaimer p, .health-disclaimer p {
            margin: 8px 0;
            padding-left: 20px;
            position: relative;
        }
        .noise-disclaimer p:before, .health-disclaimer p:before {
            content: "›";
            position: absolute;
            left: 0;
            color: #999;
        }
        .disclaimer-text {
            margin: 8px 0;
            padding-left: 20px;
            position: relative;
            font-size: 0.95rem;
            line-height: 1.5;
            font-style: italic;
            color: #888;
        }
        .disclaimer-text:before {
            content: "›";
            position: absolute;
            left: 0;
            color: #999;
            font-style: normal;
            font-weight: 500;
        }
        .list-item {
            background: white;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid #e1e4e8;
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 4px 0;
            font-size: 0.95rem;
            color: #2c3e50;
        }
        .source-text {
            font-style: normal !important;
            font-weight: 500 !important;
            color: #666 !important;
        }
        .health-grid, .noise-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
            margin: 16px 0;
        }
        .health-item, .noise-item {
            background: white;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid #e1e4e8;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
        }
        .health-item:hover, .noise-item:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        @media (max-width: 768px) {
            /* 在小螢幕上改為單列顯示 */
            .health-grid, .noise-grid {
                grid-template-columns: 1fr;
            }

            /* 減少內邊距 */
            .health-section, .noise-section {
                padding: 16px;
            }

            /* 調整字體大小 */
            .section-header {
                font-size: 1rem;
            }

            /* 調整項目內邊距 */
            .health-item, .noise-item {
                padding: 10px 14px;
            }
        }

        /* 較小的手機螢幕 */
        @media (max-width: 480px) {
            .health-grid, .noise-grid {
                gap: 8px;
            }

            .health-item, .noise-item {
                padding: 8px 12px;
                font-size: 0.9rem;
            }
        }

        .expandable-section {
            margin-top: 1rem;
        }

        .expand-header {
            cursor: pointer;
            padding: 0.5rem;
            background-color: #f3f4f6;
            border-radius: 0.375rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .expand-header:hover {
            background-color: #e5e7eb;
        }

        .expand-icon {
            transition: transform 0.2s;
        }

        .expandable-section[open] .expand-icon {
            transform: rotate(180deg);
        }

        .expanded-content {
            padding: 1rem;
            background-color: #ffffff;
            border-radius: 0.375rem;
            margin-top: 0.5rem;
        }

        .info-cards > div:first-child .tooltip .tooltip-text,
        .info-cards > div:nth-child(3n+1) .tooltip .tooltip-text {
            left: calc(100% + 20px); /* 向右移動更多 */
        }

        .info-cards > div:first-child .tooltip .tooltip-text::after,
        .info-cards > div:nth-child(3n+1) .tooltip .tooltip-text::after {
            right: calc(100% - 2px); /* 向右移動箭頭 */
        }

        .section-header h3 .tooltip .tooltip-text {
            left: calc(100% + 20px); /* 向右移動更多 */
        }

        .section-header h3 .tooltip .tooltip-text::after {
            right: calc(100% - 2px); /* 向右移動箭頭 */

        }

        .analysis-container {
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .metric-card {
            padding: 20px;
            background: #f8fafc;
            border-radius: 8px;
            text-align: center;
        }

        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2563eb;
        }

        .metric-details {
            padding: 15px;
            background: #f8fafc;
            border-radius: 8px;
            margin: 10px 0;
        }

        .metric-details h3 {
            color: #1e40af;
            margin-bottom: 10px;
        }

        .metric-details ul {
            list-style-type: none;
            padding: 0;
        }

        .metric-details li {
            margin: 5px 0;
            color: #4b5563;
        }

        """
