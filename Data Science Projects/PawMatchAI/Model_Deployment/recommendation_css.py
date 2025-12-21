# recommendation_css.py
"""
CSS 樣式模組 - 專門處理推薦結果的 CSS 樣式
將所有 CSS 定義集中管理，提高可維護性
"""

# Description Search (Find by Description) 模式的 CSS 樣式
DESCRIPTION_SEARCH_CSS = """
<style>
.recommendations-container {
    display: flex;
    flex-direction: column;
    gap: 20px;
    padding: 20px;
}

.breed-card {
    border: 2px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px;
    background: white;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
    position: relative;
}

.breed-card:hover {
    box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
    transform: translateY(-2px);
}

.rank-badge {
    position: absolute;
    top: 15px;
    left: 15px;
    padding: 8px 14px;
    border-radius: 8px;
    font-weight: 800;
    font-size: 20px;
    min-width: 45px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.rank-1 {
    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 50%, #F59E0B 100%);
    color: #92400E;
    font-size: 32px;
    font-weight: 900;
    animation: pulse 2s infinite;
    border: 3px solid rgba(251, 191, 36, 0.4);
    box-shadow: 0 6px 20px rgba(245, 158, 11, 0.3);
    text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8);
}

.rank-2 {
    background: linear-gradient(135deg, #F1F5F9 0%, #E2E8F0 50%, #94A3B8 100%);
    color: #475569;
    font-size: 30px;
    font-weight: 800;
    border: 3px solid rgba(148, 163, 184, 0.4);
    box-shadow: 0 5px 15px rgba(148, 163, 184, 0.3);
    text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8);
}

.rank-3 {
    background: linear-gradient(135deg, #FEF2F2 0%, #FED7AA 50%, #FB923C 100%);
    color: #9A3412;
    font-size: 28px;
    font-weight: 800;
    border: 3px solid rgba(251, 146, 60, 0.4);
    box-shadow: 0 4px 12px rgba(251, 146, 60, 0.3);
    text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8);
}

.rank-other {
    background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 50%, #CBD5E1 100%);
    color: #475569;
    font-size: 26px;
    font-weight: 700;
    border: 2px solid rgba(203, 213, 225, 0.6);
    box-shadow: 0 3px 8px rgba(203, 213, 225, 0.4);
    text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8);
}

@keyframes pulse {
    0% {
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.3);
        transform: scale(1);
    }
    50% {
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.5), 0 0 0 4px rgba(245, 158, 11, 0.15);
        transform: scale(1.05);
    }
    100% {
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.3);
        transform: scale(1);
    }
}

.breed-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
    padding-left: 70px;
}

.breed-name {
    font-size: 26px;
    font-weight: 800;
    color: #1F2937;
    margin: 0;
    letter-spacing: -0.025em;
    line-height: 1.2;
}

.match-score {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    padding: 12px 16px;
    background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
    border-radius: 12px;
    border: 2px solid rgba(6, 182, 212, 0.2);
    box-shadow: 0 4px 12px rgba(6, 182, 212, 0.1);
}

.match-percentage {
    font-size: 48px;
    font-weight: 900;
    margin-bottom: 8px;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    line-height: 1;
    letter-spacing: -0.02em;
}

.match-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 2px;
    opacity: 0.9;
    font-weight: 800;
    margin-bottom: 6px;
    color: #0369A1;
}

.score-excellent { color: #22C55E; }
.score-good { color: #F59E0B; }
.score-moderate { color: #6B7280; }

.score-bar {
    width: 220px;
    height: 14px;
    background: rgba(226, 232, 240, 0.8);
    border-radius: 8px;
    overflow: hidden;
    margin-top: 8px;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(6, 182, 212, 0.2);
}

.score-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 1s ease;
}

.fill-excellent { background: linear-gradient(90deg, #22C55E, #16A34A); }
.fill-good { background: linear-gradient(90deg, #F59E0B, #DC2626); }
.fill-moderate { background: linear-gradient(90deg, #6B7280, #4B5563); }

/* Tooltip styles for Find by Description */
.tooltip {
    position: relative;
    display: inline-block;
    cursor: help;
}

.tooltip-icon {
    display: inline-block;
    width: 18px;
    height: 18px;
    background: linear-gradient(135deg, #06b6d4, #0891b2);
    color: white;
    border-radius: 50%;
    text-align: center;
    line-height: 18px;
    font-size: 12px;
    font-weight: bold;
    margin-left: 8px;
    cursor: help;
    box-shadow: 0 2px 4px rgba(6, 182, 212, 0.3);
    transition: all 0.2s ease;
}

.tooltip-icon:hover {
    background: linear-gradient(135deg, #0891b2, #0e7490);
    transform: scale(1.1);
    box-shadow: 0 3px 6px rgba(6, 182, 212, 0.4);
}

.tooltip-text {
    visibility: hidden;
    width: 320px;
    background: linear-gradient(145deg, #1e293b, #334155);
    color: #f1f5f9;
    text-align: left;
    border-radius: 12px;
    padding: 16px;
    position: absolute;
    z-index: 1000;
    bottom: 125%;
    left: 50%;
    margin-left: -160px;
    opacity: 0;
    transition: opacity 0.3s ease, transform 0.3s ease;
    transform: translateY(10px);
    font-size: 14px;
    line-height: 1.5;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(148, 163, 184, 0.2);
}

.tooltip-text::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -8px;
    border-width: 8px;
    border-style: solid;
    border-color: #334155 transparent transparent transparent;
}

.tooltip:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
    transform: translateY(0);
}

.tooltip-text strong {
    color: #06b6d4;
    font-weight: 700;
    display: block;
    margin-bottom: 8px;
    font-size: 15px;
}
</style>
"""

# Criteria Search (Find by Criteria) 模式的 CSS 樣式
CRITERIA_SEARCH_CSS = """
<style>
.recommendations-container {
    display: flex;
    flex-direction: column;
    gap: 15px;
    padding: 15px;
}

.breed-card {
    border: 1px solid #d1d5db;
    border-radius: 8px;
    padding: 16px;
    background: #ffffff;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    transition: all 0.2s ease;
}

.breed-card:hover {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transform: translateY(-1px);
}

.breed-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #f3f4f6;
}

.breed-title {
    display: flex;
    align-items: center;
    gap: 16px;
    justify-content: flex-start;
}

.trophy-rank {
    font-size: 24px;
    font-weight: 800;
    color: #1f2937;
}

.breed-name {
    font-size: 42px;
    font-weight: 900;
    color: #1f2937;
    margin: 0;
    padding: 8px 16px;
    background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
    border: 2px solid #22C55E;
    border-radius: 12px;
    display: inline-block;
    box-shadow: 0 2px 8px rgba(34, 197, 94, 0.2);
}

.overall-score {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
}

.score-percentage {
    font-size: 32px;
    font-weight: 900;
    margin-bottom: 4px;
    line-height: 1;
}

.score-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.7;
    font-weight: 600;
}

.score-bar-wide {
    width: 200px;
    height: 8px;
    background: #f3f4f6;
    border-radius: 4px;
    overflow: hidden;
    margin-top: 6px;
}

.score-fill-wide {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
}

.score-excellent { color: #22C55E; }
.score-good { color: #65a30d; }
.score-moderate { color: #d4a332; }
.score-fair { color: #e67e22; }
.score-poor { color: #e74c3c; }

.fill-excellent { background: #22C55E; }
.fill-good { background: #65a30d; }
.fill-moderate { background: #d4a332; }
.fill-fair { background: #e67e22; }
.fill-poor { background: #e74c3c; }

.breed-details {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #e5e7eb;
}

/* 通用樣式（兩個模式都需要） */
.progress {
    transition: all 0.3s ease-in-out;
    border-radius: 4px;
    height: 12px;
}

.progress-bar {
    background-color: #f5f5f5;
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}

.score-item {
    margin: 10px 0;
}

.percentage {
    margin-left: 8px;
    font-weight: 500;
}

/* White Tooltip Styles */
.tooltip {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    cursor: help;
}

.tooltip .tooltip-icon {
    font-size: 14px;
    color: #666;
}

.tooltip .tooltip-text {
    visibility: hidden;
    width: 250px;
    background-color: rgba(44, 62, 80, 0.95);
    color: white;
    text-align: left;
    border-radius: 8px;
    padding: 8px 10px;
    position: absolute;
    z-index: 100;
    bottom: 150%;
    left: 50%;
    transform: translateX(-50%);
    opacity: 0;
    transition: all 0.3s ease;
    font-size: 14px;
    line-height: 1.3;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 10px;
}

.tooltip:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
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
</style>
"""

# Unified Recommendations (統一推薦結果) 的 CSS 樣式
UNIFIED_CSS = """
<style>
.unified-recommendations {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.unified-breed-card {
    background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    border: 1px solid #e2e8f0;
    transition: all 0.3s ease;
}

.unified-breed-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
}

.unified-breed-header {
    margin-bottom: 20px;
}

.unified-rank-section {
    display: flex;
    align-items: center;
    gap: 15px;
}

.unified-rank-badge {
    background: linear-gradient(135deg, #E0F2FE 0%, #BAE6FD 100%);
    color: #0C4A6E;
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 900;
    font-size: 24px;
    box-shadow: 0 2px 8px rgba(14, 165, 233, 0.2);
    border: 2px solid #0EA5E9;
    display: inline-block;
    min-width: 80px;
    text-align: center;
}

.unified-breed-info {
    flex-grow: 1;
}

.unified-breed-title {
    font-size: 24px;
    font-weight: 800;
    color: #0C4A6E;
    margin: 0;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #F0F9FF, #E0F2FE);
    padding: 12px 20px;
    border-radius: 10px;
    border: 2px solid #0EA5E9;
    display: inline-block;
    box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1);
}

.unified-match-score {
    font-size: 24px;
    font-weight: 900;
    color: #0F5132;
    background: linear-gradient(135deg, #D1FAE5, #A7F3D0);
    padding: 12px 20px;
    border-radius: 10px;
    display: inline-block;
    text-align: center;
    border: 2px solid #22C55E;
    box-shadow: 0 2px 8px rgba(34, 197, 94, 0.2);
    margin: 0;
    text-shadow: 0 1px 2px rgba(255, 255, 255, 0.5);
    letter-spacing: -0.02em;
}

.unified-overall-section {
    background: linear-gradient(135deg, #f0f9ff, #ecfeff);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
    border: 2px solid #06b6d4;
    box-shadow: 0 4px 12px rgba(6, 182, 212, 0.1);
}

.unified-dimension-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}

.unified-dimension-item {
    background: white;
    padding: 16px;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    transition: all 0.2s ease;
}

.unified-dimension-item:hover {
    background: #f8fafc;
    border-color: #cbd5e1;
}

.unified-breed-info {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin: 20px 0;
}

.unified-info-item {
    background: #f8fafc;
    padding: 12px;
    border-radius: 8px;
    border-left: 4px solid #6366f1;
}

.unified-info-label {
    font-weight: 600;
    color: #4b5563;
    font-size: 0.85em;
    margin-bottom: 4px;
}

.unified-info-value {
    color: #1f2937;
    font-weight: 500;
}

/* Tooltip styles for unified recommendations */
.tooltip {
    position: relative;
    display: inline-block;
    cursor: help;
}

.tooltip-icon {
    display: inline-block;
    width: 18px;
    height: 18px;
    background: linear-gradient(135deg, #06b6d4, #0891b2);
    color: white;
    border-radius: 50%;
    text-align: center;
    line-height: 18px;
    font-size: 12px;
    font-weight: bold;
    margin-left: 8px;
    cursor: help;
    box-shadow: 0 2px 4px rgba(6, 182, 212, 0.3);
    transition: all 0.2s ease;
}

.tooltip-icon:hover {
    background: linear-gradient(135deg, #0891b2, #0e7490);
    transform: scale(1.1);
    box-shadow: 0 3px 6px rgba(6, 182, 212, 0.4);
}

.tooltip-text {
    visibility: hidden;
    width: 320px;
    background: linear-gradient(145deg, #1e293b, #334155);
    color: #f1f5f9;
    text-align: left;
    border-radius: 12px;
    padding: 16px;
    position: absolute;
    z-index: 1000;
    bottom: 125%;
    left: 50%;
    margin-left: -160px;
    opacity: 0;
    transition: opacity 0.3s ease, transform 0.3s ease;
    transform: translateY(10px);
    font-size: 14px;
    line-height: 1.5;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(148, 163, 184, 0.2);
}

.tooltip-text::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -8px;
    border-width: 8px;
    border-style: solid;
    border-color: #334155 transparent transparent transparent;
}

.tooltip:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
    transform: translateY(0);
}

.tooltip-text strong {
    color: #06b6d4;
    font-weight: 700;
    display: block;
    margin-bottom: 8px;
    font-size: 15px;
}

.akc-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(59, 130, 246, 0.5) !important;
}
</style>
"""


def get_recommendation_css_styles(is_description_search: bool) -> str:
    """
    根據搜尋類型返回對應的推薦結果 CSS 樣式

    Args:
        is_description_search: 是否為 Description Search 模式

    Returns:
        str: 對應的 CSS 樣式字串
    """
    if is_description_search:
        return DESCRIPTION_SEARCH_CSS
    else:
        return CRITERIA_SEARCH_CSS


def get_unified_css() -> str:
    """
    獲取統一推薦結果的 CSS 樣式

    Returns:
        str: 統一推薦 CSS 樣式字串
    """
    return UNIFIED_CSS
