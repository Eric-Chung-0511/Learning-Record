# %%writefile recommendation_html_formatter.py
"""
HTML 格式化器 - 處理推薦結果的 HTML 生成
CSS 樣式已分離至 recommendation_css.py
"""
import random
from typing import List, Dict
from breed_health_info import breed_health_info, default_health_note
from breed_noise_info import breed_noise_info
from dog_database import get_dog_description
from recommendation_formatter import (
    generate_breed_characteristics_data,
    parse_noise_information,
    parse_health_information,
    calculate_breed_bonus_factors,
    generate_dimension_scores_for_display
)
from recommendation_css import (
    DESCRIPTION_SEARCH_CSS,
    CRITERIA_SEARCH_CSS,
    UNIFIED_CSS,
    get_recommendation_css_styles,
    get_unified_css
)


class RecommendationHTMLFormatter:
    """處理推薦結果的HTML格式化，CSS樣式從recommendation_css.py導入"""

    def __init__(self):
        # 從 recommendation_css.py 導入 CSS 樣式
        self.description_search_css = DESCRIPTION_SEARCH_CSS
        self.criteria_search_css = CRITERIA_SEARCH_CSS
        self.unified_css = UNIFIED_CSS

    def format_unified_percentage(self, score: float) -> str:
        """統一格式化百分比顯示，確保數值邏輯一致"""
        try:
            # 確保分數在0-1範圍內
            normalized_score = max(0.0, min(1.0, float(score)))
            # 轉換為百分比並保留一位小數
            percentage = normalized_score * 100
            return f"{percentage:.1f}%"
        except Exception as e:
            print(f"Error formatting percentage: {str(e)}")
            return "70.0%"

    def generate_unified_progress_bar(self, score: float) -> str:
        """Generate unified progress bar HTML with width directly corresponding to score"""
        try:
            # Ensure score is in 0-1 range
            normalized_score = max(0.0, min(1.0, float(score)))

            # Progress bar width with reasonable visual mapping
            # High scores get enhanced visual representation for impact
            if normalized_score >= 0.9:
                width_percentage = 85 + (normalized_score - 0.9) * 130  # 85-98% for excellent scores
            elif normalized_score >= 0.8:
                width_percentage = 70 + (normalized_score - 0.8) * 150  # 70-85% for very good scores
            elif normalized_score >= 0.7:
                width_percentage = 55 + (normalized_score - 0.7) * 150  # 55-70% for good scores
            elif normalized_score >= 0.5:
                width_percentage = 30 + (normalized_score - 0.5) * 125  # 30-55% for fair scores
            else:
                width_percentage = 8 + normalized_score * 44  # 8-30% for low scores

            # Ensure reasonable bounds
            width_percentage = max(5, min(98, width_percentage))

            # Choose color based on score with appropriate theme
            # This is used for unified recommendations (Description search)
            if normalized_score >= 0.9:
                color = '#10b981'    # Excellent (emerald green)
            elif normalized_score >= 0.8:
                color = '#06b6d4'    # Good (cyan)
            elif normalized_score >= 0.7:
                color = '#3b82f6'    # Fair (blue)
            elif normalized_score >= 0.6:
                color = '#1d4ed8'    # Average (darker blue)
            elif normalized_score >= 0.5:
                color = '#1e40af'    # Below average (dark blue)
            else:
                color = '#ef4444'    # Poor (red)

            return f'''
            <div class="progress-bar-container" style="
                background-color: #f1f5f9;
                border-radius: 8px;
                overflow: hidden;
                height: 16px;
                width: 100%;
            ">
                <div class="progress-bar-fill" style="
                    width: {width_percentage:.1f}%;
                    height: 100%;
                    background-color: {color};
                    border-radius: 8px;
                    transition: width 0.8s ease-out;
                "></div>
            </div>
            '''

        except Exception as e:
            print(f"Error generating progress bar: {str(e)}")
            return '<div class="progress-bar-container"><div class="progress-bar-fill" style="width: 70%; background-color: #d4a332;"></div></div>'

    def generate_progress_bar(self, score: float, score_type: str = None, is_percentage_display: bool = False, is_description_search: bool = False) -> dict:
        """
        Generate progress bar width and color with consistent score-to-visual mapping

        Parameters:
            score: Score value (float between 0-1 or percentage 0-100)
            score_type: Score type for special handling
            is_percentage_display: Whether the score is in percentage format

        Returns:
            dict: Dictionary containing width and color
        """
        # Normalize score to 0-1 range
        if is_percentage_display:
            normalized_score = score / 100.0  # Convert percentage to 0-1 range
        else:
            normalized_score = score

        # Ensure score is within valid range
        normalized_score = max(0.0, min(1.0, normalized_score))

        # Calculate progress bar width - simplified for Find by Criteria
        if not is_description_search and score_type != 'bonus':
            # Find by Criteria: 調整為更有說服力的視覺比例
            percentage = normalized_score * 100
            if percentage >= 95:
                width = 92 + (percentage - 95) * 1.2  # 95%+ 顯示為 92-98%
            elif percentage >= 90:
                width = 85 + (percentage - 90)  # 90-95% 顯示為 85-92%
            elif percentage >= 80:
                width = 75 + (percentage - 80) * 1.0  # 80-90% 顯示為 75-85%
            elif percentage >= 70:
                width = 60 + (percentage - 70) * 1.5  # 70-80% 顯示為 60-75%
            else:
                width = percentage * 0.8  # 70% 以下按比例縮放
            width = max(5, min(98, width))
        elif score_type == 'bonus':
            # Bonus scores are typically smaller, need amplified display
            width = max(5, min(95, normalized_score * 150))  # Amplified for visibility
        else:
            # Find by Description: 保持現有的複雜計算
            if normalized_score >= 0.8:
                width = 75 + (normalized_score - 0.8) * 115  # 75-98% range for high scores
            elif normalized_score >= 0.6:
                width = 50 + (normalized_score - 0.6) * 125  # 50-75% range for good scores
            elif normalized_score >= 0.4:
                width = 25 + (normalized_score - 0.4) * 125  # 25-50% range for fair scores
            else:
                width = 5 + normalized_score * 50  # 5-25% range for low scores

            width = max(3, min(98, width))

        # Color coding based on normalized score - Criteria uses green gradation
        if is_description_search:
            # Find by Description uses blue theme
            if normalized_score >= 0.9:
                color = '#10b981'    # Excellent (emerald green)
            elif normalized_score >= 0.85:
                color = '#06b6d4'    # Very good (cyan)
            elif normalized_score >= 0.8:
                color = '#3b82f6'    # Good (blue)
            elif normalized_score >= 0.7:
                color = '#1d4ed8'    # Fair (darker blue)
            elif normalized_score >= 0.6:
                color = '#1e40af'    # Below average (dark blue)
            elif normalized_score >= 0.5:
                color = '#f59e0b'    # Poor (amber)
            else:
                color = '#ef4444'    # Very poor (red)
        else:
            # Find by Criteria uses original green gradation
            if normalized_score >= 0.9:
                color = '#22c55e'    # Excellent (bright green)
            elif normalized_score >= 0.85:
                color = '#65a30d'    # Very good (green)
            elif normalized_score >= 0.8:
                color = '#a3a332'    # Good (yellow-green)
            elif normalized_score >= 0.7:
                color = '#d4a332'    # Fair (yellow)
            elif normalized_score >= 0.6:
                color = '#e67e22'    # Below average (orange)
            elif normalized_score >= 0.5:
                color = '#e74c3c'    # Poor (red)
            else:
                color = '#c0392b'    # Very poor (dark red)

        return {
            'width': width,
            'color': color
        }

    def get_css_styles(self, is_description_search: bool) -> str:
        """根據搜尋類型返回對應的CSS樣式"""
        if is_description_search:
            return self.description_search_css
        else:
            return self.criteria_search_css

    def generate_breed_card_header(self, breed: str, rank: int, final_score: float, is_description_search: bool) -> str:
        """生成品種卡片標題部分的HTML"""
        rank_class = f"rank-{rank}" if rank <= 3 else "rank-other"
        percentage = final_score * 100
        
        if percentage >= 90:
            score_class = "score-excellent"
            fill_class = "fill-excellent"
            match_label = "EXCELLENT MATCH"
        elif percentage >= 70:
            score_class = "score-good"
            fill_class = "fill-good"
            match_label = "GOOD MATCH"
        else:
            score_class = "score-moderate"
            fill_class = "fill-moderate"
            match_label = "MODERATE MATCH"

        if is_description_search:
            # Find by Description: 使用現有複雜設計
            return f"""
            <div class="breed-card">
                <div class="rank-badge {rank_class}">#{rank}</div>

                <div class="breed-header">
                    <h3 class="breed-name">{breed.replace('_', ' ')}</h3>
                    <div class="match-score">
                        <div class="match-percentage {score_class}">{percentage:.1f}%</div>
                        <div class="match-label">{match_label}</div>
                        <div class="score-bar">
                            <div class="score-fill {fill_class}" style="width: {percentage}%"></div>
                        </div>
                    </div>
                </div>"""
        else:
            # Find by Criteria: 使用簡潔設計，包含獎盃圖示
            # 計算進度條寬度 - 調整為更有說服力的視覺比例
            if percentage >= 95:
                score_width = 92 + (percentage - 95) * 1.2  # 95%+ 顯示為 92-98%
            elif percentage >= 90:
                score_width = 85 + (percentage - 90)  # 90-95% 顯示為 85-92%
            elif percentage >= 80:
                score_width = 75 + (percentage - 80) * 1.0  # 80-90% 顯示為 75-85%
            elif percentage >= 70:
                score_width = 60 + (percentage - 70) * 1.5  # 70-80% 顯示為 60-75%
            else:
                score_width = percentage * 0.8  # 70% 以下按比例縮放
            score_width = max(5, min(98, score_width))
            
            return f"""
            <div class="breed-card">
                <div class="breed-header">
                    <div class="breed-title">
                        <div class="trophy-rank">🏆 #{rank}</div>
                        <h3 class="breed-name" style="font-size: 28px !important;">{breed.replace('_', ' ')}</h3>
                    </div>
                    <div class="overall-score">
                        <div class="score-percentage {score_class}">{percentage:.1f}%</div>
                        <div class="score-label">OVERALL MATCH</div>
                        <div class="score-bar-wide">
                            <div class="score-fill-wide {fill_class}" style="width: {score_width:.1f}%"></div>
                        </div>
                    </div>
                </div>"""

    def generate_tooltips_section(self) -> str:
        """生成提示氣泡HTML"""
        return '''
                        <span class="tooltip">
                            <span class="tooltip-icon">ⓘ</span>
                            <span class="tooltip-text">
                                <strong>Space Compatibility Score:</strong><br>
                                • Evaluates how well the breed adapts to your living environment<br>
                                • Considers if your home (apartment/house) and yard access suit the breed's size<br>
                                • Higher score means the breed fits well in your available space.
                            </span>
                        </span>'''
    
    def generate_detailed_sections_html(self, breed: str, info: dict, 
                                      noise_characteristics: List[str], 
                                      barking_triggers: List[str], 
                                      noise_level: str,
                                      health_considerations: List[str], 
                                      health_screenings: List[str]) -> str:
        """生成詳細區段的HTML"""
        # 生成特徵和觸發因素的HTML
        noise_characteristics_html = '\n'.join([f'<li>{item}</li>' for item in noise_characteristics])
        barking_triggers_html = '\n'.join([f'<li>{item}</li>' for item in barking_triggers])
        health_considerations_html = '\n'.join([f'<li>{item}</li>' for item in health_considerations])
        health_screenings_html = '\n'.join([f'<li>{item}</li>' for item in health_screenings])

        return f"""
                <div class="breed-details-section">
                    <h3 class="subsection-title">
                        <span class="icon">📋</span> Breed Details
                    </h3>
                    <div class="details-grid">
                        <div class="detail-item">
                            <span class="tooltip">
                                <span class="icon">📏</span>
                                <span class="label">Size:</span>
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Size Categories:</strong><br>
                                    • Small: Under 20 pounds<br>
                                    • Medium: 20-60 pounds<br>
                                    • Large: Over 60 pounds
                                </span>
                                <span class="value">{info['Size']}</span>
                            </span>
                        </div>
                        <div class="detail-item">
                            <span class="tooltip">
                                <span class="icon">🏃</span>
                                <span class="label">Exercise Needs:</span>
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Exercise Needs:</strong><br>
                                    • Low: Short walks<br>
                                    • Moderate: 1-2 hours daily<br>
                                    • High: 2+ hours daily<br>
                                    • Very High: Constant activity
                                </span>
                                <span class="value">{info['Exercise Needs']}</span>
                            </span>
                        </div>
                        <div class="detail-item">
                            <span class="tooltip">
                                <span class="icon">👨‍👩‍👧‍👦</span>
                                <span class="label">Good with Children:</span>
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Child Compatibility:</strong><br>
                                    • Yes: Excellent with kids<br>
                                    • Moderate: Good with older children<br>
                                    • No: Better for adult households
                                </span>
                                <span class="value">{info['Good with Children']}</span>
                            </span>
                        </div>
                        <div class="detail-item">
                            <span class="tooltip">
                                <span class="icon">⏳</span>
                                <span class="label">Lifespan:</span>
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Average Lifespan:</strong><br>
                                    • Short: 6-8 years<br>
                                    • Average: 10-15 years<br>
                                    • Long: 12-20 years<br>
                                    • Varies by size: Larger breeds typically have shorter lifespans
                                </span>
                            </span>
                            <span class="value">{info['Lifespan']}</span>
                        </div>
                    </div>
                </div>
                <div class="description-section">
                    <h3 class="subsection-title">
                        <span class="icon">📝</span> Description
                    </h3>
                    <p class="description-text">{info.get('Description', '')}</p>
                </div>
                <div class="noise-section">
                    <h3 class="section-header">
                        <span class="icon">🔊</span> Noise Behavior
                        <span class="tooltip">
                            <span class="tooltip-icon">ⓘ</span>
                            <span class="tooltip-text">
                                <strong>Noise Behavior:</strong><br>
                                • Typical vocalization patterns<br>
                                • Common triggers and frequency<br>
                                • Based on breed characteristics
                            </span>
                        </span>
                    </h3>
                    <div class="noise-info">
                        <div class="noise-details">
                            <h4 class="section-header">Typical noise characteristics:</h4>
                            <div class="characteristics-list">
                                <div class="list-item">Moderate to high barker</div>
                                <div class="list-item">Alert watch dog</div>
                                <div class="list-item">Attention-seeking barks</div>
                                <div class="list-item">Social vocalizations</div>
                            </div>
                            <div class="noise-level-display">
                                <h4 class="section-header">Noise level:</h4>
                                <div class="level-indicator">
                                    <span class="level-text">Moderate-High</span>
                                    <div class="level-bars">
                                        <span class="bar"></span>
                                        <span class="bar"></span>
                                        <span class="bar"></span>
                                    </div>
                                </div>
                            </div>
                            <h4 class="section-header">Barking triggers:</h4>
                            <div class="triggers-list">
                                <div class="list-item">Separation anxiety</div>
                                <div class="list-item">Attention needs</div>
                                <div class="list-item">Strange noises</div>
                                <div class="list-item">Excitement</div>
                            </div>
                        </div>
                        <div class="noise-disclaimer">
                            <p class="disclaimer-text source-text">Source: Compiled from various breed behavior resources, 2024</p>
                            <p class="disclaimer-text">Individual dogs may vary in their vocalization patterns.</p>
                            <p class="disclaimer-text">Training can significantly influence barking behavior.</p>
                            <p class="disclaimer-text">Environmental factors may affect noise levels.</p>
                        </div>
                    </div>
                </div>
                <div class="health-section">
                    <h3 class="section-header">
                        <span class="icon">🏥</span> Health Insights
                        <span class="tooltip">
                            <span class="tooltip-icon">ⓘ</span>
                            <span class="tooltip-text">
                                Health information is compiled from multiple sources including veterinary resources, breed guides, and international canine health databases.
                                Each dog is unique and may vary from these general guidelines.
                            </span>
                        </span>
                    </h3>
                    <div class="health-info">
                        <div class="health-details">
                            <div class="health-block">
                                <h4 class="section-header">Common breed-specific health considerations:</h4>
                                <div class="health-grid">
                                    <div class="health-item">Patellar luxation</div>
                                    <div class="health-item">Progressive retinal atrophy</div>
                                    <div class="health-item">Von Willebrand's disease</div>
                                    <div class="health-item">Open fontanel</div>
                                </div>
                            </div>
                            <div class="health-block">
                                <h4 class="section-header">Recommended health screenings:</h4>
                                <div class="health-grid">
                                    <div class="health-item screening">Patella evaluation</div>
                                    <div class="health-item screening">Eye examination</div>
                                    <div class="health-item screening">Blood clotting tests</div>
                                    <div class="health-item screening">Skull development monitoring</div>
                                </div>
                            </div>
                        </div>
                        <div class="health-disclaimer">
                            <p class="disclaimer-text source-text">Source: Compiled from various veterinary and breed information resources, 2024</p>
                            <p class="disclaimer-text">This information is for reference only and based on breed tendencies.</p>
                            <p class="disclaimer-text">Each dog is unique and may not develop any or all of these conditions.</p>
                            <p class="disclaimer-text">Always consult with qualified veterinarians for professional advice.</p>
                        </div>
                    </div>
                </div>
                <div class="action-section">
                    <a href="https://www.akc.org/dog-breeds/{breed.lower().replace('_', '-')}/"
                       target="_blank"
                       class="akc-button">
                        <span class="icon">🌐</span>
                        Learn more about {breed.replace('_', ' ')} on AKC website
                    </a>
                </div>
        """
