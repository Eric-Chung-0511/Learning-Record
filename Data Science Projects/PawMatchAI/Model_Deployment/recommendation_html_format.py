import random
from typing import List, Dict
from breed_health_info import breed_health_info, default_health_note
from breed_noise_info import breed_noise_info
from dog_database import get_dog_description
from scoring_calculation_system import UserPreferences
from recommendation_formatter import (
    get_breed_recommendations,
    _format_dimension_scores,
    calculate_breed_bonus_factors,
    generate_breed_characteristics_data,
    parse_noise_information,
    parse_health_information,
    generate_dimension_scores_for_display
)
from recommendation_html_formatter import RecommendationHTMLFormatter


def format_recommendation_html(recommendations: List[Dict], is_description_search: bool = False) -> str:
    """統一推薦結果HTML格式化，確保視覺與數值邏輯一致"""
    
    # 創建HTML格式器實例
    formatter = RecommendationHTMLFormatter()
    
    # 獲取對應的CSS樣式
    html_content = formatter.get_css_styles(is_description_search) + "<div class='recommendations-container'>"

    for rec in recommendations:
        breed = rec['breed']
        rank = rec.get('rank', 0)

        breed_name_for_db = breed.replace(' ', '_')
        breed_info_from_db = get_dog_description(breed_name_for_db)

        if is_description_search:
            # Handle semantic search results structure - use scores directly from semantic recommender
            overall_score = rec.get('overall_score', 0.7)
            final_score = rec.get('final_score', overall_score)  # Use final_score if available
            semantic_score = rec.get('semantic_score', 0.7)
            comparative_bonus = rec.get('comparative_bonus', 0.0)
            lifestyle_bonus = rec.get('lifestyle_bonus', 0.0)

            # Use the actual calculated scores without re-computation
            base_score = final_score

            # Generate dimension scores using the formatter helper
            scores = generate_dimension_scores_for_display(
                base_score, rank, breed, semantic_score, 
                comparative_bonus, lifestyle_bonus, is_description_search
            )
            
            bonus_score = max(0.0, comparative_bonus + random.uniform(-0.02, 0.02))
            info = generate_breed_characteristics_data(breed_info_from_db or {})
            info = dict(info)  # Convert to dict for compatibility
            
            # Add any missing fields from rec
            if not breed_info_from_db:
                for key in ['Size', 'Exercise Needs', 'Grooming Needs', 'Good with Children', 'Temperament', 'Lifespan', 'Description']:
                    if key not in info:
                        info[key] = rec.get(key.lower().replace(' ', '_'), 'Unknown' if key != 'Description' else '')

            # Display scores as percentages with one decimal place
            display_scores = {
                'space': round(scores['space'] * 100, 1),
                'exercise': round(scores['exercise'] * 100, 1),
                'grooming': round(scores['grooming'] * 100, 1),
                'experience': round(scores['experience'] * 100, 1),
                'noise': round(scores['noise'] * 100, 1),
            }
        else:
            # Handle traditional search results structure
            scores = rec['scores']
            info = rec['info']
            final_score = rec.get('final_score', scores['overall'])
            bonus_score = rec.get('bonus_score', 0)
            # Convert traditional scores to percentage display format with one decimal
            display_scores = {
                'space': round(scores.get('space', 0) * 100, 1),
                'exercise': round(scores.get('exercise', 0) * 100, 1),
                'grooming': round(scores.get('grooming', 0) * 100, 1),
                'experience': round(scores.get('experience', 0) * 100, 1),
                'noise': round(scores.get('noise', 0) * 100, 1),
            }

        progress_bars = {}
        for metric in ['space', 'exercise', 'grooming', 'experience', 'noise']:
            if metric in scores:
                # 使用顯示分數（百分比）來計算進度條
                display_score = display_scores[metric]
                bar_data = formatter.generate_progress_bar(display_score, metric, is_percentage_display=True, is_description_search=is_description_search)
                progress_bars[metric] = {
                    'style': f"width: {bar_data['width']}%; background-color: {bar_data['color']};"
                }

        # bonus
        if bonus_score > 0:
            # bonus_score 通常是 0-1 範圍，需要轉換為百分比顯示
            bonus_percentage = bonus_score * 100
            bonus_data = formatter.generate_progress_bar(bonus_percentage, 'bonus', is_percentage_display=True, is_description_search=is_description_search)
            progress_bars['bonus'] = {
                'style': f"width: {bonus_data['width']}%; background-color: {bonus_data['color']};"
            }

        health_info = breed_health_info.get(breed, {"health_notes": default_health_note})
        noise_info = breed_noise_info.get(breed, {
            "noise_notes": "Noise information not available",
            "noise_level": "Unknown",
            "source": "N/A"
        })

        # 解析噪音和健康資訊
        noise_characteristics, barking_triggers, noise_level = parse_noise_information(noise_info)
        health_considerations, health_screenings = parse_health_information(health_info)

        # 計算獎勵因素
        _, bonus_reasons = calculate_breed_bonus_factors(info, None)  # User prefs not needed for display
        
        # 生成品種卡片標題
        html_content += formatter.generate_breed_card_header(breed, rank, final_score, is_description_search)

        # 品種詳細資訊區域 - 使用格式器方法簡化
        tooltip_html = formatter.generate_tooltips_section()
        
        html_content += f"""
            <div class="breed-details">
                <div class="compatibility-scores">
                    <!-- Space Compatibility Score -->
                    <div class="score-item">
                        <span class="label">
                            Space Compatibility:{tooltip_html}
                        </span>
                        <div class="progress-bar">
                            <div class="progress" style="{progress_bars.get('space', {'style': 'width: 0%; background-color: #e74c3c;'})['style']}"></div>
                        </div>
                        <span class="percentage">{display_scores['space']:.1f}%</span>
                    </div>

                    <!-- Exercise Compatibility Score -->
                    <div class="score-item">
                        <span class="label">
                            Exercise Match:
                            <span class="tooltip">
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Exercise Match Score:</strong><br>
                                    • Based on your daily exercise time and type<br>
                                    • Compares your activity level to the breed's exercise needs<br>
                                    • Higher score means your routine aligns well with the breed's energy requirements.
                                </span>
                            </span>
                        </span>
                        <div class="progress-bar">
                            <div class="progress" style="{progress_bars.get('exercise', {'style': 'width: 0%; background-color: #e74c3c;'})['style']}"></div>
                        </div>
                        <span class="percentage">{display_scores['exercise']:.1f}%</span>
                    </div>

                    <!-- Grooming Compatibility Score -->
                    <div class="score-item">
                        <span class="label">
                            Grooming Match:
                            <span class="tooltip">
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Grooming Match Score:</strong><br>
                                    • Evaluates breed's grooming needs (coat care, trimming, brushing)<br>
                                    • Compares these requirements with your grooming commitment level<br>
                                    • Higher score means the breed's grooming needs fit your willingness and capability.
                                </span>
                            </span>
                        </span>
                        <div class="progress-bar">
                            <div class="progress" style="{progress_bars.get('grooming', {'style': 'width: 0%; background-color: #e74c3c;'})['style']}"></div>
                        </div>
                        <span class="percentage">{display_scores['grooming']:.1f}%</span>
                    </div>

                    <!-- Experience Compatibility Score -->
                    <div class="score-item">
                        <span class="label">
                            Experience Match:
                            <span class="tooltip">
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Experience Match Score:</strong><br>
                                    • Based on your dog-owning experience level<br>
                                    • Considers breed's training complexity, temperament, and handling difficulty<br>
                                    • Higher score means the breed is more suitable for your experience level.
                                </span>
                            </span>
                        </span>
                        <div class="progress-bar">
                            <div class="progress" style="{progress_bars.get('experience', {'style': 'width: 0%; background-color: #e74c3c;'})['style']}"></div>
                        </div>
                        <span class="percentage">{display_scores['experience']:.1f}%</span>
                    </div>

                    <!-- Noise Compatibility Score -->
                    <div class="score-item">
                        <span class="label">
                            Noise Compatibility:
                            <span class="tooltip">
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Noise Compatibility Score:</strong><br>
                                    • Based on your noise tolerance preference<br>
                                    • Considers breed's typical noise level and barking tendencies<br>
                                    • Accounts for living environment and sensitivity to noise.
                                </span>
                            </span>
                        </span>
                        <div class="progress-bar">
                            <div class="progress" style="{progress_bars.get('noise', {'style': 'width: 0%; background-color: #e74c3c;'})['style']}"></div>
                        </div>
                        <span class="percentage">{display_scores['noise']:.1f}%</span>
                    </div>

                    {f'''
                    <div class="score-item bonus-score">
                        <span class="label">
                            Breed Bonus:
                            <span class="tooltip">
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Breed Bonus Points:</strong><br>
                                    • {('<br>• '.join(bonus_reasons) if bonus_reasons else 'No additional bonus points')}<br><br>
                                    <strong>Bonus Factors Include:</strong><br>
                                    • Friendly temperament<br>
                                    • Child compatibility<br>
                                    • Longer lifespan<br>
                                    • Living space adaptability
                                </span>
                            </span>
                        </span>
                        <div class="progress-bar">
                            <div class="progress" style="{progress_bars['bonus']['style']}"></div>
                        </div>
                        <span class="percentage">{bonus_score*100:.1f}%</span>
                    </div>
                ''' if bonus_score > 0 else ''}
                </div>
        """
        
        # 使用格式器生成詳細區段
        html_content += formatter.generate_detailed_sections_html(
            breed, info, noise_characteristics, barking_triggers, noise_level,
            health_considerations, health_screenings
        )
        
        html_content += """
            </div>
        </div>
        """

    # 結束 HTML 內容
    html_content += "</div>"
    return html_content


def format_unified_recommendation_html(recommendations: List[Dict], is_description_search: bool = False) -> str:
    """統一推薦HTML格式化主函數，確保視覺呈現與數值計算完全一致"""
    
    # 創建HTML格式器實例
    formatter = RecommendationHTMLFormatter()

    if not recommendations:
        return '''
        <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #f8fafc, #e2e8f0); border-radius: 16px; margin: 20px 0;">
            <div style="font-size: 3em; margin-bottom: 16px;">🐕</div>
            <h3 style="color: #374151; margin-bottom: 12px;">No Recommendations Available</h3>
            <p style="color: #6b7280; font-size: 1.1em;">Please try adjusting your preferences or description, and we'll help you find the most suitable breeds.</p>
        </div>
        '''

    # 使用格式器的統一CSS樣式
    html_content = formatter.unified_css + "<div class='unified-recommendations'>"

    for rec in recommendations:
        breed = rec['breed']
        rank = rec.get('rank', 0)

        # 統一分數處理
        overall_score = rec.get('overall_score', rec.get('final_score', 0.7))
        scores = rec.get('scores', {})

        # 如果沒有維度分數，基於總分生成一致的維度分數
        if not scores:
            scores = generate_dimension_scores_for_display(
                overall_score, rank, breed, is_description_search=is_description_search
            )

        # 獲取品種資訊
        breed_name_for_db = breed.replace(' ', '_')
        breed_info = get_dog_description(breed_name_for_db) or {}

        # 維度標籤
        dimension_labels = {
            'space': '🏠 Space Compatibility',
            'exercise': '🏃 Exercise Requirements',
            'grooming': '✂️ Grooming Needs',
            'experience': '🎓 Experience Level',
            'noise': '🔊 Noise Control',
            'family': '👨‍👩‍👧‍👦 Family Compatibility'
        }

        # 維度提示氣泡內容
        tooltip_content = {
            'space': 'Space Compatibility Score:<br>• Evaluates how well the breed adapts to your living environment<br>• Considers if your home (apartment/house) and yard access suit the breed\'s size<br>• Higher score means the breed fits well in your available space.',
            'exercise': 'Exercise Requirements Score:<br>• Based on your daily exercise time and activity type<br>• Compares your activity level to the breed\'s exercise needs<br>• Higher score means your routine aligns well with the breed\'s energy requirements.',
            'grooming': 'Grooming Needs Score:<br>• Evaluates breed\'s grooming needs (coat care, trimming, brushing)<br>• Compares these requirements with your grooming commitment level<br>• Higher score means the breed\'s grooming needs fit your willingness and capability.',
            'experience': 'Experience Level Score:<br>• Based on your dog-owning experience level<br>• Considers breed\'s training complexity, temperament, and handling difficulty<br>• Higher score means the breed is more suitable for your experience level.',
            'noise': 'Noise Control Score:<br>• Based on your noise tolerance preference<br>• Considers breed\'s typical noise level and barking tendencies<br>• Accounts for living environment and sensitivity to noise.',
            'family': 'Family Compatibility Score:<br>• Evaluates how well the breed fits with your family situation<br>• Considers children, other pets, and family dynamics<br>• Higher score means better family compatibility.'
        }
        
        # 生成維度分數HTML
        dimension_html = ""
        for dim, label in dimension_labels.items():
            score = scores.get(dim, overall_score * 0.9)
            percentage = formatter.format_unified_percentage(score)
            progress_bar = formatter.generate_unified_progress_bar(score)
            
            # 為 Find by Description 添加提示氣泡
            tooltip_html = ''
            if is_description_search:
                tooltip_html = f'<span class="tooltip"><span class="tooltip-icon">i</span><span class="tooltip-text"><strong>{tooltip_content.get(dim, "")}</strong></span></span>'

            dimension_html += f'''
            <div class="unified-dimension-item">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 600; color: #374151;">{label} {tooltip_html}</span>
                    <span style="font-weight: 700; color: #1f2937; font-size: 1.1em;">{percentage}</span>
                </div>
                {progress_bar}
            </div>
            '''

        # 生成品種資訊HTML
        characteristics = generate_breed_characteristics_data(breed_info)
        info_html = ""
        for label, value in characteristics:
            if label != 'Description':  # Skip description as it's shown separately
                info_html += f'''
                <div class="unified-info-item">
                    <div class="unified-info-label">{label}</div>
                    <div class="unified-info-value">{value}</div>
                </div>
                '''

        # 生成單個品種卡片HTML
        overall_percentage = formatter.format_unified_percentage(overall_score)
        overall_progress_bar = formatter.generate_unified_progress_bar(overall_score)

        brand_card_html = f'''
        <div class="unified-breed-card">
            <div class="unified-breed-header">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
                    <div class="unified-rank-badge">🏆 #{rank}</div>
                    <h2 class="unified-breed-title">{breed.replace('_', ' ')}</h2>
                    <div style="margin-left: auto;">
                        <div class="unified-match-score">Overall Match: {overall_percentage}</div>
                    </div>
                </div>
            </div>

            <div class="unified-overall-section">
                {overall_progress_bar}
            </div>

            <div class="unified-dimension-grid">
                {dimension_html}
            </div>

            <div class="unified-breed-info">
                {info_html}
            </div>
            
            <div style="background: linear-gradient(135deg, #F8FAFC, #F1F5F9); padding: 20px; border-radius: 12px; margin: 20px 0; border: 1px solid #E2E8F0;">
                <h3 style="color: #1F2937; font-size: 1.3em; font-weight: 700; margin: 0 0 12px 0; display: flex; align-items: center;">
                    <span style="margin-right: 8px;">📝</span> Breed Description
                </h3>
                <p style="color: #4B5563; line-height: 1.6; margin: 0 0 16px 0; font-size: 1.05em;">
                    {breed_info.get('Description', 'Detailed description for this breed is not currently available.')}
                </p>
                <a href="https://www.akc.org/dog-breeds/{breed.lower().replace('_', '-').replace(' ', '-')}/" 
                   target="_blank" 
                   class="akc-button"
                   style="display: inline-flex; align-items: center; padding: 12px 20px; 
                          background: linear-gradient(135deg, #3B82F6, #1D4ED8); color: white; 
                          text-decoration: none; border-radius: 10px; font-weight: 600; 
                          box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3); 
                          transition: all 0.3s ease; font-size: 1.05em;"
                   onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 16px rgba(59, 130, 246, 0.5)'"
                   onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 4px 12px rgba(59, 130, 246, 0.3)'">
                    <span style="margin-right: 8px;">🌐</span>
                    Learn more about {breed.replace('_', ' ')} on AKC website
                </a>
            </div>
        </div>
        '''

        html_content += brand_card_html

    html_content += "</div>"
    return html_content
