
import sqlite3
import traceback
from typing import List, Dict
# from breed_health_info import breed_health_info, default_health_note
# from breed_noise_info import breed_noise_info
# from dog_database import get_dog_description
# from scoring_calculation_system import  UserPreferences, calculate_compatibility_score

def format_recommendation_html(recommendations: List[Dict], is_description_search: bool = False) -> str:
    """將推薦結果格式化為HTML"""

    html_content = """
    <style>
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
    </style>
    <div class='recommendations-container'>"""

    def _convert_to_display_score(score: float, score_type: str = None) -> int:
        """
        更改為生成更明顯差異的顯示分數
        """
        try:
            # 基礎分數轉換（保持相對關係但擴大差異）
            if score_type == 'bonus':  # Breed Bonus 使用不同的轉換邏輯
                base_score = 35 + (score * 60)  # 35-95 範圍，差異更大
            else:
                # 其他類型的分數轉換
                if score <= 0.3:
                    base_score = 40 + (score * 45)  # 40-53.5 範圍
                elif score <= 0.6:
                    base_score = 55 + ((score - 0.3) * 55)  # 55-71.5 範圍
                elif score <= 0.8:
                    base_score = 72 + ((score - 0.6) * 60)  # 72-84 範圍
                else:
                    base_score = 85 + ((score - 0.8) * 50)  # 85-95 範圍

            # 添加不規則的微調，但保持相對關係
            import random
            if score_type == 'bonus':
                adjustment = random.uniform(-2, 2)
            else:
                # 根據分數範圍決定調整幅度
                if score > 0.8:
                    adjustment = random.uniform(-3, 3)
                elif score > 0.6:
                    adjustment = random.uniform(-4, 4)
                else:
                    adjustment = random.uniform(-2, 2)

            final_score = base_score + adjustment

            # 確保最終分數在合理範圍內並避免5的倍數
            final_score = min(95, max(40, final_score))
            rounded_score = round(final_score)
            if rounded_score % 5 == 0:
                rounded_score += random.choice([-1, 1])

            return rounded_score

        except Exception as e:
            print(f"Error in convert_to_display_score: {str(e)}")
            return 70


    def _generate_progress_bar(score: float, score_type: str = None) -> dict:
        """
        生成進度條的寬度和顏色

        Parameters:
            score: 原始分數 (0-1 之間的浮點數)
            score_type: 分數類型，用於特殊處理某些類型的分數

        Returns:
            dict: 包含寬度和顏色的字典
        """
        # 計算寬度
        if score_type == 'bonus':
            # Breed Bonus 特殊的計算方式
            width = min(100, max(5, 10 + (score * 300)))
        else:
            # 一般分數的計算
            if score >= 0.9:
                width = 90 + (score - 0.9) * 100
            elif score >= 0.7:
                width = 70 + (score - 0.7) * 100
            elif score >= 0.5:
                width = 40 + (score - 0.5) * 150
            elif score >= 0.3:
                width = 20 + (score - 0.3) * 100
            else:
                width = max(5, score * 66.7)

        # 根據分數決定顏色
        if score >= 0.9:
            color = '#68b36b'    # 高分段柔和綠
        elif score >= 0.7:
            color = '#9bcf74'    # 中高分段略黃綠
        elif score >= 0.5:
            color = '#d4d880'    # 中等分段黃綠
        elif score >= 0.3:
            color = '#e3b583'    # 偏低分段柔和橘
        else:
            color = '#e9a098'    # 低分段暖紅粉

        return {
            'width': width,
            'color': color
        }


    for rec in recommendations:
        breed = rec['breed']
        scores = rec['scores']
        info = rec['info']
        rank = rec.get('rank', 0)
        final_score = rec.get('final_score', scores['overall'])
        bonus_score = rec.get('bonus_score', 0)

        if is_description_search:
            display_scores = {
                'space': _convert_to_display_score(scores['space'], 'space'),
                'exercise': _convert_to_display_score(scores['exercise'], 'exercise'),
                'grooming': _convert_to_display_score(scores['grooming'], 'grooming'),
                'experience': _convert_to_display_score(scores['experience'], 'experience'),
                'noise': _convert_to_display_score(scores['noise'], 'noise')
            }
        else:
            display_scores = scores  # 圖片識別使用原始分數

        progress_bars = {}
        for metric in ['space', 'exercise', 'grooming', 'experience', 'noise']:
            if metric in scores:
                bar_data = _generate_progress_bar(scores[metric], metric)
                progress_bars[metric] = {
                    'style': f"width: {bar_data['width']}%; background-color: {bar_data['color']};"
                }

        # bonus
        if bonus_score > 0:
            bonus_data = _generate_progress_bar(bonus_score, 'bonus')
            progress_bars['bonus'] = {
                'style': f"width: {bonus_data['width']}%; background-color: {bonus_data['color']};"
            }

        health_info = breed_health_info.get(breed, {"health_notes": default_health_note})
        noise_info = breed_noise_info.get(breed, {
            "noise_notes": "Noise information not available",
            "noise_level": "Unknown",
            "source": "N/A"
        })

        # 解析噪音資訊
        noise_notes = noise_info.get('noise_notes', '').split('\n')
        noise_characteristics = []
        barking_triggers = []
        noise_level = ''

        current_section = None
        for line in noise_notes:
            line = line.strip()
            if 'Typical noise characteristics:' in line:
                current_section = 'characteristics'
            elif 'Noise level:' in line:
                noise_level = line.replace('Noise level:', '').strip()
            elif 'Barking triggers:' in line:
                current_section = 'triggers'
            elif line.startswith('•'):
                if current_section == 'characteristics':
                    noise_characteristics.append(line[1:].strip())
                elif current_section == 'triggers':
                    barking_triggers.append(line[1:].strip())

        # 生成特徵和觸發因素的HTML
        noise_characteristics_html = '\n'.join([f'<li>{item}</li>' for item in noise_characteristics])
        barking_triggers_html = '\n'.join([f'<li>{item}</li>' for item in barking_triggers])

        # 處理健康資訊
        health_notes = health_info.get('health_notes', '').split('\n')
        health_considerations = []
        health_screenings = []

        current_section = None
        for line in health_notes:
            line = line.strip()
            if 'Common breed-specific health considerations' in line:
                current_section = 'considerations'
            elif 'Recommended health screenings:' in line:
                current_section = 'screenings'
            elif line.startswith('•'):
                if current_section == 'considerations':
                    health_considerations.append(line[1:].strip())
                elif current_section == 'screenings':
                    health_screenings.append(line[1:].strip())

        health_considerations_html = '\n'.join([f'<li>{item}</li>' for item in health_considerations])
        health_screenings_html = '\n'.join([f'<li>{item}</li>' for item in health_screenings])

        # 獎勵原因計算
        bonus_reasons = []
        temperament = info.get('Temperament', '').lower()
        if any(trait in temperament for trait in ['friendly', 'gentle', 'affectionate']):
            bonus_reasons.append("Positive temperament traits")
        if info.get('Good with Children') == 'Yes':
            bonus_reasons.append("Excellent with children")
        try:
            lifespan = info.get('Lifespan', '10-12 years')
            years = int(lifespan.split('-')[0])
            if years >= 12:
                bonus_reasons.append("Above-average lifespan")
        except:
            pass

        html_content += f"""
        <div class="dog-info-card recommendation-card">
            <div class="breed-info">
                <h2 class="section-title">
                    <span class="icon">🏆</span> #{rank} {breed.replace('_', ' ')}
                    <span class="score-badge">
                        Overall Match: {final_score*100:.1f}%
                    </span>
                </h2>
                <div class="compatibility-scores">
                    <!-- 空間相容性評分 -->
                    <div class="score-item">
                        <span class="label">
                            Space Compatibility:
                            <span class="tooltip">
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Space Compatibility Score:</strong><br>
                                    • Evaluates how well the breed adapts to your living environment<br>
                                    • Considers if your home (apartment/house) and yard access suit the breed’s size<br>
                                    • Higher score means the breed fits well in your available space.
                                </span>
                            </span>
                        </span>
                        <div class="progress-bar">
                            <div class="progress" style="{progress_bars.get('space', {'style': 'width: 0%; background-color: #e74c3c;'})['style']}"></div>
                        </div>
                        <span class="percentage">{display_scores['space'] if is_description_search else scores.get('space', 0)*100:.1f}%</span>
                    </div>

                    <!-- 運動匹配度評分 -->
                    <div class="score-item">
                        <span class="label">
                            Exercise Match:
                            <span class="tooltip">
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Exercise Match Score:</strong><br>
                                    • Based on your daily exercise time and type<br>
                                    • Compares your activity level to the breed’s exercise needs<br>
                                    • Higher score means your routine aligns well with the breed’s energy requirements.
                                </span>
                            </span>
                        </span>
                        <div class="progress-bar">
                            <div class="progress" style="{progress_bars.get('exercise', {'style': 'width: 0%; background-color: #e74c3c;'})['style']}"></div>
                        </div>
                        <span class="percentage">{display_scores['exercise'] if is_description_search else scores.get('exercise', 0)*100:.1f}%</span>
                    </div>

                    <!-- 美容需求匹配度評分 -->
                    <div class="score-item">
                        <span class="label">
                            Grooming Match:
                            <span class="tooltip">
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Grooming Match Score:</strong><br>
                                    • Evaluates breed’s grooming needs (coat care, trimming, brushing)<br>
                                    • Compares these requirements with your grooming commitment level<br>
                                    • Higher score means the breed’s grooming needs fit your willingness and capability.
                                </span>
                            </span>
                        </span>
                        <div class="progress-bar">
                            <div class="progress" style="{progress_bars.get('grooming', {'style': 'width: 0%; background-color: #e74c3c;'})['style']}"></div>
                        </div>
                        <span class="percentage">{display_scores['grooming'] if is_description_search else scores.get('grooming', 0)*100:.1f}%</span>
                    </div>

                    <!-- 經驗需求匹配度評分 -->
                    <div class="score-item">
                        <span class="label">
                            Experience Match:
                            <span class="tooltip">
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Experience Match Score:</strong><br>
                                    • Based on your dog-owning experience level<br>
                                    • Considers breed’s training complexity, temperament, and handling difficulty<br>
                                    • Higher score means the breed is more suitable for your experience level.
                                </span>
                            </span>
                        </span>
                        <div class="progress-bar">
                            <div class="progress" style="{progress_bars.get('experience', {'style': 'width: 0%; background-color: #e74c3c;'})['style']}"></div>
                        </div>
                        <span class="percentage">{display_scores['experience'] if is_description_search else scores.get('experience', 0)*100:.1f}%</span>
                    </div>

                    <!-- 噪音相容性評分 -->
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
                        <span class="percentage">{display_scores['noise'] if is_description_search else scores.get('noise', 0)*100:.1f}%</span>
                    </div>

                    {f'''
                    <div class="score-item bonus-score">
                        <span class="label">
                            Breed Bonus:
                            <span class="tooltip">
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Breed Bonus Points:</strong><br>
                                    • {('<br>• '.join(bonus_reasons)) if bonus_reasons else 'No additional bonus points'}<br>
                                    <br>
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
            </div>
        </div>
        """

    html_content += "</div>"
    return html_content

def get_breed_recommendations(user_prefs: UserPreferences, top_n: int = 15) -> List[Dict]:
    """基於使用者偏好推薦狗品種，確保正確的分數排序"""
    print("Starting get_breed_recommendations")
    recommendations = []
    seen_breeds = set()

    try:
        # 獲取所有品種
        conn = sqlite3.connect('animal_detector.db')
        cursor = conn.cursor()
        cursor.execute("SELECT Breed FROM AnimalCatalog")
        all_breeds = cursor.fetchall()
        conn.close()

        # 收集所有品種的分數
        for breed_tuple in all_breeds:
            breed = breed_tuple[0]
            base_breed = breed.split('(')[0].strip()

            if base_breed in seen_breeds:
                continue
            seen_breeds.add(base_breed)

            # 獲取品種資訊
            breed_info = get_dog_description(breed)
            if not isinstance(breed_info, dict):
                continue

            if user_prefs.size_preference != "no_preference":
                breed_size = breed_info.get('Size', '').lower()
                user_size = user_prefs.size_preference.lower()
                if breed_size != user_size:
                    continue

            # 獲取噪音資訊
            noise_info = breed_noise_info.get(breed, {
                "noise_notes": "Noise information not available",
                "noise_level": "Unknown",
                "source": "N/A"
            })

            # 將噪音資訊整合到品種資訊中
            breed_info['noise_info'] = noise_info

            # 計算基礎相容性分數
            compatibility_scores = calculate_compatibility_score(breed_info, user_prefs)

            # 計算品種特定加分
            breed_bonus = 0.0

            # 壽命加分
            try:
                lifespan = breed_info.get('Lifespan', '10-12 years')
                years = [int(x) for x in lifespan.split('-')[0].split()[0:1]]
                longevity_bonus = min(0.02, (max(years) - 10) * 0.005)
                breed_bonus += longevity_bonus
            except:
                pass

            # 性格特徵加分
            temperament = breed_info.get('Temperament', '').lower()
            positive_traits = ['friendly', 'gentle', 'affectionate', 'intelligent']
            negative_traits = ['aggressive', 'stubborn', 'dominant']

            breed_bonus += sum(0.01 for trait in positive_traits if trait in temperament)
            breed_bonus -= sum(0.01 for trait in negative_traits if trait in temperament)

            # 與孩童相容性加分
            if user_prefs.has_children:
                if breed_info.get('Good with Children') == 'Yes':
                    breed_bonus += 0.02
                elif breed_info.get('Good with Children') == 'No':
                    breed_bonus -= 0.03

            # 噪音相關加分
            if user_prefs.noise_tolerance == 'low':
                if noise_info['noise_level'].lower() == 'high':
                    breed_bonus -= 0.03
                elif noise_info['noise_level'].lower() == 'low':
                    breed_bonus += 0.02
            elif user_prefs.noise_tolerance == 'high':
                if noise_info['noise_level'].lower() == 'high':
                    breed_bonus += 0.01

            # 計算最終分數
            breed_bonus = round(breed_bonus, 4)
            final_score = round(compatibility_scores['overall'] + breed_bonus, 4)

            recommendations.append({
                'breed': breed,
                'base_score': round(compatibility_scores['overall'], 4),
                'bonus_score': round(breed_bonus, 4),
                'final_score': final_score,
                'scores': compatibility_scores,
                'info': breed_info,
                'noise_info': noise_info  # 添加噪音資訊到推薦結果
            })

        # 嚴格按照 final_score 排序
        recommendations.sort(key=lambda x: (round(-x['final_score'], 4), x['breed'] ))  # 負號降序排列

        # 選擇前N名並確保正確排序
        final_recommendations = []
        last_score = None
        rank = 1

        available_breeds = len(recommendations)
        max_to_return = min(available_breeds, top_n)  # 不會超過實際可用品種數

        for rec in recommendations:
            if len(final_recommendations) >= max_to_return:
                break

            current_score = rec['final_score']
            if last_score is not None and current_score > last_score:
                continue

            rec['rank'] = rank
            final_recommendations.append(rec)
            last_score = current_score
            rank += 1

        # 驗證最終排序
        for i in range(len(final_recommendations)-1):
            current = final_recommendations[i]
            next_rec = final_recommendations[i+1]

            if current['final_score'] < next_rec['final_score']:
                print(f"Warning: Sorting error detected!")
                print(f"#{i+1} {current['breed']}: {current['final_score']}")
                print(f"#{i+2} {next_rec['breed']}: {next_rec['final_score']}")

                # 交換位置
                final_recommendations[i], final_recommendations[i+1] = \
                    final_recommendations[i+1], final_recommendations[i]

        # 打印最終結果以供驗證
        print("\nFinal Rankings:")
        for rec in final_recommendations:
            print(f"#{rec['rank']} {rec['breed']}")
            print(f"Base Score: {rec['base_score']:.4f}")
            print(f"Bonus: {rec['bonus_score']:.4f}")
            print(f"Final Score: {rec['final_score']:.4f}\n")

        return final_recommendations

    except Exception as e:
        print(f"Error in get_breed_recommendations: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return []
