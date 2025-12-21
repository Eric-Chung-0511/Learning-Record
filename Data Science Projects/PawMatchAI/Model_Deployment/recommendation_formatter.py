# %%writefile recommendation_formatter.py
import sqlite3
import traceback
import random
from typing import List, Dict
from breed_health_info import breed_health_info, default_health_note
from breed_noise_info import breed_noise_info
from dog_database import get_dog_description
from scoring_calculation_system import UserPreferences, calculate_compatibility_score

def get_breed_recommendations(user_prefs: UserPreferences, top_n: int = 15) -> List[Dict]:
    """基於使用者偏好推薦狗品種，確保正確的分數排序"""
    print(f"Starting get_breed_recommendations with top_n={top_n}")
    recommendations = []
    seen_breeds = set()

    try:
        # 獲取所有品種
        conn = sqlite3.connect('animal_detector.db')
        cursor = conn.cursor()
        cursor.execute("SELECT Breed FROM AnimalCatalog")
        all_breeds = cursor.fetchall()
        conn.close()

        print(f"Total breeds in database: {len(all_breeds)}")

        # 收集所有品種的分數
        for breed_tuple in all_breeds:
            breed = breed_tuple[0]
            base_breed = breed.split('(')[0].strip()

            # 過濾掉野生動物品種
            if base_breed == 'Dhole':
                continue

            if base_breed in seen_breeds:
                continue
            seen_breeds.add(base_breed)

            # 獲取品種資訊
            breed_info = get_dog_description(breed)
            if not isinstance(breed_info, dict):
                continue

            # 調整品種尺寸過濾邏輯，避免過度限制候選品種
            if user_prefs.size_preference != "no_preference":
                breed_size = breed_info.get('Size', '').lower()
                user_size = user_prefs.size_preference.lower()

                # 放寬尺寸匹配條件，允許相鄰尺寸的品種通過篩選
                size_compatibility = False
                if user_size == 'small':
                    size_compatibility = breed_size in ['small', 'medium']
                elif user_size == 'medium':
                    size_compatibility = breed_size in ['small', 'medium', 'large']
                elif user_size == 'large':
                    size_compatibility = breed_size in ['medium', 'large']
                else:
                    size_compatibility = True

                if not size_compatibility:
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

            # 計算最終分數並加入自然變異
            breed_hash = hash(breed)
            random.seed(breed_hash)

            # Add small natural variation to avoid identical scores
            natural_variation = random.uniform(-0.008, 0.008)
            breed_bonus = round(breed_bonus + natural_variation, 4)
            final_score = round(min(1.0, compatibility_scores['overall'] + breed_bonus), 4)

            recommendations.append({
                'breed': breed,
                'base_score': round(compatibility_scores['overall'], 4),
                'bonus_score': round(breed_bonus, 4),
                'final_score': final_score,
                'scores': compatibility_scores,
                'info': breed_info,
                'noise_info': noise_info
            })

        print(f"Breeds after filtering: {len(recommendations)}")

        # 按照 final_score 排序
        recommendations.sort(key=lambda x: (round(-x['final_score'], 4), x['breed']))

        # 修正後的推薦選擇邏輯，移除有問題的分數比較
        final_recommendations = []

        # 直接選取前 top_n 個品種，確保返回完整數量
        for i, rec in enumerate(recommendations[:top_n]):
            rec['rank'] = i + 1
            final_recommendations.append(rec)

        print(f"Final recommendations count: {len(final_recommendations)}")

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


def _format_dimension_scores(dimension_scores: Dict) -> str:
    """Format individual dimension scores as badges"""
    if not dimension_scores:
        return ""

    badges_html = '<div class="dimension-badges">'

    for dimension, score in dimension_scores.items():
        if isinstance(score, (int, float)):
            score_percent = score * 100
        else:
            score_percent = 75  # default

        if score_percent >= 80:
            badge_class = "badge-high"
        elif score_percent >= 60:
            badge_class = "badge-medium"
        else:
            badge_class = "badge-low"

        dimension_label = dimension.replace('_', ' ').title()
        badges_html += f'''
        <span class="dimension-badge {badge_class}">
            {dimension_label}: {score_percent:.0f}%
        </span>
        '''

    badges_html += '</div>'
    return badges_html


def calculate_breed_bonus_factors(breed_info: dict, user_prefs: 'UserPreferences') -> tuple:
    """計算品種額外加分因素並返回原因列表"""
    bonus = 0.0
    reasons = []
    
    # 壽命加分
    try:
        lifespan = breed_info.get('Lifespan', '10-12 years')
        years = [int(x) for x in lifespan.split('-')[0].split()[0:1]]
        if max(years) >= 12:
            bonus += 0.02
            reasons.append("Above-average lifespan")
    except:
        pass

    # 性格特徵加分  
    temperament = breed_info.get('Temperament', '').lower()
    if any(trait in temperament for trait in ['friendly', 'gentle', 'affectionate']):
        bonus += 0.01
        reasons.append("Positive temperament traits")

    # 與孩童相容性
    if breed_info.get('Good with Children') == 'Yes':
        bonus += 0.01
        reasons.append("Excellent with children")

    return bonus, reasons


def generate_breed_characteristics_data(breed_info: dict) -> List[tuple]:
    """生成品種特徵資料列表"""
    return [
        ('Size', breed_info.get('Size', 'Unknown')),
        ('Exercise Needs', breed_info.get('Exercise Needs', 'Moderate')),
        ('Grooming Needs', breed_info.get('Grooming Needs', 'Moderate')),
        ('Good with Children', breed_info.get('Good with Children', 'Yes')),
        ('Temperament', breed_info.get('Temperament', '')),
        ('Lifespan', breed_info.get('Lifespan', '10-12 years')),
        ('Description', breed_info.get('Description', ''))
    ]


def parse_noise_information(noise_info: dict) -> tuple:
    """解析噪音資訊並返回結構化資料"""
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

    return noise_characteristics, barking_triggers, noise_level


def parse_health_information(health_info: dict) -> tuple:
    """解析健康資訊並返回結構化資料"""
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

    return health_considerations, health_screenings


def generate_dimension_scores_for_display(base_score: float, rank: int, breed: str,
                                        semantic_score: float = 0.7,
                                        comparative_bonus: float = 0.0,
                                        lifestyle_bonus: float = 0.0,
                                        is_description_search: bool = False,
                                        user_input: str = "") -> dict:
    """
    為顯示生成維度分數 - 基於真實品種特性

    這個函數現在會考慮品種的實際特性來計算分數，
    而不是僅僅基於總分生成假分數。
    """
    from dog_database import get_dog_description

    # 獲取品種資訊
    breed_name = breed.replace(' ', '_')
    breed_info = get_dog_description(breed_name) or {}

    temperament = breed_info.get('Temperament', '').lower()
    size = breed_info.get('Size', 'Medium').lower()
    exercise_needs = breed_info.get('Exercise Needs', 'Moderate').lower()
    grooming_needs = breed_info.get('Grooming Needs', 'Moderate').lower()
    good_with_children = breed_info.get('Good with Children', 'Yes')
    care_level = breed_info.get('Care Level', 'Moderate').lower()

    user_text = user_input.lower() if user_input else ""

    if is_description_search:
        # === 真實維度評分 ===

        # 1. Space Compatibility
        space_score = 0.75
        if 'small' in size:
            space_score = 0.85
        elif 'medium' in size:
            space_score = 0.75
        elif 'large' in size:
            space_score = 0.65
        elif 'giant' in size:
            space_score = 0.55

        # 2. Exercise Compatibility
        exercise_score = 0.75
        if 'low' in exercise_needs:
            exercise_score = 0.85
        elif 'moderate' in exercise_needs:
            exercise_score = 0.75
        elif 'high' in exercise_needs:
            exercise_score = 0.60
        elif 'very high' in exercise_needs:
            exercise_score = 0.50

        # 3. Grooming/Maintenance
        grooming_score = 0.75
        if 'low' in grooming_needs:
            grooming_score = 0.85
        elif 'moderate' in grooming_needs:
            grooming_score = 0.70
        elif 'high' in grooming_needs:
            grooming_score = 0.55

        # 敏感品種需要額外照顧
        if 'sensitive' in temperament:
            grooming_score -= 0.10

        # 4. Experience Compatibility - 關鍵！
        # DEBUG: 如果這個函數被呼叫，experience 會基於真實品種特性計算
        experience_score = 0.70

        # 基於 care level 的基礎分數
        if 'low' in care_level:
            experience_score = 0.85
        elif 'moderate' in care_level:
            experience_score = 0.70
        elif 'high' in care_level:
            experience_score = 0.55

        # DEBUG 標記：打印正在計算的品種
        print(f"[REAL_SCORE] Calculating experience for {breed}: care_level={care_level}, temperament={temperament}")

        # 性格對經驗的影響
        difficult_traits = {
            'sensitive': -0.15,
            'stubborn': -0.12,
            'independent': -0.10,
            'dominant': -0.12,
            'aggressive': -0.20,
            'nervous': -0.12,
            'alert': -0.05,
            'shy': -0.08,
            'timid': -0.08
        }
        for trait, penalty in difficult_traits.items():
            if trait in temperament:
                experience_score += penalty

        easy_traits = {
            'friendly': 0.08,
            'gentle': 0.08,
            'eager to please': 0.10,
            'patient': 0.08,
            'calm': 0.08,
            'outgoing': 0.05,
            'playful': 0.03
        }
        for trait, bonus in easy_traits.items():
            if trait in temperament:
                experience_score += bonus

        # 5. Noise Compatibility
        noise_score = 0.75
        if any(term in temperament for term in ['quiet', 'calm', 'gentle']):
            noise_score = 0.85
        elif any(term in temperament for term in ['alert', 'vocal']):
            noise_score = 0.60

        # 6. Family Compatibility
        family_score = 0.70
        if good_with_children == 'Yes' or good_with_children == True:
            family_score = 0.80
            if any(term in temperament for term in ['gentle', 'patient', 'friendly']):
                family_score = 0.90
        elif good_with_children == 'No' or good_with_children == False:
            family_score = 0.45

        # 確保分數在合理範圍內
        scores = {
            'space': max(0.30, min(0.95, space_score)),
            'exercise': max(0.30, min(0.95, exercise_score)),
            'grooming': max(0.30, min(0.95, grooming_score)),
            'experience': max(0.30, min(0.95, experience_score)),
            'noise': max(0.30, min(0.95, noise_score)),
            'family': max(0.30, min(0.95, family_score)),
            'overall': base_score
        }
    else:
        # 傳統搜尋結果的分數結構會在呼叫處理中傳入
        scores = {'overall': base_score}

    return scores
