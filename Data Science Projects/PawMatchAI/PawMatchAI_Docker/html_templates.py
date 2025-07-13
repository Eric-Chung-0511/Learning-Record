from typing import Dict, List, Union, Any, Optional, Callable
from urllib.parse import quote
from breed_health_info import breed_health_info
from breed_noise_info import breed_noise_info

def get_akc_breeds_link(breed: str) -> str:
    """Generate AKC breed page URL with intelligent name handling."""
    breed_name = breed.lower()
    breed_name = breed_name.replace('_', '-')
    breed_name = breed_name.replace("'", '')
    breed_name = breed_name.replace(" ", '-')

    special_cases = {
        'mexican-hairless': 'xoloitzcuintli',
        'brabancon-griffon': 'brussels-griffon',
        'bull-mastiff': 'bullmastiff',
        'walker-hound': 'treeing-walker-coonhound'
    }

    breed_name = special_cases.get(breed_name, breed_name)
    return f"https://www.akc.org/dog-breeds/{breed_name}/"

def get_color_scheme(is_single_dog: bool) -> Union[str, List[str]]:
    """Get color scheme for dog detection visualization."""
    single_dog_color = '#34C759'  # 清爽的綠色作為單狗顏色
    color_list = [
        '#FF5733',  # 珊瑚紅
        '#28A745',  # 深綠色
        '#3357FF',  # 寶藍色
        '#FF33F5',  # 粉紫色
        '#FFB733',  # 橙黃色
        '#33FFF5',  # 青藍色
        '#A233FF',  # 紫色
        '#FF3333',  # 紅色
        '#33FFB7',  # 青綠色
        '#FFE033'   # 金黃色
    ]
    return single_dog_color if is_single_dog else color_list

def format_hint_html(message: str) -> str:
    """
    提示訊息的 HTML。

    Args:
        message: str, 要顯示的提示訊息
    Returns:
        str: 格式化後的 HTML 字符串
    """
    return f'''
    <div style="
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4A90E2;
    ">
        <div style="
            display: flex;
            align-items: center;
            gap: 10px;
            color: #2C5282;
            font-weight: 500;
        ">
            <span style="font-size: 24px">💡</span>
            <span>{message}</span>
        </div>
    </div>
    '''

def format_unknown_breed_message(color: str, index: int) -> str:
    """
    當狗的品種無法被辨識時（信心度低於0.2）使用此函數格式化錯誤訊息。
    這種情況通常發生在：
    1. 圖片品質不佳
    2. 狗的品種不在我們的資料集中
    3. 拍攝角度不理想
    """
    return f'''
    <div class="dog-info-card" style="border-left: 8px solid {color};">
        <div class="dog-info-header" style="background-color: {color}10;">
            <span class="dog-label" style="color: {color};">Dog {index}</span>
        </div>
        <div class="breed-info">
            <div class="warning-message">
                <span class="icon">⚠️</span>
                Unable to identify the dog breed. This breed might not be included in the dataset.
            </div>
        </div>
    </div>
    '''

def format_not_dog_message(color: str, index: int) -> str:
    """
    當YOLO模型檢測到物體不是狗時使用此函數格式化錯誤訊息。
    這是第一層的過濾機制，在進行品種分類之前就先確認是否為狗。
    """
    return f'''
    <div class="dog-info-card" style="border-left: 8px solid {color};">
        <div class="dog-info-header" style="background-color: {color}10;">
            <span class="dog-label" style="color: {color};">Object {index}</span>
        </div>
        <div class="breed-info">
            <div class="warning-message">
                <span class="icon">❌</span>
                This does not appear to be a dog. Please upload an image containing a dog.
            </div>
        </div>
    </div>
    '''

def format_description_html(description: Dict[str, Any], breed: str) -> str:
    """Format basic breed description with tooltips."""
    if not isinstance(description, dict):
        return f"<p>{description}</p>"

    fields_order = [
        "Size", "Lifespan", "Temperament", "Exercise Needs",
        "Grooming Needs", "Care Level", "Good with Children",
        "Description"
    ]

    html_parts = []
    for field in fields_order:
        if field in description:
            value = description[field]
            tooltip_html = format_tooltip(field, value)
            html_parts.append(f'<li style="margin-bottom: 10px;">{tooltip_html}</li>')

    # Add any remaining fields
    for key, value in description.items():
        if key not in fields_order and key != "Breed":
            html_parts.append(f'<li style="margin-bottom: 10px;"><strong>{key}:</strong> {value}</li>')

    return f'<ul style="list-style-type: none; padding-left: 0;">{" ".join(html_parts)}</ul>'


def format_tooltip(key: str, value: str) -> str:
    """Format tooltip with content for each field."""
    tooltip_contents = {
        "Size": {
            "title": "Size Categories",
            "items": [
                "Small: Under 20 pounds",
                "Medium: 20-60 pounds",
                "Large: Over 60 pounds",
                "Giant: Over 100 pounds",
                "Varies: Depends on variety"
            ]
        },
        "Exercise Needs": {
            "title": "Exercise Needs",
            "items": [
                "Low: Short walks and play sessions",
                "Moderate: 1-2 hours of daily activity",
                "High: Extensive exercise (2+ hours/day)",
                "Very High: Constant activity and mental stimulation needed"
            ]
        },
        "Grooming Needs": {
            "title": "Grooming Requirements",
            "items": [
                "Low: Basic brushing, occasional baths",
                "Moderate: Weekly brushing, occasional grooming",
                "High: Daily brushing, frequent professional grooming needed",
                "Professional care recommended for all levels"
            ]
        },
        "Care Level": {
            "title": "Care Level Explained",
            "items": [
                "Low: Basic care and attention needed",
                "Moderate: Regular care and routine needed",
                "High: Significant time and attention needed",
                "Very High: Extensive care, training and attention required"
            ]
        },
        "Good with Children": {
            "title": "Child Compatibility",
            "items": [
                "Yes: Excellent with kids, patient and gentle",
                "Moderate: Good with older children",
                "No: Better suited for adult households"
            ]
        },
        "Lifespan": {
            "title": "Average Lifespan",
            "items": [
                "Short: 6-8 years",
                "Average: 10-15 years",
                "Long: 12-20 years",
                "Varies by size: Larger breeds typically have shorter lifespans"
            ]
        },
        "Temperament": {
            "title": "Temperament Guide",
            "items": [
                "Describes the dog's natural behavior and personality",
                "Important for matching with owner's lifestyle",
                "Can be influenced by training and socialization"
            ]
        }
    }

    tooltip = tooltip_contents.get(key, {"title": key, "items": []})
    tooltip_content = "<br>".join([f"• {item}" for item in tooltip["items"]])

    return f'''
        <span class="tooltip">
            <strong>{key}:</strong>
            <span class="tooltip-icon">ⓘ</span>
            <span class="tooltip-text">
                <strong>{tooltip["title"]}:</strong><br>
                {tooltip_content}
            </span>
        </span> {value}
    '''

def format_single_dog_result(breed: str, description: Dict[str, Any], color: str = "#34C759") -> str:
    """Format single dog detection result into HTML."""
    # 獲取noise和health資訊
    noise_info = breed_noise_info.get(breed, {})
    health_info = breed_health_info.get(breed, {})

    # 處理噪音資訊
    noise_notes = noise_info.get('noise_notes', '').split('\n')
    noise_characteristics = []
    barking_triggers = []
    noise_level = noise_info.get('noise_level', 'Information not available')

    in_section = None
    for line in noise_notes:
        line = line.strip()
        if 'Typical noise characteristics:' in line:
            in_section = 'characteristics'
        elif 'Barking triggers:' in line:
            in_section = 'triggers'
        elif line.startswith('•'):
            if in_section == 'characteristics':
                noise_characteristics.append(line[1:].strip())
            elif in_section == 'triggers':
                barking_triggers.append(line[1:].strip())

    # 處理健康資訊
    health_notes = health_info.get('health_notes', '').split('\n')
    health_considerations = []
    health_screenings = []

    in_section = None
    for line in health_notes:
        line = line.strip()
        if 'Common breed-specific health considerations' in line:
            in_section = 'considerations'
        elif 'Recommended health screenings:' in line:
            in_section = 'screenings'
        elif line.startswith('•'):
            if in_section == 'considerations':
                health_considerations.append(line[1:].strip())
            elif in_section == 'screenings':
                health_screenings.append(line[1:].strip())
    display_breeds_name = breed.replace('_', ' ')

    return f'''
        <div class="dog-info-card" style="background: {color}10;">
            <div class="breed-title" style="display: flex; align-items: center; gap: 10px; margin: 0 0 20px 20px;">
                <span class="icon" style="font-size: 2.0em;">🐾</span>
                <h2 style="
                    margin: 0;
                    padding: 10px 20px;
                    background: ${color}15;
                    border-left: 4px solid ${color};
                    border-radius: 6px;
                    color: ${color};
                    font-weight: 600;
                    font-size: 2.0em;
                    letter-spacing: 0.5px;">
                    {display_breeds_name}
                </h2>
            </div>
            <div class="breed-info">
                <!-- Basic Information -->
                <div class="section-header" style="margin-bottom: 15px;">
                    <h3>
                        <span class="icon">📋</span> BASIC INFORMATION
                    </h3>
                </div>
                <div class="info-cards" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px;">
                    <div class="info-card" style="background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <span class="tooltip">
                            <span class="icon">📏</span>
                            <span class="label">Size</span>
                            <span class="tooltip-icon">ⓘ</span>
                            <span class="tooltip-text">
                                <strong>Size Categories:</strong><br>
                                • Small: Under 20 pounds<br>
                                • Medium: 20-60 pounds<br>
                                • Large: Over 60 pounds<br>
                                • Giant: Over 100 pounds
                            </span>
                        </span>
                        <span>{description['Size']}</span>
                    </div>
                    <div class="info-card" style="background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <span class="tooltip">
                            <span class="icon">⏳</span>
                            <span class="label">Lifespan</span>
                            <span class="tooltip-icon">ⓘ</span>
                            <span class="tooltip-text">
                                <strong>Lifespan Categories:</strong><br>
                                • Short: 6-8 years<br>
                                • Average: 10-15 years<br>
                                • Long: 12-20 years
                            </span>
                        </span>
                        <span>{description['Lifespan']}</span>
                    </div>
                </div>
                <!-- Care Requirements -->
                <div class="section-header" style="margin-bottom: 15px;">
                    <h3>
                        <span class="icon">💪</span> CARE REQUIREMENTS
                    </h3>
                </div>
                <div class="info-cards" style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 30px;">
                    <div class="info-card" style="background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <span class="tooltip">
                            <span class="icon">🏃</span>
                            <span class="label">Exercise</span>
                            <span class="tooltip-icon">ⓘ</span>
                            <span class="tooltip-text">
                                <strong>Exercise Needs:</strong><br>
                                • Low: Short walks<br>
                                • Moderate: 1-2 hours daily<br>
                                • High: 2+ hours daily
                            </span>
                        </span>
                        <span>{description['Exercise Needs']}</span>
                    </div>
                    <div class="info-card" style="background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <span class="tooltip">
                            <span class="icon">✂️</span>
                            <span class="label">Grooming</span>
                            <span class="tooltip-icon">ⓘ</span>
                            <span class="tooltip-text">
                                <strong>Grooming Requirements:</strong><br>
                                • Low: Basic brushing<br>
                                • Moderate: Weekly grooming<br>
                                • High: Daily maintenance
                            </span>
                        </span>
                        <span>{description['Grooming Needs']}</span>
                    </div>
                    <div class="info-card" style="background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <span class="tooltip">
                            <span class="icon">⭐</span>
                            <span class="label">Care Level</span>
                            <span class="tooltip-icon">ⓘ</span>
                            <span class="tooltip-text">
                                <strong>Care Level:</strong><br>
                                • Low: Basic care<br>
                                • Moderate: Regular care<br>
                                • High: Extensive care
                            </span>
                        </span>
                        <span>{description['Care Level']}</span>
                    </div>
                </div>
                <!-- Noise Behavior -->
                <div class="section-header" style="margin-bottom: 15px;">
                    <h3>
                        <span class="tooltip">
                            <span class="icon">🔊</span>
                            <span>NOISE BEHAVIOR</span>
                            <span class="tooltip-icon">ⓘ</span>
                            <span class="tooltip-text">
                                <strong>Noise Behavior:</strong><br>
                                • Typical vocalization patterns<br>
                                • Common triggers and frequency<br>
                                • Based on breed characteristics
                            </span>
                        </span>
                    </h3>
                </div>
                <div class="noise-section" style="background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 30px;">
                    <div class="noise-level" style="margin-bottom: 15px;">
                        <span class="label">Noise Level:</span>
                        <span class="value">{noise_level}</span>
                    </div>
                    <div class="characteristics-list">
                        {format_noise_items(noise_characteristics[:2])}
                    </div>
                    <details class="expandable-section" style="margin-top: 15px;">
                        <summary class="expand-header" style="cursor: pointer; padding: 10px; background: #f8f9fa; border-radius: 6px;">
                            Show Complete Noise Information
                            <span class="expand-icon">▼</span>
                        </summary>
                        <div class="expanded-content" style="margin-top: 15px;">
                            <div class="characteristics-list">
                                <h4>All Characteristics:</h4>
                                {format_noise_items(noise_characteristics)}
                            </div>
                            <div class="triggers-list">
                                <h4>Barking Triggers:</h4>
                                {format_noise_items(barking_triggers)}
                            </div>
                            <div class="disclaimer-section" style="margin-top: 15px; font-size: 0.9em; color: #666;">
                                <p class="disclaimer-text source-text">Source: Compiled from various breed behavior resources, 2025</p>
                                <p class="disclaimer-text">Individual dogs may vary in their vocalization patterns.</p>
                                <p class="disclaimer-text">Training can significantly influence barking behavior.</p>
                                <p class="disclaimer-text">Environmental factors may affect noise levels.</p>
                            </div>
                        </div>
                    </details>
                </div>
                <!-- Health Insights -->
                <div class="section-header" style="margin-bottom: 15px;">
                    <h3>
                        <span class="tooltip">
                            <span class="icon">🏥</span>
                            <span>HEALTH INSIGHTS</span>
                            <span class="tooltip-icon">ⓘ</span>
                            <span class="tooltip-text">
                                <strong>Health Information:</strong><br>
                                • Common breed-specific conditions<br>
                                • Recommended health screenings<br>
                                • General health considerations
                            </span>
                        </span>
                    </h3>
                </div>
                <div class="health-section" style="background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 30px;">
                    <div class="key-considerations">
                        {format_health_items(health_considerations[:2])}
                    </div>
                    <details class="expandable-section" style="margin-top: 15px;">
                        <summary class="expand-header" style="cursor: pointer; padding: 10px; background: #f8f9fa; border-radius: 6px;">
                            Show Complete Health Information
                            <span class="expand-icon">▼</span>
                        </summary>
                        <div class="expanded-content" style="margin-top: 15px;">
                            <div class="considerations-list">
                                <h4>All Health Considerations:</h4>
                                {format_health_items(health_considerations)}
                            </div>
                            <div class="screenings-list">
                                <h4>Recommended Screenings:</h4>
                                {format_health_items(health_screenings)}
                            </div>
                            <div class="disclaimer-section" style="margin-top: 15px; font-size: 0.9em; color: #666;">
                                <p class="disclaimer-text source-text">Source: Compiled from various veterinary and breed information resources, 2025</p>
                                <p class="disclaimer-text">This information is for reference only and based on breed tendencies.</p>
                                <p class="disclaimer-text">Each dog is unique and may not develop any or all of these conditions.</p>
                                <p class="disclaimer-text">Always consult with qualified veterinarians for professional advice.</p>
                            </div>
                        </div>
                    </details>
                </div>
                <!-- Description -->
                <div class="section-header" style="margin-bottom: 15px;">
                    <h3>
                        <span class="icon">📝</span> DESCRIPTION
                    </h3>
                </div>
                <div class="description-section" style="background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 30px;">
                    <p style="line-height: 1.6;">{description.get('Description', '')}</p>
                </div>
                <!-- Action Section -->
                <div class="action-section" style="text-align: center; margin-top: 30px;">
                    <a href="{get_akc_breeds_link(breed)}"
                       target="_blank"
                       class="akc-button"
                       style="display: inline-block; padding: 12px 24px; background: linear-gradient(90deg, #4299e1, #48bb78); color: white; text-decoration: none; border-radius: 6px;">
                        <span class="icon">🌐</span>
                        Learn more about {display_breeds_name} on AKC website
                    </a>
                </div>
            </div>
        </div>
    '''


def format_noise_items(items: List[str]) -> str:
    """Format noise-related items into HTML list items."""
    if not items:
        return "<div class='list-item'>Information not available</div>"
    return "\n".join([f"<div class='list-item'>• {item}</div>" for item in items])

def format_health_items(items: List[str]) -> str:
    """Format health-related items into HTML list items."""
    if not items:
        return "<div class='list-item'>Information not available</div>"
    return "\n".join([f"<div class='list-item'>• {item}</div>" for item in items])


def format_multiple_breeds_result(
    topk_breeds: List[str],
    relative_probs: List[str],
    color: str,
    index: int,
    get_dog_description: Callable
) -> str:
    """Format multiple breed predictions into HTML with complete information."""
    display_breeds = [breed.replace('_', ' ') for breed in topk_breeds]

    result = f'''
        <!-- 主標題區塊 -->
        <div style="background: {color}10; border-radius: 12px 12px 0 0; padding: 20px;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 1.6em; color: {color}; margin-right: 12px;">🐾</span>
                <span style="font-size: 1.4em; font-weight: 600; color: {color};">Dog {index}</span>
            </div>
        </div>
        <!-- 內容區塊 -->
        <div style="padding: 20px;">
            <!-- 不確定性提示 -->
            <div style="margin-bottom: 20px;">
                <div style="display: flex; align-items: center; gap: 8px; padding: 12px; background: #f8f9fa; border-radius: 8px;">
                    <span>ℹ️</span>
                    <span>Note: The model is showing some uncertainty in its predictions.
                    Here are the most likely breeds based on the available visual features.</span>
                </div>
            </div>
            <!-- 品種列表容器 -->
            <div class="breeds-list" style="display: grid; gap: 20px;">
    '''

    for j, (breed, display_name, prob) in enumerate(zip(topk_breeds, display_breeds, relative_probs)):
        description = get_dog_description(breed)
        noise_info = breed_noise_info.get(breed, {})
        health_info = breed_health_info.get(breed, {})

        # 處理噪音資訊
        noise_notes = noise_info.get('noise_notes', '').split('\n')
        noise_characteristics = []
        barking_triggers = []
        noise_level = noise_info.get('noise_level', 'Information not available')

        in_section = None
        for line in noise_notes:
            line = line.strip()
            if 'Typical noise characteristics:' in line:
                in_section = 'characteristics'
            elif 'Barking triggers:' in line:
                in_section = 'triggers'
            elif line.startswith('•'):
                if in_section == 'characteristics':
                    noise_characteristics.append(line[1:].strip())
                elif in_section == 'triggers':
                    barking_triggers.append(line[1:].strip())

        # 處理健康資訊
        health_notes = health_info.get('health_notes', '').split('\n')
        health_considerations = []
        health_screenings = []

        in_section = None
        for line in health_notes:
            line = line.strip()
            if 'Common breed-specific health considerations' in line:
                in_section = 'considerations'
            elif 'Recommended health screenings:' in line:
                in_section = 'screenings'
            elif line.startswith('•'):
                if in_section == 'considerations':
                    health_considerations.append(line[1:].strip())
                elif in_section == 'screenings':
                    health_screenings.append(line[1:].strip())

        result += f'''
          <div class="dog-info-card" style="background: #fff; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 48px;">
              <div style="border-left: 8px solid {color};">
                  <!-- Title Section -->
                  <div class="breed-title" style="padding: 24px; background: #f8f9fa;">
                      <div style="display: flex; align-items: center; justify-content: space-between; padding: 12px; background: #fff; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                          <div style="display: flex; align-items: center; gap: 14px;">
                              <span style="font-size: 1.4em;">🐾</span>
                              <h2 style="margin: 0; font-size: 1.8em; color: #1a202c; font-weight: 600;">
                                  {'Option ' + str(j+1) + ': ' if prob else ''}{display_name}
                              </h2>
                          </div>
                          {f'<span style="background: {color}12; color: {color}; padding: 8px 16px; border-radius: 8px; font-size: 1em; font-weight: 500; box-shadow: 0 1px 2px {color}20;">Confidence: {prob}</span>' if prob else ''}
                      </div>
                  </div>
                  <div class="breed-info" style="padding: 24px;">
                      <!-- Basic Information -->
                      <div style="margin-bottom: 32px;">
                          <h3 style="display: flex; align-items: center; gap: 10px; margin: 0 0 20px 0; padding: 12px; background: #f8f9fa; border-radius: 6px;">
                              <span style="font-size: 1.2em;">📋</span>
                              <span style="font-size: 1.2em; font-weight: 600; color: #2d3748;">BASIC INFORMATION</span>
                          </h3>
                          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
                              <!-- Size -->
                              <div style="padding: 16px; border-radius: 10px; background: #fff; border: 1px solid #e2e8f0;">
                                  <div style="display: flex; align-items: center; gap: 10px;">
                                      <span style="font-size: 1.1em;">📏</span>
                                      <span style="font-weight: 500;">Size</span>
                                      <div style="position: relative; display: inline-block;"
                                          onmouseover="this.querySelector('.tooltip-content').style.visibility='visible';this.querySelector('.tooltip-content').style.opacity='1';"
                                          onmouseout="this.querySelector('.tooltip-content').style.visibility='hidden';this.querySelector('.tooltip-content').style.opacity='0';">
                                          <span style="cursor: help; color: #718096;">ⓘ</span>
                                          <div class="tooltip-content" style="
                                              visibility: hidden;
                                              opacity: 0;
                                              position: absolute;
                                              background: #2C3E50;
                                              color: white;
                                              padding: 12px;
                                              border-radius: 8px;
                                              font-size: 14px;
                                              width: 250px;
                                              {f'top: -130px; left: 0;' if not prob else 'top: 50%; right: -270px; transform: translateY(-50%); left: auto;'};
                                              z-index: 99999;
                                              box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                                              pointer-events: none;">
                                              <strong style="display: block; margin-bottom: 8px; color: white;">Size Categories:</strong>
                                              • Small: Under 20 pounds<br>
                                              • Medium: 20-60 pounds<br>
                                              • Large: Over 60 pounds<br>
                                              • Giant: Over 100 pounds
                                              <div style="position: absolute;
                                                  {f'left: 20px; bottom: -8px; border-top: 8px solid #2C3E50;' if not prob else 'top: 50%; right: 100%; transform: translateY(-50%); border-right: 8px solid #2C3E50;'};
                                                  width: 0;
                                                  height: 0;
                                                  border-left: 8px solid transparent;
                                                  border-right: 8px solid transparent;">
                                              </div>
                                          </div>
                                      </div>
                                  </div>
                                  <span style="display: block; margin-top: 8px; color: #4a5568;">{description['Size']}</span>
                              </div>
                              <!-- Lifespan -->
                              <div style="padding: 16px; border-radius: 10px; background: #fff; border: 1px solid #e2e8f0;">
                                  <div style="display: flex; align-items: center; gap: 10px;">
                                      <span style="font-size: 1.1em;">⏳</span>
                                      <span style="font-weight: 500;">Lifespan</span>
                                      <div style="position: relative; display: inline-block;"
                                          onmouseover="this.querySelector('.tooltip-content').style.visibility='visible';this.querySelector('.tooltip-content').style.opacity='1';"
                                          onmouseout="this.querySelector('.tooltip-content').style.visibility='hidden';this.querySelector('.tooltip-content').style.opacity='0';">
                                          <span style="cursor: help; color: #718096;">ⓘ</span>
                                          <div class="tooltip-content" style="
                                              visibility: hidden;
                                              opacity: 0;
                                              position: absolute;
                                              background: #2C3E50;
                                              color: white;
                                              padding: 12px;
                                              border-radius: 8px;
                                              font-size: 14px;
                                              width: 250px;
                                              {f'top: -130px; left: 0;' if not prob else 'top: 50%; right: -270px; transform: translateY(-50%); left: auto;'};
                                              z-index: 99999;
                                              box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                                              pointer-events: none;">
                                              <strong style="display: block; margin-bottom: 8px; color: white;">Lifespan Categories:</strong>
                                              • Short: 6-8 years<br>
                                              • Average: 10-15 years<br>
                                              • Long: 12-20 years
                                              <div style="position: absolute;
                                                  {f'left: 20px; bottom: -8px; border-top: 8px solid #2C3E50;' if not prob else 'top: 50%; right: 100%; transform: translateY(-50%); border-right: 8px solid #2C3E50;'};
                                                  width: 0;
                                                  height: 0;
                                                  border-left: 8px solid transparent;
                                                  border-right: 8px solid transparent;">
                                              </div>
                                          </div>
                                      </div>
                                  </div>
                                  <span style="display: block; margin-top: 8px; color: #4a5568;">{description['Lifespan']}</span>
                              </div>
                          </div>
                      </div>
                      <!-- Care Requirements -->
                      <div style="margin-bottom: 32px;">
                          <h3 style="display: flex; align-items: center; gap: 10px; margin: 0 0 20px 0; padding: 12px; background: #f8f9fa; border-radius: 6px;">
                              <span style="font-size: 1.2em;">💪</span>
                              <span style="font-size: 1.2em; font-weight: 600; color: #2d3748;">CARE REQUIREMENTS</span>
                          </h3>
                          <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 24px;">
                              <!-- Exercise -->
                              <div style="padding: 16px; border-radius: 10px; background: #fff; border: 1px solid #e2e8f0;">
                                  <div style="display: flex; align-items: center; gap: 10px;">
                                      <span style="font-size: 1.1em;">🏃</span>
                                      <span style="font-weight: 500;">Exercise</span>
                                      <div style="position: relative; display: inline-block;"
                                          onmouseover="this.querySelector('.tooltip-content').style.visibility='visible';this.querySelector('.tooltip-content').style.opacity='1';"
                                          onmouseout="this.querySelector('.tooltip-content').style.visibility='hidden';this.querySelector('.tooltip-content').style.opacity='0';">
                                          <span style="cursor: help; color: #718096;">ⓘ</span>
                                          <div class="tooltip-content" style="
                                              visibility: hidden;
                                              opacity: 0;
                                              position: absolute;
                                              background: #2C3E50;
                                              color: white;
                                              padding: 12px;
                                              border-radius: 8px;
                                              font-size: 14px;
                                              width: 250px;
                                              {f'top: -130px; left: 0;' if not prob else 'top: 50%; right: -270px; transform: translateY(-50%); left: auto;'};
                                              z-index: 99999;
                                              box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                                              pointer-events: none;">
                                              <strong style="display: block; margin-bottom: 8px; color: white;">Exercise Needs:</strong>
                                              • Low: Short walks<br>
                                              • Moderate: 1-2 hours daily<br>
                                              • High: 2+ hours daily
                                              <div style="position: absolute;
                                                  {f'left: 20px; bottom: -8px; border-top: 8px solid #2C3E50;' if not prob else 'top: 50%; right: 100%; transform: translateY(-50%); border-right: 8px solid #2C3E50;'};
                                                  width: 0;
                                                  height: 0;
                                                  border-left: 8px solid transparent;
                                                  border-right: 8px solid transparent;">
                                              </div>
                                          </div>
                                      </div>
                                  </div>
                                  <span style="display: block; margin-top: 8px; color: #4a5568;">{description['Exercise Needs']}</span>
                              </div>
                              <!-- Grooming -->
                              <div style="padding: 16px; border-radius: 10px; background: #fff; border: 1px solid #e2e8f0;">
                                  <div style="display: flex; align-items: center; gap: 10px;">
                                      <span style="font-size: 1.1em;">✂️</span>
                                      <span style="font-weight: 500;">Grooming</span>
                                      <div style="position: relative; display: inline-block;"
                                          onmouseover="this.querySelector('.tooltip-content').style.visibility='visible';this.querySelector('.tooltip-content').style.opacity='1';"
                                          onmouseout="this.querySelector('.tooltip-content').style.visibility='hidden';this.querySelector('.tooltip-content').style.opacity='0';">
                                          <span style="cursor: help; color: #718096;">ⓘ</span>
                                          <div class="tooltip-content" style="
                                              visibility: hidden;
                                              opacity: 0;
                                              position: absolute;
                                              background: #2C3E50;
                                              color: white;
                                              padding: 12px;
                                              border-radius: 8px;
                                              font-size: 14px;
                                              width: 250px;
                                              {f'top: -130px; left: 0;' if not prob else 'top: 50%; right: -270px; transform: translateY(-50%); left: auto;'};
                                              z-index: 99999;
                                              box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                                              pointer-events: none;">
                                              <strong style="display: block; margin-bottom: 8px; color: white;">Grooming Requirements:</strong>
                                              • Low: Basic brushing<br>
                                              • Moderate: Weekly grooming<br>
                                              • High: Daily maintenance
                                              <div style="position: absolute;
                                                  {f'left: 20px; bottom: -8px; border-top: 8px solid #2C3E50;' if not prob else 'top: 50%; right: 100%; transform: translateY(-50%); border-right: 8px solid #2C3E50;'};
                                                  width: 0;
                                                  height: 0;
                                                  border-left: 8px solid transparent;
                                                  border-right: 8px solid transparent;">
                                              </div>
                                          </div>
                                      </div>
                                  </div>
                                  <span style="display: block; margin-top: 8px; color: #4a5568;">{description['Grooming Needs']}</span>
                              </div>
                              <!-- Care Level -->
                              <div style="padding: 16px; border-radius: 10px; background: #fff; border: 1px solid #e2e8f0;">
                                  <div style="display: flex; align-items: center; gap: 10px;">
                                      <span style="font-size: 1.1em;">⭐</span>
                                      <span style="font-weight: 500;">Care Level</span>
                                      <div style="position: relative; display: inline-block;"
                                          onmouseover="this.querySelector('.tooltip-content').style.visibility='visible';this.querySelector('.tooltip-content').style.opacity='1';"
                                          onmouseout="this.querySelector('.tooltip-content').style.visibility='hidden';this.querySelector('.tooltip-content').style.opacity='0';">
                                          <span style="cursor: help; color: #718096;">ⓘ</span>
                                          <div class="tooltip-content" style="
                                              visibility: hidden;
                                              opacity: 0;
                                              position: absolute;
                                              background: #2C3E50;
                                              color: white;
                                              padding: 12px;
                                              border-radius: 8px;
                                              font-size: 14px;
                                              width: 250px;
                                              {f'top: -130px; left: 0;' if not prob else 'top: 50%; right: -270px; transform: translateY(-50%); left: auto;'};
                                              z-index: 99999;
                                              box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                                              pointer-events: none;">
                                              <strong style="display: block; margin-bottom: 8px; color: white;">Care Level:</strong>
                                              • Low: Basic care<br>
                                              • Moderate: Regular care<br>
                                              • High: Extensive care
                                              <div style="position: absolute;
                                                  {f'left: 20px; bottom: -8px; border-top: 8px solid #2C3E50;' if not prob else 'top: 50%; right: 100%; transform: translateY(-50%); border-right: 8px solid #2C3E50;'};
                                                  width: 0;
                                                  height: 0;
                                                  border-left: 8px solid transparent;
                                                  border-right: 8px solid transparent;">
                                              </div>
                                          </div>
                                      </div>
                                  </div>
                                  <span style="display: block; margin-top: 8px; color: #4a5568;">{description['Care Level']}</span>
                              </div>
                          </div>
                      </div>
                      <!-- Noise Behavior -->
                      <div style="margin-bottom: 32px;">
                          <h3 style="display: flex; align-items: center; gap: 10px; margin: 0 0 20px 0; padding: 12px; background: #f8f9fa; border-radius: 6px;">
                              <span style="font-size: 1.2em;">🔊</span>
                              <span style="font-size: 1.2em; font-weight: 600; color: #2d3748;">NOISE BEHAVIOR</span>
                              <div style="position: relative; display: inline-block;"
                                  onmouseover="this.querySelector('.tooltip-content').style.visibility='visible';this.querySelector('.tooltip-content').style.opacity='1';"
                                  onmouseout="this.querySelector('.tooltip-content').style.visibility='hidden';this.querySelector('.tooltip-content').style.opacity='0';">
                                  <span style="cursor: help; color: #718096;">ⓘ</span>
                                  <div class="tooltip-content" style="
                                      visibility: hidden;
                                      opacity: 0;
                                      position: absolute;
                                      background: #2C3E50;
                                      color: white;
                                      padding: 12px;
                                      border-radius: 8px;
                                      font-size: 14px;
                                      width: 250px;
                                      {f'top: -130px; left: 0;' if not prob else 'top: 50%; right: -270px; transform: translateY(-50%); left: auto;'};
                                      z-index: 99999;
                                      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                                      pointer-events: none;">
                                      <strong style="display: block; margin-bottom: 8px; color: white;">Noise Behavior:</strong>
                                      • Typical vocalization patterns<br>
                                      • Common triggers and frequency<br>
                                      • Based on breed characteristics
                                      <div style="position: absolute;
                                          {f'left: 20px; bottom: -8px; border-top: 8px solid #2C3E50;' if not prob else 'top: 50%; right: 100%; transform: translateY(-50%); border-right: 8px solid #2C3E50;'};
                                          width: 0;
                                          height: 0;
                                          border-left: 8px solid transparent;
                                          border-right: 8px solid transparent;">
                                      </div>
                                  </div>
                              </div>
                          </h3>
                          <div style="background: #fff; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0;">
                              <div style="margin-bottom: 16px;">
                                  <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                                      <span style="font-weight: 500;">Noise Level:</span>
                                      <span>{noise_level}</span>
                                  </div>
                                  <div style="margin-bottom: 16px;">
                                      {format_noise_items(noise_characteristics[:2])}
                                  </div>
                              </div>
                              <details style="margin-top: 20px;">
                                  <summary style="cursor: pointer; padding: 12px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e2e8f0; outline: none; list-style-type: none;">
                                      <span style="display: flex; align-items: center; justify-content: space-between;">
                                          <span style="font-weight: 500;">Show Complete Noise Information</span>
                                          <span style="transition: transform 0.2s;">▶</span>
                                      </span>
                                  </summary>
                                  <!-- 展開內容部分 -->
                                  <div style="margin-top: 16px; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; background: #fff;">
                                      <div style="margin-bottom: 24px;">
                                          <h4 style="margin: 0 0 12px 0; color: #2d3748;">All Characteristics</h4>
                                          {format_noise_items(noise_characteristics)}
                                      </div>
                                      <div style="margin-bottom: 24px;">
                                          <h4 style="margin: 0 0 12px 0; color: #2d3748;">Barking Triggers</h4>
                                          {format_noise_items(barking_triggers)}
                                      </div>
                                      <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid #e2e8f0;">
                                          <p style="margin: 0 0 8px 0; color: #4a5568; font-size: 0.9em;">Source: Compiled from various breed behavior resources, 2025</p>
                                          <p style="margin: 0 0 8px 0; color: #4a5568; font-size: 0.9em;">Individual dogs may vary in their vocalization patterns.</p>
                                          <p style="margin: 0 0 8px 0; color: #4a5568; font-size: 0.9em;">Training can significantly influence barking behavior.</p>
                                          <p style="margin: 0 0 8px 0; color: #4a5568; font-size: 0.9em;">Environmental factors may affect noise levels.</p>
                                      </div>
                                  </div>
                              </details>
                          </div>
                      </div>
                      <!-- Health Insights -->
                      <div style="margin-bottom: 32px;">
                          <h3 style="display: flex; align-items: center; gap: 10px; margin: 0 0 20px 0; padding: 12px; background: #f8f9fa; border-radius: 6px;">
                              <span style="font-size: 1.2em;">🏥</span>
                              <span style="font-size: 1.2em; font-weight: 600; color: #2d3748;">HEALTH INSIGHTS</span>
                              <div style="position: relative; display: inline-block;"
                                  onmouseover="this.querySelector('.tooltip-content').style.visibility='visible';this.querySelector('.tooltip-content').style.opacity='1';"
                                  onmouseout="this.querySelector('.tooltip-content').style.visibility='hidden';this.querySelector('.tooltip-content').style.opacity='0';">
                                  <span style="cursor: help; color: #718096;">ⓘ</span>
                                  <div class="tooltip-content" style="
                                      visibility: hidden;
                                      opacity: 0;
                                      position: absolute;
                                      background: #2C3E50;
                                      color: white;
                                      padding: 12px;
                                      border-radius: 8px;
                                      font-size: 14px;
                                      width: 250px;
                                      {f'top: -130px; left: 0;' if not prob else 'top: 50%; right: -270px; transform: translateY(-50%); left: auto;'};
                                      z-index: 99999;
                                      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                                      pointer-events: none;">
                                      <strong style="display: block; margin-bottom: 8px; color: white;">Health Information:</strong>
                                      • Common breed-specific conditions<br>
                                      • Recommended health screenings<br>
                                      • General health considerations
                                      <div style="position: absolute;
                                          {f'left: 20px; bottom: -8px; border-top: 8px solid #2C3E50;' if not prob else 'top: 50%; right: 100%; transform: translateY(-50%); border-right: 8px solid #2C3E50;'};
                                          width: 0;
                                          height: 0;
                                          border-left: 8px solid transparent;
                                          border-right: 8px solid transparent;">
                                      </div>
                                  </div>
                              </div>
                          </h3>
                          <div style="background: #fff; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0;">
                              <div style="margin-bottom: 16px;">
                                  {format_health_items(health_considerations[:2])}
                              </div>
                              <details style="margin-top: 20px;">
                                  <summary style="cursor: pointer; padding: 12px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e2e8f0; outline: none; list-style-type: none;">
                                      <span style="display: flex; align-items: center; justify-content: space-between;">
                                          <span style="font-weight: 500;">Show Complete Health Information</span>
                                          <span style="transition: transform 0.2s;">▶</span>
                                      </span>
                                  </summary>
                                  <div style="margin-top: 16px; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; background: #fff;">
                                      <div style="margin-bottom: 24px;">
                                          <h4 style="margin: 0 0 12px 0; color: #2d3748;">All Health Considerations</h4>
                                          {format_health_items(health_considerations)}
                                      </div>
                                      <div style="margin-bottom: 24px;">
                                          <h4 style="margin: 0 0 12px 0; color: #2d3748;">Recommended Screenings</h4>
                                          {format_health_items(health_screenings)}
                                      </div>
                                      <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid #e2e8f0;">
                                          <p style="margin: 0 0 8px 0; color: #4a5568; font-size: 0.9em;">Source: Compiled from veterinary resources and breed health studies, 2025</p>
                                          <p style="margin: 0 0 8px 0; color: #4a5568; font-size: 0.9em;">Regular vet check-ups are essential for all breeds.</p>
                                          <p style="margin: 0 0 8px 0; color: #4a5568; font-size: 0.9em;">Early detection and prevention are key to managing health issues.</p>
                                          <p style="margin: 0 0 8px 0; color: #4a5568; font-size: 0.9em;">Not all dogs will develop these conditions.</p>
                                      </div>
                                  </div>
                              </details>
                          </div>
                      </div>
                      <!-- Description -->
                      <div style="margin-bottom: 32px;">
                          <h3 style="display: flex; align-items: center; gap: 10px; margin: 0 0 20px 0; padding: 12px; background: #f8f9fa; border-radius: 6px;">
                              <span style="font-size: 1.2em;">📝</span>
                              <span style="font-size: 1.2em; font-weight: 600; color: #2d3748;">DESCRIPTION</span>
                          </h3>
                          <div style="background: #fff; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0;">
                              <p style="line-height: 1.6; margin: 0; color: #2d3748;">{description.get('Description', '')}</p>
                          </div>
                      </div>
                      <!-- Action Section -->
                      <div style="text-align: center; margin-top: 32px;">
                          <a href="{get_akc_breeds_link(breed)}"
                              target="_blank"
                              rel="noopener noreferrer"
                              style="display: inline-flex; align-items: center; gap: 8px; padding: 14px 28px; background: linear-gradient(90deg, #4299e1, #48bb78); color: white; text-decoration: none; border-radius: 8px; font-weight: 500; transition: opacity 0.2s; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                              <span style="font-size: 1.2em;">🌐</span>
                              Learn more about {display_name} on AKC website
                          </a>
                      </div>
                  </div>
              </div>
          </div>
          '''

    return result


def format_multi_dog_container(dogs_info: str) -> str:
    """Wrap multiple dog detection results in a container."""
    return f"""
        <div class="dog-info-card">
            {dogs_info}
        </div>
    """

def format_breed_details_html(description: Dict[str, Any], breed: str) -> str:
    """Format breed details for the show_details_html function."""
    return f"""
    <div class="dog-info">
        <h2>{breed}</h2>
        <div class="breed-details">
            {format_description_html(description, breed)}
            <div class="action-section">
                <a href="{get_akc_breeds_link(breed)}" target="_blank" class="akc-button">
                    <span class="icon">🌐</span>
                    Learn more about {breed} on AKC website
                </a>
            </div>
        </div>
    </div>
    """

def format_comparison_result(breed1: str, breed2: str, comparison_data: Dict) -> str:
    """Format breed comparison results into HTML."""
    return f"""
    <div class="comparison-container">
        <div class="comparison-header">
            <h3>Comparison: {breed1} vs {breed2}</h3>
        </div>
        <div class="comparison-content">
            <div class="breed-column">
                <h4>{breed1}</h4>
                {format_comparison_details(comparison_data[breed1])}
            </div>
            <div class="breed-column">
                <h4>{breed2}</h4>
                {format_comparison_details(comparison_data[breed2])}
            </div>
        </div>
    </div>
    """

def format_comparison_details(breed_data: Dict) -> str:
    """Format individual breed details for comparison."""
    original_data = breed_data.get('Original_Data', {})
    return f"""
        <div class="comparison-details">
            <p><strong>Size:</strong> {original_data.get('Size', 'N/A')}</p>
            <p><strong>Exercise Needs:</strong> {original_data.get('Exercise Needs', 'N/A')}</p>
            <p><strong>Care Level:</strong> {original_data.get('Care Level', 'N/A')}</p>
            <p><strong>Grooming Needs:</strong> {original_data.get('Grooming Needs', 'N/A')}</p>
            <p><strong>Good with Children:</strong> {original_data.get('Good with Children', 'N/A')}</p>
            <p><strong>Temperament:</strong> {original_data.get('Temperament', 'N/A')}</p>
        </div>
    """
