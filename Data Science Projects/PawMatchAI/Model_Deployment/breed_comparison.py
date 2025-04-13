import gradio as gr
import sqlite3
from dog_database import get_dog_description
from breed_health_info import breed_health_info
from breed_noise_info import breed_noise_info

def create_comparison_tab(dog_breeds, get_dog_description, breed_noise_info, breed_health_info):
    # 每個選項是一個元組：(顯示值, 實際值)
    # 對整個列表進行排序，基於顯示值（即沒有底線的品種名稱）
    breed_choices = [(breed.replace('_', ' '), breed) for breed in sorted(dog_breeds)]
    with gr.TabItem("Breed Comparison"):
        gr.HTML("""
            <div style='
                text-align: center;
                padding: 20px 0;
                margin: 15px 0;
                background: linear-gradient(to right, rgba(66, 153, 225, 0.1), rgba(72, 187, 120, 0.1));
                border-radius: 10px;
            '>
                <p style='
                    font-size: 1.2em;
                    margin: 0;
                    padding: 0 20px;
                    line-height: 1.5;
                    background: linear-gradient(90deg, #4299e1, #48bb78);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-weight: 600;
                '>
                    Select two dog breeds to compare their characteristics and care requirements.
                </p>
            </div>
        """)

        with gr.Row():
            breed1_dropdown = gr.Dropdown(
                choices=breed_choices,
                label="Select First Breed",
                value=breed_choices[0][1] if breed_choices else None # 使用第一個品種的實際值作為預設值
            )
            breed2_dropdown = gr.Dropdown(
                choices=breed_choices,
                label="Select Second Breed",
                value=breed_choices[1][1] if len(breed_choices) > 1 else None  # 使用第二個品種的實際值作為預設值
            )

        compare_btn = gr.Button("Compare Breeds", elem_classes="custom-compare-button")
        comparison_output = gr.HTML(label="Comparison Results")

        def format_noise_data(notes):
            characteristics = []
            triggers = []
            noise_level = "Moderate"  # 預設值

            if isinstance(notes, str):
                lines = notes.strip().split('\n')
                section = ""
                for line in lines:
                    line = line.strip()
                    if "Typical noise characteristics:" in line:
                        section = "characteristics"
                    elif "Barking triggers:" in line:
                        section = "triggers"
                    elif "Noise level:" in line:
                        noise_level = line.split(':')[1].strip()
                    elif line.startswith('•'):
                        if section == "characteristics":
                            characteristics.append(line[1:].strip())
                        elif section == "triggers":
                            triggers.append(line[1:].strip())

            return {
                'characteristics': characteristics,
                'triggers': triggers,
                'noise_level': noise_level
            }

        def format_health_data(notes):
            considerations = []
            screenings = []

            if isinstance(notes, str):
                lines = notes.strip().split('\n')
                current_section = None

                for line in lines:
                    line = line.strip()
                    # 修正字串比對
                    if "Common breed-specific health considerations" in line:
                        current_section = "considerations"
                    elif "Recommended health screenings:" in line:
                        current_section = "screenings"
                    elif line.startswith('•'):
                        item = line[1:].strip()
                        if current_section == "considerations":
                            considerations.append(item)
                        elif current_section == "screenings":
                            screenings.append(item)

            # 只有當真的沒有資料時才返回 "Information not available"
            if not considerations and not screenings:
                return {
                    'considerations': ["Information not available"],
                    'screenings': ["Information not available"]
                }

            return {
                'considerations': considerations,
                'screenings': screenings
            }

        def get_comparison_styles():
            return """
            .comparison-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 24px;
                padding: 20px;
            }
            .breed-column {
                background: white;
                border-radius: 10px;
                padding: 24px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .section-title {
                font-size: 24px;
                color: #2D3748;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #E2E8F0;
            }
            .info-section {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 16px;
                margin-bottom: 24px;
            }
            .info-item {
                position: relative;
                background: #F8FAFC;
                padding: 16px;
                border-radius: 8px;
                border: 1px solid #E2E8F0;
            }
            .info-label {
                display: flex;
                align-items: center;
                gap: 8px;
                color: #4A5568;
                font-size: 0.9em;
                margin-bottom: 4px;
            }
            .info-icon {
                cursor: help;
                background: #E2E8F0;
                width: 18px;
                height: 18px;
                border-radius: 50%;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                color: #4A5568;
                margin-left: 4px;
            }
            .info-icon:hover + .tooltip-content {
                display: block;
            }
            .tooltip-content {
                display: none;
                position: absolute;
                background: #2D3748;
                color: #FFFFFF;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 14px;
                line-height: 1.3;
                width: max-content;
                max-width: 280px;
                z-index: 1000;
                top: 0;           /* 修改位置 */
                left: 100%;
                margin-left: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                white-space: normal;  /* 允許文字換行 */
            }
            .tooltip-content,
            .tooltip-content *,
            .tooltip-content strong,
            .tooltip-content li,
            .tooltip-content ul,
            .tooltip-content p,
            .tooltip-content span,
            .tooltip-content div {
                color: #FFFFFF !important;
            }
            .tooltip-content::before {
                content: '';
                position: absolute;
                left: -6px;
                top: 14px;        /* 配合上方位置調整 */
                border-width: 6px;
                border-style: solid;
                border-color: transparent #2D3748 transparent transparent;
            }
            .tooltip-content strong {
                color: #FFFFFF;
                display: block;
                margin-bottom: 4px;
                font-weight: 600;
            }
            .tooltip-content ul {
                margin: 0;
                padding-left: 16px;
                color: #FFFFFF;
            }
            .tooltip-content * {
                color: #FFFFFF;
            }
            .tooltip-content li {
                margin-bottom: 2px;
                color: #FFFFFF;
            }
            .tooltip-content li::before {
                color: #FFFFFF !important;
            }
            .tooltip-content br {
                display: block;
                margin: 2px 0;
            }
            .info-value {
                color: #2D3748;
                font-weight: 500;
            }
            .characteristic-section {
                background: #F8FAFC;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            .subsection-title {
                font-size: 18px;
                color: #2D3748;
                margin-bottom: 16px;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .noise-level {
                background: #EDF2F7;
                padding: 16px;
                border-radius: 6px;
                margin: 16px 0;
                border: 1px solid #CBD5E0;
            }
            .level-label {
                color: #4A5568;
                font-size: 1.1em;
                font-weight: 500;
                margin-bottom: 8px;
            }
            .level-value {
                color: #2D3748;
                font-size: 1.2em;
                font-weight: 600;
            }
            .characteristics-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 12px;
                margin-top: 12px;
            }
            .characteristic-item {
                background: white;
                padding: 12px;
                border-radius: 6px;
                border: 1px solid #E2E8F0;
                color: #4A5568;
            }
            .health-insights {
                margin-top: 24px;
            }
            .health-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 8px;
            }
            .health-item {
                background: white;
                padding: 12px;
                border-radius: 6px;
                border: 1px solid #E2E8F0;
                color: #E53E3E;
            }
            .screening-item {
                background: white;
                padding: 12px;
                border-radius: 6px;
                border: 1px solid #E2E8F0;
                color: #38A169;
            }
            .learn-more-btn {
                display: inline-block;
                margin-top: 20px;
                padding: 12px 24px;
                background: linear-gradient(90deg, #4299e1, #48bb78);
                color: white !important;
                text-decoration: none;
                border-radius: 6px;
                transition: all 0.3s ease;
                text-align: center;
                width: 100%;
                font-weight: 500;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .learn-more-btn:hover {
                background: linear-gradient(135deg, #3182ce, #38a169);
                transform: translateY(-2px);
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                color: white !important;
            }
            .info-disclaimer {
                margin-top: 24px;
                padding: 16px;
                background: #F7FAFC;
                border-radius: 8px;
                font-size: 0.9em;
                color: #4A5568;
                line-height: 1.5;
                border-left: 4px solid #4299E1;
            }
            @media (max-width: 768px) {
                .comparison-container {
                    grid-template-columns: 1fr;
                }
                .info-section {
                    grid-template-columns: 1fr;
                }
                .characteristics-grid {
                    grid-template-columns: 1fr;
                }
            }
            .custom-compare-button {
                background: linear-gradient(135deg, #4299e1, #48bb78) !important;
                border: none !important;
                padding: 12px 30px !important;
                border-radius: 8px !important;
                font-size: 1.1em !important;
                font-weight: 600 !important;
                color: white !important;
                cursor: pointer !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 4px 6px rgba(66, 153, 225, 0.2) !important;
                margin: 20px auto !important;
                display: block !important;
                width: auto !important;
                min-width: 200px !important;
                text-transform: uppercase !important;
                letter-spacing: 0.5px !important;
            }
            .custom-compare-button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 12px rgba(66, 153, 225, 0.3) !important;
                background: linear-gradient(135deg, #48bb78, #4299e1) !important;
            }
            .custom-compare-button:active {
                transform: translateY(1px) !important;
                box-shadow: 0 2px 4px rgba(66, 153, 225, 0.2) !important;
            }
            """

        gr.HTML(f"<style>{get_comparison_styles()}</style>")

        def show_comparison(breed1, breed2):
            if not breed1 or not breed2:
                return "Please select two breeds to compare"

            breed1_info = get_dog_description(breed1)
            breed2_info = get_dog_description(breed2)
            breed1_noise = breed_noise_info.get(breed1, {})
            breed2_noise = breed_noise_info.get(breed2, {})
            breed1_health = breed_health_info.get(breed1, {})
            breed2_health = breed_health_info.get(breed2, {})

            def create_info_item(label, value, tooltip_text=""):
                tooltip = f"""
                    <div class="info-label">
                        <span>{label}</span>
                        <div class="tooltip">
                            <span class="info-icon">i</span>
                            <div class="tooltip-content">
                                {tooltip_text}
                            </div>
                        </div>
                    </div>
                """ if tooltip_text else f'<div class="info-label">{label}</div>'

                return f"""
                <div class="info-item" style="position: relative;">
                    {tooltip}
                    <div class="info-value">{value}</div>
                </div>
                """

            def create_breed_section(breed, info, noise_info, health_info):
                # 建立提示文字
                section_tooltips = {
                    'Size': """
                        <strong>Size Categories:</strong><br>
                        • Small: Under 20 pounds<br>
                        • Medium: 20-60 pounds<br>
                        • Large: Over 60 pounds
                    """,
                    'Exercise': """
                        <strong>Exercise Needs:</strong><br>
                        • Low: Short walks suffice<br>
                        • Moderate: 1-2 hours daily<br>
                        • High: 2+ hours daily activity<br>
                        • Very High: Intensive daily exercise
                    """,
                    'Grooming': """
                        <strong>Grooming Requirements:</strong><br>
                        • Low: Occasional brushing<br>
                        • Moderate: Weekly grooming<br>
                        • High: Daily maintenance needed
                    """,
                    'Children': """
                        <strong>Compatibility with Children:</strong><br>
                        • Yes: Excellent with kids<br>
                        • Moderate: Good with supervision<br>
                        • No: Better with older children
                    """,
                    'Lifespan': """
                        <strong>Average Lifespan Range:</strong><br>
                        Typical lifespan for this breed with proper care and genetics
                    """,
                    'noise': """
                        <strong>Noise Behavior Information:</strong><br>
                        • Noise Level indicates typical vocalization intensity<br>
                        • Characteristics describe common vocal behaviors<br>
                        • Triggers list common causes of barking or vocalization
                    """,
                    'health': """
                        <strong>Health Information:</strong><br>
                        • Health considerations are breed-specific concerns<br>
                        • Screenings are recommended preventive tests<br>
                        • Always consult with veterinary professionals
                    """
                }

                noise_data = format_noise_data(noise_info.get('noise_notes', ''))
                health_data = format_health_data(health_info.get('health_notes', ''))

                def create_section_header(title, icon, tooltip_text):
                    return f"""
                    <div class="section-header">
                        <span>{icon}</span>
                        <span>{title}</span>
                        <span class="tooltip">
                            <span class="tooltip-icon">ⓘ</span>
                            <span class="tooltip-text">{tooltip_text}</span>
                        </span>
                    </div>
                    """

                return f"""
                <div class="breed-column">
                    <h2 class="section-title">🐕 {breed.replace('_', ' ')}</h2>
                    <div class="info-section">
                        {create_info_item('Size', info['Size'], section_tooltips['Size'])}
                        {create_info_item('Exercise', info['Exercise Needs'], section_tooltips['Exercise'])}
                        {create_info_item('Grooming', info['Grooming Needs'], section_tooltips['Grooming'])}
                        {create_info_item('With Children', info['Good with Children'], section_tooltips['Children'])}
                        {create_info_item('Lifespan', info['Lifespan'], section_tooltips['Lifespan'])}
                    </div>
                    <div class="characteristic-section">
                        {create_section_header('Noise Behavior', '🔊', section_tooltips['noise'])}
                        <div class="noise-level">
                            <div class="level-label">Noise Level</div>
                            <div class="level-value">{noise_data['noise_level'].upper()}</div>
                        </div>
                        <div class="subsection">
                            <h4>Typical Characteristics</h4>
                            <div class="characteristics-grid">
                                {' '.join([f'<div class="characteristic-item">{char}</div>'
                                        for char in noise_data['characteristics']])}
                            </div>
                        </div>
                        <div class="subsection">
                            <h4>Barking Triggers</h4>
                            <div class="characteristics-grid">
                                {' '.join([f'<div class="characteristic-item">{trigger}</div>'
                                        for trigger in noise_data['triggers']])}
                            </div>
                        </div>
                    </div>
                    <div class="characteristic-section health-insights">
                        {create_section_header('Health Insights', '🏥', section_tooltips['health'])}
                        <div class="subsection">
                            <h4>Health Considerations</h4>
                            <div class="health-grid">
                                {' '.join([f'<div class="health-item">{item}</div>'
                                        for item in health_data['considerations']])}
                            </div>
                        </div>
                        <div class="subsection">
                            <h4>Recommended Screenings</h4>
                            <div class="health-grid">
                                {' '.join([f'<div class="screening-item">{item}</div>'
                                        for item in health_data['screenings']])}
                            </div>
                        </div>
                    </div>
                    <a href="https://www.akc.org/dog-breeds/{breed.lower().replace('_', '-')}/"
                    class="learn-more-btn"
                    target="_blank">
                        🌐 Learn more about {breed.replace('_', ' ')} on AKC
                    </a>
                </div>
                """

            html_output = f"""
            <div class="comparison-container">
                {create_breed_section(breed1, breed1_info, breed1_noise, breed1_health)}
                {create_breed_section(breed2, breed2_info, breed2_noise, breed2_health)}
            </div>
            <div class="info-disclaimer">
                <strong>Note:</strong> The health and behavioral information provided is for general reference only.
                Individual dogs may vary, and characteristics can be influenced by training, socialization, and genetics.
                Always consult with veterinary professionals for specific health advice and professional trainers for
                behavioral guidance.
            </div>
            <style>{get_comparison_styles()}</style>
            """

            return html_output

        compare_btn.click(
            show_comparison,
            inputs=[breed1_dropdown, breed2_dropdown],
            outputs=comparison_output
        )

    return {
        'breed1_dropdown': breed1_dropdown,
        'breed2_dropdown': breed2_dropdown,
        'compare_btn': compare_btn,
        'comparison_output': comparison_output
    }
