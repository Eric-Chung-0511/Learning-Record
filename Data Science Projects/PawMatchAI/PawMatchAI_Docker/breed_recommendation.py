import sqlite3
import gradio as gr
from typing import Generator
from dog_database import get_dog_description, dog_data
from breed_health_info import breed_health_info
from breed_noise_info import breed_noise_info
from scoring_calculation_system import UserPreferences, calculate_compatibility_score
from recommendation_html_format import format_recommendation_html, get_breed_recommendations
from search_history import create_history_tab, create_history_component

def create_custom_button_style():
    return """
        <style>
        /* 確保有匹配到 */
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
            background: rgba(200, 200, 200, 0.1);  # 中性的背景色
            transition: opacity 0.3s ease;  # 平滑過渡效果
        }
        /* 強制覆蓋任何其他樣式 */
        .gradio-button {
            position: relative !important;
            overflow: visible !important;
        }
        </style>
    """

def create_recommendation_tab(UserPreferences, get_breed_recommendations, format_recommendation_html, history_component):

    with gr.TabItem("Breed Recommendation"):
        with gr.Tabs():
            with gr.Tab("Find by Criteria"):
                gr.HTML("""
                    <div style='
                        text-align: center;
                        position: relative;
                        padding: 20px 0;
                        margin: 15px 0;
                        background: linear-gradient(to right, rgba(66, 153, 225, 0.1), rgba(72, 187, 120, 0.1));
                        border-radius: 10px;
                    '>
                        <!-- BETA 標籤 -->
                        <div style='
                            position: absolute;
                            top: 10px;
                            right: 20px;
                            background: linear-gradient(90deg, #4299e1, #48bb78);
                            color: white;
                            padding: 4px 12px;
                            border-radius: 15px;
                            font-size: 0.85em;
                            font-weight: 600;
                            letter-spacing: 1px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        '>BETA</div>

                        <!-- 主標題 -->
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
                            Tell us about your lifestyle, and we'll recommend the perfect dog breeds for you!
                        </p>
                        <!-- 提示訊息 -->
                        <div style='
                            margin-top: 15px;
                            padding: 10px 20px;
                            background: linear-gradient(to right, rgba(66, 153, 225, 0.15), rgba(72, 187, 120, 0.15));
                            border-radius: 8px;
                            font-size: 0.9em;
                            color: #2D3748;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            gap: 8px;
                        '>
                            <span style="font-size: 1.2em;">🔬</span>
                            <span style="
                                letter-spacing: 0.3px;
                                line-height: 1.4;
                            "><strong>Beta Feature:</strong> Our matching algorithm is continuously improving. Results are for reference only.</span>
                        </div>
                    </div>
                """)

                with gr.Row():
                    with gr.Column():
                        living_space = gr.Radio(
                            choices=["apartment", "house_small", "house_large"],
                            label="What type of living space do you have?",
                            info="Choose your current living situation",
                            value="apartment"
                        )

                        yard_access = gr.Radio(
                            choices=["no_yard", "shared_yard", "private_yard"],
                            label="Yard Access Type",
                            info="Available outdoor space",
                            value="no_yard"
                        )

                        exercise_time = gr.Slider(
                            minimum=0,
                            maximum=180,
                            value=60,
                            label="Daily exercise time (minutes)",
                            info="Consider walks, play time, and training"
                        )

                        exercise_type = gr.Radio(
                            choices=["light_walks", "moderate_activity", "active_training"],
                            label="Exercise Style",
                            info="What kind of activities do you prefer?",
                            value="moderate_activity"
                        )


                        grooming_commitment = gr.Radio(
                            choices=["low", "medium", "high"],
                            label="Grooming commitment level",
                            info="Low: monthly, Medium: weekly, High: daily",
                            value="medium"
                        )

                    with gr.Column():
                        size_preference = gr.Radio(
                            choices=["no_preference", "small", "medium", "large", "giant"],
                            label="Preference Dog Size",
                            info="Select your preferred dog size - this will strongly filter the recommendations",
                            value = "no_preference"
                        )
                        experience_level = gr.Radio(
                            choices=["beginner", "intermediate", "advanced"],
                            label="Dog ownership experience",
                            info="Be honest - this helps find the right match",
                            value="beginner"
                        )

                        time_availability = gr.Radio(
                            choices=["limited", "moderate", "flexible"],
                            label="Time Availability",
                            info="Time available for dog care daily",
                            value="moderate"
                        )

                        has_children = gr.Checkbox(
                            label="Have children at home",
                            info="Helps recommend child-friendly breeds"
                        )

                        children_age = gr.Radio(
                            choices=["toddler", "school_age", "teenager"],
                            label="Children's Age Group",
                            info="Helps match with age-appropriate breeds",
                            visible=False  # 默認隱藏，只在has_children=True時顯示
                        )

                        noise_tolerance = gr.Radio(
                            choices=["low", "medium", "high"],
                            label="Noise tolerance level",
                            info="Some breeds are more vocal than others",
                            value="medium"
                        )

                def update_children_age_visibility(has_children):
                    return gr.update(visible=has_children)

                has_children.change(
                    fn=update_children_age_visibility,
                    inputs=has_children,
                    outputs=children_age
                )
                gr.HTML(create_custom_button_style())

                get_recommendations_btn = gr.Button(
                        "Find My Perfect Match! 🔍",
                        elem_id="find-match-btn"
                    )

                search_status = gr.HTML(
                        '<div id="search-status">🔍 Sniffing out your perfect furry companion...</div>',
                        visible=False,
                        elem_id="search-status-container"
                    )

                recommendation_output = gr.HTML(
                    label="Breed Recommendations",
                    visible=True,  # 確保可見性
                    elem_id="recommendation-output"
                )

        def on_find_match_click(*args):
            try:
                print("Starting breed matching process...")
                user_prefs = UserPreferences(
                living_space=args[0],
                yard_access=args[1],
                exercise_time=args[2],
                exercise_type=args[3],
                grooming_commitment=args[4],
                size_preference=args[5],
                experience_level=args[6],
                time_availability=args[7],
                has_children=args[8],
                children_age=args[9] if args[8] else None,
                noise_tolerance=args[10],
                space_for_play=True if args[0] != "apartment" else False,
                other_pets=False,
                climate="moderate",
                health_sensitivity="medium",
                barking_acceptance=args[10]
            )

                recommendations = get_breed_recommendations(user_prefs, top_n=15)

                history_results = [{
                    'breed': rec['breed'],
                    'rank': rec['rank'],
                    'overall_score': rec['final_score'],
                    'base_score': rec['base_score'],
                    'bonus_score': rec['bonus_score'],
                    'scores': rec['scores']
                } for rec in recommendations]

                history_component.save_search(
                    user_preferences={
                        'living_space': args[0],
                        'yard_access': args[1],
                        'exercise_time': args[2],
                        'exercise_type': args[3],
                        'grooming_commitment': args[4],
                        'size_preference': args[5],
                        'experience_level': args[6],
                        'time_availability': args[7],
                        'has_children': args[8],
                        'children_age': args[9] if args[8] else None,
                        'noise_tolerance': args[10],
                        'search_type': 'Criteria'
                    },
                    results=history_results
                )

                return [
                    format_recommendation_html(recommendations, is_description_search=False),
                    gr.update(visible=False)
                ]

            except Exception as e:
                print(f"Error in find match: {str(e)}")
                import traceback
                print(traceback.format_exc())
                return ["Error getting recommendations", gr.HTML.update(visible=False)]

        def update_status_and_process(*args):
            return [
                gr.update(value=None, visible=True),  # 更新可見性
                gr.update(visible=True)
            ]

        get_recommendations_btn.click(
            fn=update_status_and_process,  # 先執行狀態更新
            outputs=[recommendation_output, search_status],
            queue=False  # 確保狀態更新立即執行
        ).then(  # 然後執行主要處理邏輯
            fn=on_find_match_click,
            inputs=[
                living_space,
                yard_access,
                exercise_time,
                exercise_type,
                grooming_commitment,
                size_preference,
                experience_level,
                time_availability,
                has_children,
                children_age,
                noise_tolerance
            ],
            outputs=[recommendation_output, search_status]
        )

    return {
        'living_space': living_space,
        'exercise_time': exercise_time,
        'grooming_commitment': grooming_commitment,
        'experience_level': experience_level,
        'has_children': has_children,
        'noise_tolerance': noise_tolerance,
        'get_recommendations_btn': get_recommendations_btn,
        'recommendation_output': recommendation_output,
    }
