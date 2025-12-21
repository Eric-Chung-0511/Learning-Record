# %%writefile breed_recommendation_enhanced.py
import gradio as gr
from typing import Dict, List, Any, Optional
import traceback
import spaces
from semantic_breed_recommender import get_breed_recommendations_by_description, get_enhanced_recommendations_with_unified_scoring
from natural_language_processor import get_nlp_processor
from recommendation_html_format import format_unified_recommendation_html

def create_description_examples():
    """Create HTML for description examples with dynamic visibility"""
    return """
        <div style='
            background: linear-gradient(135deg, rgba(66, 153, 225, 0.1) 0%, rgba(72, 187, 120, 0.1) 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            border-left: 4px solid #4299e1;
            display: block;
        '>
            <h4 style='
                color: #2d3748;
                margin: 0 0 15px 0;
                font-size: 1.1em;
                font-weight: 600;
            '>💡 Example Descriptions - Try These Expression Styles:</h4>
            <div style='
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin-top: 10px;
            '>
                <!-- 左上：冷色（藍） -->
                <div style='
                    background: white;
                    padding: 12px;
                    border-radius: 8px;
                    border: 1px solid #e2e8f0;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                '>
                    <strong style='color: #4299e1;'>🏡 Priority: Quiet Environment</strong><br>
                    <span style='color: #4a5568; font-size: 0.9em;'>
                        "Most importantly I need a quiet dog. I live in a small apartment with thin walls, and my neighbors are very noise sensitive."
                    </span>
                </div>
                <!-- 右上：暖色（橘） -->
                <div style='
                    background: white;
                    padding: 12px;
                    border-radius: 8px;
                    border: 1px solid #e2e8f0;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                '>
                    <strong style='color: #ed8936;'>🎾 Multiple Priorities:</strong><br>
                    <span style='color: #4a5568; font-size: 0.9em;'>
                        "First I need a dog that's good with kids, second prefer low maintenance grooming, and third would like an active breed for weekend hiking."
                    </span>
                </div>
                <!-- 左下：冷色（紫） -->
                <div style='
                    background: white;
                    padding: 12px;
                    border-radius: 8px;
                    border: 1px solid #e2e8f0;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                '>
                    <strong style='color: #805ad5;'>🏠 Beginner Owner:</strong><br>
                    <span style='color: #4a5568; font-size: 0.9em;'>
                        "This is my first dog. I live in a house with a small yard, work full time, and really want a low-maintenance breed that's easy to train."
                    </span>
                </div>
                <!-- 右下：暖色（琥珀橘） -->
                <div style='
                    background: white;
                    padding: 12px;
                    border-radius: 8px;
                    border: 1px solid #e2e8f0;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                '>
                    <strong style='color: #276749;'>🤫 Active Lifestyle Priority:</strong><br>
                    <span style='color: #4a5568; font-size: 0.9em;'>
                        "I absolutely need an energetic dog for daily running and hiking. Size doesn't matter, but the dog must be able to keep up with intense exercise."
                    </span>
                </div>
            </div>
            <div style='
                margin-top: 15px;
                padding: 12px;
                background: rgba(255, 255, 255, 0.8);
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            '>
                <strong style='color: #2d3748;'>🔍 Tips:</strong>
                <span style='color: #4a5568; font-size: 0.9em;'>
                    Please describe in English, including living environment, preferred breeds, family situation, activity needs, etc. The more detailed your description, the more accurate the recommendations!
                </span>
            </div>
        </div>
    """

def create_recommendation_tab(
    UserPreferences,
    get_breed_recommendations,
    format_recommendation_html,
    history_component
):
    """Create the enhanced breed recommendation tab with natural language support"""

    with gr.TabItem("Breed Recommendation"):
        with gr.Tabs():
            # --------------------------
            # Find by Criteria
            # --------------------------
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
                            ">The matching algorithm is continuously improving. Results are for reference only.</span>
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
                            value="no_preference"
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
                            visible=False
                        )

                        noise_tolerance = gr.Radio(
                            choices=["low", "medium", "high"],
                            label="Noise tolerance level",
                            info="Some breeds are more vocal than others",
                            value="medium"
                        )

                def update_children_age_visibility(has_children_val):
                    """Update children age visibility based on has_children checkbox"""
                    return gr.update(visible=has_children_val)

                has_children.change(
                    fn=update_children_age_visibility,
                    inputs=[has_children],
                    outputs=[children_age]
                )

                # --------- 條件搜尋---------
                def find_breed_matches(
                    living_space, yard_access, exercise_time, exercise_type,
                    grooming_commitment, size_preference, experience_level,
                    time_availability, has_children, children_age, noise_tolerance
                ):
                    """Process criteria-based breed matching and persist history"""
                    try:
                        # 1) 建立偏好
                        user_prefs = UserPreferences(
                            living_space=living_space,
                            yard_access=yard_access,
                            exercise_time=exercise_time,
                            exercise_type=exercise_type,
                            grooming_commitment=grooming_commitment,
                            size_preference=size_preference,
                            experience_level=experience_level,
                            time_availability=time_availability,
                            has_children=has_children,
                            children_age=children_age if has_children else None,
                            noise_tolerance=noise_tolerance,
                            # 其他欄位依原始設計
                            space_for_play=(living_space != "apartment"),
                            other_pets=False,
                            climate="moderate",
                            health_sensitivity="medium",
                            barking_acceptance=noise_tolerance
                        )

                        # 2) 取得推薦
                        recommendations = get_breed_recommendations(user_prefs)
                        print(f"[CRITERIA] generated={len(recommendations) if recommendations else 0}")

                        if not recommendations:
                            return format_recommendation_html([], is_description_search=False)

                        # 3) 準備歷史資料（final_score / overall_score 同步）
                        history_results = []
                        for idx, rec in enumerate(recommendations, start=1):
                            final_score = rec.get("final_score", rec.get("overall_score", 0))
                            overall_score = final_score  # Ensure consistency
                            history_results.append({
                                "breed": rec.get("breed", "Unknown"),
                                "rank": rec.get("rank", idx),
                                "final_score": final_score,
                                "overall_score": overall_score,
                                "base_score": rec.get("base_score", 0),
                                "bonus_score": rec.get("bonus_score", 0),
                                "scores": rec.get("scores", {})
                            })

                        prefs_dict = user_prefs.__dict__ if hasattr(user_prefs, "__dict__") else user_prefs

                        # 4) 寫入歷史（criteria）
                        try:
                            ok = history_component.save_search(
                                user_preferences=prefs_dict,
                                results=history_results,
                                search_type="criteria",
                                description=None
                            )
                            print(f"[CRITERIA SAVE] ok={ok}, saved={len(history_results)}")
                        except Exception as e:
                            print(f"[CRITERIA SAVE][ERROR] {str(e)}")

                        # 5) 顯示結果
                        return format_recommendation_html(recommendations, is_description_search=False)

                    except Exception as e:
                        print(f"[CRITERIA][ERROR] {str(e)}")
                        print(traceback.format_exc())
                        return f"""
                        <div style="text-align: center; padding: 20px; color: #e53e3e;">
                            <h3>⚠️ Error generating recommendations</h3>
                            <p>We encountered an issue while processing your preferences.</p>
                            <p><strong>Error details:</strong> {str(e)}</p>
                        </div>
                        """

                find_button = gr.Button("🔍 Find My Perfect Match!", elem_id="find-match-btn", size="lg")
                criteria_results = gr.HTML(label="Breed Recommendations")
                find_button.click(
                    fn=find_breed_matches,
                    inputs=[living_space, yard_access, exercise_time, exercise_type,
                            grooming_commitment, size_preference, experience_level,
                            time_availability, has_children, children_age, noise_tolerance],
                    outputs=criteria_results
                )

            # --------------------------
            # Find by Description
            # --------------------------
            with gr.Tab("Find by Description") as description_tab:
                gr.HTML("""
                    <div style='
                        text-align: center;
                        position: relative;
                        padding: 20px 0;
                        margin: 15px 0;
                        background: linear-gradient(to right, rgba(66, 153, 225, 0.1), rgba(72, 187, 120, 0.1));
                        border-radius: 10px;
                    '>
                        <div style='
                            position: absolute;
                            top: 10px;
                            right: 20px;
                            background: linear-gradient(90deg, #ff6b6b, #feca57);
                            color: white;
                            padding: 4px 12px;
                            border-radius: 15px;
                            font-size: 0.85em;
                            font-weight: 600;
                            letter-spacing: 1px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        '>NEW</div>
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
                            Describe your needs in natural language, and AI will find the most suitable breeds!
                        </p>
                        <div style='
                            margin-top: 15px;
                            padding: 10px 20px;
                            background: linear-gradient(to right, rgba(255, 107, 107, 0.15), rgba(254, 202, 87, 0.15));
                            border-radius: 8px;
                            font-size: 0.9em;
                            color: #2D3748;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            gap: 8px;
                        '>
                            <span style="font-size: 1.2em;">🚀</span>
                            <span style="
                                letter-spacing: 0.3px;
                                line-height: 1.4;
                            "><strong>New Feature:</strong> Based on advanced semantic understanding technology, making search more aligned with your real needs!</span>
                        </div>
                    </div>
                """)

                examples_display = gr.HTML(create_description_examples())

                description_input = gr.Textbox(
                    label="🗣️ Please describe your needs",
                    placeholder=("Example: I live in an apartment and need a quiet, small dog that's good with children. "
                                 "I prefer Border Collies and Golden Retrievers..."),
                    lines=4,
                    max_lines=6,
                    elem_classes=["description-input"]
                )

                validation_status = gr.HTML(visible=False)

                # Accuracy disclaimer
                gr.HTML("""
                    <div style='
                        background: linear-gradient(135deg, rgba(34, 197, 94, 0.08) 0%, rgba(59, 130, 246, 0.08) 100%);
                        border: 1px solid rgba(34, 197, 94, 0.2);
                        border-radius: 10px;
                        padding: 16px 20px;
                        margin: 16px 0 20px 0;
                        display: flex;
                        align-items: center;
                        gap: 12px;
                    '>
                        <div style='
                            background: linear-gradient(135deg, #22c55e, #3b82f6);
                            border-radius: 50%;
                            width: 32px;
                            height: 32px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            flex-shrink: 0;
                        '>
                            <span style='color: white; font-size: 16px; font-weight: bold;'>ⓘ</span>
                        </div>
                        <div style='flex: 1;'>
                            <div style='
                                color: #1f2937;
                                font-weight: 600;
                                font-size: 0.95em;
                                margin-bottom: 4px;
                                line-height: 1.4;
                            '>
                                Accuracy Continuously Improving - Use as Reference Guide
                            </div>
                            <div style='
                                color: #4b5563;
                                font-size: 0.88em;
                                line-height: 1.5;
                                letter-spacing: 0.2px;
                            '>
                                The AI recommendation system is constantly learning and improving. Use these recommendations as a helpful reference for your pet adoption.
                            </div>
                        </div>
                    </div>
                """)

                def validate_description_input(text):
                    """Validate description input"""
                    try:
                        nlp = get_nlp_processor()
                        validation = nlp.validate_input(text)
                        if validation.get("is_valid", True):
                            return gr.update(visible=False), True
                        else:
                            error_html = f"""
                            <div style='
                                background: #fed7d7;
                                border: 1px solid #fc8181;
                                color: #c53030;
                                padding: 10px;
                                border-radius: 8px;
                                margin: 10px 0;
                            '>
                                <strong>⚠️ {validation.get('error', 'Invalid input')}</strong><br>
                                {"<br>".join(f"• {s}" for s in validation.get('suggestions', []))}
                            </div>
                            """
                            return gr.update(value=error_html, visible=True), False
                    except Exception as e:
                        # 無 NLP 驗證也可放行
                        print(f"[DESC][VALIDATE][WARN] {str(e)}")
                        return gr.update(visible=False), True

                @spaces.GPU
                def find_breeds_by_description(description_text):
                    """Find breeds based on description and persist history"""
                    try:
                        if not description_text or not description_text.strip():
                            return """
                            <div style="text-align: center; padding: 20px; color: #718096;">
                                <p>Please enter your description to get personalized recommendations</p>
                            </div>
                            """

                        # 驗證（若可用）
                        try:
                            nlp = get_nlp_processor()
                            validation = nlp.validate_input(description_text)
                            if not validation.get("is_valid", True):
                                return f"""
                                <div style="text-align: center; padding: 20px; color: #e53e3e;">
                                    <h3>⚠️ Input validation failed</h3>
                                    <p>{validation.get('error','Invalid input')}</p>
                                    <ul style="text-align: left; display: inline-block;">
                                        {"".join(f"<li>{s}</li>" for s in validation.get('suggestions', []))}
                                    </ul>
                                </div>
                                """
                        except Exception as e:
                            print(f"[DESC][VALIDATE][WARN] {str(e)} (skip validation)")

                        # 取得增強語意推薦
                        recommendations = get_enhanced_recommendations_with_unified_scoring(
                            user_description=description_text,
                            top_k=15
                        )
                        print(f"[DESC] generated={len(recommendations) if recommendations else 0}")

                        if not recommendations:
                            return """
                            <div style="text-align: center; padding: 20px; color: #e53e3e;">
                                <h3>😔 No matching breeds found</h3>
                                <p>No dog breeds match your specific requirements. Please try:</p>
                                <ul style="text-align: left; display: inline-block; color: #4a5568;">
                                    <li>Providing a more general description</li>
                                    <li>Relaxing some specific requirements</li>
                                    <li>Including different breed preferences</li>
                                </ul>
                            </div>
                            """

                        # 準備歷史資料
                        def _to_float(x, default=0.0):
                            try:
                                return float(x)
                            except Exception:
                                return default

                        history_results = []
                        for i, rec in enumerate(recommendations, start=1):
                            final_score = _to_float(rec.get("final_score", rec.get("overall_score", 0)))
                            overall_score = final_score  # Ensure consistency between final_score and overall_score
                            history_results.append({
                                "breed": str(rec.get("breed", "Unknown")),
                                "rank": int(rec.get("rank", i)),
                                "final_score": final_score,
                                "overall_score": overall_score,
                                "semantic_score": _to_float(rec.get("semantic_score", 0)),
                                "comparative_bonus": _to_float(rec.get("comparative_bonus", 0)),
                                "lifestyle_bonus": _to_float(rec.get("lifestyle_bonus", 0)),
                                "size": str(rec.get("size", "Unknown")),
                                "scores": rec.get("scores", {})
                            })

                        # 寫入歷史（description）
                        try:
                            ok = history_component.save_search(
                                user_preferences=None,
                                results=history_results,
                                search_type="description",
                                description=description_text
                            )
                            print(f"[DESC SAVE] ok={ok}, saved={len(history_results)}")
                        except Exception as e:
                            print(f"[DESC SAVE][ERROR] {str(e)}")

                        # 使用統一HTML格式化器顯示增強推薦結果
                        html_output = format_unified_recommendation_html(recommendations, is_description_search=True)
                        return html_output

                    except RuntimeError as e:
                        error_msg = str(e)
                        print(f"[DESC][RUNTIME_ERROR] {error_msg}")
                        return f"""
                        <div style="text-align: center; padding: 20px; color: #e53e3e;">
                            <h3>🔧 System Configuration Issue</h3>
                            <p style="color: #4a5568; text-align: left; max-width: 600px; margin: 0 auto;">
                                {error_msg.replace(chr(10), '<br>').replace('•', '&bull;')}
                            </p>
                            <div style="margin-top: 15px; padding: 10px; background-color: #f7fafc; border-radius: 8px;">
                                <p style="color: #2d3748; font-weight: 500;">💡 What you can try:</p>
                                <ul style="text-align: left; color: #4a5568; margin: 10px 0;">
                                    <li>Restart the application</li>
                                    <li>Use the "Find by Criteria" tab instead</li>
                                    <li>Contact support if the issue persists</li>
                                </ul>
                            </div>
                        </div>
                        """
                    except ValueError as e:
                        error_msg = str(e)
                        print(f"[DESC][VALUE_ERROR] {error_msg}")
                        return f"""
                        <div style="text-align: center; padding: 20px; color: #e53e3e;">
                            <h3>🔍 No Matching Results</h3>
                            <p style="color: #4a5568; text-align: left; max-width: 600px; margin: 0 auto;">
                                {error_msg}
                            </p>
                            <div style="margin-top: 15px; padding: 10px; background-color: #f0f9ff; border-radius: 8px;">
                                <p style="color: #2d3748; font-weight: 500;">💡 Suggestions to get better results:</p>
                                <ul style="text-align: left; color: #4a5568; margin: 10px 0;">
                                    <li>Try describing your lifestyle more generally</li>
                                    <li>Mention multiple breed preferences</li>
                                    <li>Include both what you want and what you can accommodate</li>
                                    <li>Consider using the "Find by Criteria" tab for structured search</li>
                                </ul>
                            </div>
                        </div>
                        """
                    except Exception as e:
                        error_msg = str(e)
                        print(f"[DESC][ERROR] {error_msg}")
                        print(traceback.format_exc())
                        return f"""
                        <div style="text-align: center; padding: 20px; color: #e53e3e;">
                            <h3>⚠️ Unexpected Error</h3>
                            <p style="color: #4a5568;">An unexpected error occurred while processing your description.</p>
                            <details style="margin-top: 15px; text-align: left; max-width: 600px; margin-left: auto; margin-right: auto;">
                                <summary style="cursor: pointer; color: #2d3748; font-weight: 500;">Show technical details</summary>
                                <pre style="background-color: #f7fafc; padding: 10px; border-radius: 4px; font-size: 12px; color: #4a5568; margin-top: 10px; overflow: auto;">{error_msg}</pre>
                            </details>
                            <p style="margin-top: 15px; color: #4a5568; font-size: 14px;">Please try the "Find by Criteria" tab or contact support.</p>
                        </div>
                        """

                description_input.change(
                    fn=lambda x: validate_description_input(x)[0],
                    inputs=[description_input],
                    outputs=[validation_status]
                )

                description_button = gr.Button("🤖 Smart Breed Recommendation", elem_id="find-by-description-btn", size="lg")
                description_results = gr.HTML(label="AI Breed Recommendations")

                description_button.click(
                    fn=find_breeds_by_description,
                    inputs=[description_input],
                    outputs=[description_results]
                )

    return {
        'criteria_results': locals().get('criteria_results'),
        'description_results': locals().get('description_results'),
        'description_input': locals().get('description_input')
    }