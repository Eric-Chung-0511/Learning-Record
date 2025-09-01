import gradio as gr
import traceback
from typing import  Optional , Dict , List
from history_manager import UserHistoryManager

class SearchHistoryComponent:
    def __init__(self):
        """初始化搜尋歷史組件"""
        self.history_manager = UserHistoryManager()

    def format_history_html(self, history_data: Optional[List[Dict]] = None) -> str:
        try:
            if history_data is None:
                history_data = self.history_manager.get_history()

            if not history_data:
                return """
                <div style='text-align: center; padding: 40px 20px;'>
                    <p>No search history yet. Try making some breed recommendations!</p>
                </div>
                """

            html = "<div class='history-container'>"

            # 最新的顯示在前面
            for entry in reversed(history_data):
                timestamp = entry.get('timestamp', 'Unknown time')
                search_type = entry.get('search_type', 'criteria')
                results = entry.get('results', [])

                # 標籤樣式
                if search_type == "description":
                    border_color = "#4299e1"
                    tag_color = "#4299e1"
                    tag_bg = "rgba(66, 153, 225, 0.1)"
                    tag_text = "Description Search"
                    icon = "🤖"
                else:
                    border_color = "#48bb78"
                    tag_color = "#48bb78"
                    tag_bg = "rgba(72, 187, 120, 0.1)"
                    tag_text = "Criteria Search"
                    icon = "🔍"

                # header
                html += f"""
                <div class="history-entry">
                    <div class="history-header" style="border-left: 4px solid {border_color}; padding-left: 10px;">
                        <span class="timestamp">🕒 {timestamp}</span>
                        <span class="search-type" style="
                            background: {tag_bg};
                            color: {tag_color};
                            padding: 4px 8px;
                            border-radius: 12px;
                            font-size: 0.8em;
                            font-weight: 600;
                            margin-left: 10px;
                            display: inline-flex;
                            align-items: center;
                            gap: 4px;
                        ">
                            {icon} {tag_text}
                        </span>
                    </div>
                """

                # 參數/描述
                if search_type == "criteria":
                    prefs = entry.get('preferences', {})
                    html += f"""
                    <div class="params-list" style="background: #f8fafc; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
                        <h4 style="margin-bottom: 12px;">Search Parameters:</h4>
                        <ul style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                            <li>Living Space: {prefs.get('living_space', 'N/A')}</li>
                            <li>Exercise Time: {prefs.get('exercise_time', 'N/A')} minutes</li>
                            <li>Grooming: {prefs.get('grooming_commitment', 'N/A')}</li>
                            <li>Size Preference: {prefs.get('size_preference', 'N/A')}</li>
                            <li>Experience: {prefs.get('experience_level', 'N/A')}</li>
                            <li>Children at Home: {"Yes" if prefs.get('has_children') else "No"}</li>
                            <li>Noise Tolerance: {prefs.get('noise_tolerance', 'N/A')}</li>
                        </ul>
                    </div>
                    """
                elif search_type == "description":
                    description = entry.get('user_description', '')
                    html += f"""
                    <div class="params-list" style="background: #f0f8ff; padding: 16px; border-radius: 8px; margin-bottom: 16px; border: 1px solid rgba(66, 153, 225, 0.2);">
                        <h4 style="margin-bottom: 12px; color: #4299e1;">User Description:</h4>
                        <div style="
                            background: white;
                            padding: 12px;
                            border-radius: 6px;
                            border-left: 3px solid #4299e1;
                            font-style: italic;
                            color: #2d3748;
                            line-height: 1.5;
                        ">
                            "{description}"
                        </div>
                    </div>
                    """

                # 結果區
                if results:
                    html += """
                    <div class="results-list" style="margin-top: 16px;">
                        <h4 style="margin-bottom: 12px;">Top 15 Breed Matches:</h4>
                        <div class="breed-list">
                    """

                    for i, result in enumerate(results[:15], 1):
                        breed = result.get('breed', 'Unknown breed')

                        # ★ 分數回退順序：final_score → overall_score → semantic_score
                        score_val = (
                            result.get('final_score', None)
                            if result.get('final_score', None) not in [None, ""]
                            else result.get('overall_score', None)
                        )
                        if score_val in [None, ""]:
                            score_val = result.get('semantic_score', 0)

                        try:
                            score_pct = float(score_val) * 100.0
                        except Exception:
                            score_pct = 0.0

                        html += f"""
                        <div class="breed-item" style="margin-bottom: 8px;">
                            <div class="breed-info" style="display: flex; align-items: center; justify-content: space-between; padding: 8px; background: #f8fafc; border-radius: 6px;">
                                <span class="breed-rank" style="background: linear-gradient(135deg, #4299e1, #48bb78); color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; min-width: 40px; text-align: center;">#{i}</span>
                                <span class="breed-name" style="font-weight: 500; color: #2D3748; margin: 0 12px;">{breed.replace('_', ' ')}</span>
                                <span class="breed-score" style="background: #F0FFF4; color: #48BB78; padding: 4px 8px; border-radius: 4px; font-weight: 600;">{score_pct:.1f}%</span>
                            </div>
                        </div>
                        """

                    html += """
                        </div>
                    </div>
                    """

                html += "</div>"  # 關閉 .history-entry

            html += "</div>"  # 關閉 .history-container
            return html

        except Exception as e:
            print(f"Error formatting history: {str(e)}")
            print(traceback.format_exc())
            return f"""
            <div style='text-align: center; padding: 20px; color: #dc2626;'>
                Error formatting history. Please try refreshing the page.
                <br>Error details: {str(e)}
            </div>
            """

    def clear_history(self) -> str:
        try:
            success = self.history_manager.clear_all_history()
            print(f"Clear history result: {success}")
            return self.format_history_html()
        except Exception as e:
            print(f"Error in clear_history: {str(e)}")
            print(traceback.format_exc())
            return "Error clearing history"

    def refresh_history(self) -> str:
        try:
            return self.format_history_html()
        except Exception as e:
            print(f"Error in refresh_history: {str(e)}")
            return "Error refreshing history"

    def save_search(self,
                    user_preferences: Optional[dict] = None,
                    results: list = None,
                    search_type: str = "criteria",
                    description: str = None) -> bool:
        """參數原樣透傳給 history_manager"""
        return self.history_manager.save_history(
            user_preferences=user_preferences,
            results=results,
            search_type=search_type,
            description=description,
            user_description=description
        )

def  create_history_component ():
    """只建立實例"""
    return SearchHistoryComponent()

def  create_history_tab ( history_component: SearchHistoryComponent ):
    """創建歷史紀錄的頁面
    Args:
        history_component:
    """
    with gr.TabItem( "Recommendation Search History" ):
        gr.HTML( """
            <style>
                .custom-btn {
                    padding: 10px 20px !important;
                    border-radius: 8px !important;
                    border: none !important;
                    font-weight: 500 !important;
                    transition: all 0.2s ease !important;
                    color: white !important;
                    font-size: 0.95em !important;
                    cursor: pointer !important;
                    width: 100% !important;
                    min-height: 42px !important;
                }

                /* Clear History 的按鈕*/
                .clear-btn {
                    background: linear-gradient(135deg, #FF6B6B 0%, #FF9B8B 100%) !important;
                    box-shadow: 0 2px 4px rgba(255, 107, 107, 0.15) !important;
                }

                .clear-btn:hover {
                    background: linear-gradient(135deg, #FF5252, #FF8B7B) !important;
                    transform: translateY(-1px);
                }

                .clear-btn:active {
                    transform: translateY(1px) scale(0.98);
                    background: linear-gradient(135deg, #FF4242, #FF7B6B) !important;
                }

                /* Refresh 的按鈕*/
                .refresh-btn {
                    background: linear-gradient(135deg, #4FB5E5 0%, #32CCBC 100%) !important;
                    box-shadow: 0 2px 4px rgba(79, 181, 229, 0.15) !important;
                }

                .refresh-btn:hover {
                    background: linear-gradient(135deg, #45A5D5, #2DBCAC) !important;
                    transform: translateY(-1px);
                }

                .refresh-btn:active {
                    transform: translateY(1px) scale(0.98);
                    background: linear-gradient(135deg, #3B95C5, #28AC9C) !important;
                }

                /* 懸浮的效果*/
                .custom-btn:hover {
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
                }

                /* 點擊的效果*/
                .custom-btn:active {
                    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
                }
            </style>

            <div style='text-align: center; padding: 20px;'>
                <h3 style='
                    color: #2D3748;
                    margin-bottom: 10px;
                    font-size: 1.5em;
                    background: linear-gradient(90deg, #4299e1, #48bb78);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-weight: 600;
                '>Search History</h3>
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
                        View your previous breed recommendations and search preferences
                    </p>
                </div>
            </div>
        """ )

        with gr.Row():
            with gr.Column(scale= 4 ):
                history_display = gr.HTML(value=history_component.format_history_html())
                with gr.Row(equal_height= True ):
                    with gr.Column(scale= 1 ):
                        clear_history_btn = gr.Button(
                            "🗑️ Clear History" ,
                            variant= "primary" ,
                            elem_classes= "custom-btn clear-btn"
                        )
                    with gr.Column(scale= 1 ):
                        refresh_btn = gr.Button(
                            "🔄 Refresh" ,
                            variant= "primary" ,
                            elem_classes= "custom-btn refresh-btn"
                        )

                clear_history_btn.click(
                    fn=history_component.clear_history,
                    outputs=[history_display],
                    api_name= "clear_history"
                )

                refresh_btn.click(
                    fn=history_component.refresh_history,
                    outputs=[history_display],
                    api_name= "refresh_history"
                )
