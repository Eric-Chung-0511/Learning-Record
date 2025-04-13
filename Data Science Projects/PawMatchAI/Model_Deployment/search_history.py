import gradio as gr
import traceback
from typing import  Optional , Dict , List
# from history_manager import UserHistoryManager

class  SearchHistoryComponent :
    def  __init__ ( self ):
        """初始化搜尋歷史組件"""
        self.history_manager = UserHistoryManager()

    def  format_history_html ( self, history_data: Optional [ List [ Dict ]] = None ) -> str :
        try :
            if history_data is  None :
                history_data = self.history_manager.get_history()

            if  not history_data:
                return  """
                <div style='text-align: center; padding: 40px 20px;'>
                    <p>No search history yet. Try making some breed recommendations!</p>
                </div>
                """

            html = "<div class='history-container'>"

            # 對歷史記錄進行反轉，最新的顯示在前面
            for entry in  reversed (history_data):
                timestamp = entry.get( 'timestamp' , 'Unknown time' )
                search_type = entry.get( 'search_type' , 'criteria' )
                results = entry.get( 'results' , [])

                # 顯示時間戳記和搜尋類型
                html += f"""
                <div class="history-entry">
                    <div class="history-header" style="border-left: 4px solid #4299e1; padding-left: 10px;">
                        <span class="timestamp">🕒 {timestamp} </span>
                        <span class="search-type" style="color: #4299e1; font-weight: bold; margin-left: 10px;">
                            Search History
                        </span>
                    </div>
                """

                # 顯示搜尋參數
                if search_type == "criteria" :
                    prefs = entry.get( 'preferences' , {})
                    html += f"""
                    <div class="params-list" style="background: #f8fafc; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
                        <h4 style="margin-bottom: 12px;">Search Parameters:</h4>
                        <ul style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                            <li>Living Space: {prefs.get( 'living_space' , 'N/A' )} </li>
                            <li>Exercise Time: {prefs.get( 'exercise_time' , 'N/A' )} minutes</li>
                            <li>Grooming: {prefs.get( 'grooming_commitment' , 'N/A' )} </li>
                            <li>Size Preference: {prefs.get( 'size_preference' , 'N/A' )} </li>
                            <li>Experience: {prefs.get( 'experience_level' , 'N/A' )} </li>
                            <li>Children at Home: { "Yes"  if prefs.get( 'has_children' ) else  "No" } </li>
                            <li>Noise Tolerance: {prefs.get( 'noise_tolerance' , 'N/A' )} </li>
                        </ul>
                    </div>
                    """

                # 關鍵修改：確保結果部分始終顯示
                if results:   # 只有在有結果時才顯示結果區域
                    html += """
                    <div class="results-list" style="margin-top: 16px;">
                        <h4 style="margin-bottom: 12px;">Top 15 Breed Matches:</h4>
                        <div class="breed-list">
                    """

                    # 顯示每個推薦結果
                    for i, result in  enumerate (results[: 15 ], 1 ):
                        breed = result.get( 'breed' , 'Unknown breed' )
                        score = result.get( 'overall_score' , 0 )   # 改用overall_score
                        if  isinstance (score, ( int , float )):   # 確保分數是數字
                            score = float (score) * 100   # 轉換為百分比

                        html += f"""
                        <div class="breed-item" style="margin-bottom: 8px;">
                            <div class="breed-info" style="display: flex; align-items: center; justify-content: space-between; padding: 8px; background: #f8fafc; border-radius: 6px;">
                                <span class="breed-rank" style="background: linear-gradient(135deg, #4299e1, #48bb78); color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; min-width: 40px; text-align: center;">#{i}</span>
                                <span class="breed-name" style="font-weight: 500; color: #2D3748; margin: 0 12px;">{breed.replace('_', ' ')}</span>
                                <span class="breed-score" style="background: #F0FFF4; color: #48BB78; padding: 4px 8px; border-radius: 4px; font-weight: 600;">{score:.1f}%</span>
                            </div>
                        </div>
                        """

                    html += """
                        </div>
                    </div>
                    """

                html += "</div>"   # 關閉history-entry div

            html += "</div>"   # 關閉history-container div
            return html

        except Exception as e:
            print ( f"Error formatting history: { str (e)} " )
            print (traceback.format_exc())
            return  f"""
            <div style='text-align: center; padding: 20px; color: #dc2626;'>
                Error formatting history. Please try refreshing the page.
                <br>Error details: { str (e)}
            </div>
            """

    def  clear_history ( self ) -> str :
        """清除所有搜尋紀錄"""
        try :
            success = self.history_manager.clear_all_history()
            print ( f"Clear history result: {success} " )
            return self.format_history_html()
        except Exception as e:
            print ( f"Error in clear_history: { str (e)} " )
            print (traceback.format_exc())
            return  "Error clearing history"

    def  refresh_history ( self ) -> str :
        """刷新歷史記錄顯示"""
        try :
            return self.format_history_html()
        except Exception as e:
            print ( f"Error in refresh_history: { str (e)} " )
            return  "Error refreshing history"

    def  save_search ( self, user_preferences: Optional [ dict ] = None ,
                results: list = None ,
                search_type: str = "criteria" ,
                description: str = None ) -> bool :
        """
        儲存搜尋結果到歷史記錄
        這個方法負責處理搜尋結果的保存，並確保只保存前15個最相關的推薦結果。
        在儲存之前，會處理結果資料確保格式正確且包含所需的所有資訊。
        Args:
            user_preferences: 使用者偏好設定(僅用於criteria搜尋)
                包含所有搜尋條件如居住空間、運動時間等
            results: 推薦結果列表
                包含所有推薦的狗品種及其評分
            search_type: 搜尋類型("criteria" 或"description")
                用於標識搜尋方式
            description: 用戶輸入的描述(僅用於description搜尋)
                用於自然語言搜尋時的描述文本

        Returns:
            bool: 表示保存是否成功
        """
        # 首先確保結果不為空且為列表
        if results and  isinstance (results, list ):
            # 只取前15個結果
            processed_results = []
            for result in results[: 15 ]:   # 限制為前15個結果
                # 確保每個結果都包含必要的信息
                if  isinstance (result, dict ):
                    processed_result = {
                        'breed' : result.get( 'breed' , 'Unknown' ),
                        'overall_score' : result.get( 'overall_score' , result.get( 'final_score' , 0 )),
                        'rank' : result.get( 'rank' , 0 ),
                        'base_score' : result.get( 'base_score' , 0 ),
                        'bonus_score' : result.get( 'bonus_score' , 0 ),
                        'scores' : result.get( 'scores' , {})
                    }
                    processed_results.append(processed_result)
        else :
            # 如果沒有結果，創建空列表
            processed_results = []

        # 調用history_manager 的save_history 方法保存處理過的結果
        return self.history_manager.save_history(
            user_preferences=user_preferences,
            results=processed_results,   # 使用處理過的結果
            search_type= 'criteria'
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
                history_display = gr.HTML()
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

                history_display.value = history_component.format_history_html()

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
