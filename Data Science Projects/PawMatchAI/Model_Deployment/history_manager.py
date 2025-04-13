from datetime import datetime
import json
import os
import pytz
import traceback

class UserHistoryManager:
    def __init__(self):
        """初始化歷史紀錄管理器"""
        self.history_file = "user_history.json"
        print(f"Initializing UserHistoryManager with file: {os.path.abspath(self.history_file)}")
        self._init_file()

    def _init_file(self):
        """初始化JSON檔案"""
        try:
            if not os.path.exists(self.history_file):
                print(f"Creating new history file: {self.history_file}")
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
            else:
                print(f"History file exists: {self.history_file}")
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"Current history entries: {len(data)}")
        except Exception as e:
            print(f"Error in _init_file: {str(e)}")
            print(traceback.format_exc())

    def save_history(self, user_preferences: dict = None, results: list = None, search_type: str = "criteria", description: str = None) -> bool:
        """
        保存搜尋歷史，確保結果資料被完整保存
        
        Args:
            user_preferences: 使用者的搜尋偏好設定
            results: 品種推薦結果列表
            search_type: 搜尋類型
            description: 搜尋描述
        
        Returns:
            bool: 保存是否成功
        """
        try:
            # 初始化時區和當前時間
            taipei_tz = pytz.timezone('Asia/Taipei')
            current_time = datetime.now(taipei_tz)
            
            # 創建歷史紀錄項目並包含時間戳記
            history_entry = {
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "search_type": search_type
            }
            
            # 確保結果資料的完整性
            if results and isinstance(results, list):
                processed_results = []
                for result in results[:15]:
                    # 確保每個結果都包含必要的欄位
                    if isinstance(result, dict):
                        processed_result = {
                            'breed': result.get('breed', 'Unknown'),
                            'overall_score': result.get('overall_score', 0),
                            'rank': result.get('rank', 0),
                            'size': result.get('size', 'Unknown') 
                        }
                        processed_results.append(processed_result)
                history_entry["results"] = processed_results
            
            # 加入使用者偏好設定（如果有的話）
            if user_preferences:  
                formatted_preferences = {
                'living_space': user_preferences.get('living_space'),
                'exercise_time': user_preferences.get('exercise_time'),
                'grooming_commitment': user_preferences.get('grooming_commitment'),
                'experience_level': user_preferences.get('experience_level'),  
                'has_children': user_preferences.get('has_children'),
                'noise_tolerance': user_preferences.get('noise_tolerance'),
                'size_preference': user_preferences.get('size_preference')  
                }
                history_entry["preferences"] = user_preferences
            
            # 讀取現有歷史
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            # 加入新紀錄並保持歷史限制
            history.append(history_entry)
            if len(history) > 20:  # 保留最近 20 筆
                history = history[-20:]
            
            # 儲存更新後的歷史
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            
            print(f"Successfully saved history entry: {history_entry}")
            return True
        
        except Exception as e:
            print(f"Error saving history: {str(e)}")
            print(traceback.format_exc())
            return False

    
    def get_history(self) -> list:
        """獲取搜尋歷史"""
        try:
            print("Attempting to read history")  # Debug
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Read {len(data)} history entries")  # Debug
                return data if isinstance(data, list) else []
        except Exception as e:
            print(f"Error reading history: {str(e)}")
            print(traceback.format_exc())
            return []

    def clear_all_history(self) -> bool:
        """清除所有歷史紀錄"""
        try:
            print("Attempting to clear all history")  # Debug
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            print("History cleared successfully")  # Debug
            return True
        except Exception as e:
            print(f"Error clearing history: {str(e)}")
            print(traceback.format_exc())
            return False
