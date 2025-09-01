from datetime import datetime
import json
import os
import pytz
import traceback

class UserHistoryManager:
    def __init__(self):
        """Initialize history record manager"""
        self.history_file = "user_history.json"
        print(f"Initializing UserHistoryManager with file: {os.path.abspath(self.history_file)}")
        self._init_file()

    def _init_file(self):
        """Initialize JSON file"""
        try:
            if not os.path.exists(self.history_file):
                print(f"Creating new history file: {self.history_file}")
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
            else:
                print(f"History file exists: {self.history_file}")
                # Added a check for empty file before loading
                if os.path.getsize(self.history_file) > 0:
                    with open(self.history_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        print(f"Current history entries: {len(data)}")
                else:
                    print("History file is empty.")
        except Exception as e:
            print(f"Error in _init_file: {str(e)}")
            print(traceback.format_exc())

    def save_history(self, user_preferences: dict = None, results: list = None, search_type: str = "criteria", description: str = None, user_description: str = None) -> bool:
        """
        Save search history with complete result data
        """
        try:
            taipei_tz = pytz.timezone('Asia/Taipei')
            current_time = datetime.now(taipei_tz)

            history_entry = {
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "search_type": search_type
            }

            description_text = user_description or description
            if search_type == "description" and description_text:
                history_entry["user_description"] = description_text[:200] + "..." if len(description_text) > 200 else description_text

            def _to_float(x, default=0.0):
                try:
                    return float(x)
                except Exception:
                    return default

            def _to_int(x, default=0):
                try:
                    return int(x)
                except Exception:
                    return default

            if results and isinstance(results, list):
                processed_results = []
                for i, r in enumerate(results[:15], start=1):
                    processed_results.append({
                        "breed": str(r.get("breed", "Unknown")),
                        "rank": _to_int(r.get("rank", i)),
                        # 先拿 overall_score，沒有就退 final_score，都轉成 float
                        "overall_score": _to_float(r.get("overall_score", r.get("final_score", 0))),
                        # 描述搜尋常見附加分，也一併安全轉型
                        "semantic_score": _to_float(r.get("semantic_score", 0)),
                        "comparative_bonus": _to_float(r.get("comparative_bonus", 0)),
                        "lifestyle_bonus": _to_float(r.get("lifestyle_bonus", 0)),
                        "size": str(r.get("size", "Unknown")),
                    })
                history_entry["results"] = processed_results

            if user_preferences:
                history_entry["preferences"] = {
                    'living_space': user_preferences.get('living_space'),
                    'exercise_time': user_preferences.get('exercise_time'),
                    'grooming_commitment': user_preferences.get('grooming_commitment'),
                    'experience_level': user_preferences.get('experience_level'),
                    'has_children': user_preferences.get('has_children'),
                    'noise_tolerance': user_preferences.get('noise_tolerance'),
                    'size_preference': user_preferences.get('size_preference')
                }

            try:
                history = []
                if os.path.exists(self.history_file) and os.path.getsize(self.history_file) > 0:
                    with open(self.history_file, 'r', encoding='utf-8') as f:
                        history = json.load(f)
            except json.JSONDecodeError as e:
                print(f"JSON decode error when reading history: {str(e)}")
                backup_file = f"{self.history_file}.backup.{int(datetime.now().timestamp())}"
                if os.path.exists(self.history_file):
                    os.rename(self.history_file, backup_file)
                    print(f"Backed up corrupted file to {backup_file}")
                history = []

            history.append(history_entry)
            history = history[-20:] # Keep recent 20 entries

            temp_file = f"{self.history_file}.tmp"
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)
                os.rename(temp_file, self.history_file)
            except Exception as e:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                raise

            print(f"Successfully saved history entry for {search_type} search.")
            return True

        except Exception as e:
            print(f"Error saving history: {str(e)}")
            print(traceback.format_exc())
            return False

    # get_history, clear_all_history, and format_history_for_display methods remain the same as you provided
    def get_history(self) -> list:
        """Get search history"""
        try:
            print("Attempting to read history")  # Debug

            # Check if file exists and is not empty
            if not os.path.exists(self.history_file):
                print("History file does not exist, creating empty file")
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                return []

            # Check file size
            if os.path.getsize(self.history_file) == 0:
                print("History file is empty, initializing with empty array")
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                return []

            # Try to read with error recovery
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        print("File content is empty, returning empty list")
                        return []
                    data = json.loads(content)
                    print(f"Read {len(data)} history entries")  # Debug
                    return data if isinstance(data, list) else []
            except json.JSONDecodeError as je:
                print(f"JSON decode error: {str(je)}")
                print(f"Corrupted content near position {je.pos}")
                # Backup corrupted file and create new one
                backup_file = f"{self.history_file}.backup"
                os.rename(self.history_file, backup_file)
                print(f"Backed up corrupted file to {backup_file}")
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                return []

        except Exception as e:
            print(f"Error reading history: {str(e)}")
            print(traceback.format_exc())
            return []

    def clear_all_history(self) -> bool:
        """Clear all history records"""
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

    def format_history_for_display(self) -> str:
        """
        Format history records for HTML display

        Returns:
            str: Formatted HTML string
        """
        try:
            history = self.get_history()

            if not history:
                return """
                <div style="text-align: center; padding: 20px; color: #718096;">
                    <p>No search history yet</p>
                </div>
                """

            html_parts = []
            html_parts.append("""
                <div style="max-height: 400px; overflow-y: auto;">
            """)

            for i, entry in enumerate(reversed(history)):  # Latest entries first
                search_type = entry.get('search_type', 'criteria')
                timestamp = entry.get('timestamp', 'Unknown time')
                results = entry.get('results', [])

                # Set tag color based on search type
                if search_type == 'description':
                    tag_color = "#4299e1"  # Blue
                    tag_bg = "rgba(66, 153, 225, 0.1)"
                    tag_text = "Description Search"
                    icon = "🤖"
                else:
                    tag_color = "#48bb78"  # Green
                    tag_bg = "rgba(72, 187, 120, 0.1)"
                    tag_text = "Criteria Search"
                    icon = "🔍"

                # Search content preview
                preview_content = ""
                if search_type == 'description':
                    user_desc = entry.get('user_description', '')
                    if user_desc:
                        preview_content = f"Description: {user_desc}"
                else:
                    prefs = entry.get('preferences', {})
                    if prefs:
                        living = prefs.get('living_space', '')
                        size = prefs.get('size_preference', '')
                        exercise = prefs.get('exercise_time', '')
                        preview_content = f"Living: {living}, Size: {size}, Exercise: {exercise}min"

                # Result summary
                result_summary = ""
                if results:
                    top_breeds = [r.get('breed', 'Unknown') for r in results[:3]]
                    result_summary = f"Recommended: {', '.join(top_breeds)}"
                    if len(results) > 3:
                        result_summary += f" and {len(results)} breeds total"

                html_parts.append(f"""
                    <div style="
                        border: 1px solid #e2e8f0;
                        border-radius: 8px;
                        padding: 12px;
                        margin: 8px 0;
                        background: white;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    ">
                        <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 8px;">
                            <div style="
                                background: {tag_bg};
                                color: {tag_color};
                                padding: 4px 8px;
                                border-radius: 12px;
                                font-size: 0.8em;
                                font-weight: 600;
                                display: inline-flex;
                                align-items: center;
                                gap: 4px;
                            ">
                                {icon} {tag_text}
                            </div>
                            <div style="font-size: 0.8em; color: #718096;">
                                {timestamp}
                            </div>
                        </div>

                        {f'<div style="font-size: 0.9em; color: #4a5568; margin: 4px 0;">{preview_content}</div>' if preview_content else ''}
                        {f'<div style="font-size: 0.9em; color: #2d3748; font-weight: 500;">{result_summary}</div>' if result_summary else ''}
                    </div>
                """)

            html_parts.append("</div>")

            return ''.join(html_parts)

        except Exception as e:
            print(f"Error formatting history for display: {str(e)}")
            return f"""
            <div style="text-align: center; padding: 20px; color: #e53e3e;">
                <p>Error loading history records: {str(e)}</p>
            </div>
            """

    def get_search_statistics(self) -> dict:
        """
        Get search statistics information

        Returns:
            dict: Statistics information
        """
        try:
            history = self.get_history()

            stats = {
                'total_searches': len(history),
                'criteria_searches': 0,
                'description_searches': 0,
                'most_searched_breeds': {},
                'search_frequency_by_day': {}
            }

            for entry in history:
                search_type = entry.get('search_type', 'criteria')
                if search_type == 'description':
                    stats['description_searches'] += 1
                else:
                    stats['criteria_searches'] += 1

                # Count breed search frequency
                results = entry.get('results', [])
                for result in results:
                    breed = result.get('breed', 'Unknown')
                    stats['most_searched_breeds'][breed] = stats['most_searched_breeds'].get(breed, 0) + 1

                # Count search frequency by date
                timestamp = entry.get('timestamp', '')
                if timestamp:
                    date = timestamp.split(' ')[0]
                    stats['search_frequency_by_day'][date] = stats['search_frequency_by_day'].get(date, 0) + 1

            return stats

        except Exception as e:
            print(f"Error getting search statistics: {str(e)}")
            return {
                'total_searches': 0,
                'criteria_searches': 0,
                'description_searches': 0,
                'most_searched_breeds': {},
                'search_frequency_by_day': {}
            }
