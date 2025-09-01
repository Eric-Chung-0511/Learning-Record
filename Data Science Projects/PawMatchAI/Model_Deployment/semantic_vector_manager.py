import random
import hashlib
import numpy as np
import sqlite3
import re
import traceback
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from dog_database import get_dog_description
from breed_health_info import breed_health_info
from breed_noise_info import breed_noise_info

@dataclass
class BreedDescriptionVector:
    """品種描述向量的資料結構"""
    breed_name: str
    description_text: str
    embedding: np.ndarray
    characteristics: Dict[str, Any]

class SemanticVectorManager:
    """
    語義向量管理器
    處理 SBERT 模型初始化、品種向量化建構和品種描述生成
    """

    def __init__(self):
        """初始化語義向量管理器"""
        self.model_name = 'all-MiniLM-L6-v2'  
        self.sbert_model = None
        self._sbert_loading_attempted = False
        self.breed_vectors = {}
        self.breed_list = self._get_breed_list()
        # 延遲SBERT模型載入直到需要時才在GPU環境中進行
        print("SemanticVectorManager initialized (SBERT loading deferred)")

    def _get_breed_list(self) -> List[str]:
        """從資料庫獲取品種清單"""
        try:
            conn = sqlite3.connect('animal_detector.db')
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT Breed FROM AnimalCatalog")
            breeds = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            # 過濾掉野生動物品種
            breeds = [breed for breed in breeds if breed != 'Dhole']
            return breeds
        except Exception as e:
            print(f"Error getting breed list: {str(e)}")
            return ['Labrador_Retriever', 'German_Shepherd', 'Golden_Retriever',
                   'Bulldog', 'Poodle', 'Beagle', 'Rottweiler', 'Yorkshire_Terrier']

    def _initialize_model(self):
        """初始化 SBERT 模型，包含容錯機制 - 設計用於ZeroGPU相容性"""
        if self.sbert_model is not None or self._sbert_loading_attempted:
            return self.sbert_model
            
        try:
            print("Loading SBERT model in GPU context...")
            # 如果主要模型失敗，嘗試不同的模型名稱
            model_options = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'all-MiniLM-L12-v2']

            for model_name in model_options:
                try:
                    # 明確指定設備以處理ZeroGPU環境
                    import torch
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    self.sbert_model = SentenceTransformer(model_name, device=device)
                    self.model_name = model_name
                    print(f"SBERT model {model_name} loaded successfully on {device}")
                    return self.sbert_model
                except Exception as model_e:
                    print(f"Failed to load {model_name}: {str(model_e)}")
                    continue

            # 如果所有模型都失敗
            print("All SBERT models failed to load. Using basic text matching fallback.")
            self.sbert_model = None
            return None

        except Exception as e:
            print(f"Failed to initialize any SBERT model: {str(e)}")
            print(traceback.format_exc())
            print("Will provide basic text-based recommendations without embeddings")
            self.sbert_model = None
            return None
        finally:
            self._sbert_loading_attempted = True

    def _create_breed_description(self, breed: str) -> str:
        """為品種創建包含所有關鍵特徵的全面自然語言描述"""
        try:
            # 獲取所有信息來源
            breed_info = get_dog_description(breed) or {}
            health_info = breed_health_info.get(breed, {}) if breed_health_info else {}
            noise_info = breed_noise_info.get(breed, {}) if breed_noise_info else {}

            breed_display_name = breed.replace('_', ' ')
            description_parts = []

            # 1. 基本尺寸和身體特徵
            size = breed_info.get('Size', 'medium').lower()
            description_parts.append(f"{breed_display_name} is a {size} sized dog breed")

            # 2. 氣質和個性（匹配的關鍵因素）
            temperament = breed_info.get('Temperament', '')
            if temperament:
                description_parts.append(f"with a {temperament.lower()} temperament")

            # 3. 運動和活動水平（公寓居住的關鍵因素）
            exercise_needs = breed_info.get('Exercise Needs', 'moderate').lower()
            if 'high' in exercise_needs or 'very high' in exercise_needs:
                description_parts.append("requiring high daily exercise and mental stimulation")
            elif 'low' in exercise_needs or 'minimal' in exercise_needs:
                description_parts.append("with minimal exercise requirements, suitable for apartment living")
            else:
                description_parts.append("with moderate exercise needs")

            # 4. 噪音特徵（安靜需求的關鍵因素）
            noise_level = noise_info.get('noise_level', 'moderate').lower()
            if 'low' in noise_level or 'quiet' in noise_level:
                description_parts.append("known for being quiet and rarely barking")
            elif 'high' in noise_level or 'loud' in noise_level:
                description_parts.append("tends to be vocal and bark frequently")
            else:
                description_parts.append("with moderate barking tendencies")

            # 5. 居住空間相容性
            if size in ['small', 'tiny']:
                description_parts.append("excellent for small apartments and limited spaces")
            elif size in ['large', 'giant']:
                description_parts.append("requiring large living spaces and preferably a yard")
            else:
                description_parts.append("adaptable to various living situations")

            # 6. 美容和維護
            grooming_needs = breed_info.get('Grooming Needs', 'moderate').lower()
            if 'high' in grooming_needs:
                description_parts.append("requiring regular professional grooming")
            elif 'low' in grooming_needs:
                description_parts.append("with minimal grooming requirements")
            else:
                description_parts.append("with moderate grooming needs")

            # 7. 家庭相容性
            good_with_children = breed_info.get('Good with Children', 'Yes')
            if good_with_children == 'Yes':
                description_parts.append("excellent with children and families")
            else:
                description_parts.append("better suited for adult households")

            # 8. 智力和可訓練性（從資料庫描述中提取）
            intelligence_keywords = []
            description_text = breed_info.get('Description', '').lower()

            if description_text:
                # 從描述中提取智力指標
                if any(word in description_text for word in ['intelligent', 'smart', 'clever', 'quick to learn']):
                    intelligence_keywords.extend(['highly intelligent', 'trainable', 'quick learner'])
                elif any(word in description_text for word in ['stubborn', 'independent', 'difficult to train']):
                    intelligence_keywords.extend(['independent minded', 'requires patience', 'challenging to train'])
                else:
                    intelligence_keywords.extend(['moderate intelligence', 'trainable with consistency'])

                # 從描述中提取工作/用途特徵
                if any(word in description_text for word in ['working', 'herding', 'guard', 'hunting']):
                    intelligence_keywords.extend(['working breed', 'purpose-driven', 'task-oriented'])
                elif any(word in description_text for word in ['companion', 'lap', 'toy', 'decorative']):
                    intelligence_keywords.extend(['companion breed', 'affectionate', 'people-focused'])

                # 添加智力背景到描述中
                if intelligence_keywords:
                    description_parts.append(f"characterized as {', '.join(intelligence_keywords[:2])}")

            # 9. 特殊特徵和用途（使用資料庫挖掘進行增強）
            if breed_info.get('Description'):
                desc = breed_info.get('Description', '')[:150]  # 增加到 150 字元以提供更多背景
                if desc:
                    # 從描述中提取關鍵特徵以便更好的語義匹配
                    desc_lower = desc.lower()
                    key_traits = []

                    # 從描述中提取關鍵行為特徵
                    if 'friendly' in desc_lower:
                        key_traits.append('friendly')
                    if 'gentle' in desc_lower:
                        key_traits.append('gentle')
                    if 'energetic' in desc_lower or 'active' in desc_lower:
                        key_traits.append('energetic')
                    if 'calm' in desc_lower or 'peaceful' in desc_lower:
                        key_traits.append('calm')
                    if 'protective' in desc_lower or 'guard' in desc_lower:
                        key_traits.append('protective')

                    trait_text = f" and {', '.join(key_traits)}" if key_traits else ""
                    description_parts.append(f"Known for: {desc.lower()}{trait_text}")

            # 10. 照護水平需求
            try:
                care_level = breed_info.get('Care Level', 'moderate')
                if isinstance(care_level, str):
                    description_parts.append(f"requiring {care_level.lower()} overall care level")
                else:
                    description_parts.append("requiring moderate overall care level")
            except Exception as e:
                print(f"Error processing care level for {breed}: {str(e)}")
                description_parts.append("requiring moderate overall care level")

            # 11. 壽命資訊
            try:
                lifespan = breed_info.get('Lifespan', '10-12 years')
                if lifespan and isinstance(lifespan, str) and lifespan.strip():
                    description_parts.append(f"with a typical lifespan of {lifespan}")
                else:
                    description_parts.append("with a typical lifespan of 10-12 years")
            except Exception as e:
                print(f"Error processing lifespan for {breed}: {str(e)}")
                description_parts.append("with a typical lifespan of 10-12 years")

            # 創建全面的描述
            full_description = '. '.join(description_parts) + '.'

            # 添加全面的關鍵字以便更好的語義匹配
            keywords = []

            # 基本品種名稱關鍵字
            keywords.extend([word.lower() for word in breed_display_name.split()])

            # 氣質關鍵字
            if temperament:
                keywords.extend([word.lower().strip(',') for word in temperament.split()])

            # 基於尺寸的關鍵字
            if 'small' in size or 'tiny' in size:
                keywords.extend(['small', 'tiny', 'compact', 'little', 'apartment', 'indoor', 'lap'])
            elif 'large' in size or 'giant' in size:
                keywords.extend(['large', 'big', 'giant', 'huge', 'yard', 'space', 'outdoor'])
            else:
                keywords.extend(['medium', 'moderate', 'average', 'balanced'])

            # 活動水平關鍵字
            exercise_needs = breed_info.get('Exercise Needs', 'moderate').lower()
            if 'high' in exercise_needs:
                keywords.extend(['active', 'energetic', 'exercise', 'outdoor', 'hiking', 'running', 'athletic'])
            elif 'low' in exercise_needs:
                keywords.extend(['calm', 'low-energy', 'indoor', 'relaxed', 'couch', 'sedentary'])
            else:
                keywords.extend(['moderate', 'balanced', 'walks', 'regular'])

            # 噪音水平關鍵字
            noise_level = noise_info.get('noise_level', 'moderate').lower()
            if 'quiet' in noise_level or 'low' in noise_level:
                keywords.extend(['quiet', 'silent', 'calm', 'peaceful', 'low-noise'])
            elif 'high' in noise_level or 'loud' in noise_level:
                keywords.extend(['vocal', 'barking', 'loud', 'alert', 'watchdog'])

            # 居住情況關鍵字
            if size in ['small', 'tiny'] and 'low' in exercise_needs:
                keywords.extend(['apartment', 'city', 'urban', 'small-space'])
            if size in ['large', 'giant'] or 'high' in exercise_needs:
                keywords.extend(['house', 'yard', 'suburban', 'rural', 'space'])

            # 家庭關鍵字
            good_with_children = breed_info.get('Good with Children', 'Yes')
            if good_with_children == 'Yes':
                keywords.extend(['family', 'children', 'kids', 'friendly', 'gentle'])

            # 智力和可訓練性關鍵字（從資料庫描述挖掘）
            if intelligence_keywords:
                keywords.extend([word.lower() for phrase in intelligence_keywords for word in phrase.split()])

            # 美容相關關鍵字（增強）
            grooming_needs = breed_info.get('Grooming Needs', 'moderate').lower()
            if 'high' in grooming_needs:
                keywords.extend(['high-maintenance', 'professional-grooming', 'daily-brushing', 'coat-care'])
            elif 'low' in grooming_needs:
                keywords.extend(['low-maintenance', 'minimal-grooming', 'easy-care', 'wash-and-go'])
            else:
                keywords.extend(['moderate-grooming', 'weekly-brushing', 'regular-care'])

            # 基於壽命的關鍵字
            lifespan = breed_info.get('Lifespan', '10-12 years')
            if lifespan and isinstance(lifespan, str):
                try:
                    # 從壽命字符串中提取年數（例如 "10-12 years" 或 "12-15 years"）
                    import re
                    years = re.findall(r'\d+', lifespan)
                    if years:
                        avg_years = sum(int(y) for y in years) / len(years)
                        if avg_years >= 14:
                            keywords.extend(['long-lived', 'longevity', 'durable', 'healthy-lifespan'])
                        elif avg_years <= 8:
                            keywords.extend(['shorter-lifespan', 'health-considerations', 'special-care'])
                        else:
                            keywords.extend(['average-lifespan', 'moderate-longevity'])
                except:
                    keywords.extend(['average-lifespan'])

            # 將關鍵字添加到描述中以便更好的語義匹配
            unique_keywords = list(set(keywords))
            keyword_text = ' '.join(unique_keywords)
            full_description += f" Additional context: {keyword_text}"

            return full_description

        except Exception as e:
            print(f"Error creating description for {breed}: {str(e)}")
            return f"{breed.replace('_', ' ')} is a dog breed with unique characteristics."

    def _build_breed_vectors(self):
        """為所有品種建立向量表示 - 延遲調用當需要時"""
        try:
            print("Building breed vector database...")

            # 初始化模型如果尚未完成
            if self.sbert_model is None:
                self._initialize_model()
                
            # 如果模型不可用則跳過
            if self.sbert_model is None:
                print("SBERT model not available, skipping vector building")
                return

            for breed in self.breed_list:
                description = self._create_breed_description(breed)

                # 生成嵌入向量
                embedding = self.sbert_model.encode(description, convert_to_tensor=False)

                # 獲取品種特徵
                breed_info = get_dog_description(breed)
                characteristics = {
                    'size': breed_info.get('Size', 'Medium') if breed_info else 'Medium',
                    'exercise_needs': breed_info.get('Exercise Needs', 'Moderate') if breed_info else 'Moderate',
                    'grooming_needs': breed_info.get('Grooming Needs', 'Moderate') if breed_info else 'Moderate',
                    'good_with_children': breed_info.get('Good with Children', 'Yes') if breed_info else 'Yes',
                    'temperament': breed_info.get('Temperament', '') if breed_info else ''
                }

                self.breed_vectors[breed] = BreedDescriptionVector(
                    breed_name=breed,
                    description_text=description,
                    embedding=embedding,
                    characteristics=characteristics
                )

            print(f"Successfully built {len(self.breed_vectors)} breed vectors")

        except Exception as e:
            print(f"Error building breed vectors: {str(e)}")
            print(traceback.format_exc())
            raise

    def get_breed_vectors(self) -> Dict[str, BreedDescriptionVector]:
        """獲取所有品種向量"""
        # 確保向量已建構
        if not self.breed_vectors:
            self._build_breed_vectors()
        return self.breed_vectors

    def get_sbert_model(self) -> Optional[SentenceTransformer]:
        """獲取 SBERT 模型"""
        return self.sbert_model

    def get_breed_list(self) -> List[str]:
        """獲取品種清單"""
        return self.breed_list

    def is_model_available(self) -> bool:
        """檢查 SBERT 模型是否可用"""
        return self.sbert_model is not None

    def encode_text(self, text: str) -> np.ndarray:
        """使用 SBERT 模型編碼文本"""
        # 初始化模型如果尚未完成
        if self.sbert_model is None:
            self._initialize_model()
            
        if self.sbert_model is None:
            raise RuntimeError("SBERT model not available")
        return self.sbert_model.encode(text, convert_to_tensor=False)
