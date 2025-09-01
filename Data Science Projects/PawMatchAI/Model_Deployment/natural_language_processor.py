import re
import string
from typing import Dict, List, Tuple, Optional, Any
import traceback

class NaturalLanguageProcessor:
    """
    Natural language processing utility class
    Handles text preprocessing and keyword extraction for user input
    """

    def __init__(self):
        """Initialize the natural language processor"""
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'i', 'me', 'my', 'we', 'us',
            'our', 'you', 'your', 'they', 'them', 'their'
        }

        # Breed name mappings (common aliases to standard names)
        self.breed_aliases = {
            'lab': 'labrador_retriever',
            'labrador': 'labrador_retriever',
            'golden': 'golden_retriever',
            'retriever': ['labrador_retriever', 'golden_retriever'],
            'german shepherd': 'german_shepherd',
            'shepherd': 'german_shepherd',
            'border collie': 'border_collie',
            'collie': ['border_collie', 'collie'],
            'bulldog': ['french_bulldog', 'english_bulldog'],
            'french bulldog': 'french_bulldog',
            'poodle': ['standard_poodle', 'miniature_poodle', 'toy_poodle'],
            'husky': 'siberian_husky',
            'siberian husky': 'siberian_husky',
            'beagle': 'beagle',
            'yorkshire terrier': 'yorkshire_terrier',
            'yorkie': 'yorkshire_terrier',
            'chihuahua': 'chihuahua',
            'dachshund': 'dachshund',
            'wiener dog': 'dachshund',
            'rottweiler': 'rottweiler',
            'rottie': 'rottweiler',
            'boxer': 'boxer',
            'great dane': 'great_dane',
            'dane': 'great_dane',
            'mastiff': ['bull_mastiff', 'tibetan_mastiff'],
            'pitbull': 'american_staffordshire_terrier',
            'pit bull': 'american_staffordshire_terrier',
            'shih tzu': 'shih-tzu',
            'maltese': 'maltese_dog',
            'pug': 'pug',
            'basset hound': 'basset',
            'bloodhound': 'bloodhound',
            'australian shepherd': 'kelpie',
            'aussie': 'kelpie'
        }

        # Lifestyle keyword mappings
        self.lifestyle_keywords = {
            'living_space': {
                'apartment': ['apartment', 'flat', 'condo', 'small space', 'city living', 'urban'],
                'house': ['house', 'home', 'yard', 'garden', 'suburban', 'large space'],
                'farm': ['farm', 'rural', 'country', 'acreage', 'ranch']
            },
            'activity_level': {
                'very_high': ['very active', 'extremely energetic', 'marathon runner', 'athlete'],
                'high': ['active', 'energetic', 'exercise', 'hiking', 'running', 'outdoor activities',
                        'sports', 'jogging', 'biking', 'adventure'],
                'moderate': ['moderate exercise', 'some activity', 'weekend walks', 'occasional exercise'],
                'low': ['calm', 'lazy', 'indoor', 'low energy', 'couch potato', 'sedentary', 'quiet lifestyle']
            },
            'family_situation': {
                'children': ['children', 'kids', 'toddlers', 'babies', 'family with children', 'young family'],
                'elderly': ['elderly', 'senior', 'old', 'retired', 'senior citizen'],
                'single': ['single', 'alone', 'individual', 'bachelor', 'solo'],
                'couple': ['couple', 'two people', 'pair', 'duo']
            },
            'noise_tolerance': {
                'low': ['quiet', 'silent', 'noise-sensitive', 'peaceful', 'no barking', 'minimal noise'],
                'moderate': ['some noise ok', 'moderate barking', 'normal noise'],
                'high': ['loud ok', 'barking fine', 'noise tolerant', 'doesn\'t mind noise']
            },
            'size_preference': {
                'small': ['small', 'tiny', 'little', 'compact', 'lap dog', 'petite', 'miniature'],
                'medium': ['medium', 'moderate size', 'average', 'mid-size'],
                'large': ['large', 'big', 'huge', 'giant', 'massive', 'substantial'],
                'varies': ['any size', 'size doesn\'t matter', 'flexible on size']
            },
            'experience_level': {
                'beginner': ['first time', 'beginner', 'new to dogs', 'inexperienced', 'never had'],
                'some': ['some experience', 'had dogs before', 'moderate experience'],
                'experienced': ['experienced', 'expert', 'very experienced', 'professional', 'trainer']
            },
            'grooming_commitment': {
                'low': ['low maintenance', 'easy care', 'minimal grooming', 'wash and go'],
                'moderate': ['moderate grooming', 'some brushing', 'regular care'],
                'high': ['high maintenance', 'lots of grooming', 'professional grooming', 'daily brushing']
            },
            'special_needs': {
                'guard': ['guard dog', 'protection', 'security', 'watchdog', 'guardian'],
                'therapy': ['therapy dog', 'emotional support', 'comfort', 'calm companion'],
                'hypoallergenic': ['hypoallergenic', 'allergies', 'non-shedding', 'allergy friendly'],
                'working': ['working dog', 'job', 'task', 'service dog'],
                'companion': ['companion', 'friend', 'buddy', 'lap dog', 'cuddle']
            }
        }

        # Comparative preference keywords
        self.preference_indicators = {
            'love': 1.0,
            'prefer': 0.9,
            'like': 0.8,
            'want': 0.8,
            'interested in': 0.7,
            'considering': 0.6,
            'ok with': 0.5,
            'don\'t mind': 0.4,
            'not interested': 0.2,
            'dislike': 0.1,
            'hate': 0.0
        }

        # Order keywords
        self.order_keywords = {
            'first': 1.0, 'most': 1.0, 'primary': 1.0, 'main': 1.0,
            'second': 0.8, 'then': 0.8, 'next': 0.8,
            'third': 0.6, 'also': 0.6, 'additionally': 0.6,
            'last': 0.4, 'least': 0.4, 'finally': 0.4
        }

    def preprocess_text(self, text: str) -> str:
        """
        Text preprocessing

        Args:
            text: Raw text

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower().strip()

        # Remove punctuation (keep some meaningful ones)
        text = re.sub(r'[^\w\s\-\']', ' ', text)

        # Handle extra whitespace
        text = re.sub(r'\s+', ' ', text)

        return text

    def extract_breed_mentions(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract mentioned breeds and their preference levels from text

        Args:
            text: Input text

        Returns:
            List of (breed_name, preference_score) tuples
        """
        text = self.preprocess_text(text)
        breed_mentions = []

        try:
            # Check each breed alias
            for alias, standard_breed in self.breed_aliases.items():
                if alias in text:
                    # Find surrounding preference indicators
                    preference_score = self._find_preference_score(text, alias)

                    if isinstance(standard_breed, list):
                        # If alias maps to multiple breeds, add all
                        for breed in standard_breed:
                            breed_mentions.append((breed, preference_score))
                    else:
                        breed_mentions.append((standard_breed, preference_score))

            # Deduplicate and merge scores
            breed_scores = {}
            for breed, score in breed_mentions:
                if breed in breed_scores:
                    breed_scores[breed] = max(breed_scores[breed], score)
                else:
                    breed_scores[breed] = score

            return list(breed_scores.items())

        except Exception as e:
            print(f"Error extracting breed mentions: {str(e)}")
            return []

    def _find_preference_score(self, text: str, breed_mention: str) -> float:
        """
        Find preference score near breed mention

        Args:
            text: Text
            breed_mention: Breed mention

        Returns:
            Preference score (0.0-1.0)
        """
        try:
            # Find breed mention position
            mention_pos = text.find(breed_mention)
            if mention_pos == -1:
                return 0.5  # Default neutral score

            # Check context (50 characters before and after)
            context_start = max(0, mention_pos - 50)
            context_end = min(len(text), mention_pos + len(breed_mention) + 50)
            context = text[context_start:context_end]

            # Find preference indicators
            max_score = 0.5  # Default score

            for indicator, score in self.preference_indicators.items():
                if indicator in context:
                    max_score = max(max_score, score)

            # Find order keywords
            for order_word, multiplier in self.order_keywords.items():
                if order_word in context:
                    max_score = max(max_score, max_score * multiplier)

            return max_score

        except Exception as e:
            print(f"Error finding preference score: {str(e)}")
            return 0.5

    def extract_lifestyle_preferences(self, text: str) -> Dict[str, Dict[str, float]]:
        """
        Extract lifestyle preferences from text

        Args:
            text: Input text

        Returns:
            Lifestyle preferences dictionary
        """
        text = self.preprocess_text(text)
        preferences = {}

        try:
            for category, keywords_dict in self.lifestyle_keywords.items():
                preferences[category] = {}

                for preference_type, keywords in keywords_dict.items():
                    score = 0.0
                    count = 0

                    for keyword in keywords:
                        if keyword in text:
                            # Calculate keyword occurrence intensity
                            keyword_count = text.count(keyword)
                            score += keyword_count
                            count += keyword_count

                    if count > 0:
                        # Normalize score
                        preferences[category][preference_type] = min(score / max(count, 1), 1.0)

            return preferences

        except Exception as e:
            print(f"Error extracting lifestyle preferences: {str(e)}")
            return {}

    def generate_search_keywords(self, text: str) -> List[str]:
        """
        Generate keyword list for search

        Args:
            text: Input text

        Returns:
            List of keywords
        """
        text = self.preprocess_text(text)
        keywords = []

        try:
            # Tokenize and filter stop words
            words = text.split()
            for word in words:
                if len(word) > 2 and word not in self.stop_words:
                    keywords.append(word)

            # Extract important phrases
            phrases = self._extract_phrases(text)
            keywords.extend(phrases)

            # Remove duplicates
            keywords = list(set(keywords))

            return keywords

        except Exception as e:
            print(f"Error generating search keywords: {str(e)}")
            return []

    def _extract_phrases(self, text: str) -> List[str]:
        """
        Extract important phrases

        Args:
            text: Input text

        Returns:
            List of phrases
        """
        phrases = []

        # Define important phrase patterns
        phrase_patterns = [
            r'good with \w+',
            r'apartment \w+',
            r'family \w+',
            r'exercise \w+',
            r'grooming \w+',
            r'noise \w+',
            r'training \w+',
            r'health \w+',
            r'\w+ friendly',
            r'\w+ tolerant',
            r'\w+ maintenance',
            r'\w+ energy',
            r'\w+ barking',
            r'\w+ shedding'
        ]

        for pattern in phrase_patterns:
            matches = re.findall(pattern, text)
            phrases.extend(matches)

        return phrases

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze text sentiment

        Args:
            text: Input text

        Returns:
            Sentiment analysis results {'positive': 0.0-1.0, 'negative': 0.0-1.0, 'neutral': 0.0-1.0}
        """
        text = self.preprocess_text(text)

        positive_words = [
            'love', 'like', 'want', 'prefer', 'good', 'great', 'excellent',
            'perfect', 'ideal', 'wonderful', 'amazing', 'fantastic'
        ]

        negative_words = [
            'hate', 'dislike', 'bad', 'terrible', 'awful', 'horrible',
            'not good', 'don\'t want', 'avoid', 'against', 'problem'
        ]

        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        total_words = len(text.split())

        if total_words == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        neutral_ratio = 1.0 - positive_ratio - negative_ratio

        return {
            'positive': positive_ratio,
            'negative': negative_ratio,
            'neutral': max(0.0, neutral_ratio)
        }

    def extract_implicit_preferences(self, text: str) -> Dict[str, Any]:
        """
        Extract implicit preferences from text

        Args:
            text: Input text

        Returns:
            Dictionary of implicit preferences
        """
        text = self.preprocess_text(text)
        implicit_prefs = {}

        try:
            # Infer preferences from mentioned activities
            if any(activity in text for activity in ['hiking', 'running', 'jogging', 'outdoor']):
                implicit_prefs['exercise_needs'] = 'high'
                implicit_prefs['size_preference'] = 'medium_to_large'

            # Infer from living environment
            if any(env in text for env in ['apartment', 'small space', 'city']):
                implicit_prefs['size_preference'] = 'small_to_medium'
                implicit_prefs['noise_tolerance'] = 'low'
                implicit_prefs['exercise_needs'] = 'moderate'

            # Infer from family situation
            if 'children' in text or 'kids' in text:
                implicit_prefs['temperament'] = 'gentle_patient'
                implicit_prefs['good_with_children'] = True

            # Infer from experience level
            if any(exp in text for exp in ['first time', 'beginner', 'new to']):
                implicit_prefs['care_level'] = 'low_to_moderate'
                implicit_prefs['training_difficulty'] = 'easy'

            # Infer from time commitment
            if any(time in text for time in ['busy', 'no time', 'low maintenance']):
                implicit_prefs['grooming_needs'] = 'low'
                implicit_prefs['care_level'] = 'low'
                implicit_prefs['exercise_needs'] = 'low_to_moderate'

            return implicit_prefs

        except Exception as e:
            print(f"Error extracting implicit preferences: {str(e)}")
            return {}

    def validate_input(self, text: str) -> Dict[str, Any]:
        """
        Validate input text validity

        Args:
            text: Input text

        Returns:
            Validation results dictionary
        """
        if not text or not text.strip():
            return {
                'is_valid': False,
                'error': 'Empty input',
                'suggestions': ['Please provide a description of your preferences']
            }

        text = text.strip()

        # Check length
        if len(text) < 10:
            return {
                'is_valid': False,
                'error': 'Input too short',
                'suggestions': ['Please provide more details about your preferences']
            }

        if len(text) > 1000:
            return {
                'is_valid': False,
                'error': 'Input too long',
                'suggestions': ['Please provide a more concise description']
            }

        # Check for meaningful content
        processed_text = self.preprocess_text(text)
        meaningful_words = [word for word in processed_text.split()
                          if len(word) > 2 and word not in self.stop_words]

        if len(meaningful_words) < 3:
            return {
                'is_valid': False,
                'error': 'Not enough meaningful content',
                'suggestions': ['Please provide more specific details about your lifestyle and preferences']
            }

        return {
            'is_valid': True,
            'word_count': len(meaningful_words),
            'suggestions': []
        }

def get_nlp_processor():
    """Get natural language processor instance"""
    try:
        return NaturalLanguageProcessor()
    except Exception as e:
        print(f"Error creating NLP processor: {str(e)}")
        return None
