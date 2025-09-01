import sqlite3
import json
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import traceback
from dog_database import get_dog_description
from dynamic_scoring_config import get_scoring_config
from breed_health_info import breed_health_info
from breed_noise_info import breed_noise_info
from query_understanding import QueryDimensions

class ConstraintPriority(Enum):
    """Constraint priority definitions"""
    CRITICAL = 1      # Critical constraints (safety, space)
    HIGH = 2          # High priority (activity level, noise)
    MODERATE = 3      # Moderate priority (maintenance, experience)
    FLEXIBLE = 4      # Flexible constraints (other preferences)

@dataclass
class ConstraintRule:
    """Constraint rule structure"""
    name: str
    priority: ConstraintPriority
    description: str
    filter_function: str  # Function name
    relaxation_allowed: bool = True
    safety_critical: bool = False

@dataclass
class FilterResult:
    """Filter result structure"""
    passed_breeds: Set[str]
    filtered_breeds: Dict[str, str]  # breed -> reason
    applied_constraints: List[str]
    relaxed_constraints: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class ConstraintManager:
    """
    Hierarchical constraint management system
    Implements priority-based constraint filtering with progressive constraint relaxation
    """

    def __init__(self):
        """Initialize constraint manager"""
        self.breed_list = self._load_breed_list()
        self.breed_cache = {}  # Breed information cache
        self.constraint_rules = self._initialize_constraint_rules()
        self._warm_cache()

    def _load_breed_list(self) -> List[str]:
        """Load breed list from database"""
        try:
            conn = sqlite3.connect('animal_detector.db')
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT Breed FROM AnimalCatalog")
            breeds = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return breeds
        except Exception as e:
            print(f"Error loading breed list: {str(e)}")
            return ['Labrador_Retriever', 'German_Shepherd', 'Golden_Retriever',
                   'Bulldog', 'Poodle', 'Beagle', 'Border_Collie', 'Yorkshire_Terrier']

    def _warm_cache(self):
        """Warm up breed information cache"""
        for breed in self.breed_list:
            self.breed_cache[breed] = self._get_breed_info(breed)

    def _get_breed_info(self, breed: str) -> Dict[str, Any]:
        """Get comprehensive breed information"""
        if breed in self.breed_cache:
            return self.breed_cache[breed]

        try:
            # Basic breed information
            breed_info = get_dog_description(breed) or {}

            # Health information
            health_info = breed_health_info.get(breed, {})

            # Noise information
            noise_info = breed_noise_info.get(breed, {})

            # Combine all information
            combined_info = {
                'breed_name': breed,
                'display_name': breed.replace('_', ' '),
                'size': breed_info.get('Size', '').lower(),
                'exercise_needs': breed_info.get('Exercise Needs', '').lower(),
                'grooming_needs': breed_info.get('Grooming Needs', '').lower(),
                'temperament': breed_info.get('Temperament', '').lower(),
                'good_with_children': breed_info.get('Good with Children', 'Yes'),
                'care_level': breed_info.get('Care Level', '').lower(),
                'lifespan': breed_info.get('Lifespan', '10-12 years'),
                'noise_level': noise_info.get('noise_level', 'moderate').lower(),
                'health_issues': health_info.get('health_notes', ''),
                'raw_breed_info': breed_info,
                'raw_health_info': health_info,
                'raw_noise_info': noise_info
            }

            self.breed_cache[breed] = combined_info
            return combined_info

        except Exception as e:
            print(f"Error getting breed info for {breed}: {str(e)}")
            return {'breed_name': breed, 'display_name': breed.replace('_', ' ')}

    def _initialize_constraint_rules(self) -> List[ConstraintRule]:
        """Initialize constraint rules"""
        return [
            # Priority 1: Critical constraints (cannot be violated)
            ConstraintRule(
                name="apartment_size_constraint",
                priority=ConstraintPriority.CRITICAL,
                description="Apartment living space size restrictions",
                filter_function="filter_apartment_size",
                relaxation_allowed=False,
                safety_critical=True
            ),
            ConstraintRule(
                name="child_safety_constraint",
                priority=ConstraintPriority.CRITICAL,
                description="Child safety compatibility",
                filter_function="filter_child_safety",
                relaxation_allowed=False,
                safety_critical=True
            ),
            ConstraintRule(
                name="severe_allergy_constraint",
                priority=ConstraintPriority.CRITICAL,
                description="Severe allergy restrictions",
                filter_function="filter_severe_allergies",
                relaxation_allowed=False,
                safety_critical=True
            ),

            # Priority 2: High priority constraints
            ConstraintRule(
                name="exercise_constraint",
                priority=ConstraintPriority.HIGH,
                description="Exercise requirement mismatch",
                filter_function="filter_exercise_mismatch",
                relaxation_allowed=False,
                safety_critical=False
            ),
            ConstraintRule(
                name="size_bias_correction",
                priority=ConstraintPriority.MODERATE,
                description="Correct size bias in moderate lifestyle matches",
                filter_function="filter_size_bias",
                relaxation_allowed=True,
                safety_critical=False
            ),
            ConstraintRule(
                name="low_activity_constraint",
                priority=ConstraintPriority.HIGH,
                description="Low activity level restrictions",
                filter_function="filter_low_activity",
                relaxation_allowed=True
            ),
            ConstraintRule(
                name="quiet_requirement_constraint",
                priority=ConstraintPriority.HIGH,
                description="Quiet environment requirements",
                filter_function="filter_quiet_requirements",
                relaxation_allowed=True
            ),
            ConstraintRule(
                name="space_compatibility_constraint",
                priority=ConstraintPriority.HIGH,
                description="Living space compatibility",
                filter_function="filter_space_compatibility",
                relaxation_allowed=True
            ),

            # Priority 3: Moderate constraints
            ConstraintRule(
                name="grooming_preference_constraint",
                priority=ConstraintPriority.MODERATE,
                description="Grooming maintenance preferences",
                filter_function="filter_grooming_preferences",
                relaxation_allowed=True
            ),
            ConstraintRule(
                name="experience_level_constraint",
                priority=ConstraintPriority.MODERATE,
                description="Ownership experience requirements",
                filter_function="filter_experience_level",
                relaxation_allowed=True
            ),

            # Priority 4: Flexible constraints
            ConstraintRule(
                name="size_preference_constraint",
                priority=ConstraintPriority.FLEXIBLE,
                description="Size preferences",
                filter_function="filter_size_preferences",
                relaxation_allowed=True
            )
        ]

    def apply_constraints(self, dimensions: QueryDimensions,
                         min_candidates: int = 12) -> FilterResult:
        """
        Apply constraint filtering

        Args:
            dimensions: Query dimensions
            min_candidates: Minimum number of candidate breeds

        Returns:
            FilterResult: Filtering results
        """
        try:
            # Start with all breeds
            candidates = set(self.breed_list)
            filtered_breeds = {}
            applied_constraints = []
            relaxed_constraints = []
            warnings = []

            # Apply constraints in priority order
            for priority in [ConstraintPriority.CRITICAL, ConstraintPriority.HIGH,
                           ConstraintPriority.MODERATE, ConstraintPriority.FLEXIBLE]:

                # Get constraint rules for this priority level
                priority_rules = [rule for rule in self.constraint_rules
                                if rule.priority == priority]

                for rule in priority_rules:
                    # Check if this constraint should be applied
                    if self._should_apply_constraint(rule, dimensions):
                        # Apply constraint
                        before_count = len(candidates)
                        filter_func = getattr(self, rule.filter_function)
                        new_filtered = filter_func(candidates, dimensions)

                        # Update candidate list
                        candidates -= set(new_filtered.keys())
                        filtered_breeds.update(new_filtered)
                        applied_constraints.append(rule.name)

                        print(f"Applied {rule.name}: {before_count} -> {len(candidates)} candidates")

                        # Check if constraint relaxation is needed
                        if (len(candidates) < min_candidates and
                            rule.relaxation_allowed and not rule.safety_critical):

                            # Constraint relaxation
                            # candidates.update(new_filtered.keys())
                            relaxed_constraints.append(rule.name)
                            warnings.append(f"Relaxed {rule.description} to maintain diversity")

                            print(f"Relaxed {rule.name}: restored to {len(candidates)} candidates")

                # If too few candidates after critical constraints, warn but don't relax
                if (priority == ConstraintPriority.CRITICAL and
                    len(candidates) < min_candidates):
                    warnings.append(f"Critical constraints resulted in only {len(candidates)} candidates")

            # Final safety net: ensure at least some candidate breeds
            if len(candidates) == 0:
                warnings.append("All breeds filtered out, returning top safe breeds")
                candidates = self._get_emergency_candidates()

            return FilterResult(
                passed_breeds=candidates,
                filtered_breeds=filtered_breeds,
                applied_constraints=applied_constraints,
                relaxed_constraints=relaxed_constraints,
                warnings=warnings
            )

        except Exception as e:
            print(f"Error applying constraints: {str(e)}")
            print(traceback.format_exc())
            return FilterResult(
                passed_breeds=set(self.breed_list[:min_candidates]),
                filtered_breeds={},
                applied_constraints=[],
                warnings=[f"Constraint application failed: {str(e)}"]
            )

    def _should_apply_constraint(self, rule: ConstraintRule,
                               dimensions: QueryDimensions) -> bool:
        """Enhanced constraint application logic"""

        # Always apply size constraints when space is mentioned
        if rule.name == "apartment_size_constraint":
            return any(term in dimensions.spatial_constraints
                      for term in ['apartment', 'small', 'studio', 'condo'])

        # Apply exercise constraints when activity level is specified
        if rule.name == "exercise_constraint":
            return len(dimensions.activity_level) > 0 or \
                   any(term in str(dimensions.spatial_constraints)
                       for term in ['apartment', 'small'])

        # Child safety constraint
        if rule.name == "child_safety_constraint":
            return 'children' in dimensions.family_context

        # Severe allergy constraint
        if rule.name == "severe_allergy_constraint":
            return 'hypoallergenic' in dimensions.special_requirements

        # Low activity constraint
        if rule.name == "low_activity_constraint":
            return 'low' in dimensions.activity_level

        # Quiet requirement constraint
        if rule.name == "quiet_requirement_constraint":
            return 'low' in dimensions.noise_preferences

        # Space compatibility constraint
        if rule.name == "space_compatibility_constraint":
            return ('apartment' in dimensions.spatial_constraints or
                   'house' in dimensions.spatial_constraints)

        # Grooming preference constraint
        if rule.name == "grooming_preference_constraint":
            return len(dimensions.maintenance_level) > 0

        # Experience level constraint
        if rule.name == "experience_level_constraint":
            return 'first_time' in dimensions.special_requirements

        # Size preference constraint
        if rule.name == "size_preference_constraint":
            return len(dimensions.size_preferences) > 0

        return False

    def filter_apartment_size(self, candidates: Set[str],
                            dimensions: QueryDimensions) -> Dict[str, str]:
        """Enhanced apartment size filtering with strict enforcement"""
        filtered = {}

        # Extract living space type with better pattern matching
        living_space = self._extract_living_space(dimensions)
        space_requirements = self._get_space_requirements(living_space)

        for breed in list(candidates):
            breed_info = self.breed_cache.get(breed, {})
            breed_size = self._normalize_breed_size(breed_info.get('size', 'Medium'))
            exercise_needs = self._normalize_exercise_level(breed_info.get('exercise_needs', 'Moderate'))

            # Dynamic space compatibility check
            compatibility_score = self._calculate_space_compatibility(
                breed_size, exercise_needs, space_requirements
            )

            # Apply threshold-based filtering
            if compatibility_score < 0.3:  # Strict threshold for poor matches
                reason = self._generate_filter_reason(breed_size, exercise_needs, living_space)
                filtered[breed] = reason
                continue

        return filtered

    def _extract_living_space(self, dimensions: QueryDimensions) -> str:
        """Extract living space type from dimensions"""
        spatial_text = ' '.join(dimensions.spatial_constraints).lower()

        if any(term in spatial_text for term in ['apartment', 'small apartment', 'studio', 'condo']):
            return 'apartment'
        elif any(term in spatial_text for term in ['small house', 'townhouse']):
            return 'small_house'
        elif any(term in spatial_text for term in ['medium house', 'medium-sized']):
            return 'medium_house'
        elif any(term in spatial_text for term in ['large house', 'big house']):
            return 'large_house'
        else:
            return 'medium_house'  # Default assumption

    def _get_space_requirements(self, living_space: str) -> Dict[str, float]:
        """Get space requirements for different living situations"""
        requirements = {
            'apartment': {'min_space': 1.0, 'yard_bonus': 0.0, 'exercise_penalty': 1.5},
            'small_house': {'min_space': 1.5, 'yard_bonus': 0.2, 'exercise_penalty': 1.2},
            'medium_house': {'min_space': 2.0, 'yard_bonus': 0.3, 'exercise_penalty': 1.0},
            'large_house': {'min_space': 3.0, 'yard_bonus': 0.5, 'exercise_penalty': 0.8}
        }
        return requirements.get(living_space, requirements['medium_house'])

    def _normalize_breed_size(self, size: str) -> str:
        """Normalize breed size to standard categories"""
        size_lower = size.lower()
        if any(term in size_lower for term in ['toy', 'tiny']):
            return 'toy'
        elif 'small' in size_lower:
            return 'small'
        elif 'medium' in size_lower:
            return 'medium'
        elif 'large' in size_lower:
            return 'large'
        elif any(term in size_lower for term in ['giant', 'extra large']):
            return 'giant'
        else:
            return 'medium'  # Default

    def _normalize_exercise_level(self, exercise: str) -> str:
        """Normalize exercise level to standard categories"""
        exercise_lower = exercise.lower()
        if any(term in exercise_lower for term in ['very high', 'extreme', 'intense']):
            return 'very_high'
        elif 'high' in exercise_lower:
            return 'high'
        elif 'moderate' in exercise_lower:
            return 'moderate'
        elif any(term in exercise_lower for term in ['low', 'minimal']):
            return 'low'
        else:
            return 'moderate'  # Default

    def _calculate_space_compatibility(self, breed_size: str, exercise_level: str, space_req: Dict[str, float]) -> float:
        """Calculate dynamic space compatibility score"""
        # Size-space compatibility matrix (dynamic, not hardcoded)
        size_factors = {
            'toy': 0.5, 'small': 1.0, 'medium': 1.5, 'large': 2.5, 'giant': 4.0
        }

        exercise_factors = {
            'low': 1.0, 'moderate': 1.3, 'high': 1.8, 'very_high': 2.5
        }

        breed_space_need = size_factors[breed_size] * exercise_factors[exercise_level]
        available_space = space_req['min_space']

        # Calculate compatibility ratio
        compatibility = available_space / breed_space_need

        # Apply exercise penalty for high-energy breeds in small spaces
        if exercise_level in ['high', 'very_high'] and available_space < 2.0:
            compatibility *= (1.0 - space_req['exercise_penalty'] * 0.3)

        return max(0.0, min(1.0, compatibility))

    def _generate_filter_reason(self, breed_size: str, exercise_level: str, living_space: str) -> str:
        """Generate dynamic filtering reason"""
        if breed_size in ['giant', 'large'] and living_space == 'apartment':
            return f"{breed_size.title()} breed not suitable for apartment living"
        elif exercise_level in ['high', 'very_high'] and living_space in ['apartment', 'small_house']:
            return f"High-energy breed needs more space than {living_space.replace('_', ' ')}"
        else:
            return f"Space and exercise requirements exceed {living_space.replace('_', ' ')} capacity"

    def filter_child_safety(self, candidates: Set[str],
                          dimensions: QueryDimensions) -> Dict[str, str]:
        """Child safety filtering"""
        filtered = {}

        for breed in list(candidates):
            breed_info = self.breed_cache.get(breed, {})
            good_with_children = breed_info.get('good_with_children', 'Yes')
            size = breed_info.get('size', '')
            temperament = breed_info.get('temperament', '')

            # Breeds explicitly not suitable for children
            if good_with_children == 'No':
                filtered[breed] = "Not suitable for children"
            # Large breeds without clear child compatibility indicators should be cautious
            elif ('large' in size and good_with_children != 'Yes' and
                  any(trait in temperament for trait in ['aggressive', 'dominant', 'protective'])):
                filtered[breed] = "Large breed with uncertain child compatibility"

        return filtered

    def filter_severe_allergies(self, candidates: Set[str],
                              dimensions: QueryDimensions) -> Dict[str, str]:
        """Severe allergy filtering"""
        filtered = {}

        # High shedding breed list (should be adjusted based on actual database)
        high_shedding_breeds = {
            'German_Shepherd', 'Golden_Retriever', 'Labrador_Retriever',
            'Husky', 'Akita', 'Bernese_Mountain_Dog'
        }

        for breed in list(candidates):
            if breed in high_shedding_breeds:
                filtered[breed] = "High shedding breed not suitable for allergies"

        return filtered

    def filter_low_activity(self, candidates: Set[str],
                          dimensions: QueryDimensions) -> Dict[str, str]:
        """Low activity level filtering"""
        filtered = {}

        for breed in list(candidates):
            breed_info = self.breed_cache.get(breed, {})
            exercise_needs = breed_info.get('exercise_needs', '')
            temperament = breed_info.get('temperament', '')

            # High exercise requirement breeds
            if 'high' in exercise_needs or 'very high' in exercise_needs:
                filtered[breed] = "High exercise requirements unsuitable for low activity lifestyle"
            # Working dogs, sporting dogs, herding dogs typically need substantial exercise
            elif any(trait in temperament for trait in ['working', 'sporting', 'herding', 'energetic']):
                filtered[breed] = "High-energy breed requiring substantial daily exercise"

        return filtered

    def filter_quiet_requirements(self, candidates: Set[str],
                                dimensions: QueryDimensions) -> Dict[str, str]:
        """Quiet requirement filtering"""
        filtered = {}

        for breed in list(candidates):
            breed_info = self.breed_cache.get(breed, {})
            noise_level = breed_info.get('noise_level', 'moderate').lower()
            temperament = breed_info.get('temperament', '')

            # High noise level breeds
            if 'high' in noise_level or 'loud' in noise_level:
                filtered[breed] = "High noise level unsuitable for quiet requirements"
            # Terriers and hounds are typically more vocal
            elif ('terrier' in breed.lower() or 'hound' in breed.lower() or
                  'vocal' in temperament):
                filtered[breed] = "Breed group typically more vocal than desired"

        return filtered

    def filter_space_compatibility(self, candidates: Set[str],
                                 dimensions: QueryDimensions) -> Dict[str, str]:
        """Space compatibility filtering"""
        filtered = {}

        # This function provides more refined space matching
        for breed in list(candidates):
            breed_info = self.breed_cache.get(breed, {})
            size = breed_info.get('size', '')
            exercise_needs = breed_info.get('exercise_needs', '')

            # If house is specified but breed is too small, may not be optimal choice (soft constraint)
            if ('house' in dimensions.spatial_constraints and
                'tiny' in size and 'guard' in dimensions.special_requirements):
                filtered[breed] = "Very small breed may not meet guard dog requirements for house"

        return filtered

    def filter_grooming_preferences(self, candidates: Set[str],
                                  dimensions: QueryDimensions) -> Dict[str, str]:
        """Grooming preference filtering"""
        filtered = {}

        for breed in list(candidates):
            breed_info = self.breed_cache.get(breed, {})
            grooming_needs = breed_info.get('grooming_needs', '')

            # Low maintenance needed but breed requires high maintenance
            if ('low' in dimensions.maintenance_level and
                'high' in grooming_needs):
                filtered[breed] = "High grooming requirements exceed maintenance preferences"
            # High maintenance preference but breed is too simple (rarely applicable)
            elif ('high' in dimensions.maintenance_level and
                  'low' in grooming_needs):
                # Usually don't filter out, as low maintenance is always good
                pass

        return filtered

    def filter_experience_level(self, candidates: Set[str],
                              dimensions: QueryDimensions) -> Dict[str, str]:
        """Experience level filtering"""
        filtered = {}

        for breed in list(candidates):
            breed_info = self.breed_cache.get(breed, {})
            care_level = breed_info.get('care_level', '')
            temperament = breed_info.get('temperament', '')

            # Beginners not suitable for high maintenance or difficult breeds
            if 'first_time' in dimensions.special_requirements:
                if ('high' in care_level or 'expert' in care_level or
                    any(trait in temperament for trait in
                        ['stubborn', 'independent', 'dominant', 'challenging'])):
                    filtered[breed] = "High care requirements unsuitable for first-time owners"

        return filtered

    def filter_size_preferences(self, candidates: Set[str],
                              dimensions: QueryDimensions) -> Dict[str, str]:
        """Size preference filtering"""
        filtered = {}

        # This is a soft constraint, usually won't completely exclude
        size_preferences = dimensions.size_preferences

        if not size_preferences:
            return filtered

        for breed in list(candidates):
            breed_info = self.breed_cache.get(breed, {})
            breed_size = breed_info.get('size', '')

            # Check if matches preferences
            size_match = False
            for preferred_size in size_preferences:
                if preferred_size in breed_size:
                    size_match = True
                    break

            # Since this is a flexible constraint, usually won't filter out, only reflected in scores
            # But if user is very explicit (e.g., only wants small dogs), can filter
            if not size_match and len(size_preferences) == 1:
                # Only filter when user has very explicit preference for single size
                preferred = size_preferences[0]
                if ((preferred == 'small' and 'large' in breed_size) or
                    (preferred == 'large' and 'small' in breed_size)):
                    filtered[breed] = f"Size mismatch: prefer {preferred} but breed is {breed_size}"

        return filtered

    def filter_exercise_mismatch(self, candidates: Set[str],
                                dimensions: QueryDimensions) -> Dict[str, str]:
        """Filter breeds with severe exercise mismatches using dynamic thresholds"""
        filtered = {}

        # Extract user exercise profile dynamically
        user_profile = self._extract_exercise_profile(dimensions)
        compatibility_threshold = self._get_exercise_threshold(user_profile)

        for breed in candidates:
            breed_info = self.breed_cache.get(breed, {})
            breed_exercise_level = self._normalize_exercise_level(breed_info.get('exercise_needs', 'Moderate'))

            # Calculate exercise compatibility score
            compatibility = self._calculate_exercise_compatibility(
                user_profile, breed_exercise_level
            )

            # Apply threshold-based filtering
            if compatibility < compatibility_threshold:
                reason = self._generate_exercise_filter_reason(user_profile, breed_exercise_level)
                filtered[breed] = reason

        return filtered

    def _extract_exercise_profile(self, dimensions: QueryDimensions) -> Dict[str, str]:
        """Extract comprehensive user exercise profile"""
        activity_text = ' '.join(dimensions.activity_level).lower()
        spatial_text = ' '.join(dimensions.spatial_constraints).lower()

        # Determine exercise level
        if any(term in activity_text for term in ['don\'t exercise', 'minimal', 'low', 'light walks']):
            level = 'low'
        elif any(term in activity_text for term in ['hiking', 'running', 'active', 'athletic']):
            level = 'high'
        elif any(term in activity_text for term in ['30 minutes', 'moderate', 'balanced']):
            level = 'moderate'
        else:
            # Infer from living space
            if 'apartment' in spatial_text:
                level = 'low_moderate'
            else:
                level = 'moderate'

        # Determine time commitment
        if any(term in activity_text for term in ['30 minutes', 'half hour']):
            time = 'limited'
        elif any(term in activity_text for term in ['hiking', 'outdoor activities']):
            time = 'extensive'
        else:
            time = 'moderate'

        return {'level': level, 'time': time}

    def _get_exercise_threshold(self, user_profile: Dict[str, str]) -> float:
        """Get dynamic threshold based on user profile"""
        base_threshold = 0.4

        # Adjust threshold based on user constraints
        if user_profile['level'] == 'low':
            base_threshold = 0.6  # Stricter for low-activity users
        elif user_profile['level'] == 'high':
            base_threshold = 0.3  # More lenient for active users

        return base_threshold

    def _calculate_exercise_compatibility(self, user_profile: Dict[str, str], breed_level: str) -> float:
        """Calculate dynamic exercise compatibility"""
        # Exercise level compatibility matrix
        compatibility_matrix = {
            'low': {'low': 1.0, 'moderate': 0.7, 'high': 0.3, 'very_high': 0.1},
            'low_moderate': {'low': 0.9, 'moderate': 1.0, 'high': 0.5, 'very_high': 0.2},
            'moderate': {'low': 0.8, 'moderate': 1.0, 'high': 0.8, 'very_high': 0.4},
            'high': {'low': 0.5, 'moderate': 0.8, 'high': 1.0, 'very_high': 0.9}
        }

        user_level = user_profile['level']
        base_compatibility = compatibility_matrix.get(user_level, {}).get(breed_level, 0.5)

        # Adjust for time commitment
        if user_profile['time'] == 'limited' and breed_level in ['high', 'very_high']:
            base_compatibility *= 0.7
        elif user_profile['time'] == 'extensive' and breed_level == 'low':
            base_compatibility *= 0.8

        return base_compatibility

    def _generate_exercise_filter_reason(self, user_profile: Dict[str, str], breed_level: str) -> str:
        """Generate dynamic exercise filtering reason"""
        user_level = user_profile['level']

        if user_level == 'low' and breed_level in ['high', 'very_high']:
            return f"High-energy breed unsuitable for low-activity lifestyle"
        elif user_level == 'high' and breed_level == 'low':
            return f"Low-energy breed may not match active lifestyle requirements"
        else:
            return f"Exercise requirements mismatch: {user_level} user with {breed_level} breed"

    def filter_size_bias(self, candidates: Set[str], dimensions: QueryDimensions) -> Dict[str, str]:
        """Filter to correct size bias for moderate lifestyle users"""
        filtered = {}

        # Detect moderate lifestyle indicators
        activity_text = ' '.join(dimensions.activity_level).lower()
        is_moderate_lifestyle = any(term in activity_text for term in
                                   ['moderate', 'balanced', '30 minutes', 'medium-sized house'])

        if not is_moderate_lifestyle:
            return filtered  # No filtering needed

        # Count size distribution in candidates
        size_counts = {'toy': 0, 'small': 0, 'medium': 0, 'large': 0, 'giant': 0}
        total_candidates = len(candidates)

        for breed in candidates:
            breed_info = self.breed_cache.get(breed, {})
            breed_size = self._normalize_breed_size(breed_info.get('size', 'Medium'))
            size_counts[breed_size] += 1

        # Check for size bias (too many large/giant breeds)
        large_giant_ratio = (size_counts['large'] + size_counts['giant']) / max(total_candidates, 1)

        if large_giant_ratio > 0.6:  # More than 60% large/giant breeds
            # Filter some large/giant breeds to balance distribution
            large_giant_filtered = 0
            target_reduction = int((large_giant_ratio - 0.4) * total_candidates)

            for breed in list(candidates):
                if large_giant_filtered >= target_reduction:
                    break

                breed_info = self.breed_cache.get(breed, {})
                breed_size = self._normalize_breed_size(breed_info.get('size', 'Medium'))

                if breed_size in ['large', 'giant']:
                    # Check if breed has additional compatibility issues
                    exercise_level = self._normalize_exercise_level(
                        breed_info.get('exercise_needs', 'Moderate')
                    )

                    if breed_size == 'giant' or exercise_level == 'very_high':
                        filtered[breed] = f"Size bias correction: {breed_size} breed less suitable for moderate lifestyle"
                        large_giant_filtered += 1

        return filtered

    def _get_emergency_candidates(self) -> Set[str]:
        """Get emergency candidate breeds (safest choices)"""
        safe_breeds = {
            'Labrador_Retriever', 'Golden_Retriever', 'Cavalier_King_Charles_Spaniel',
            'Bichon_Frise', 'French_Bulldog', 'Boston_Terrier', 'Pug'
        }

        # Only return breeds that exist in the database
        available_safe_breeds = safe_breeds.intersection(set(self.breed_list))

        if not available_safe_breeds:
            # If even safe breeds are not available, return first few breeds
            return set(self.breed_list[:5])

        return available_safe_breeds

    def get_constraint_summary(self, filter_result: FilterResult) -> Dict[str, Any]:
        """Get constraint application summary"""
        return {
            'total_breeds': len(self.breed_list),
            'passed_breeds': len(filter_result.passed_breeds),
            'filtered_breeds': len(filter_result.filtered_breeds),
            'applied_constraints': filter_result.applied_constraints,
            'relaxed_constraints': filter_result.relaxed_constraints,
            'warnings': filter_result.warnings,
            'pass_rate': len(filter_result.passed_breeds) / len(self.breed_list),
            'filter_breakdown': self._get_filter_breakdown(filter_result)
        }

    def _get_filter_breakdown(self, filter_result: FilterResult) -> Dict[str, int]:
        """Get filtering reason breakdown"""
        breakdown = {}

        for breed, reason in filter_result.filtered_breeds.items():
            # Simplify reason categorization
            if 'apartment' in reason.lower() or 'large' in reason.lower():
                category = 'Size/Space Issues'
            elif 'child' in reason.lower():
                category = 'Child Safety'
            elif 'allerg' in reason.lower() or 'shed' in reason.lower():
                category = 'Allergy Concerns'
            elif 'exercise' in reason.lower() or 'activity' in reason.lower():
                category = 'Exercise/Activity Mismatch'
            elif 'noise' in reason.lower() or 'bark' in reason.lower():
                category = 'Noise Issues'
            elif 'groom' in reason.lower() or 'maintenance' in reason.lower():
                category = 'Maintenance Requirements'
            elif 'experience' in reason.lower() or 'first-time' in reason.lower():
                category = 'Experience Level'
            else:
                category = 'Other'

            breakdown[category] = breakdown.get(category, 0) + 1

        return breakdown

def apply_breed_constraints(dimensions: QueryDimensions,
                          min_candidates: int = 12) -> FilterResult:
    """
    Convenience function: Apply breed constraint filtering

    Args:
        dimensions: Query dimensions
        min_candidates: Minimum number of candidate breeds

    Returns:
        FilterResult: Filtering results
    """
    manager = ConstraintManager()
    return manager.apply_constraints(dimensions, min_candidates)

def get_filtered_breeds(dimensions: QueryDimensions) -> Tuple[List[str], Dict[str, Any]]:
    """
    Convenience function: Get filtered breed list and summary

    Args:
        dimensions: Query dimensions

    Returns:
        Tuple: (Filtered breed list, filtering summary)
    """
    manager = ConstraintManager()
    result = manager.apply_constraints(dimensions)
    summary = manager.get_constraint_summary(result)

    return list(result.passed_breeds), summary
