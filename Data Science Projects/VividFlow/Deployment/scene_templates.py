import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SceneTemplate:
    """Data class representing a scene template"""
    key: str
    name: str
    prompt: str
    negative_extra: str
    category: str
    icon: str
    guidance_scale: float = 7.5


class SceneTemplateManager:
    """
    Manages curated scene templates for background generation.
    Provides categorized presets that users can select with one click.
    """

    # Scene template definitions
    TEMPLATES: Dict[str, SceneTemplate] = {
        # Professional Category
        "office_modern": SceneTemplate(
            key="office_modern",
            name="Modern Office",
            prompt="modern minimalist office interior, clean white desk, large floor-to-ceiling windows, natural daylight, professional corporate environment, soft shadows, contemporary furniture",
            negative_extra="messy, cluttered, dark, old",
            category="Professional",
            icon="🏢",
            guidance_scale=7.5
        ),
        "office_executive": SceneTemplate(
            key="office_executive",
            name="Executive Suite",
            prompt="luxurious executive office, mahogany desk, leather chair, city skyline view through windows, warm ambient lighting, bookshelf, elegant professional setting",
            negative_extra="cheap, cramped, messy",
            category="Professional",
            icon="👔",
            guidance_scale=7.5
        ),
        "studio_white": SceneTemplate(
            key="studio_white",
            name="White Studio",
            prompt="clean white photography studio background, professional lighting setup, seamless white backdrop, soft diffused light, minimal shadows",
            negative_extra="colored, textured, dirty",
            category="Professional",
            icon="📷",
            guidance_scale=8.0
        ),
        "coworking": SceneTemplate(
            key="coworking",
            name="Coworking Space",
            prompt="modern coworking space, open plan office, plants, exposed brick, industrial chic design, natural light, collaborative environment",
            negative_extra="empty, dark, boring",
            category="Professional",
            icon="💼",
            guidance_scale=7.0
        ),
        "conference": SceneTemplate(
            key="conference",
            name="Conference Room",
            prompt="modern conference room, large meeting table, glass walls, professional presentation screen, bright corporate lighting, clean minimal design",
            negative_extra="small, cramped, outdated",
            category="Professional",
            icon="🤝",
            guidance_scale=7.5
        ),

        # Nature Category
        "beach_sunset": SceneTemplate(
            key="beach_sunset",
            name="Sunset Beach",
            prompt="beautiful tropical beach at golden hour sunset, palm trees silhouette, calm turquoise ocean waves, warm orange and pink sky, soft sand, paradise vacation vibes",
            negative_extra="storm, rain, crowded, trash",
            category="Nature",
            icon="🏖️",
            guidance_scale=7.0
        ),
        "forest_enchanted": SceneTemplate(
            key="forest_enchanted",
            name="Enchanted Forest",
            prompt="magical enchanted forest, sunlight streaming through tall trees, lush green foliage, mystical atmosphere, morning mist, fairy tale woodland",
            negative_extra="dead trees, dark, scary, barren",
            category="Nature",
            icon="🌲",
            guidance_scale=7.0
        ),
        "mountain_scenic": SceneTemplate(
            key="mountain_scenic",
            name="Mountain Vista",
            prompt="breathtaking mountain landscape, snow-capped peaks, alpine meadow, clear blue sky, majestic scenic view, pristine nature, peaceful atmosphere",
            negative_extra="industrial, polluted, crowded",
            category="Nature",
            icon="🏔️",
            guidance_scale=7.5
        ),
        "garden_spring": SceneTemplate(
            key="garden_spring",
            name="Spring Garden",
            prompt="beautiful spring flower garden, colorful blooming flowers, roses and tulips, manicured hedges, sunny day, botanical paradise, fresh and vibrant",
            negative_extra="dead, winter, wilted, dry",
            category="Nature",
            icon="🌸",
            guidance_scale=7.0
        ),
        "lake_serene": SceneTemplate(
            key="lake_serene",
            name="Serene Lake",
            prompt="peaceful serene lake at dawn, mirror-like water reflection, surrounding mountains, soft morning light, tranquil atmosphere, pristine natural beauty",
            negative_extra="stormy, polluted, industrial",
            category="Nature",
            icon="🏞️",
            guidance_scale=7.0
        ),
        "cherry_blossom": SceneTemplate(
            key="cherry_blossom",
            name="Cherry Blossom",
            prompt="stunning cherry blossom trees in full bloom, pink sakura petals falling gently, Japanese garden aesthetic, soft spring sunlight, romantic atmosphere",
            negative_extra="winter, dead, brown, wilted",
            category="Nature",
            icon="🌸",
            guidance_scale=7.0
        ),

        # Urban Category
        "city_skyline": SceneTemplate(
            key="city_skyline",
            name="City Skyline",
            prompt="modern city skyline at blue hour, impressive skyscrapers, glass buildings reflecting sunset, urban metropolitan view, cinematic atmosphere",
            negative_extra="slums, dirty, abandoned, ruins",
            category="Urban",
            icon="🌆",
            guidance_scale=7.5
        ),
        "cafe_cozy": SceneTemplate(
            key="cafe_cozy",
            name="Cozy Cafe",
            prompt="warm cozy coffee shop interior, wooden furniture, ambient lighting, exposed brick walls, plants, comfortable atmosphere, artisan cafe vibes",
            negative_extra="fast food, plastic, harsh lighting",
            category="Urban",
            icon="☕",
            guidance_scale=7.0
        ),
        "street_european": SceneTemplate(
            key="street_european",
            name="European Street",
            prompt="charming European cobblestone street, historic buildings, outdoor cafe, flowers on balconies, warm afternoon light, romantic Paris or Rome vibes",
            negative_extra="modern, industrial, ugly, dirty",
            category="Urban",
            icon="🏛️",
            guidance_scale=7.0
        ),
        "night_neon": SceneTemplate(
            key="night_neon",
            name="Neon Nightlife",
            prompt="vibrant city nightlife scene, neon lights and signs, urban night atmosphere, colorful reflections on wet street, cyberpunk aesthetic, electric energy",
            negative_extra="daytime, boring, plain",
            category="Urban",
            icon="🌃",
            guidance_scale=6.5
        ),
        "rooftop_view": SceneTemplate(
            key="rooftop_view",
            name="Rooftop Terrace",
            prompt="luxury rooftop terrace, city panoramic view, modern outdoor furniture, string lights, sunset golden hour, sophisticated urban oasis",
            negative_extra="cheap, dirty, crowded",
            category="Urban",
            icon="🏙️",
            guidance_scale=7.5
        ),

        # Artistic Category
        "gradient_soft": SceneTemplate(
            key="gradient_soft",
            name="Soft Gradient",
            prompt="smooth soft gradient background, pastel colors blending beautifully, pink to blue to purple transition, dreamy aesthetic, professional portrait backdrop",
            negative_extra="harsh, noisy, textured, busy",
            category="Artistic",
            icon="🎨",
            guidance_scale=8.0
        ),
        "abstract_modern": SceneTemplate(
            key="abstract_modern",
            name="Modern Abstract",
            prompt="modern abstract art background, geometric shapes, bold colors, contemporary design, artistic composition, museum gallery aesthetic",
            negative_extra="realistic, plain, boring",
            category="Artistic",
            icon="🖼️",
            guidance_scale=6.5
        ),
        "vintage_retro": SceneTemplate(
            key="vintage_retro",
            name="Vintage Retro",
            prompt="vintage retro aesthetic background, warm sepia tones, nostalgic 70s vibes, film grain texture, classic photography style, timeless elegance",
            negative_extra="modern, digital, cold, harsh",
            category="Artistic",
            icon="📻",
            guidance_scale=7.0
        ),
        "watercolor_dream": SceneTemplate(
            key="watercolor_dream",
            name="Watercolor Dream",
            prompt="beautiful watercolor painting background, soft flowing colors, artistic brush strokes, dreamy ethereal atmosphere, delicate artistic aesthetic",
            negative_extra="digital, sharp, photorealistic",
            category="Artistic",
            icon="🖌️",
            guidance_scale=6.5
        ),

        # Seasonal Category
        "autumn_foliage": SceneTemplate(
            key="autumn_foliage",
            name="Autumn Foliage",
            prompt="beautiful autumn scenery, vibrant fall foliage, orange red and golden leaves, maple trees, warm sunlight filtering through, cozy seasonal atmosphere",
            negative_extra="spring, summer, green, snow",
            category="Seasonal",
            icon="🍂",
            guidance_scale=7.0
        ),
        "winter_snow": SceneTemplate(
            key="winter_snow",
            name="Winter Wonderland",
            prompt="magical winter wonderland, fresh white snow covering everything, snow-laden pine trees, soft snowfall, peaceful cold atmosphere, holiday season vibes",
            negative_extra="summer, green, rain, mud",
            category="Seasonal",
            icon="❄️",
            guidance_scale=7.0
        ),
        "summer_tropical": SceneTemplate(
            key="summer_tropical",
            name="Tropical Summer",
            prompt="vibrant tropical summer scene, lush palm trees, bright sunny day, exotic flowers, paradise vacation destination, warm and inviting atmosphere",
            negative_extra="winter, cold, snow, gray",
            category="Seasonal",
            icon="🌴",
            guidance_scale=7.0
        ),
        "spring_meadow": SceneTemplate(
            key="spring_meadow",
            name="Spring Meadow",
            prompt="beautiful spring meadow, wildflowers blooming, fresh green grass, butterflies, soft warm sunlight, renewal and new beginnings, pastoral beauty",
            negative_extra="winter, autumn, dead, dry",
            category="Seasonal",
            icon="🌷",
            guidance_scale=7.0
        ),
    }

    # Category display order
    CATEGORIES = ["Professional", "Nature", "Urban", "Artistic", "Seasonal"]

    def __init__(self):
        """Initialize the scene template manager"""
        logger.info(f"SceneTemplateManager initialized with {len(self.TEMPLATES)} templates")

    def get_all_templates(self) -> Dict[str, SceneTemplate]:
        """Get all available templates"""
        return self.TEMPLATES

    def get_template(self, key: str) -> Optional[SceneTemplate]:
        """Get a specific template by key"""
        return self.TEMPLATES.get(key)

    def get_templates_by_category(self, category: str) -> List[SceneTemplate]:
        """Get all templates in a specific category"""
        return [t for t in self.TEMPLATES.values() if t.category == category]

    def get_categories(self) -> List[str]:
        """Get list of all categories in display order"""
        return self.CATEGORIES

    def get_template_choices_sorted(self) -> List[str]:
        """
        Get template choices formatted for Gradio dropdown.
        Returns list of display strings sorted A-Z: "🏢 Modern Office"
        """
        display_list = []
        for key, template in self.TEMPLATES.items():
            display_name = f"{template.icon} {template.name}"
            display_list.append(display_name)

        # Sort alphabetically by name (ignoring emoji)
        display_list.sort(key=lambda x: x.split(' ', 1)[1] if ' ' in x else x)
        return display_list

    def get_template_key_from_display(self, display_name: str) -> Optional[str]:
        """
        Get template key from display name.
        Example: "🏢 Modern Office" -> "office_modern"
        """
        if not display_name:
            return None

        for key, template in self.TEMPLATES.items():
            if f"{template.icon} {template.name}" == display_name:
                return key
        return None

    def get_prompt_for_template(self, key: str) -> Optional[str]:
        """Get the prompt string for a template"""
        template = self.get_template(key)
        return template.prompt if template else None

    def get_negative_prompt_for_template(
        self,
        key: str,
        base_negative: str = "blurry, low quality, distorted, people, characters"
    ) -> str:
        """Get combined negative prompt for a template"""
        template = self.get_template(key)
        if template and template.negative_extra:
            return f"{base_negative}, {template.negative_extra}"
        return base_negative

    def get_guidance_scale_for_template(self, key: str) -> float:
        """Get the recommended guidance scale for a template"""
        template = self.get_template(key)
        return template.guidance_scale if template else 7.5

    def build_gallery_html(self) -> str:
        """
        Build HTML for the scene template gallery.
        Returns HTML string for display in Gradio.
        """
        html_parts = ['<div class="scene-gallery">']

        for category in self.CATEGORIES:
            templates = self.get_templates_by_category(category)
            if not templates:
                continue

            html_parts.append(f'''
            <div class="scene-category">
                <h4 class="scene-category-title">{category}</h4>
                <div class="scene-grid">
            ''')

            for template in templates:
                html_parts.append(f'''
                <button class="scene-card" data-template="{template.key}" onclick="selectTemplate('{template.key}')">
                    <span class="scene-icon">{template.icon}</span>
                    <span class="scene-name">{template.name}</span>
                </button>
                ''')

            html_parts.append('</div></div>')

        html_parts.append('</div>')
        return ''.join(html_parts)

    def get_gallery_css(self) -> str:
        """Get CSS styles for the scene gallery"""
        return """
        /* Scene Gallery Styles */
        .scene-gallery {
            margin: 16px 0;
        }

        .scene-category {
            margin-bottom: 20px;
        }

        .scene-category-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: #475569;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e2e8f0;
        }

        .scene-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 8px;
        }

        .scene-card {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 12px 8px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            min-height: 70px;
        }

        .scene-card:hover {
            background: #dbeafe;
            border-color: #3b82f6;
            transform: translateY(-2px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .scene-card.selected {
            background: #dbeafe;
            border-color: #3b82f6;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
        }

        .scene-icon {
            font-size: 1.5rem;
            margin-bottom: 4px;
        }

        .scene-name {
            font-size: 0.75rem;
            font-weight: 500;
            color: #1e293b;
            text-align: center;
            line-height: 1.2;
        }

        @media (max-width: 768px) {
            .scene-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        """
