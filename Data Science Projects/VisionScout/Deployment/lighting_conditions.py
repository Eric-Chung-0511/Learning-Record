
LIGHTING_CONDITIONS = {
    "time_descriptions": {
        "day_clear": {
        "general": "The scene is captured during clear daylight hours with bright natural lighting.",
        "bright": "The scene is brightly lit with strong, clear daylight.",
        "medium": "The scene is illuminated with moderate daylight under clear conditions.",
        "dim": "The scene is captured in soft daylight on a clear day."
        },
        "day_cloudy": {
        "general": "The scene is captured during daytime under overcast conditions.",
        "bright": "The scene has the diffused bright lighting of an overcast day.",
        "medium": "The scene has even, soft lighting typical of a cloudy day.",
        "dim": "The scene has the muted lighting of a heavily overcast day."
        },
         "day_cloudy_gray": {
        "general": "The scene is captured during an overcast day with muted gray lighting.",
        "bright": "The scene has bright but diffused gray daylight from heavy cloud cover.",
        "medium": "The scene has even, muted lighting typical of a gray, overcast day.",
        "dim": "The scene has subdued lighting under thick gray clouds."
        },
        "indoor_residential_natural": {
            "general": "The scene is captured in a residential setting with natural window lighting.",
            "bright": "The residential space is brightly lit with abundant natural light from windows.",
            "medium": "The home interior has comfortable natural lighting complemented by artificial sources.",
            "dim": "The residential space has soft natural lighting creating a cozy atmosphere."
        },
        "indoor_designer_residential": {
            "general": "The scene is captured in a well-designed residential space with curated lighting.",
            "bright": "The residential interior features bright, designer lighting creating an elegant atmosphere.",
            "medium": "The home space has thoughtfully planned lighting balancing aesthetics and functionality.",
            "dim": "The residential area has sophisticated mood lighting enhancing the design elements."
        },
        "indoor_bright_natural_mix": {
            "general": "The scene is captured indoors with a blend of natural and artificial lighting.",
            "bright": "The indoor space combines bright natural window light with artificial illumination.",
            "medium": "The interior has balanced mixed lighting from windows and electric sources.",
            "dim": "The indoor area has gentle mixed lighting creating comfortable illumination."
        },
        "indoor_restaurant_bar": {
            "general": "The scene is captured inside a restaurant or bar with characteristic warm lighting.",
            "bright": "The dining establishment is well-lit with warm illumination emphasizing ambiance.",
            "medium": "The restaurant/bar has moderate warm lighting creating a comfortable social atmosphere.",
            "dim": "The establishment features soft, warm lighting creating an intimate dining or social atmosphere."
        },
        "sunset/sunrise": {
        "general": "The scene is captured during golden hour with warm lighting.",
        "bright": "The scene is illuminated with bright golden hour light with long shadows.",
        "medium": "The scene has the warm orange-yellow glow typical of sunset or sunrise.",
        "dim": "The scene has soft, warm lighting characteristic of early sunrise or late sunset."
        },
        "night": {
        "general": "The scene is captured at night with limited natural lighting.",
        "bright": "The scene is captured at night but well-lit with artificial lighting.",
        "medium": "The scene is captured at night with moderate artificial lighting.",
        "dim": "The scene is captured in low-light night conditions with minimal illumination."
        },
        "indoor_bright": {
        "general": "The scene is captured indoors with ample lighting.",
        "bright": "The indoor space is brightly lit, possibly with natural light from windows.",
        "medium": "The indoor space has good lighting conditions.",
        "dim": "The indoor space has adequate lighting."
        },
        "indoor_moderate": {
        "general": "The scene is captured indoors with moderate lighting.",
        "bright": "The indoor space has comfortable, moderate lighting.",
        "medium": "The indoor space has standard interior lighting.",
        "dim": "The indoor space has somewhat subdued lighting."
        },
        "indoor_dim": {
        "general": "The scene is captured indoors with dim or mood lighting.",
        "bright": "The indoor space has dim but sufficient lighting.",
        "medium": "The indoor space has low, atmospheric lighting.",
        "dim": "The indoor space has very dim, possibly mood-oriented lighting."
        },
        "beach_daylight": {
            "general": "The scene is captured during daytime at a beach with bright natural sunlight.",
            "bright": "The beach scene is intensely illuminated by direct sunlight.",
            "medium": "The coastal area has even natural daylight.",
            "dim": "The beach has softer lighting, possibly from a partially cloudy sky."
        },
        "sports_arena": {
            "general": "The scene is captured in a sports venue with specialized arena lighting.",
            "bright": "The sports facility is brightly illuminated with powerful overhead lights.",
            "medium": "The venue has standard sports event lighting providing clear visibility.",
            "dim": "The sports area has reduced illumination, possibly before or after an event."
        },
        "kitchen_working": {
            "general": "The scene is captured in a professional kitchen with task-oriented lighting.",
            "bright": "The kitchen is intensely illuminated with clear, functional lighting.",
            "medium": "The culinary space has standard working lights focused on preparation areas.",
            "dim": "The kitchen has reduced lighting, possibly during off-peak hours."
        },
        "unknown": {
        "general": "The lighting conditions in this scene are not easily determined."
        }
    },
    "template_modifiers": {
        "day_clear": "brightly-lit",
        "day_cloudy": "softly-lit",
        "sunset/sunrise": "warmly-lit",
        "night": "night-time",
        "indoor_bright": "well-lit indoor",
        "indoor_moderate": "indoor",
        "indoor_dim": "dimly-lit indoor",
        "indoor_commercial": "retail-lit",
        "indoor_restaurant": "atmospherically-lit",
        "neon_night": "neon-illuminated",
        "stadium_lighting": "flood-lit",
        "mixed_lighting": "transitionally-lit",
        "beach_lighting": "sun-drenched",
        "sports_venue_lighting": "arena-lit",
        "professional_kitchen_lighting": "kitchen-task lit",
        "day_cloudy_gray": "gray-lit",
        "indoor_residential_natural": "naturally-lit residential",
        "indoor_designer_residential": "designer-lit residential",
        "indoor_bright_natural_mix": "mixed-lit indoor",
        "unknown": ""
    },
    "activity_modifiers": {
        "day_clear": ["active", "lively", "busy"],
        "day_cloudy": ["calm", "relaxed", "casual"],
        "sunset/sunrise": ["peaceful", "transitional", "atmospheric"],
        "night": ["quiet", "subdued", "nocturnal"],
        "indoor_bright": ["focused", "productive", "engaged"],
        "indoor_moderate": ["comfortable", "social", "casual"],
        "indoor_dim": ["intimate", "relaxed", "private"],
        "indoor_commercial": ["shopping", "browsing", "consumer-oriented"],
        "indoor_restaurant": ["dining", "social", "culinary"],
        "neon_night": ["vibrant", "energetic", "night-life"],
        "stadium_lighting": ["event-focused", "spectator-oriented", "performance-based"],
        "mixed_lighting": ["transitional", "adaptable", "variable"],
        "unknown": []
    },
    "indoor_commercial": {
    "general": "The scene is captured inside a commercial setting with retail-optimized lighting.",
    "bright": "The space is brightly illuminated with commercial display lighting to highlight merchandise.",
    "medium": "The commercial interior has standard retail lighting that balances visibility and ambiance.",
    "dim": "The commercial space has subdued lighting creating an upscale or intimate shopping atmosphere."
    },
    "indoor_restaurant": {
        "general": "The scene is captured inside a restaurant with characteristic dining lighting.",
        "bright": "The restaurant is well-lit with clear illumination emphasizing food presentation.",
        "medium": "The dining space has moderate lighting striking a balance between functionality and ambiance.",
        "dim": "The restaurant features soft, low lighting creating an intimate dining atmosphere."
    },
    "neon_night": {
        "general": "The scene is captured at night with colorful neon lighting typical of entertainment districts.",
        "bright": "The night scene is illuminated by vibrant neon signs creating a lively, colorful atmosphere.",
        "medium": "The evening setting features moderate neon lighting creating a characteristic urban nightlife scene.",
        "dim": "The night area has subtle neon accents against the darkness, creating a moody urban atmosphere."
    },
    "stadium_lighting": {
        "general": "The scene is captured under powerful stadium lights designed for spectator events.",
        "bright": "The venue is intensely illuminated by stadium floodlights creating daylight-like conditions.",
        "medium": "The sports facility has standard event lighting providing clear visibility across the venue.",
        "dim": "The stadium has reduced illumination typical of pre-event or post-event conditions."
    },
    "mixed_lighting": {
        "general": "The scene features a mix of indoor and outdoor lighting creating transitional illumination.",
        "bright": "The space blends bright natural and artificial light sources across indoor-outdoor boundaries.",
        "medium": "The area combines moderate indoor lighting with outdoor illumination in a balanced way.",
        "dim": "The transition space features subtle lighting gradients between indoor and outdoor zones."
    },
    "stadium_or_floodlit_area": {
    "general": "The scene is captured under powerful floodlights creating uniform bright illumination.",
    "bright": "The area is intensely illuminated by floodlights, similar to stadium conditions.",
    "medium": "The space has even, powerful lighting typical of sports facilities or outdoor events.",
    "dim": "The area has moderate floodlight illumination providing consistent lighting across the space."
    }
}
