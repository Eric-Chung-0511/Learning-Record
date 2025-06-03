
# 場景類型提示
SCENE_TYPE_PROMPTS = {
    # 基本室內場景
    "living_room": "A photo of a living room with furniture and entertainment systems.",
    "bedroom": "A photo of a bedroom with a bed and personal items.",
    "dining_area": "A photo of a dining area with a table and chairs for meals.",
    "kitchen": "A photo of a kitchen with cooking appliances and food preparation areas.",
    "office_workspace": "A photo of an office workspace with desk, computer and work equipment.",
    "meeting_room": "A photo of a meeting room with a conference table and multiple chairs.",

    # 基本室外/城市場景
    "city_street": "A photo of a city street with traffic, pedestrians and urban buildings.",
    "parking_lot": "A photo of a parking lot with multiple parked vehicles.",
    "park_area": "A photo of a park or recreational area with greenery and outdoor facilities.",
    "retail_store": "A photo of a retail store with merchandise displays and shopping areas.",
    "supermarket": "A photo of a supermarket with food items, aisles and shopping carts.",

    # 特殊室內場景
    "upscale_dining": "A photo of an upscale dining area with elegant furniture and refined decor.",
    "conference_room": "A photo of a professional conference room with presentation equipment and seating.",
    "classroom": "A photo of a classroom with desks, chairs and educational equipment.",
    "library": "A photo of a library with bookshelves, reading areas and study spaces.",

    # 亞洲特色場景
    "asian_commercial_street": "A photo of an Asian commercial street with dense signage, shops and pedestrians.",
    "asian_night_market": "A photo of an Asian night market with food stalls, crowds and colorful lights.",
    "asian_temple_area": "A photo of an Asian temple with traditional architecture and cultural elements.",

    # 交通相關場景
    "financial_district": "A photo of a financial district with tall office buildings and business activity.",
    "urban_intersection": "A photo of an urban intersection with crosswalks, traffic lights and pedestrians crossing.",
    "transit_hub": "A photo of a transportation hub with multiple modes of public transit and passengers.",
    "bus_stop": "A photo of a bus stop with people waiting and buses arriving or departing.",
    "bus_station": "A photo of a bus terminal with multiple buses and traveler facilities.",
    "train_station": "A photo of a train station with platforms, trains and passenger activity.",
    "airport": "A photo of an airport with planes, terminals and traveler activity.",

    # 商業場景
    "shopping_district": "A photo of a shopping district with multiple retail stores and consumer activity.",
    "cafe": "A photo of a cafe with coffee service, seating and casual dining.",
    "restaurant": "A photo of a restaurant with dining tables, food service and eating areas.",

    # 空中視角場景
    "aerial_view_intersection": "An aerial view of an intersection showing crosswalks and traffic patterns from above.",
    "aerial_view_commercial_area": "An aerial view of a commercial area showing shopping districts from above.",
    "aerial_view_plaza": "An aerial view of a public plaza or square showing patterns of people movement from above.",

    # 娛樂場景
    "zoo": "A photo of a zoo with animal enclosures, exhibits and visitors.",
    "playground": "A photo of a playground with recreational equipment and children playing.",
    "sports_field": "A photo of a sports field with playing surfaces and athletic equipment.",
    "sports_stadium": "A photo of a sports stadium with spectator seating and athletic facilities.",

    # 水相關場景
    "harbor": "A photo of a harbor with boats, docks and waterfront activity.",
    "beach_water_recreation": "A photo of a beach area with water activities, sand and recreational equipment like surfboards.",

    # 文化時間特定場景
    "nighttime_street": "A photo of a street at night with artificial lighting and evening activity.",
    "nighttime_commercial_district": "A photo of a commercial district at night with illuminated signs and evening shopping.",
    "european_plaza": "A photo of a European-style plaza with historic architecture and public gathering spaces.",

    # 混合環境場景
    "indoor_outdoor_cafe": "A photo of a cafe with both indoor seating and outdoor patio areas.",
    "transit_station_platform": "A photo of a transit station platform with waiting areas and arriving vehicles.",

    # 工作場景
    "construction_site": "A photo of a construction site with building materials, equipment and workers.",
    "medical_facility": "A photo of a medical facility with healthcare equipment and professional staff.",
    "educational_setting": "A photo of an educational setting with learning spaces and academic resources.",
    "professional_kitchen": "A photo of a professional commercial kitchen with industrial cooking equipment and food preparation stations."
}

# 文化特定場景提示
CULTURAL_SCENE_PROMPTS = {
    "asian_commercial_street": [
        "A busy Asian shopping street with neon signs and dense storefronts.",
        "A commercial street in Asia with multi-level signage and narrow walkways.",
        "A street scene in Taiwan or Hong Kong with vertical signage and compact shops.",
        "A crowded commercial alley in an Asian city with signs in Chinese characters.",
        "A narrow shopping street in Asia with small shops on both sides.",
        "An outdoor shopping district in an East Asian city with electronic billboards.",
        "A bustling commercial street in Taiwan with food vendors and retail shops.",
        "A pedestrian shopping area with Korean or Chinese signs and storefronts.",
        "A daytime shopping street in an Asian urban center with vertical development."
    ],
    "asian_night_market": [
        "A vibrant night market in Asia with food stalls and large crowds.",
        "An evening street market in Taiwan with street food vendors and bright lights.",
        "A busy night bazaar in Asia with illuminated stalls and local food.",
        "A crowded night street food market in an Asian city with vendor carts.",
        "An Asian night market with steam from cooking food and hanging lanterns.",
        "A nocturnal food street in East Asia with vendor canopies and neon lights.",
        "A bustling evening market with rows of food stalls and plastic stools.",
        "A lively Asian street food scene at night with cooking stations and crowds."
    ],
    "asian_temple_area": [
        "A traditional Asian temple with ornate roof details and religious symbols.",
        "A Buddhist temple complex in East Asia with multiple pavilions and prayer areas.",
        "A sacred site in Asia with incense burners and ceremonial elements.",
        "A temple courtyard with stone statues and traditional Asian architecture.",
        "A spiritual center in East Asia with pagoda-style structures and visitors.",
        "An ancient temple site with Asian architectural elements and cultural symbols.",
        "A religious compound with characteristic Asian roof curves and decorative features."
    ],
    "european_plaza": [
        "A historic European city square with classical architecture and cafes.",
        "An old-world plaza in Europe with cobblestone paving and historic buildings.",
        "A public square in a European city with fountains and surrounding architecture.",
        "A central plaza in Europe with outdoor seating areas and historic monuments.",
        "A traditional European town square with surrounding shops and restaurants.",
        "A historic gathering space in Europe with distinctive architecture and pedestrians."
    ]
}

# 對比類別提示
COMPARATIVE_PROMPTS = {
    "indoor_vs_outdoor": [
        "An indoor shopping mall corridor with controlled lighting and storefronts.",
        "An outdoor commercial street with natural lighting and urban storefronts.",
        "An enclosed shopping gallery with artificial lighting and climate control.",
        "An open-air market street with natural light and weather exposure."
    ],
    "professional_vs_home": [
        "A professional commercial kitchen with stainless steel equipment and workstations.",
        "A home kitchen with residential appliances and family cooking space.",
        "A restaurant kitchen with multiple cooking stations and chef activity.",
        "A family kitchen with standard household equipment and personal touches."
    ],
    "sports_venue_vs_park": [
        "A professional sports stadium with designated playing areas and audience seating.",
        "A public park with casual recreation space and community greenery.",
        "An athletic venue with specialized sports equipment and competitive playing surfaces.",
        "An outdoor community space with general purpose areas and natural elements."
    ],
    "asian_vs_western_commercial": [
        "An Asian shopping street with vertical signage and compact multi-level shops.",
        "A Western commercial street with horizontal storefronts and wider sidewalks.",
        "An East Asian retail area with dense signage in Asian scripts and narrow walkways."
        "A Western shopping district with uniform building heights and Latin alphabetic signs."
    ],
    "daytime_vs_nighttime": [
        "A daytime urban scene with natural sunlight illuminating streets and buildings.",
        "A nighttime city scene with artificial lighting from stores, signs and streetlights.",
        "A commercial district during daylight hours with natural shadows and visibility.",
        "An evening urban setting with illuminated storefronts and light patterns on streets."
    ],
    "aerial_vs_street_level": [
        "An aerial view showing urban patterns and layouts from above.",
        "A street-level view showing pedestrian perspective and immediate surroundings.",
        "A bird's-eye view of city organization and movement patterns from high above.",
        "An eye-level perspective showing direct human interaction with urban elements."
    ]
}

# 環境條件文本提示
LIGHTING_CONDITION_PROMPTS = {
    "day_clear": "A photo taken during daytime with clear skies and direct sunlight.",
    "day_cloudy": "A photo taken during daytime with overcast conditions and diffused light.",
    "sunset/sunrise": "A photo taken during sunset or sunrise with warm golden lighting and long shadows.",
    "night": "A photo taken at night with minimal natural light and artificial illumination.",
    "indoor_bright": "An indoor photo with bright, even artificial lighting throughout the space.",
    "indoor_moderate": "An indoor photo with moderate lighting creating a balanced indoor atmosphere.",
    "indoor_dim": "An indoor photo with low lighting levels creating a subdued environment.",
    "neon_night": "A night scene with colorful neon lighting creating vibrant illumination patterns.",
    "indoor_commercial": "An indoor retail environment with directed display lighting highlighting products.",
    "indoor_restaurant": "An indoor dining space with ambient mood lighting for atmosphere.",
    "stadium_lighting": "A sports venue with powerful floodlights creating intense, even illumination.",
    "mixed_lighting": "A scene with combined natural and artificial light sources creating transition zones.",
    "beach_daylight": "A photo taken at a beach with bright natural sunlight and reflections from water.",
    "sports_arena_lighting": "A photo of a sports venue illuminated by powerful overhead lighting systems.",
    "kitchen_task_lighting": "A photo of a kitchen with focused lighting concentrated on work surfaces."
}

# 針對新場景類型的特殊提示
SPECIALIZED_SCENE_PROMPTS = {
    "beach_water_recreation": [
        "A coastal beach scene with people surfing and sunbathing on sandy shores.",
        "Active water sports participants at a beach with surfboards and swimming areas.",
        "A sunny beach destination with recreational water equipment and beachgoers.",
        "A shoreline recreation area with surf gear and coastal activities.",
        "An oceanfront scene with people engaging in water sports and beach leisure.",
        "A popular beach spot with swimming areas and surfing zones.",
        "A coastal recreation setting with beach umbrellas and water activities."
    ],
    "sports_venue": [
        "An indoor sports arena with professional equipment and competition spaces.",
        "A sports stadium with marked playing areas and spectator seating arrangement.",
        "A specialized athletic venue with competition equipment and performance areas.",
        "A professional sports facility with game-related apparatus and audience zones.",
        "An organized sports center with competitive play areas and athletic equipment.",
        "A competition venue with sport-specific markings and professional setup.",
        "A formal athletic facility with standardized equipment and playing surfaces."
    ],
    "professional_kitchen": [
        "A commercial restaurant kitchen with multiple cooking stations and food prep areas.",
        "A professional culinary workspace with industrial appliances and chef activity.",
        "A busy restaurant back-of-house with stainless steel equipment and meal preparation.",
        "A commercial food service kitchen with chef workstations and specialized zones.",
        "An industrial kitchen facility with specialized cooking equipment and prep surfaces.",
        "A high-volume food production kitchen with professional-grade appliances.",
        "A restaurant kitchen with distinct cooking areas and culinary workflow design."
    ],
    "urban_intersection": [
        "A city intersection with crosswalks and traffic signals controlling movement.",
        "A busy urban crossroad with pedestrian crossings and vehicle traffic.",
        "A regulated street intersection with crosswalk markings and waiting pedestrians.",
        "A metropolitan junction with traffic lights and pedestrian crossing zones.",
        "A city street crossing with safety features for pedestrians and traffic flow.",
        "A controlled urban intersection with movement patterns for vehicles and people.",
        "A city center crossroad with traffic management features and pedestrian areas."
    ],
    "financial_district": [
        "A downtown business area with tall office buildings and commercial activity.",
        "An urban financial center with skyscrapers and professional environment.",
        "A city's business district with corporate headquarters and office towers.",
        "A metropolitan financial zone with high-rise buildings and business traffic.",
        "A corporate district in a city center with professional architecture.",
        "An urban area dominated by office buildings and business establishments.",
        "A city's economic center with banking institutions and corporate offices."
    ],
    "aerial_view_intersection": [
        "A bird's-eye view of a city intersection showing crossing patterns from above.",
        "An overhead perspective of an urban crossroad showing traffic organization.",
        "A top-down view of a street intersection revealing pedestrian crosswalks.",
        "An aerial shot of a city junction showing the layout of roads and crossings.",
        "A high-angle view of an intersection showing traffic and pedestrian flow patterns.",
        "A drone perspective of urban crossing design viewed from directly above.",
        "A vertical view of a street intersection showing crossing infrastructure."
    ]
}

VIEWPOINT_PROMPTS = {
    "eye_level": "A photo taken from normal human eye level showing a direct frontal perspective.",
    "aerial": "A photo taken from high above looking directly down at the scene below.",
    "elevated": "A photo taken from a higher than normal position looking down at an angle.",
    "low_angle": "A photo taken from a low position looking upward at the scene.",
    "bird_eye": "A photo taken from very high above showing a complete overhead perspective.",
    "street_level": "A photo taken from the perspective of someone standing on the street.",
    "interior": "A photo taken from inside a building showing the internal environment.",
    "vehicular": "A photo taken from inside or mounted on a moving vehicle."
}

OBJECT_COMBINATION_PROMPTS = {
    "dining_setting": "A scene with tables, chairs, plates, and eating utensils arranged for meals.",
    "office_setup": "A scene with desks, chairs, computers, and office supplies for work.",
    "living_space": "A scene with sofas, coffee tables, TVs, and comfortable seating arrangements.",
    "transportation_hub": "A scene with vehicles, waiting areas, passengers, and transit information.",
    "retail_environment": "A scene with merchandise displays, shoppers, and store fixtures.",
    "crosswalk_scene": "A scene with street markings, pedestrians crossing, and traffic signals.",
    "cooking_area": "A scene with stoves, prep surfaces, cooking utensils, and food items.",
    "recreational_space": "A scene with sports equipment, play areas, and activity participants."
}

ACTIVITY_PROMPTS = {
    "shopping": "People looking at merchandise, carrying shopping bags, and browsing stores.",
    "dining": "People eating food, sitting at tables, and using dining utensils.",
    "commuting": "People waiting for transportation, boarding vehicles, and traveling.",
    "working": "People using computers, attending meetings, and engaged in professional tasks.",
    "exercising": "People engaged in physical activities, using sports equipment, and training.",
    "cooking": "People preparing food, using kitchen equipment, and creating meals.",
    "crossing_street": "People walking across designated crosswalks and navigating intersections.",
    "recreational_activity": "People engaged in leisure activities, games, and social recreation."
}
