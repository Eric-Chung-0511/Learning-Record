
SCENE_TYPES = {
    "living_room": {
        "name": "Living Room",
        "required_objects": [57, 62],  # couch, tv
        "optional_objects": [56, 60, 73, 75],  # chair, dining table, book, vase
        "minimum_required": 2,
        "description": "A living room area with furniture for relaxation and entertainment"
    },
    "bedroom": {
        "name": "Bedroom",
        "required_objects": [59],  # bed
        "optional_objects": [56, 60, 73, 74, 75],  # chair, dining table, book, clock, vase
        "minimum_required": 1,
        "description": "A bedroom with sleeping furniture"
    },
    "dining_area": {
        "name": "Dining Area",
        "required_objects": [60],  # dining table
        "optional_objects": [56, 39, 41, 42, 43, 44, 45],  # chair, bottle, cup, fork, knife, spoon, bowl
        "minimum_required": 1,
        "description": "A dining area for meals"
    },
    "kitchen": {
        "name": "Kitchen",
        "required_objects": [72, 68, 69, 71],  # refrigerator, microwave, oven, sink
        "optional_objects": [39, 41, 42, 43, 44, 45],  # bottle, cup, fork, knife, spoon, bowl
        "minimum_required": 1,
        "description": "A kitchen area for food preparation"
    },
    "office_workspace": {
        "name": "Office Workspace",
        "required_objects": [56, 63, 66, 64, 73],  # chair, laptop, keyboard, mouse, book
        "optional_objects": [60, 74, 75, 67],  # dining table, clock, vase, cell phone
        "minimum_required": 2,
        "description": "A workspace with computer equipment for office work"
    },
    "meeting_room": {
        "name": "Meeting Room",
        "required_objects": [56, 60],  # chair, dining table
        "optional_objects": [63, 62, 67],  # laptop, tv, cell phone
        "minimum_required": 2,
        "description": "A room set up for meetings with multiple seating"
    },
    "city_street": {
        "name": "City Street",
        "required_objects": [0, 1, 2, 3, 5, 7, 9],  # person, bicycle, car, motorcycle, bus, truck, traffic light
        "optional_objects": [10, 11, 12, 24, 25, 26, 28],  # fire hydrant, stop sign, parking meter, backpack, umbrella, handbag, suitcase
        "minimum_required": 2,
        "description": "A city street with traffic and pedestrians"
    },
    "parking_lot": {
        "name": "Parking Lot",
        "required_objects": [2, 3, 5, 7],  # car, motorcycle, bus, truck
        "optional_objects": [0, 11, 12],  # person, stop sign, parking meter
        "minimum_required": 3,
        "description": "A parking area with multiple vehicles"
    },
    "park_area": {
        "name": "Park or Recreation Area",
        "required_objects": [0, 13],  # person, bench
        "optional_objects": [1, 14, 16, 25, 33],  # bicycle, bird, dog, umbrella, kite
        "minimum_required": 2,
        "description": "An outdoor recreational area for leisure activities"
    },
    "retail_store": {
        "name": "Retail Store",
        "required_objects": [0, 24, 26, 28],  # person, backpack, handbag, suitcase
        "optional_objects": [39, 45, 67],  # bottle, bowl, cell phone
        "minimum_required": 2,
        "description": "A retail environment with shoppers and merchandise"
    },
    "supermarket": {
        "name": "Supermarket",
        "required_objects": [0, 24, 39, 46, 47, 49],  # person, backpack, bottle, banana, apple, orange
        "optional_objects": [26, 37, 45, 48, 51, 52, 53, 54, 55],  # handbag, surfboard, bowl, sandwich, carrot, hot dog, pizza, donut, cake
        "minimum_required": 3,
        "description": "A supermarket with food items and shoppers"
    },
    "classroom": {
        "name": "Classroom",
        "required_objects": [56, 60, 73],  # chair, dining table, book
        "optional_objects": [63, 66, 67],  # laptop, keyboard, cell phone
        "minimum_required": 2,
        "description": "A classroom environment set up for educational activities"
    },
    "conference_room": {
        "name": "Conference Room",
        "required_objects": [56, 60, 63],  # chair, dining table, laptop
        "optional_objects": [62, 67, 73],  # tv, cell phone, book
        "minimum_required": 2,
        "description": "A conference room designed for meetings and presentations"
    },
    "cafe": {
        "name": "Cafe",
        "required_objects": [56, 60, 41],  # chair, dining table, cup
        "optional_objects": [39, 40, 63, 67, 73],  # bottle, wine glass, laptop, cell phone, book
        "minimum_required": 2,
        "description": "A cafe setting with seating and beverages"
    },
    "library": {
        "name": "Library",
        "required_objects": [56, 60, 73],  # chair, dining table, book
        "optional_objects": [63, 67, 75],  # laptop, cell phone, vase
        "minimum_required": 2,
        "description": "A library with books and reading areas"
    },
    "gym": {
        "name": "Gym",
        "required_objects": [0, 32],  # person, sports ball
        "optional_objects": [24, 25, 28, 38],  # backpack, umbrella, suitcase, tennis racket
        "minimum_required": 1,
        "description": "A gym or fitness area for physical activities"
    },
    "beach": {
        "name": "Beach",
        "required_objects": [0, 25, 29, 33, 37],  # person, umbrella, frisbee, kite, surfboard
        "optional_objects": [1, 24, 26, 38],  # bicycle, backpack, handbag, tennis racket
        "minimum_required": 2,
        "description": "A beach area with people and recreational items"
    },
    "restaurant": {
        "name": "Restaurant",
        "required_objects": [56, 60, 41, 42, 43, 44, 45],  # chair, dining table, cup, fork, knife, spoon, bowl
        "optional_objects": [39, 40, 48, 49, 50, 51, 52, 53, 54, 55],  # bottle, wine glass, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
        "minimum_required": 3,
        "description": "A restaurant setting for dining with tables and eating utensils"
    },
    "train_station": {
        "name": "Train Station",
        "required_objects": [0, 6],  # person, train
        "optional_objects": [1, 2, 24, 28, 67],  # bicycle, car, backpack, suitcase, cell phone
        "minimum_required": 1,
        "description": "A train station with train and passengers"
    },
    "airport": {
        "name": "Airport",
        "required_objects": [0, 4, 28],  # person, airplane, suitcase
        "optional_objects": [24, 25, 26, 67],  # backpack, umbrella, handbag, cell phone
        "minimum_required": 2,
        "description": "An airport with planes and travelers carrying luggage"
    },
      "upscale_dining": {
        "name": "Upscale Dining Area",
        "required_objects": [56, 60, 40, 41],  # chair, dining table, wine glass, cup
        "optional_objects": [39, 42, 43, 44, 45, 62, 75],  # bottle, fork, knife, spoon, bowl, tv, vase
        "minimum_required": 2,
        "description": "An elegantly designed dining space with refined furniture and decorative elements"
    },
    "asian_commercial_street": {
        "name": "Asian Commercial Street",
        "required_objects": [0, 67],  # person, cell phone
        "optional_objects": [1, 2, 3, 24, 25, 26, 28],  # bicycle, car, motorcycle, backpack, umbrella, handbag, suitcase
        "minimum_required": 1,
        "description": "A bustling commercial street with shops, signage, and pedestrians in an Asian urban setting"
    },
    "financial_district": {
        "name": "Financial District",
        "required_objects": [2, 5, 7, 9],  # car, bus, truck, traffic light
        "optional_objects": [0, 1, 3, 8],  # person, bicycle, motorcycle, boat
        "minimum_required": 2,
        "description": "A major thoroughfare in a business district with high-rise buildings and traffic"
    },
    "urban_intersection": {
        "name": "Urban Intersection",
        "required_objects": [0, 9],  # person, traffic light
        "optional_objects": [1, 2, 3, 5, 7],  # bicycle, car, motorcycle, bus, truck
        "minimum_required": 1,
        "description": "A busy urban crossroad with pedestrian crossings and multiple traffic flows"
    },
    "transit_hub": {
        "name": "Transit Hub",
        "required_objects": [0, 5, 6, 7],  # person, bus, train, truck
        "optional_objects": [1, 2, 3, 9, 24, 28],  # bicycle, car, motorcycle, traffic light, backpack, suitcase
        "minimum_required": 2,
        "description": "A transportation center where multiple modes of transit converge"
    },
    "shopping_district": {
        "name": "Shopping District",
        "required_objects": [0, 24, 26],  # person, backpack, handbag
        "optional_objects": [1, 2, 3, 25, 27, 28, 39, 67],  # bicycle, car, motorcycle, umbrella, tie, suitcase, bottle, cell phone
        "minimum_required": 2,
        "description": "A retail-focused area with shops, pedestrians, and commercial activity"
    },
     "bus_stop": {
        "name": "Bus Stop",
        "required_objects": [0, 5],  # person, bus
        "optional_objects": [1, 2, 7, 24],  # bicycle, car, truck, backpack
        "minimum_required": 2,
        "description": "A roadside bus stop with waiting passengers and buses"
    },
    "bus_station": {
        "name": "Bus Station",
        "required_objects": [0, 5, 7],  # person, bus, truck
        "optional_objects": [24, 28, 67],  # backpack, suitcase, cell phone
        "minimum_required": 2,
        "description": "A bus terminal with multiple buses and travelers"
    },
    "zoo": {
        "name": "Zoo",
        "required_objects": [20, 22, 23],  # elephant, zebra, giraffe
        "optional_objects": [0, 14, 16],  # person, bird, dog
        "minimum_required": 2,
        "description": "A zoo environment featuring large animal exhibits and visitors"
    },
    "harbor": {
        "name": "Harbor",
        "required_objects": [8],  # boat
        "optional_objects": [0, 2, 3, 39],  # person, car, motorcycle, bottle
        "minimum_required": 1,
        "description": "A harbor area with boats docked and surrounding traffic"
    },
    "playground": {
        "name": "Playground",
        "required_objects": [0, 32],  # person, sports ball
        "optional_objects": [33, 24, 1],  # kite, backpack, bicycle
        "minimum_required": 1,
        "description": "An outdoor playground with people playing sports and games"
    },
    "sports_field": {
        "name": "Sports Field",
        "required_objects": [32],  # sports ball
        "optional_objects": [38, 34, 35],  # tennis racket, baseball bat, baseball glove
        "minimum_required": 1,
        "description": "A sports field set up for various ball games"
    },
     "narrow_commercial_alley": {
        "name": "Narrow Commercial Alley",
        "required_objects": [0, 3],  # person, motorcycle
        "optional_objects": [2, 7, 24, 26],  # car, truck, backpack, handbag
        "minimum_required": 2,
        "description": "A tight urban alley lined with shops, with pedestrians and light vehicles"
    },
    "daytime_shopping_street": {
        "name": "Daytime Shopping Street",
        "required_objects": [0, 2],  # person, car
        "optional_objects": [1, 3, 24, 26],  # bicycle, motorcycle, backpack, handbag
        "minimum_required": 2,
        "description": "A busy pedestrian street during daytime, featuring shops, vehicles, and shoppers"
    },
    "urban_pedestrian_crossing": {
        "name": "Urban Pedestrian Crossing",
        "required_objects": [0, 9],  # person, traffic light
        "optional_objects": [2, 3, 5],  # car, motorcycle, bus
        "minimum_required": 1,
        "description": "A city street crossing with pedestrians and traffic signals"
    },
    "aerial_view_intersection": {
    "name": "Aerial View Intersection",
    "required_objects": [0, 9],  # person, traffic light
    "optional_objects": [1, 2, 3, 5, 7],  # bicycle, car, motorcycle, bus, truck
    "minimum_required": 1,
    "description": "An intersection viewed from above, showing crossing patterns and pedestrian movement"
    },
    "aerial_view_commercial_area": {
        "name": "Aerial View Commercial Area",
        "required_objects": [0, 2],  # person, car
        "optional_objects": [1, 3, 5, 7, 24, 26],  # bicycle, motorcycle, bus, truck, backpack, handbag
        "minimum_required": 2,
        "description": "A commercial or shopping area viewed from above showing pedestrians and urban layout"
    },
    "aerial_view_plaza": {
        "name": "Aerial View Plaza",
        "required_objects": [0],  # person
        "optional_objects": [1, 2, 24, 25, 26],  # bicycle, car, backpack, umbrella, handbag
        "minimum_required": 1,
        "description": "An urban plaza or public square viewed from above with pedestrian activity"
    },

    # specific cultural item
    "asian_night_market": {
        "name": "Asian Night Market",
        "required_objects": [0, 67],  # person, cell phone
        "optional_objects": [1, 3, 24, 26, 39, 41],  # bicycle, motorcycle, backpack, handbag, bottle, cup
        "minimum_required": 1,
        "description": "A vibrant night market scene typical in Asian cities with food stalls and crowds"
    },
    "asian_temple_area": {
        "name": "Asian Temple Area",
        "required_objects": [0],  # person
        "optional_objects": [24, 25, 26, 67, 75],  # backpack, umbrella, handbag, cell phone, vase
        "minimum_required": 1,
        "description": "A traditional Asian temple complex with visitors and cultural elements"
    },

    # specific time item
    "nighttime_street": {
        "name": "Nighttime Street",
        "required_objects": [0, 9],  # person, traffic light
        "optional_objects": [1, 2, 3, 5, 7, 67],  # bicycle, car, motorcycle, bus, truck, cell phone
        "minimum_required": 1,
        "description": "An urban street at night with artificial lighting and nighttime activity"
    },
    "nighttime_commercial_district": {
        "name": "Nighttime Commercial District",
        "required_objects": [0, 67],  # person, cell phone
        "optional_objects": [1, 2, 3, 24, 26],  # bicycle, car, motorcycle, backpack, handbag
        "minimum_required": 1,
        "description": "A commercial district illuminated at night with neon signs and evening activity"
    },

    # mixture enviroment item
    "indoor_outdoor_cafe": {
        "name": "Indoor-Outdoor Cafe",
        "required_objects": [56, 60, 41],  # chair, dining table, cup
        "optional_objects": [39, 40, 63, 67, 73],  # bottle, wine glass, laptop, cell phone, book
        "minimum_required": 2,
        "description": "A cafe setting with both indoor elements and outdoor patio or sidewalk seating"
    },
    "transit_station_platform": {
        "name": "Transit Station Platform",
        "required_objects": [0],  # person
        "optional_objects": [5, 6, 7, 24, 28, 67],  # bus, train, truck, backpack, suitcase, cell phone
        "minimum_required": 1,
        "description": "A transit platform with waiting passengers and arriving/departing vehicles"
    },
    "sports_stadium": {
        "name": "Sports Stadium",
        "required_objects": [0, 32],  # person, sports ball
        "optional_objects": [24, 38, 39, 41, 67],  # backpack, tennis racket, bottle, cup, cell phone
        "minimum_required": 1,
        "description": "A sports stadium or arena with spectators and athletic activities"
    },
    "construction_site": {
        "name": "Construction Site",
        "required_objects": [0, 7],  # person, truck
        "optional_objects": [2, 3, 11, 76, 77, 78],  # car, motorcycle, fire hydrant, scissors, teddy bear, hair drier
        "minimum_required": 1,
        "description": "A construction site with workers, equipment, and building materials"
    },
    "medical_facility": {
        "name": "Medical Facility",
        "required_objects": [0, 56, 60],  # person, chair, dining table
        "optional_objects": [63, 64, 66, 67, 73],  # laptop, mouse, keyboard, cell phone, book
        "minimum_required": 2,
        "description": "A medical facility such as hospital, clinic or doctor's office with medical staff and patients"
    },
    "educational_setting": {
        "name": "Educational Setting",
        "required_objects": [0, 56, 60, 73],  # person, chair, dining table, book
        "optional_objects": [63, 64, 66, 67, 74],  # laptop, mouse, keyboard, cell phone, clock
        "minimum_required": 2,
        "description": "An educational environment such as classroom, lecture hall or study area"
    },
    "aerial_view_intersection": {
        "name": "Aerial View Intersection",
        "required_objects": [0, 9],  # person, traffic light
        "optional_objects": [1, 2, 3, 5, 7],  # bicycle, car, motorcycle, bus, truck
        "minimum_required": 1,
        "description": "An intersection viewed from above, showing crossing patterns and pedestrian movement",
        "viewpoint_indicator": "aerial", # view side
        "key_features": ["crosswalk_pattern", "pedestrian_flow", "intersection_layout"],  # key feature
        "detection_priority": 10  # priority
    },
    "perpendicular_crosswalk_intersection": {
        "name": "Perpendicular Crosswalk Intersection",
        "required_objects": [0],  # person
        "optional_objects": [1, 2, 3, 5, 7, 9],  # bicycle, car, motorcycle, bus, truck, traffic light
        "minimum_required": 1,
        "description": "An intersection with perpendicular crosswalks where pedestrians cross in multiple directions",
        "viewpoint_indicator": "aerial",
        "key_features": ["perpendicular_crosswalks", "pedestrian_crossing", "multi_directional_movement"],
        "pattern_detection": True, # specific pattern
        "detection_priority": 15  #
    },
    "beach_water_recreation": {
    "name": "Beach/Water Recreation Area",
    "required_objects": [0, 37],  # person, surfboard
    "optional_objects": [25, 33, 1, 8, 29, 24, 26, 39, 41],  # umbrella, kite, bicycle, boat, frisbee, backpack, handbag, bottle, cup
    "minimum_required": 2,
    "description": "A beach or water recreation area with water sports equipment and beach accessories"
    },
    "sports_venue": {
    "name": "Sports Venue",
    "required_objects": [0, 32],  # person, sports ball
    "optional_objects": [34, 35, 38, 25, 24, 26, 39, 41],  # baseball bat, baseball glove, tennis racket, umbrella, backpack, handbag, bottle, cup
    "minimum_required": 2,
    "description": "A professional sports venue with specialized sports equipment and spectator areas"
    },
    "professional_kitchen": {
    "name": "Professional Kitchen",
    "required_objects": [43, 44, 45],  # knife, spoon, bowl
    "optional_objects": [42, 39, 41, 68, 69, 71, 72, 0],  # fork, bottle, cup, microwave, oven, sink, refrigerator, person
    "minimum_required": 3,
    "description": "A commercial kitchen with professional cooking equipment and food preparation areas"
    },
    "tourist_landmark": {
        "name": "Tourist Landmark",
        "required_objects": [0],  # person
        "optional_objects": [24, 26, 67],  # backpack, handbag, cell phone
        "minimum_required": 0,  # 可能沒有人，但仍然是地標
        "description": "A location featuring a famous landmark with tourist activity",
        "priority": 1.2  # 提高優先級
    },
    "natural_landmark": {
        "name": "Natural Landmark",
        "required_objects": [0],  # person
        "optional_objects": [24, 26, 67],  # backpack, handbag, cell phone
        "minimum_required": 0,
        "description": "A natural landmark site with scenic views",
        "priority": 1.2
    },
    "historical_monument": {
        "name": "Historical Monument",
        "required_objects": [0],  # person
        "optional_objects": [24, 26, 67],  # backpack, handbag, cell phone
        "minimum_required": 0,
        "description": "A historical monument or heritage site",
        "priority": 1.2
    },
    "general_indoor_space": {
        "name": "General Indoor Space",
        "required_objects": [], # No strict required objects, depends on combination
        "optional_objects": [
            56, # chair
            57, # couch
            58, # potted plant
            59, # bed
            60, # dining table
            61, # toilet
            62, # tv
            63, # laptop
            66, # keyboard
            67, # cell phone
            73, # book
            74, # clock
            75, # vase
            39, # bottle
            41, # cup
        ],
        "minimum_required": 2, # Needs at least a few common indoor items
        "description": "An indoor area with various common household or functional items.",
        "priority": 0.8 # Lower priority than more specific scenes
    },
    "generic_street_view": {
        "name": "Generic Street View",
        "required_objects": [], # More about the combination
        "optional_objects": [
            0,  # person
            1,  # bicycle
            2,  # car
            3,  # motorcycle
            5,  # bus
            7,  # truck
            9,  # traffic light
            10, # fire hydrant
            11, # stop sign
            13, # bench
            # Consider adding building if YOLO detects it (not a standard COCO class for YOLOv8, but some custom models might)
        ],
        "minimum_required": 2, # e.g., a car and a person, or multiple vehicles
        "description": "An outdoor street view, likely in an urban or suburban setting, with vehicles and/or pedestrians.",
        "priority": 0.85
    },
    "desk_area_workspace": {
        "name": "Desk Area / Workspace",
        "required_objects": [
            63, # laptop or 62 (tv as monitor) or 66 (keyboard)
        ],
        "optional_objects": [
            56, # chair
            60, # dining table (often used as a desk)
            64, # mouse
            66, # keyboard
            73, # book
            41, # cup
            67, # cell phone
            74, # clock
        ],
        "minimum_required": 2, # e.g., laptop and chair, or table and keyboard
        "description": "A workspace or desk area, typically featuring a computer and related accessories.",
        "priority": 0.9
    },
    "outdoor_gathering_spot": {
        "name": "Outdoor Gathering Spot",
        "required_objects": [
            0,  # person
        ],
        "optional_objects": [
            13, # bench
            32, # sports ball
            24, # backpack
            25, # umbrella
            29, # frisbee
            33, # kite
            58, # potted plant (if in a more structured park area)
        ],
        "minimum_required": 2, # e.g., person and bench, or multiple people
        "description": "An outdoor area where people might gather for leisure or activity.",
        "priority": 0.8
    },
    "kitchen_counter_or_utility_area": {
        "name": "Kitchen Counter or Utility Area",
        "required_objects": [],
        "optional_objects": [
            39, # bottle
            41, # cup
            44, # spoon
            45, # bowl
            68, # microwave
            69, # oven
            70, # toaster
            71, # sink
            72, # refrigerator
        ],
        "minimum_required": 2, # e.g., sink and microwave, or refrigerator and bottles
        "description": "An area likely used for food preparation or kitchen utilities.",
        "priority": 0.9
    }
}
