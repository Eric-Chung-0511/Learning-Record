
"""
Landmark database for zero-shot classification using CLIP
"""

LANDMARK_DATA = {
    # 亞洲地標
    "asia": {
        "taipei_101": {
            "name": "Taipei 101",
            "aliases": ["Taipei 101 Tower", "Taipei Financial Center", "台北101大樓", "Taipei World Financial Center"],
            "location": "Taipei, Taiwan",
            "prompts": [
                "the iconic green bamboo-shaped Taipei 101 skyscraper against city skyline",
                "Taipei 101 tower with its distinctive stacked pagoda design and vibrant green glass facade",
                "world-famous Taipei 101 skyscraper with its segmented exterior resembling a bamboo stalk and prominent observation deck",
                "close-up of Taipei 101's architectural details showing the ruyi-inspired motifs and bamboo-inspired segments",
                "aerial view of Taipei 101 standing tall as a landmark in Taipei's urban landscape",
                "Taipei 101 illuminated at night with a spectrum of colorful light displays, often themed for holidays",
                "Taipei 101 seen from street level, emphasizing its immense height and unique green-tinted curtain wall",
                "the 508-meter tall Taipei 101 tower, an engineering marvel with its unique tiered design and spire",
                "Taipei 101's distinctive green glass exterior reflecting the sky, with visible bamboo segment patterns",
                "famous Taipei skyscraper, its characteristic pagoda-like stacked sections symbolizing growth and prosperity",
                "Taipei 101 featuring its massive tuned mass damper, visible through architectural openings, designed to counteract wind and seismic activity",
                "modern Asian skyscraper Taipei 101, showcasing a fusion of contemporary architecture and traditional Asian symbolism",
                "Taipei's iconic 101-story skyscraper with its tapered pinnacle reaching towards the sky, often against a backdrop of mountains or blue sky",
                "Taipei 101 viewed from Elephant Mountain, offering a classic panoramic shot of the tower and city",
                "close view of Taipei 101's signature green glass exterior with its intricate geometric patterns and reflective surface",
                "Taipei 101 with its distinct floor segments that resemble ancient Chinese ingots or a blossoming bamboo",
                "Taiwan's tallest building, Taipei 101, with its square base, tapered form, and prominent antenna",
                "the tall, slender green Taipei 101 skyscraper, a prominent feature in the city's skyline, day or night",
                "Taipei 101 with its unique green segmented tower design, a masterpiece of modern engineering and cultural symbolism, against a clear blue sky",
                "distinctive green glass Taipei 101 tower, a symbol of Taiwan's modernity, dominating the city view",
                "Taipei 101's iconic green tiered skyscraper structure with its high-speed elevators and 360-degree observation deck"
            ]
        },
        "taroko_gorge": {
            "name": "Taroko Gorge",
            "aliases": ["Taroko National Park", "太魯閣國家公園"],
            "location": "Hualien, Taiwan",
            "prompts": [
                "a photo of Taroko Gorge in Taiwan, showcasing its deep marble canyons and the Liwu River",
                "sheer marble cliffs of Taroko Gorge, with tunnels carved through the rock like the Tunnel of Nine Turns",
                "Taroko Gorge national park landscape with turquoise river, lush vegetation, and towering rock walls",
                "Swallow Grotto (Yanzikou) trail in Taroko Gorge, with views of the river and cliff formations",
                "the Cihmu Bridge, a distinctive red suspension bridge within Taroko Gorge"
            ]
        },
        "sun_moon_lake": {
            "name": "Sun Moon Lake",
            "aliases": ["日月潭", "Lake Candidius"],
            "location": "Nantou, Taiwan",
            "prompts": [
                "a photo of Sun Moon Lake in Taiwan, showing its calm, clear waters and surrounding mountains",
                "the serene Sun Moon Lake with Lalu Island, its sacred ancestral ground, dividing the lake into sun and moon shapes",
                "Sun Moon Lake surrounded by mist-covered mountains, with traditional temples like Wenwu Temple or Ci En Pagoda visible",
                "boats and ferries on Sun Moon Lake, with cyclists on lakeside paths",
                "aerial view of Sun Moon Lake highlighting its unique shape and the lush greenery of the region"
            ]
        },
        "jiufen_old_street": {
            "name": "Jiufen Old Street",
            "aliases": ["九份老街", "Chiufen"],
            "location": "New Taipei City, Taiwan",
            "prompts": [
                "narrow, winding alleys of Jiufen Old Street, lined with glowing red lanterns, traditional teahouses, and local food stalls",
                "atmospheric Jiufen Old Street at dusk or night, with countless red lanterns illuminating the historic hillside town and views of the coast",
                "historic gold mining town architecture in Jiufen, with wooden buildings and steep staircases, reminiscent of scenes from 'Spirited Away'",
                "A-Mei Tea House in Jiufen, a famous landmark with its traditional facade and lantern decorations",
                "street food and bustling crowds in the vibrant Jiufen Old Street market"
            ]
        },
        "kenting_national_park": {
            "name": "Kenting National Park",
            "aliases": ["墾丁國家公園"],
            "location": "Pingtung, Taiwan",
            "prompts": [
                "tropical beaches with white sand and clear blue water in Kenting National Park, southern Taiwan",
                "Eluanbi Lighthouse, the iconic white lighthouse at Taiwan's southernmost point, within Kenting National Park",
                "diverse coastal landscapes of Kenting, including coral reefs, rock formations like Sail Rock (Chuanfanshi), and lush forests",
                "vibrant marine life and water activities like snorkeling or surfing in Kenting National Park",
                "Longpan Park in Kenting, featuring dramatic grassy cliffs and coastline views"
            ]
        },
        "national_palace_museum_tw": { # Added _tw to differentiate from Beijing's
            "name": "National Palace Museum (Taipei)",
            "aliases": ["國立故宮博物院", "Taipei Palace Museum"],
            "location": "Taipei, Taiwan",
            "prompts": [
                "the grand exterior of the National Palace Museum in Taipei, a traditional Chinese palace-style building housing a vast collection of imperial artifacts",
                "iconic exhibits like the Jadeite Cabbage, Meat-shaped Stone, and Mao Gong Ding at the National Palace Museum in Taipei",
                "classical Chinese palace architecture of the National Palace Museum building in Taipei, with green tiled roofs and moon gates",
                "interior halls of the National Palace Museum displaying ancient Chinese ceramics, bronzes, and calligraphy",
                "gardens surrounding the National Palace Museum in Taipei, such as Zhishan Garden"
            ]
        },
        "alishan_national_scenic_area": {
            "name": "Alishan National Scenic Area",
            "aliases": ["阿里山國家風景區", "Mount Ali"],
            "location": "Chiayi, Taiwan",
            "prompts": [
                "sea of clouds phenomenon at Alishan National Scenic Area, viewed from high mountain peaks at sunrise or sunset",
                "Alishan Forest Railway trains, with their distinctive red carriages, winding through misty forests of ancient cypress and cedar trees",
                "sunrise views over Yushan (Jade Mountain), Taiwan's highest peak, from Alishan's Chushan or Ogasawara Mountain observation points",
                "tea plantations on the rolling hills of Alishan, known for its high-mountain oolong tea",
                "hiking trails through Alishan's giant tree groves and serene forests, like the Sister Ponds"
            ]
        },
        "shilin_night_market": {
            "name": "Shilin Night Market",
            "aliases": ["士林夜市"],
            "location": "Taipei, Taiwan",
            "prompts": [
                "bustling and vibrant atmosphere of Shilin Night Market in Taipei, one of Taiwan's largest and most famous night markets, packed with people",
                "a wide variety of Taiwanese street food stalls offering delicacies like oyster omelets, stinky tofu, and giant fried chicken cutlets at Shilin Night Market",
                "crowds of people exploring the maze-like alleys of Shilin Night Market, filled with food vendors, game stalls, and small shops",
                "brightly lit signs and food aromas filling the air at the lively Shilin Night Market",
                "underground food court area of Shilin Night Market offering a diverse range of local dishes"
            ]
        },
        "tokyo_tower": {
            "name": "Tokyo Tower",
            "aliases": ["東京タワー", "Tokyo Tower Landmark"],
            "location": "Tokyo, Japan",
            "prompts": [
                "a photo of Tokyo Tower in Japan, its distinctive red and white lattice structure inspired by the Eiffel Tower",
                "Tokyo Tower with its vibrant orange and white paint scheme, standing out against the Tokyo skyline",
                "the iconic Tokyo Tower illuminated at night, often with seasonal or special light displays",
                "view from Tokyo Tower's observation deck overlooking the sprawling metropolis of Tokyo",
                "Tokyo Tower as a symbol of post-war Japan's rebirth and a prominent communications tower"
            ]
        },
        "mount_fuji": {
            "name": "Mount Fuji",
            "aliases": ["富士山", "Fujisan"],
            "location": "Honshu, Japan",
            "prompts": [
                "a photo of Mount Fuji in Japan, its perfectly symmetrical snow-capped conical volcanic peak",
                "the snow-capped peak of Mount Fuji, often with a clear blue sky or surrounded by clouds",
                "Mount Fuji with cherry blossoms (sakura) in the foreground during spring, or reflected in one of the Fuji Five Lakes (Fujigoko)",
                "iconic view of Mount Fuji from the Chureito Pagoda",
                "Mount Fuji, Japan's highest mountain and an active stratovolcano, a symbol of Japan"
            ]
        },
        "kinkaku_ji": {
            "name": "Kinkaku-ji",
            "aliases": ["Golden Pavilion", "金閣寺", "Rokuon-ji"],
            "location": "Kyoto, Japan",
            "prompts": [
                "a photo of Kinkaku-ji temple (Golden Pavilion) in Kyoto, its top two floors completely covered in gold leaf",
                "the Golden Pavilion Kinkaku-ji reflected perfectly in the Mirror Pond (Kyōko-chi) surrounding it",
                "Kinkaku-ji, a Zen Buddhist temple in Kyoto, set within a beautiful Japanese stroll garden",
                "the three-tiered structure of Kinkaku-ji, each floor representing a different architectural style",
                "Kinkaku-ji in autumn with colorful foliage, or in winter dusted with snow"
            ]
        },
        "fushimi_inari_shrine": {
            "name": "Fushimi Inari Shrine",
            "aliases": ["伏見稲荷大社", "Thousand Torii Gates"],
            "location": "Kyoto, Japan",
            "prompts": [
                "a photo of Fushimi Inari Shrine in Kyoto, famous for its thousands of vibrant red/orange torii gates",
                "thousands of vermilion torii gates forming tunnels along a network of trails at Fushimi Inari Taisha",
                "pathways lined with densely packed red torii gates winding up a mountainside at Fushimi Inari Shrine",
                "fox statues (kitsune), messengers of Inari, found throughout Fushimi Inari Shrine",
                "the main shrine buildings of Fushimi Inari at the base of the mountain, with more torii gates leading upwards"
            ]
        },
        "shibuya_crossing": {
            "name": "Shibuya Crossing",
            "aliases": ["Shibuya Scramble Crossing", "澀谷十字路口"],
            "location": "Tokyo, Japan",
            "prompts": [
                "massive pedestrian scramble at Shibuya Crossing, Tokyo, with crowds of people crossing from all directions, surrounded by neon signs and large video screens",
                "bird's-eye view of the crowded Shibuya intersection with its iconic starburst pedestrian walkways and Hachiko statue nearby",
                "throngs of people crossing the multi-directional Shibuya intersection surrounded by modern buildings, department stores, and vibrant advertisements",
                "Shibuya Crossing at night with dazzling, vibrant lights from billboards and a continuous sea of people",
                "the energetic and iconic Shibuya Crossing, a symbol of modern Tokyo and urban life"
            ]
        },
        "tokyo_skytree": {
            "name": "Tokyo Skytree",
            "aliases": ["東京晴空塔"],
            "location": "Tokyo, Japan",
            "prompts": [
                "the slender, futuristic Tokyo Skytree, a broadcasting and observation tower in Sumida, Tokyo, with its lattice steel structure",
                "Tokyo Skytree illuminated in its signature pale blue (Iki) or purple (Miyabi) lights against the night sky",
                "modern lattice structure of the Tokyo Skytree, the world's tallest freestanding tower, dominating the city's skyline",
                "panoramic view from Tokyo Skytree's Tembo Deck or Tembo Galleria overlooking the sprawling city of Tokyo and beyond",
                "Tokyo Skytree with the Sumida River and surrounding urban landscape"
            ]
        },
        "senso_ji_temple": {
            "name": "Senso-ji Temple",
            "aliases": ["淺草寺", "Asakusa Kannon Temple"],
            "location": "Tokyo, Japan",
            "prompts": [
                "the vibrant red Senso-ji Temple in Asakusa, Tokyo, with its iconic Kaminarimon (Thunder Gate) featuring a massive red paper lantern",
                "traditional Japanese temple architecture of Senso-ji, Tokyo's oldest temple, including its five-story pagoda and main hall",
                "incense smoke billowing from a large cauldron and worshippers at Senso-ji Temple",
                "Nakamise-dori, the bustling market street leading to Senso-ji Temple, lined with traditional souvenir stalls and food vendors",
                "the Hozomon Gate with its giant waraji (straw sandals) at Senso-ji Temple"
            ]
        },
        "osaka_castle": {
            "name": "Osaka Castle",
            "aliases": ["大阪城", "Osaka-jo"],
            "location": "Osaka, Japan",
            "prompts": [
                "the majestic Osaka Castle with its distinctive white walls, green tiled roofs, golden embellishments, and surrounding moat and massive stone walls",
                "Osaka Castle Park with the imposing castle keep (tenshukaku) in the background, especially beautiful during cherry blossom season or autumn foliage",
                "historic Japanese castle, Osaka Castle, reconstructed with modern interiors, showcasing its historical significance",
                "Osaka Castle illuminated at night, reflecting in its expansive moat, creating a stunning visual",
                "view of Osaka Castle from Nishinomaru Garden, offering a picturesque perspective"
            ]
        },
        "dotonbori": {
            "name": "Dotonbori",
            "aliases": ["道頓堀"],
            "location": "Osaka, Japan",
            "prompts": [
                "the vibrant and eclectic Dotonbori entertainment district in Osaka, famous for its extravagant, oversized 3D signage like the Glico Running Man and Kani Doraku crab",
                "canal view of Dotonbori at night with a dazzling array of neon lights from billboards and signs reflecting on the Dotonbori River",
                "colorful and brightly lit billboards, including the iconic Glico Running Man, lining the Dotonbori canal, a symbol of Osaka's energy",
                "bustling atmosphere of Dotonbori with street food stalls offering takoyaki and okonomiyaki, and throngs of people",
                "the Ebisu Bridge over the Dotonbori canal, a popular meeting spot and photo location"
            ]
        },
        "arashiyama_bamboo_grove": {
            "name": "Arashiyama Bamboo Grove",
            "aliases": ["嵐山竹林", "Sagano Bamboo Forest"],
            "location": "Kyoto, Japan",
            "prompts": [
                "towering stalks of green bamboo creating a dense, immersive canopy over a pathway in Arashiyama Bamboo Grove, Kyoto",
                "sunlight filtering magically through the leaves of the tall, closely packed bamboo forest in Arashiyama",
                "serene and tranquil walking path through the iconic Arashiyama Bamboo Grove, with the distinctive sound of rustling bamboo leaves",
                "people in traditional kimono or yukata walking through the Arashiyama Bamboo Grove",
                "the unique, otherworldly atmosphere of the Arashiyama Bamboo Grove, a natural wonder"
            ]
        },
        "itsukushima_shrine": {
            "name": "Itsukushima Shrine",
            "aliases": ["嚴島神社", "Miyajima Shrine"],
            "location": "Miyajima Island, Hiroshima, Japan",
            "prompts": [
                "the iconic vermilion 'floating' torii gate of Itsukushima Shrine in Miyajima, appearing to float on the water during high tide",
                "Itsukushima Shrine complex, a UNESCO World Heritage site, built on stilts over the sea, with its distinctive red-lacquered corridors and halls",
                "sacred wild deer roaming freely around Miyajima Island with the Itsukushima Shrine and its torii gate in the background",
                "Itsukushima Shrine and its torii gate illuminated at night, creating a mystical scene",
                "view of Itsukushima Shrine from Mount Misen or from a ferry approaching Miyajima Island"
            ]
        },
        "gyeongbokgung_palace": {
            "name": "Gyeongbokgung Palace",
            "aliases": ["경복궁", "Gyeongbok Palace"],
            "location": "Seoul, South Korea",
            "prompts": [
                "a photo of Gyeongbokgung Palace in Seoul, the largest and most stunning of Seoul's five grand Joseon Dynasty palaces",
                "the main royal palace of the Joseon dynasty, Gyeongbokgung, with its grand Gwanghwamun Gate and Heungnyemun Gate",
                "traditional Korean architecture at Gyeongbokgung Palace, featuring intricate Dancheong (colorful painted patterns) on wooden structures and elegant curved roofs",
                "the impressive Geunjeongjeon Hall (Imperial Throne Hall) and the picturesque Gyeonghoeru Pavilion (Royal Banquet Hall) on a pond at Gyeongbokgung Palace",
                "Changing of the Royal Guard ceremony (Sumunjang Gyedaeui) taking place at Gyeongbokgung Palace"
            ]
        },
        "n_seoul_tower": {
            "name": "N Seoul Tower",
            "aliases": ["N서울타워", "YTN Seoul Tower", "Namsan Tower"],
            "location": "Seoul, South Korea",
            "prompts": [
                "a photo of N Seoul Tower perched atop Namsan Mountain, offering panoramic views of Seoul",
                "N Seoul Tower illuminated with vibrant LED lights at night, changing colors and patterns, visible from across the city",
                "the iconic N Seoul Tower in South Korea, a popular landmark with its observation deck and 'love locks' fences",
                "view from the N Seoul Tower looking down on the sprawling cityscape of Seoul, day or night",
                "cable car ascending Namsan Mountain towards the N Seoul Tower"
            ]
        },
        "bukchon_hanok_village": {
            "name": "Bukchon Hanok Village",
            "aliases": ["북촌한옥마을", "Traditional Korean Village"],
            "location": "Seoul, South Korea",
            "prompts": [
                "a photo of Bukchon Hanok Village in Seoul, a preserved traditional Korean village with hundreds of hanok (traditional Korean houses)",
                "traditional Korean houses (hanok) in Bukchon Village, featuring distinctive tiled roofs, wooden beams, and courtyards, nestled on a hillside",
                "narrow, winding alleyways and cobblestone streets of Bukchon Hanok Village, offering glimpses into historic Seoul",
                "view of modern Seoul skyline contrasting with the traditional rooftops of Bukchon Hanok Village",
                "people wearing Hanbok (traditional Korean attire) walking through Bukchon Hanok Village"
            ]
        },
        "myeongdong_shopping_street": {
            "name": "Myeongdong Shopping Street",
            "aliases": ["명동 쇼핑거리", "Myeong-dong"],
            "location": "Seoul, South Korea",
            "prompts": [
                "bustling Myeongdong shopping street in Seoul, a paradise for cosmetics, fashion, and K-pop merchandise, packed with shoppers and vibrant storefronts",
                "crowds of shoppers navigating the pedestrian-friendly streets of Myeongdong, with bright neon signs and music from stores creating a lively atmosphere, especially at night",
                "vibrant street food scene in Myeongdong, Seoul, with numerous stalls offering popular Korean snacks like tteokbokki, gyeranppang, and tornado potatoes",
                "Myeongdong Cathedral, a historic Gothic-style church, standing amidst the bustling modern shopping district",
                "large department stores and international brands alongside local boutiques in Myeongdong"
            ]
        },
        "dmz_korea": {
            "name": "DMZ (Korean Demilitarized Zone)",
            "aliases": ["韓國非軍事區", "Panmunjom", "JSA"],
            "location": "Gyeonggi-do, South Korea / North Korea",
            "prompts": [
                "the heavily fortified Korean Demilitarized Zone (DMZ) separating North and South Korea, with barbed wire fences and guard posts",
                "Joint Security Area (JSA) or Panmunjom at the DMZ, with soldiers from North and South Korea facing each other across the Military Demarcation Line",
                "the iconic blue conference buildings straddling the border within the JSA at the DMZ",
                "tense and somber atmosphere of the DMZ, a symbol of the Korean War and the divided peninsula",
                "observation posts like Dora Observatory overlooking North Korean territory from the DMZ"
            ]
        },
        "busan_gamcheon_culture_village": {
            "name": "Busan Gamcheon Culture Village",
            "aliases": ["부산 감천문화마을", "Machu Picchu of Busan", "Taegukdo Village"],
            "location": "Busan, South Korea",
            "prompts": [
                "colorful houses built in terraced fashion on a steep hillside in Gamcheon Culture Village, Busan, often called the 'Machu Picchu of Busan'",
                "narrow, winding alleyways adorned with vibrant street art, colorful murals, and art installations in Gamcheon Culture Village",
                "panoramic view of the brightly painted houses of Gamcheon Culture Village cascading down to the sea, creating a unique urban landscape",
                "sculptures like 'The Little Prince and the Desert Fox' overlooking the village in Gamcheon Culture Village",
                "artistic and quirky atmosphere of Gamcheon Culture Village, a regenerated slum transformed into an art hub"
            ]
        },
        "jeju_island": {
            "name": "Jeju Island",
            "aliases": ["제주도", "Jejudo"],
            "location": "Jeju Province, South Korea",
            "prompts": [
                "volcanic landscape of Jeju Island, South Korea, a UNESCO World Heritage site, featuring black basalt rock formations (like Jusangjeolli Cliff) and lush greenery",
                "Seongsan Ilchulbong (Sunrise Peak), a dramatic tuff cone crater rising from the sea on the eastern coast of Jeju Island",
                "beautiful sandy beaches like Hyeopjae Beach, waterfalls such as Cheonjeyeon Falls, and lava tubes like Manjanggul Cave on Jeju Island",
                "Hallasan National Park, home to South Korea's highest mountain, Hallasan, a dormant shield volcano, on Jeju Island",
                "Dol Hareubang (stone grandfathers), iconic black lava stone statues found throughout Jeju Island"
            ]
        },
        "changdeokgung_palace_secret_garden": {
            "name": "Changdeokgung Palace & Secret Garden",
            "aliases": ["창덕궁과 후원", "Donggwol (East Palace)"],
            "location": "Seoul, South Korea",
            "prompts": [
                "the beautiful Changdeokgung Palace, a UNESCO World Heritage site in Seoul, known for its harmonious design that blends with the natural landscape",
                "traditional Korean palace architecture of Changdeokgung, including the Injeongjeon (main throne hall) and Donhwamun (main gate)",
                "the serene and picturesque Secret Garden (Huwon) of Changdeokgung Palace, a vast rear garden with pavilions, ponds, and ancient trees, requiring a guided tour",
                "Buyongjeong Pavilion and Buyongji Pond in the Secret Garden of Changdeokgung Palace",
                "Changdeokgung Palace as one of the best-preserved Joseon Dynasty palaces"
            ]
        },
        "great_wall": {
            "name": "Great Wall of China",
            "aliases": ["长城", "The Great Wall", "萬里長城"],
            "location": "China",
            "prompts": [
                "a photo of the Great Wall of China, the massive ancient fortification winding across rugged mountains and diverse terrains",
                "the Great Wall stretching for thousands of miles, with watchtowers and battlements, a testament to ancient Chinese engineering",
                "sections of the Great Wall like Badaling or Mutianyu, showing its impressive scale and historical significance",
                "the Great Wall of China snaking along steep mountain ridges, often covered in snow or surrounded by lush greenery",
                "iconic man-made structure, the Great Wall, visible from afar, symbolizing China's strength and history"
            ]
        },
        "forbidden_city": {
            "name": "Forbidden City",
            "aliases": ["紫禁城", "Palace Museum", "故宫博物院"],
            "location": "Beijing, China",
            "prompts": [
                "a photo of the Forbidden City in Beijing, the vast imperial palace complex with its iconic red walls and yellow-tiled roofs",
                "the imperial palace complex of the Forbidden City, former home to Chinese emperors, showcasing classical Chinese palatial architecture",
                "courtyards, halls, and gates of the Forbidden City, such as the Hall of Supreme Harmony, arranged along a central axis",
                "intricate details and symbolic ornamentation of the Forbidden City's buildings, reflecting imperial power and Chinese cosmology",
                "the Meridian Gate (Wumen), the main entrance to the Forbidden City, or the Corner Towers with their complex roof structures"
            ]
        },
        "terracotta_army": {
            "name": "Terracotta Army",
            "aliases": ["兵马俑", "Terracotta Warriors and Horses"],
            "location": "Xi'an, China",
            "prompts": [
                "a photo of the Terracotta Army in Xi'an, rows of life-sized terracotta warriors, archers, and chariots in vast pits",
                "thousands of life-sized terracotta warriors and horses, each with unique facial expressions and armor, part of Qin Shi Huang's mausoleum",
                "archaeological site of the Terracotta Army, a UNESCO World Heritage site, showcasing the incredible artistry and scale of ancient Chinese burial practices",
                "close-up of the detailed faces and uniforms of the Terracotta Warriors",
                "excavation pits containing the Terracotta Army, demonstrating the ongoing discovery and preservation efforts"
            ]
        },
        "the_bund": {
            "name": "The Bund",
            "aliases": ["外滩", "Waitan"],
            "location": "Shanghai, China",
            "prompts": [
                "a photo of The Bund waterfront in Shanghai, with its impressive collection of historic colonial-era buildings in various architectural styles",
                "skyline of The Bund featuring iconic early 20th-century buildings, contrasting with the modern skyscrapers of Pudong across the Huangpu River",
                "The Bund overlooking the Huangpu River, with boats and ferries, and the Oriental Pearl Tower and Shanghai Tower visible on the Pudong side",
                "pedestrian promenade along The Bund, a popular spot for tourists and locals to view Shanghai's historic and modern skylines",
                "The Bund at night, with both the historic buildings and the Pudong skyline brilliantly illuminated"
            ]
        },
        "li_river_guilin": {
            "name": "Li River, Guilin",
            "aliases": ["漓江", "Guilin Karst Landscape"],
            "location": "Guilin, Guangxi, China",
            "prompts": [
                "surreal karst mountain landscape along the Li River in Guilin, China, with picturesque limestone peaks rising dramatically from the flat terrain",
                "bamboo rafts and cruise boats navigating the scenic Li River, surrounded by mist-shrouded, unusually shaped karst formations",
                "iconic view of the Li River, often depicted in traditional Chinese landscape paintings, particularly the scene on the 20 Yuan banknote (Yellow Cloth Shoal)",
                "cormorant fishermen on bamboo rafts on the Li River",
                "lush green vegetation covering the steep karst hills along the banks of the winding Li River"
            ]
        },
        "potala_palace": {
            "name": "Potala Palace",
            "aliases": ["布達拉宮"],
            "location": "Lhasa, Tibet, China",
            "prompts": [
                "the majestic Potala Palace, former winter residence of the Dalai Lama, dramatically situated on Red Hill (Marpo Ri) in Lhasa, Tibet",
                "distinctive white (White Palace) and red (Red Palace) architecture of the Potala Palace against a clear blue sky and surrounding mountains",
                "the imposing Potala Palace, a UNESCO World Heritage site, a symbol of Tibetan Buddhism and a marvel of Tibetan architecture with its many chapels and statues",
                "Potala Palace with prayer flags fluttering in the foreground",
                "view of Potala Palace from afar, showcasing its grand scale and dominant position over Lhasa"
            ]
        },
        "zhangjiajie_national_forest_park": {
            "name": "Zhangjiajie National Forest Park",
            "aliases": ["張家界國家森林公園", "Avatar Mountains"],
            "location": "Hunan, China",
            "prompts": [
                "tall, pillar-like quartz-sandstone formations, often shrouded in mist, in Zhangjiajie National Forest Park, inspiration for the 'Avatar Hallelujah Mountains'",
                "the 'Avatar Hallelujah Mountains' (formerly Southern Sky Column) in Zhangjiajie, known for their gravity-defying appearance and lush vegetation",
                "glass bridges like the Zhangjiajie Grand Canyon Glass Bridge, and cable cars offering stunning, vertigo-inducing views of the unique landscape of Zhangjiajie",
                "Bailong Elevator (Hundred Dragons Elevator), a massive glass elevator built onto the side of a cliff in Zhangjiajie",
                "dense forests and deep ravines characterize the otherworldly scenery of Zhangjiajie National Forest Park"
            ]
        },
        "west_lake_hangzhou": {
            "name": "West Lake, Hangzhou",
            "aliases": ["西湖"],
            "location": "Hangzhou, China",
            "prompts": [
                "the serene and beautiful West Lake in Hangzhou, China, a UNESCO World Heritage site, famous for its scenic beauty, pagodas, islands, and causeways",
                "iconic landmarks of West Lake such as the Broken Bridge (Duan Qiao), Leifeng Pagoda, Su Causeway, and Bai Causeway",
                "traditional Chinese pavilions, arched bridges, and gardens surrounding West Lake, often depicted in Chinese art and poetry",
                "boats gently gliding on West Lake, with willow trees lining its shores and lotus flowers blooming in summer",
                "Three Ponds Mirroring the Moon, one of the most famous sights of West Lake"
            ]
        },
        "summer_palace_beijing": {
            "name": "Summer Palace, Beijing",
            "aliases": ["頤和園", "Yiheyuan"],
            "location": "Beijing, China",
            "prompts": [
                "the vast imperial garden of the Summer Palace in Beijing, a UNESCO World Heritage site, featuring Kunming Lake and Longevity Hill",
                "ornate palaces, temples, bridges, and pavilions within the Summer Palace, such as the Marble Boat, the Long Corridor (Chang Lang), and the Tower of Buddhist Incense (Foxiang Ge)",
                "traditional Chinese landscape garden design of the Summer Palace, showcasing harmony between man-made structures and nature",
                "Kunming Lake in the Summer Palace, with the Seventeen-Arch Bridge leading to Nanhu Island",
                "Longevity Hill at the Summer Palace, crowned by impressive imperial buildings"
            ]
        },
        "petronas_towers": {
            "name": "Petronas Twin Towers",
            "aliases": ["KLCC Twin Towers", "Menara Petronas", "Petronas Towers"],
            "location": "Kuala Lumpur, Malaysia",
            "prompts": [
                "a photo of the Petronas Twin Towers in Kuala Lumpur, the world's tallest twin skyscrapers with their distinctive postmodern Islamic architectural motifs",
                "the iconic Petronas Towers connected by a double-decker skybridge on the 41st and 42nd floors",
                "Petronas Twin Towers brilliantly illuminated at night, a striking landmark in Kuala Lumpur's skyline",
                "the multi-faceted, tapering design of the Petronas Twin Towers, inspired by Islamic geometric patterns",
                "view of the Petronas Twin Towers from KLCC Park, with its fountains and gardens"
            ]
        },
        "marina_bay_sands": {
            "name": "Marina Bay Sands",
            "aliases": ["MBS", "Singapore Skypark"],
            "location": "Singapore",
            "prompts": [
                "a photo of Marina Bay Sands in Singapore, the iconic integrated resort featuring three soaring hotel towers topped by a massive cantilevered SkyPark",
                "the Marina Bay Sands resort with its distinctive three towers and the ship-like Sands SkyPark housing an infinity pool, observation deck, and gardens",
                "Marina Bay Sands overlooking the Singapore skyline and Marina Bay, often seen with the ArtScience Museum in the foreground",
                "the impressive architecture of Marina Bay Sands, a symbol of modern Singapore, illuminated at night",
                "Spectra light and water show in front of Marina Bay Sands"
            ]
        },
        "gardens_by_the_bay": {
            "name": "Gardens by the Bay",
            "aliases": ["Singapore Gardens", "Supertree Grove"],
            "location": "Singapore",
            "prompts": [
                "a photo of Gardens by the Bay in Singapore, a futuristic nature park known for its iconic Supertree Grove",
                "the towering Supertree Grove at Gardens by the Bay, tree-like vertical gardens that light up spectacularly during the Garden Rhapsody show at night",
                "Flower Dome and Cloud Forest, two massive cooled conservatories at Gardens by the Bay, housing diverse plant life from around the world",
                "the OCBC Skyway, an aerial walkway offering panoramic views of the Supertrees and surrounding gardens",
                "lush landscapes and innovative horticultural displays within Gardens by the Bay"
            ]
        },
        "taj_mahal": {
            "name": "Taj Mahal",
            "aliases": ["Crown of Palaces", "Mumtaz Mahal"],
            "location": "Agra, India",
            "prompts": [
                "a photo of the Taj Mahal in Agra, the iconic ivory-white marble mausoleum renowned for its beauty and symmetry",
                "the ivory-white marble mausoleum of the Taj Mahal, a UNESCO World Heritage site, built by Mughal emperor Shah Jahan",
                "Taj Mahal reflected perfectly in its long सामने (frontal) water feature (reflecting pool), flanked by symmetrical gardens",
                "the intricate marble inlay work (pietra dura) and calligraphy on the facade of the Taj Mahal",
                "the central dome and four minarets of the Taj Mahal, a masterpiece of Mughal architecture"
            ]
        },
        "angkor_wat": {
            "name": "Angkor Wat",
            "aliases": ["吳哥窟", "City of Temples"],
            "location": "Siem Reap, Cambodia",
            "prompts": [
                "the magnificent temple complex of Angkor Wat in Cambodia, a UNESCO World Heritage site, with its iconic five lotus-bud shaped towers reflecting in a surrounding moat",
                "intricate bas-reliefs and stone carvings depicting Hindu epics like the Ramayana and Mahabharata, and apsaras (celestial dancers) at Angkor Wat",
                "sunrise over Angkor Wat, casting a golden glow on its ancient stone structures, the world's largest religious monument",
                "the grand scale and symmetrical design of Angkor Wat, an architectural marvel of the Khmer Empire",
                "causeway leading to the main entrance of Angkor Wat, often with monks in saffron robes"
            ]
        },
        "ha_long_bay": {
            "name": "Ha Long Bay",
            "aliases": ["下龍灣", "Vịnh Hạ Long"],
            "location": "Quảng Ninh Province, Vietnam",
            "prompts": [
                "thousands of towering limestone karsts and islets rising dramatically from the emerald green waters of Ha Long Bay, Vietnam, a UNESCO World Heritage site",
                "traditional Vietnamese junk boats sailing majestically through the stunning seascape of Ha Long Bay",
                "caves and grottoes, such as Thien Cung Cave or Dau Go Cave, hidden within the limestone islands of Ha Long Bay",
                "floating fishing villages and pearl farms nestled among the karsts in Ha Long Bay",
                "misty and ethereal atmosphere of Ha Long Bay, especially at dawn or dusk"
            ]
        },
        "mount_everest": {
            "name": "Mount Everest",
            "aliases": ["聖母峰", "Sagarmatha", "Chomolungma"],
            "location": "Mahalangur Himal, Nepal/China border",
            "prompts": [
                "the snow-covered, pyramid-shaped peak of Mount Everest, the world's highest mountain, towering above the Himalayan range",
                "climbers and expeditions at Mount Everest Base Camp, with prayer flags and views of the Khumbu Icefall",
                "majestic and formidable landscape of Mount Everest and surrounding Himalayan peaks like Lhotse and Nuptse, often with wispy clouds",
                "the challenging south face or north face routes leading to the summit of Mount Everest",
                "breathtaking aerial view of Mount Everest, highlighting its immense scale and icy terrain"
            ]
        },
        "bagan": {
            "name": "Bagan",
            "aliases": ["蒲甘", "Pagan"],
            "location": "Mandalay Region, Myanmar",
            "prompts": [
                "thousands of ancient Buddhist temples, pagodas, and stupas, mostly in reddish-brown brick, spread across the plains of Bagan, Myanmar, a UNESCO World Heritage site",
                "hot air balloons drifting gracefully over the temple-studded landscape of Bagan at sunrise or sunset, creating a magical scene",
                "archaeological zone of Bagan with its diverse stupas (like Shwezigon Pagoda) and temples (like Ananda Temple) dating back to the 9th-13th centuries",
                "sunlight illuminating the ancient temples of Bagan, with intricate carvings and Buddha statues inside",
                "view from a high temple in Bagan, offering a panoramic vista of countless religious structures"
            ]
        },
        "grand_palace_wat_phra_kaew": {
            "name": "The Grand Palace & Wat Phra Kaew",
            "aliases": ["曼谷大皇宮與玉佛寺", "Royal Palace Bangkok"],
            "location": "Bangkok, Thailand",
            "prompts": [
                "ornate and glittering architecture of the Grand Palace in Bangkok, former residence of the Kings of Siam, with its intricate details and golden spires",
                "Wat Phra Kaew (Temple of the Emerald Buddha) within the Grand Palace complex, housing the highly revered Emerald Buddha statue carved from a single jade stone",
                "intricate details, golden chedis (stupas), colorful mosaics made of glass and porcelain, and mythical guardian statues (yakshas) of Thai traditional architecture at the Grand Palace",
                "the dazzling exteriors of the royal halls and temples within the Grand Palace grounds",
                "crowds of visitors and worshippers at the magnificent Grand Palace and Wat Phra Kaew"
            ]
        }
    },

    # 歐洲地標
    "europe": {
        "eiffel_tower": {
            "name": "Eiffel Tower",
            "aliases": ["Tour Eiffel", "The Iron Lady"],
            "location": "Paris, France",
            "prompts": [
                "a photo of the Eiffel Tower in Paris, the iconic wrought-iron lattice tower on the Champ de Mars",
                "the iconic Eiffel Tower structure, its intricate ironwork and graceful curves against the Paris skyline",
                "Eiffel Tower illuminated at night with its sparkling light show, a beacon in the City of Lights",
                "view from the top of the Eiffel Tower overlooking Paris, including the Seine River and landmarks like the Arc de Triomphe",
                "Eiffel Tower seen from the Trocadéro, providing a classic photographic angle"
            ]
        },
        "louvre_museum": {
            "name": "Louvre Museum",
            "aliases": ["Musée du Louvre", "The Louvre Pyramid"],
            "location": "Paris, France",
            "prompts": [
                "a photo of the Louvre Museum in Paris, with its iconic glass pyramid designed by I. M. Pei contrasting with the historic Louvre Palace",
                "the Louvre Pyramid at the entrance of the museum, reflecting the sky and surrounding palace wings",
                "exterior of the historic Louvre Palace, a former royal palace, now one of the world's largest art museums",
                "crowds of visitors entering the Louvre Museum through the pyramid or its underground entrance",
                "the Louvre Museum at night, with the pyramid and palace illuminated"
            ]
        },
        "mont_saint_michel": {
            "name": "Mont Saint-Michel",
            "aliases": ["Saint Michael's Mount (France)", "Abbaye du Mont-Saint-Michel"],
            "location": "Normandy, France",
            "prompts": [
                "a photo of Mont Saint-Michel in France, the stunning tidal island commune topped by a medieval Benedictine abbey",
                "the tidal island and abbey of Mont Saint-Michel rising dramatically from the bay, surrounded by water at high tide or sand flats at low tide",
                "Mont Saint-Michel at high tide, appearing as a fairytale castle floating on the water",
                "the Gothic architecture of the Abbey of Mont Saint-Michel, with its towering spire and fortified walls",
                "narrow winding streets and historic buildings leading up to the abbey on Mont Saint-Michel"
            ]
        },
        "arc_de_triomphe": {
            "name": "Arc de Triomphe",
            "aliases": ["Triumphal Arch of the Star", "Arc de Triomphe de l'Étoile"],
            "location": "Paris, France",
            "prompts": [
                "a photo of the Arc de Triomphe in Paris, the monumental triumphal arch at the center of Place Charles de Gaulle (Place de l'Étoile)",
                "the iconic Arc de Triomphe at the western end of the Champs-Élysées, adorned with intricate sculptures depicting Napoleonic victories",
                "view from the top of the Arc de Triomphe, looking down the twelve radiating avenues, including the Champs-Élysées towards the Louvre",
                "the Eternal Flame burning beneath the Arc de Triomphe at the Tomb of the Unknown Soldier",
                "Arc de Triomphe illuminated at night, a symbol of French national pride"
            ]
        },
        "big_ben": {
            "name": "Big Ben (Elizabeth Tower)", # Clarified name
            "aliases": ["Elizabeth Tower", "Westminster Clock Tower", "Clock Tower, Palace of Westminster"],
            "location": "London, UK",
            "prompts": [
                "a photo of Big Ben (Elizabeth Tower) clock tower with the Houses of Parliament (Palace of Westminster) in London, on the bank of the River Thames",
                "the iconic Gothic Revival clock tower Big Ben with its four massive, illuminated clock faces in London",
                "Big Ben tower with its distinctive golden clock face, intricate stonework, and cast-iron spire, a symbol of London and the UK",
                "the famous Westminster clock tower Big Ben in Gothic revival style, meticulously restored with its original blue clock hands",
                "close-up of Big Ben's clock face showing the Roman numerals and detailed craftsmanship"
            ]
        },
        "stonehenge": {
            "name": "Stonehenge",
            "aliases": ["Prehistoric Monument", "Ring of Stones"],
            "location": "Wiltshire, UK",
            "prompts": [
                "a photo of Stonehenge in the UK, the mysterious prehistoric monument of massive standing stones arranged in a circular formation",
                "the prehistoric stone circle of Stonehenge, a UNESCO World Heritage site, with its sarsens and bluestones",
                "Stonehenge at sunset or sunrise, with dramatic lighting casting long shadows over the ancient stones",
                "the unique trilithons (two upright stones topped by a lintel) of Stonehenge",
                "the enigmatic landscape surrounding Stonehenge on Salisbury Plain"
            ]
        },
        "tower_of_london": {
            "name": "Tower of London",
            "aliases": ["His Majesty's Royal Palace and Fortress of the Tower of London", "The Tower"],
            "location": "London, UK",
            "prompts": [
                "a photo of the Tower of London, the historic medieval castle on the north bank of the River Thames",
                "the imposing White Tower, the central keep of the Tower of London, with its Norman architecture",
                "Crown Jewels of the United Kingdom on display within the Tower of London",
                "Yeoman Warders ('Beefeaters') in their traditional Tudor uniforms at the Tower of London",
                "Traitors' Gate at the Tower of London, a famous water gate leading from the Thames"
            ]
        },
        "buckingham_palace": {
            "name": "Buckingham Palace",
            "aliases": ["British Royal Residence", "The Palace"],
            "location": "London, UK",
            "prompts": [
                "a photo of Buckingham Palace in London, the official residence and administrative headquarters of the monarch of the United Kingdom",
                "the iconic facade of Buckingham Palace with the Queen's Guard (King's Guard) in their red tunics and bearskin hats",
                "Changing of the Guard ceremony taking place in the forecourt of Buckingham Palace, a popular tourist attraction",
                "the Victoria Memorial statue in front of Buckingham Palace",
                "Buckingham Palace with the Royal Standard flag flying, indicating the monarch is in residence"
            ]
        },
        "colosseum": {
            "name": "Colosseum",
            "aliases": ["Roman Colosseum", "Flavian Amphitheatre", "Colosseo"],
            "location": "Rome, Italy",
            "prompts": [
                "a photo of the Colosseum in Rome, the massive ancient Roman amphitheater, an icon of Imperial Rome",
                "the ancient Roman Colosseum structure, with its elliptical shape, tiered seating, and arched exterior, partly in ruins",
                "historic Colosseum amphitheater in Italy, where gladiatorial contests and public spectacles were held",
                "the interior of the Colosseum, showing the hypogeum (underground structures) and remaining seating areas",
                "the Colosseum illuminated at night, a powerful symbol of Roman history"
            ]
        },
        "leaning_tower_of_pisa": {
            "name": "Leaning Tower of Pisa",
            "aliases": ["Torre pendente di Pisa", "Tower of Pisa"],
            "location": "Pisa, Tuscany, Italy",
            "prompts": [
                "a photo of the Leaning Tower of Pisa in Italy, the iconic white marble freestanding bell tower (campanile) of Pisa Cathedral, famous for its significant unintended tilt",
                "the world-famous leaning cylindrical campanile of Pisa Cathedral, built of white marble with Romanesque architecture and multiple tiers of arched colonnades, noticeably tilted to one side",
                "tourists taking humorous forced perspective photos with the Leaning Tower of Pisa",
                "the Leaning Tower of Pisa located in the Piazza dei Miracoli (Square of Miracles), alongside the Duomo (cathedral) and Baptistery",
                "the white marble cylindrical structure of the Leaning Tower of Pisa with its distinctive arched galleries, showing its dramatic lean",
                "the iconic leaning cylindrical bell tower with six tiers of open galleries with arches and columns, against a blue sky, in Pisa, Italy",
                "a global tourist landmark: Italy's Leaning Tower of Pisa, a tilted white marble tower with intricate Romanesque arcades",
                "side view emphasizing the dramatic four-degree angle of the Leaning Tower's tilt, showcasing its unique structural imbalance",
                "ornate white marble cylindrical tower with multiple levels of columns, visibly leaning to one side, a masterpiece of medieval engineering",
                "the famous tilted bell tower of Pisa with its many columned galleries viewed from below, highlighting its precarious stance",
                "detailed close-up of the Leaning Tower of Pisa's distinctive multi-tiered arched colonnades and ornate architectural details in white marble",
                "the freestanding belltower of Pisa Cathedral set on the green grass of Piazza dei Miracoli, with its visible foundation on the low side where it sinks into the ground",
                "a photo of the white marble Leaning Tower of Pisa, known for its nearly four-degree lean, a UNESCO World Heritage Site in Tuscany"
            ]
        },
        "trevi_fountain": {
            "name": "Trevi Fountain",
            "aliases": ["Fontana di Trevi"],
            "location": "Rome, Italy",
            "prompts": [
                "a photo of the Trevi Fountain in Rome, the largest Baroque fountain in the city and one of the most famous fountains in the world",
                "the spectacular Baroque Trevi Fountain with its grand sculptures, including Oceanus, tritons, and horses, set against the Palazzo Poli",
                "people throwing coins over their shoulders into the Trevi Fountain, a tradition said to ensure a return to Rome",
                "the Trevi Fountain illuminated at night, showcasing its dramatic statues and cascading water",
                "the vibrant turquoise water of the Trevi Fountain contrasting with its white travertine stone"
            ]
        },
        "st_peters_basilica": {
            "name": "St. Peter's Basilica",
            "aliases": ["Basilica di San Pietro", "Vatican Basilica"],
            "location": "Vatican City",
            "prompts": [
                "a photo of St. Peter's Basilica in Vatican City, the immense Renaissance church, one of the largest and most renowned churches in the world",
                "the magnificent dome of St. Peter's Basilica, designed by Michelangelo, dominating the skyline of Rome and Vatican City",
                "St. Peter's Square (Piazza San Pietro), designed by Bernini, with its grand colonnades, obelisk, and fountains, leading to the basilica",
                "the lavish interior of St. Peter's Basilica, featuring masterpieces like Michelangelo's Pietà and Bernini's Baldachin",
                "St. Peter's Basilica viewed from Via della Conciliazione or from the top of its dome"
            ]
        },
        "sagrada_familia": {
            "name": "Sagrada Familia",
            "aliases": ["Basílica de la Sagrada Família", "Gaudi's Church"],
            "location": "Barcelona, Spain",
            "prompts": [
                "a photo of Sagrada Familia in Barcelona, Antoni Gaudí's unfinished masterpiece of Catalan Modernism, a Roman Catholic minor basilica",
                "the unique and highly ornate architecture of Gaudí's Sagrada Familia, with its towering spires, intricate facades (Nativity, Passion, Glory), and organic forms",
                "construction cranes still present on the perpetually evolving Sagrada Familia",
                "the stunning interior of Sagrada Familia, with its tree-like columns and vibrant stained-glass windows creating a forest of light",
                "close-up details of the sculptural elements and symbolic decorations on the facades of Sagrada Familia"
            ]
        },
        "alhambra": {
            "name": "Alhambra",
            "aliases": ["Alhambra Palace", "The Red Fortress", "Alhambra of Granada"],
            "location": "Granada, Spain",
            "prompts": [
                "a photo of the Alhambra palace and fortress complex in Granada, Spain, a stunning example of Moorish architecture",
                "exquisite Moorish architecture of the Alhambra, featuring intricate stucco work, geometric tile patterns (azulejos), and delicate courtyards like the Court of the Lions",
                "courtyards, palaces (Nasrid Palaces), and gardens (Generalife) of the Alhambra, showcasing Islamic art and design",
                "the Alhambra perched on a hill overlooking the city of Granada, with the Sierra Nevada mountains in the background",
                "the red-hued walls of the Alcazaba fortress within the Alhambra complex"
            ]
        },
        "brandenburg_gate": {
            "name": "Brandenburg Gate",
            "aliases": ["Brandenburger Tor", "Berlin Gate"],
            "location": "Berlin, Germany",
            "prompts": [
                "a photo of the Brandenburg Gate in Berlin, the iconic neoclassical triumphal arch, a symbol of German reunification and peace",
                "the neoclassical Brandenburg Gate monument, with its Doric columns and the Quadriga (a chariot drawn by four horses) скульптура наверху",
                "Brandenburg Gate illuminated at night, standing at the end of Unter den Linden boulevard",
                "historical significance of the Brandenburg Gate, once a symbol of division during the Cold War",
                "Pariser Platz in front of the Brandenburg Gate, a lively public square"
            ]
        },
        "neuschwanstein_castle": {
            "name": "Neuschwanstein Castle",
            "aliases": ["Schloss Neuschwanstein", "Fairy Tale Castle", "Mad King Ludwig's Castle"],
            "location": "Bavaria, Germany",
            "prompts": [
                "a photo of Neuschwanstein Castle in Germany, the quintessential fairytale castle with its white limestone facade and blue turrets, inspiring Disney's Sleeping Beauty Castle",
                "the fairytale Neuschwanstein Castle dramatically nestled in the Bavarian Alps, perched on a rugged hill overlooking the Hohenschwangau valley",
                "Neuschwanstein Castle on a hill, often viewed from Marienbrücke (Mary's Bridge) for a classic postcard shot",
                "the Romanesque Revival architecture of Neuschwanstein Castle, commissioned by King Ludwig II of Bavaria",
                "Neuschwanstein Castle surrounded by autumn foliage or dusted with snow in winter"
            ]
        },
        "acropolis_of_athens": {
            "name": "Acropolis of Athens",
            "aliases": ["Ακρόπολη Αθηνών", "Parthenon", "Sacred Rock"],
            "location": "Athens, Greece",
            "prompts": [
                "a photo of the Acropolis of Athens in Greece, the ancient citadel located on a rocky outcrop above the city, crowned by the Parthenon",
                "the Parthenon, the iconic Doric temple dedicated to the goddess Athena, standing majestically on the Acropolis",
                "ancient ruins of the Acropolis overlooking the sprawling city of Athens, including the Erechtheion with its Porch of the Caryatids, and the Propylaea",
                "the Acropolis illuminated at night, a symbol of ancient Greek civilization and democracy",
                "view of the Acropolis from Filopappou Hill or Lycabettus Hill"
            ]
        },
        "santorini_oia": { # Specified Oia for iconic view
            "name": "Santorini (Oia)",
            "aliases": ["Thera", "Greek Islands", "Oia village Santorini"],
            "location": "Cyclades, Greece",
            "prompts": [
                "a photo of Oia village in Santorini island, Greece, famous for its whitewashed cave houses and blue-domed churches clinging to cliffs above the Aegean Sea",
                "iconic whitewashed villages with blue domes in Oia, Santorini, overlooking the volcanic caldera",
                "breathtaking sunset over the caldera in Oia, Santorini, a world-famous romantic spectacle",
                "narrow winding pathways and steps through the picturesque village of Oia",
                "bougainvillea flowers adding splashes of color to the white buildings of Santorini"
            ]
        },
        "canals_of_venice": {
            "name": "Canals of Venice",
            "aliases": ["Venetian Canals", "Rialto Bridge Venice"],
            "location": "Venice, Italy",
            "prompts": [
                "gondolas gliding gracefully through the narrow, winding canals of Venice, Italy, lined with historic, colorful buildings",
                "the Grand Canal in Venice, its main waterway, bustling with vaporettos (water buses), water taxis, and gondolas, spanned by the iconic Rialto Bridge",
                "romantic atmosphere of Venetian canals, reflecting the unique architecture of the sinking city, with smaller bridges connecting walkways",
                "picturesque scenes of Venice's canals, often with laundry hanging between buildings or flower boxes on windowsills",
                "Rio di Palazzo with the Bridge of Sighs connecting the Doge's Palace to the prisons"
            ]
        },
        "florence_cathedral_duomo": {
            "name": "Florence Cathedral (Duomo)",
            "aliases": ["Cattedrale di Santa Maria del Fiore", "Il Duomo di Firenze", "Brunelleschi's Dome"],
            "location": "Florence, Italy",
            "prompts": [
                "Brunelleschi's massive, iconic red-tiled dome atop the Florence Cathedral (Santa Maria del Fiore), dominating the city skyline of Florence",
                "the intricate Gothic and Renaissance facade of the Florence Duomo, made of white, green, and pink marble, alongside Giotto's Campanile (bell tower) and the Baptistery of St. John",
                "panoramic view of Florence from the top of Brunelleschi's Dome or Giotto's Campanile, showcasing the sea of red-tiled roofs and the Arno River",
                "the octagonal Florence Baptistery with its famous bronze doors, particularly Ghiberti's 'Gates of Paradise'",
                "the grand interior of Florence Cathedral, with its vast nave and frescoes, including Vasari's 'Last Judgment' inside the dome"
            ]
        },
        "anne_frank_house": {
            "name": "Anne Frank House",
            "aliases": ["Anne Frank Huis", "Secret Annex Amsterdam"],
            "location": "Amsterdam, Netherlands",
            "prompts": [
                "the unassuming exterior of the Anne Frank House, a canal house on the Prinsengracht in Amsterdam, where Anne Frank and her family hid during WWII",
                "the secret annex (Achterhuis) hidden behind a movable bookcase in the Anne Frank House, where the Frank family lived in hiding",
                "poignant historical site of the Anne Frank House, now a biographical museum dedicated to Jewish wartime diarist Anne Frank and the Holocaust",
                "long queues of visitors outside the Anne Frank House, waiting to enter the museum",
                "the preserved rooms and exhibits within the Anne Frank House, telling the story of Anne's life and diary"
            ]
        },
        "canals_of_amsterdam": {
            "name": "Canals of Amsterdam",
            "aliases": ["Grachtengordel Amsterdam", "Amsterdam Canal Ring"],
            "location": "Amsterdam, Netherlands",
            "prompts": [
                "picturesque canals of Amsterdam, part of the Grachtengordel (canal belt), a UNESCO World Heritage site, lined with narrow, gabled canal houses and houseboats",
                "bicycles parked along the charming bridges that cross Amsterdam's numerous canals, often adorned with flowers",
                "canal cruise boats navigating the historic waterways of Amsterdam, offering views of 17th-century architecture",
                "tree-lined canals of Amsterdam in different seasons, reflecting the elegant facades of the canal houses",
                "the distinctive architecture of Amsterdam's canal houses, with their narrow fronts and decorative gables"
            ]
        },
        "charles_bridge_prague": {
            "name": "Charles Bridge",
            "aliases": ["Karlův most", "Prague Stone Bridge"],
            "location": "Prague, Czech Republic",
            "prompts": [
                "the historic Charles Bridge in Prague, a medieval stone arch bridge adorned with 30 statues of saints, crossing the Vltava River",
                "view of Prague Castle and St. Vitus Cathedral from Charles Bridge, with artists, musicians, and vendors lining the bridge",
                "Gothic Old Town Bridge Tower and Lesser Town Bridge Towers guarding both ends of Charles Bridge",
                "Charles Bridge at dawn or dusk, with fewer crowds and atmospheric lighting, a symbol of Prague",
                "statues on Charles Bridge, such as the statue of St. John of Nepomuk, often touched for good luck"
            ]
        },
        "red_square_st_basils_cathedral": {
            "name": "Red Square & St. Basil's Cathedral",
            "aliases": ["Красная площадь", "Собор Василия Блаженного", "Moscow Kremlin"],
            "location": "Moscow, Russia",
            "prompts": [
                "the iconic, vibrantly colored onion domes of St. Basil's Cathedral (Cathedral of Vasily the Blessed) in Red Square, Moscow, a unique masterpiece of Russian architecture",
                "Red Square, the historic central square of Moscow, with Lenin's Mausoleum, the fortified walls of the Kremlin, the State Historical Museum, and the GUM department store",
                "historic and vast Red Square, a UNESCO World Heritage site, a focal point of Russian history and culture",
                "St. Basil's Cathedral illuminated at night, its swirling patterns and bright colors standing out against the dark sky",
                "the imposing Spasskaya Tower of the Moscow Kremlin overlooking Red Square"
            ]
        },
        "edinburgh_castle": {
            "name": "Edinburgh Castle",
            "aliases": ["Castle Rock Edinburgh"],
            "location": "Edinburgh, UK",
            "prompts": [
                "Edinburgh Castle perched dramatically atop Castle Rock, an extinct volcano, dominating the skyline of Edinburgh, Scotland",
                "historic Scottish fortress, Edinburgh Castle, with its ancient ramparts, Crown Jewels of Scotland (Honours of Scotland), and St. Margaret's Chapel",
                "view of the Royal Mile leading up to Edinburgh Castle, the historic heart of Edinburgh's Old Town",
                "the One O'Clock Gun firing from Edinburgh Castle, a daily tradition",
                "Edinburgh Castle illuminated at night, overlooking the city"
            ]
        },
        "matterhorn": {
            "name": "Matterhorn",
            "aliases": ["Monte Cervino", "The Horn"],
            "location": "Zermatt, Switzerland / Breuil-Cervinia, Italy",
            "prompts": [
                "the distinctive, sharply defined pyramidal peak of the Matterhorn mountain in the Pennine Alps on the border between Switzerland and Italy",
                "snow-covered Matterhorn against a clear blue sky, often reflected in a tranquil alpine lake like Riffelsee or Stellisee",
                "iconic alpine scenery surrounding the Matterhorn, a world-famous mountaineering challenge and symbol of the Alps",
                "the village of Zermatt, Switzerland, with traditional chalets and views of the Matterhorn",
                "Matterhorn at sunrise or sunset (alpenglow), when the peak is bathed in golden or reddish light"
            ]
        },
        "palace_of_versailles": {
            "name": "Palace of Versailles",
            "aliases": ["Château de Versailles", "Versailles Palace"],
            "location": "Versailles, France",
            "prompts": [
                "the opulent Palace of Versailles, former principal royal residence of France, with its grand Hall of Mirrors (Galerie des Glaces) and lavish state apartments",
                "expansive formal gardens of Versailles designed by André Le Nôtre, featuring geometric patterns, fountains (like the Latona Fountain), canals, and groves",
                "luxurious Baroque architecture and lavish interiors of the Palace of Versailles, a UNESCO World Heritage site symbolizing absolute monarchy",
                "the Grand Trianon and Petit Trianon, smaller palaces within the estate of Versailles, and Marie Antoinette's Hamlet",
                "the facade of the Palace of Versailles overlooking the Place d'Armes"
            ]
        }
    },

    # 北美地標
    "north_america": {
        "statue_of_liberty": {
            "name": "Statue of Liberty",
            "aliases": ["Liberty Enlightening the World", "Lady Liberty"],
            "location": "New York Harbor, New York, USA",
            "prompts": [
                "a photo of the Statue of Liberty in New York, the colossal neoclassical sculpture on Liberty Island, a symbol of freedom and democracy",
                "the iconic Statue of Liberty with her torch held high and tabula ansata (tablet), a gift from France to the USA",
                "Statue of Liberty on Liberty Island, with the Manhattan skyline or Ellis Island in the background",
                "close-up of Lady Liberty's crowned head or her copper-green patina",
                "ferry approaching the Statue of Liberty, offering panoramic views"
            ]
        },
        "golden_gate_bridge": {
            "name": "Golden Gate Bridge",
            "aliases": ["Golden Gate", "GGB"],
            "location": "San Francisco, California, USA",
            "prompts": [
                "a photo of the Golden Gate Bridge in San Francisco, its iconic Art Deco suspension bridge painted in 'International Orange'",
                "the vibrant red-orange Golden Gate Bridge spanning the Golden Gate strait, often partially shrouded in fog",
                "Golden Gate Bridge with its distinctive twin towers, soaring cables, and views of Alcatraz Island or the San Francisco skyline",
                "view of the Golden Gate Bridge from various vantage points like Battery Spencer, Vista Point, or Baker Beach",
                "cyclists or pedestrians crossing the Golden Gate Bridge"
            ]
        },
        "grand_canyon": {
            "name": "Grand Canyon",
            "aliases": ["Grand Canyon National Park", "The Canyon"],
            "location": "Arizona, USA",
            "prompts": [
                "a photo of the Grand Canyon in Arizona, a massive, steep-sided canyon carved by the Colorado River, showcasing layers of colorful rock",
                "vast, awe-inspiring landscape of the Grand Canyon, with its immense scale, depth, and intricate formations",
                "Colorado River flowing through the bottom of the Grand Canyon, visible from viewpoints like Mather Point or Yavapai Point on the South Rim",
                "sunset or sunrise over the Grand Canyon, painting the canyon walls in vibrant hues of red, orange, and purple",
                "hiking trails along the rim or into the Grand Canyon, such as Bright Angel Trail"
            ]
        },
        "hollywood_sign": {
            "name": "Hollywood Sign",
            "aliases": ["Hollywoodland Sign"],
            "location": "Mount Lee, Los Angeles, California, USA",
            "prompts": [
                "a photo of the Hollywood Sign in Los Angeles, the iconic white capital letters spelling out 'HOLLYWOOD' on the side of Mount Lee",
                "the iconic Hollywood Sign overlooking the sprawling cityscape of Hollywood and Los Angeles",
                "view of the Hollywood Sign from Griffith Observatory, Lake Hollywood Park, or a helicopter tour",
                "the large, distinctive letters of the Hollywood Sign, a symbol of the American film industry",
                "Hollywood Sign against a clear blue sky or at sunset with city lights below"
            ]
        },
        "white_house": {
            "name": "White House",
            "aliases": ["President's House", "Executive Mansion", "1600 Pennsylvania Avenue"],
            "location": "Washington D.C., USA",
            "prompts": [
                "a photo of the White House in Washington D.C., the official residence and workplace of the President of the United States",
                "the iconic neoclassical facade of the White House, with its white columns and porticoes (North Portico and South Portico)",
                "the North Portico of the White House facing Pennsylvania Avenue, or the South Lawn with the Oval Office view",
                "the White House surrounded by its meticulously manicured gardens and security fencing",
                "Marine One helicopter landing on the South Lawn of the White House"
            ]
        },
        "mount_rushmore": {
            "name": "Mount Rushmore",
            "aliases": ["Mount Rushmore National Memorial", "Presidents' Mountain"],
            "location": "Keystone, South Dakota, USA",
            "prompts": [
                "a photo of Mount Rushmore National Memorial, featuring the colossal carved faces of U.S. Presidents George Washington, Thomas Jefferson, Theodore Roosevelt, and Abraham Lincoln",
                "the sculpted faces of four presidents carved into the granite face of Mount Rushmore in the Black Hills of South Dakota",
                "Mount Rushmore with the Avenue of Flags leading to the Grand View Terrace",
                "the immense scale and detailed carving of the presidential heads on Mount Rushmore",
                "Mount Rushmore illuminated at night during the evening lighting ceremony"
            ]
        },
        "times_square": {
            "name": "Times Square",
            "aliases": ["The Crossroads of the World", "The Great White Way"],
            "location": "New York City, New York, USA",
            "prompts": [
                "a photo of Times Square in New York City, the bustling commercial intersection and entertainment hub, famous for its dazzling array of brightly lit billboards and advertisements",
                "bright, massive digital billboards and flashing neon lights of Times Square at night, creating a vibrant and energetic atmosphere",
                "bustling crowds of tourists and locals, yellow taxis, and costumed characters in Times Square",
                "the New Year's Eve ball drop ceremony in Times Square",
                "the TKTS booth with its red steps in Times Square, a popular spot for discounted Broadway tickets"
            ]
        },
        "cn_tower": {
            "name": "CN Tower",
            "aliases": ["Canadian National Tower", "Toronto Tower"],
            "location": "Toronto, Ontario, Canada",
            "prompts": [
                "a photo of the CN Tower in Toronto, the iconic slender communications and observation tower dominating the city's skyline",
                "the tall, freestanding CN Tower with its distinctive main pod housing observation decks, a revolving restaurant, and the EdgeWalk",
                "view from the top of the CN Tower, looking down through the glass floor or out at Lake Ontario and the Toronto Islands",
                "CN Tower illuminated at night with programmable LED lights, often changing colors for special occasions",
                "Toronto skyline featuring the CN Tower as its centerpiece"
            ]
        },
        "chichen_itza": {
            "name": "Chichen Itza",
            "aliases": ["El Castillo", "Pyramid of Kukulcan", "Chichén Itzá"],
            "location": "Yucatan Peninsula, Mexico",
            "prompts": [
                "a photo of Chichen Itza in Mexico, the ancient Mayan city and UNESCO World Heritage site, with its iconic El Castillo (Pyramid of Kukulcan)",
                "the massive step-pyramid El Castillo at Chichen Itza, famous for the serpent shadow effect during the equinoxes",
                "ancient ruins of Chichen Itza, including the Temple of Warriors, the Great Ball Court, and the Observatory (El Caracol)",
                "intricate stone carvings and Mayan hieroglyphs found on the structures of Chichen Itza",
                "the sacred cenote (sinkhole) at Chichen Itza, used for sacrifices"
            ]
        },
        "niagara_falls": {
            "name": "Niagara Falls",
            "aliases": ["Horseshoe Falls", "American Falls", "Bridal Veil Falls"],
            "location": "Ontario, Canada / New York, USA",
            "prompts": [
                "massive cascades of Niagara Falls, including the powerful Horseshoe Falls (Canadian Falls), the American Falls, and the smaller Bridal Veil Falls",
                "mist rising dramatically from the thundering Niagara Falls, with tour boats like the Maid of the Mist or Hornblower navigating the turbulent waters below",
                "Rainbow Bridge connecting Canada and the USA, with panoramic views of Niagara Falls",
                "Niagara Falls illuminated with colorful lights at night, or with fireworks displays",
                "Goat Island separating the American Falls and Horseshoe Falls, offering close-up views"
            ]
        },
        "central_park": {
            "name": "Central Park",
            "aliases": ["Manhattan Central Park"],
            "location": "New York City, New York, USA",
            "prompts": [
                "vast green expanse of Central Park in Manhattan, New York City, an urban oasis surrounded by the towering skyscrapers of the city skyline",
                "iconic locations within Central Park such as Bethesda Terrace and Fountain, Strawberry Fields (John Lennon memorial), Wollman Rink (ice skating), or the Central Park Carousel",
                "people enjoying recreational activities like picnicking, boating on The Lake, jogging, or horse-drawn carriage rides in Central Park",
                "lush lawns, wooded areas, walking paths, and picturesque bridges (like Bow Bridge or Gapstow Bridge) within Central Park",
                "aerial view of Central Park, highlighting its rectangular shape amidst the dense urban grid of Manhattan"
            ]
        },
        "las_vegas_strip": {
            "name": "Las Vegas Strip",
            "aliases": ["The Strip Las Vegas", "Las Vegas Boulevard South"],
            "location": "Las Vegas, Nevada, USA",
            "prompts": [
                "the dazzling Las Vegas Strip at night, a vibrant spectacle of illuminated mega-resorts, opulent casinos, and world-class entertainment venues",
                "iconic landmarks and themed hotels on the Las Vegas Strip such as the Bellagio fountains, the Eiffel Tower replica at Paris Las Vegas, the High Roller observation wheel, or the Venetian's canals",
                "bustling energy, flashing neon signs, and extravagant architecture characterizing the world-famous Las Vegas Strip",
                "pedestrians walking along the crowded sidewalks of the Las Vegas Strip, taking in the sights and sounds",
                "the Fountains of Bellagio water show on the Las Vegas Strip"
            ]
        },
        "yellowstone_national_park": {
            "name": "Yellowstone National Park",
            "aliases": ["Old Faithful Yellowstone", "Grand Prismatic Spring"],
            "location": "Wyoming, Montana, Idaho, USA",
            "prompts": [
                "geothermal features of Yellowstone National Park, including the iconic Old Faithful geyser erupting steam and hot water into the air",
                "the vibrant, rainbow-like colors of the Grand Prismatic Spring, the largest hot spring in the United States, in Yellowstone's Midway Geyser Basin",
                "wildlife such as bison herds, elk, bears, and wolves roaming freely in the diverse landscapes of Yellowstone National Park, including forests, meadows, and rivers",
                "the Grand Canyon of the Yellowstone, a dramatic canyon with impressive waterfalls like the Lower Falls",
                "Mammoth Hot Springs in Yellowstone, with its terraced travertine formations"
            ]
        },
        "banff_national_park_lake_louise": { # Specified Lake Louise
            "name": "Banff National Park (Lake Louise / Moraine Lake)",
            "aliases": ["Lake Louise Banff", "Moraine Lake Banff", "Canadian Rockies"],
            "location": "Alberta, Canada",
            "prompts": [
                "the stunning turquoise glacial waters of Lake Louise in Banff National Park, with the majestic Victoria Glacier and Fairmont Chateau Lake Louise in the background",
                "the equally breathtaking Moraine Lake in Banff National Park, with its vivid blue water backed by the rugged Valley of the Ten Peaks",
                "stunning Canadian Rockies mountain scenery in Banff National Park, a UNESCO World Heritage site, with snow-capped peaks, alpine meadows, and pristine forests",
                "canoeing or hiking around Lake Louise or Moraine Lake",
                "wildlife like elk or grizzly bears sometimes spotted in Banff National Park"
            ]
        },
        "space_needle_seattle": {
            "name": "Space Needle",
            "aliases": ["Seattle Space Needle"],
            "location": "Seattle, Washington, USA",
            "prompts": [
                "the futuristic Space Needle observation tower in Seattle, Washington, with its distinctive saucer-shaped top and slender hourglass silhouette",
                "panoramic view of Seattle's skyline, Puget Sound, Elliott Bay, and surrounding mountains like Mount Rainier from the Space Needle's observation deck or revolving restaurant",
                "Space Needle as an icon of the Pacific Northwest and a legacy of the 1962 World's Fair",
                "the Space Needle illuminated at night, a prominent feature of Seattle's cityscape",
                "Chihuly Garden and Glass exhibit located at the base of the Space Needle"
            ]
        }
    },

    # 南美地標
    "south_america": {
        "machu_picchu": {
            "name": "Machu Picchu",
            "aliases": ["Lost City of the Incas", "Machu Pikchu"],
            "location": "Cusco Region, Peru",
            "prompts": [
                "a photo of Machu Picchu in Peru, the breathtaking ancient Inca citadel set high in the Andes Mountains, often shrouded in mist",
                "the well-preserved ruins of the Inca city of Machu Picchu, with its intricate stone masonry, temples, plazas, and agricultural terraces",
                "panoramic view of Machu Picchu ruins with Huayna Picchu mountain rising prominently in the background",
                "llamas grazing among the ancient stone structures of Machu Picchu",
                "sunrise over Machu Picchu, revealing its mystical beauty and stunning mountain setting"
            ]
        },
        "christ_the_redeemer": {
            "name": "Christ the Redeemer",
            "aliases": ["Cristo Redentor", "Rio Jesus Statue", "Corcovado Statue"],
            "location": "Rio de Janeiro, Brazil",
            "prompts": [
                "a photo of Christ the Redeemer statue in Rio de Janeiro, the colossal Art Deco statue of Jesus Christ with outstretched arms, standing atop Corcovado Mountain",
                "the iconic soapstone statue of Jesus Christ on Corcovado Mountain, overlooking the city of Rio de Janeiro, Sugarloaf Mountain, and Guanabara Bay",
                "Christ the Redeemer as a symbol of Christianity and a global icon of Rio de Janeiro and Brazil",
                "view from the base of Christ the Redeemer statue, offering breathtaking panoramic vistas",
                "Christ the Redeemer statue illuminated at night or silhouetted against a vibrant sunset"
            ]
        },
        "iguazu_falls": {
            "name": "Iguazu Falls",
            "aliases": ["Iguaçu Falls", "Cataratas del Iguazú", "Cataratas do Iguaçu", "Devil's Throat Iguazu"],
            "location": "Misiones Province, Argentina / Paraná State, Brazil",
            "prompts": [
                "expansive network of hundreds of powerful waterfalls at Iguazu Falls, spanning the border of Argentina and Brazil, surrounded by lush subtropical rainforest",
                "the immense and thunderous Devil's Throat (Garganta del Diablo / Garganta do Diabo), the largest and most dramatic cataract of Iguazu Falls",
                "walkways and viewpoints offering close-up, immersive experiences of the mighty Iguazu Falls, often with rainbows forming in the mist",
                "boat tours venturing near the base of the waterfalls at Iguazu Falls",
                "diverse wildlife like coatis and colorful birds in the Iguazu National Park surrounding the falls"
            ]
        },
        "galapagos_islands": {
            "name": "Galapagos Islands",
            "aliases": ["Archipiélago de Colón", "Darwin's Islands"],
            "location": "Ecuador",
            "prompts": [
                "unique and fearless wildlife of the Galapagos Islands, such as giant tortoises roaming freely, marine iguanas basking on volcanic rocks, and blue-footed boobies performing their mating dance",
                "pristine volcanic landscapes, lava fields, and beautiful beaches (like Tortuga Bay) of the Galapagos Islands, a UNESCO World Heritage site that inspired Charles Darwin's theory of evolution",
                "snorkeling or diving with sea lions, penguins, and diverse marine life in the clear waters surrounding the Galapagos Islands",
                "various endemic species found only in the Galapagos, like Darwin's finches or flightless cormorants",
                "cruise ships or small yachts exploring the different islands of the Galapagos archipelago"
            ]
        },
        "torres_del_paine_national_park": {
            "name": "Torres del Paine National Park",
            "aliases": ["Parque Nacional Torres del Paine", "Paine Towers"],
            "location": "Patagonia, Chile",
            "prompts": [
                "the iconic granite peaks (Horns or Towers of Paine) of the Torres del Paine massif in Chilean Patagonia, often reflecting in turquoise glacial lakes like Pehoé or Nordenskjöld",
                "stunning and wild landscapes of glaciers (like Grey Glacier), vibrant blue lakes, rivers, and mountains in Torres del Paine National Park, a UNESCO Biosphere Reserve",
                "hiking trails like the 'W' trek or 'O' circuit offering breathtaking views of the dramatic Patagonian scenery in Torres del Paine",
                "guanacos, condors, and other Patagonian wildlife in their natural habitat within Torres del Paine",
                "the dramatic, windswept environment of Torres del Paine, known for its unpredictable weather"
            ]
        },
        "angel_falls": {
            "name": "Angel Falls",
            "aliases": ["Salto Ángel", "Kerepakupai Merú"],
            "location": "Canaima National Park, Venezuela",
            "prompts": [
                "Angel Falls, the world's tallest uninterrupted waterfall, cascading spectacularly from the sheer cliff face of Auyán-Tepui, a massive table-top mountain (tepui) in Venezuela's Canaima National Park",
                "remote and dramatic jungle landscape surrounding Angel Falls, with tepuis rising from the savanna and rainforest",
                "aerial view of Angel Falls plunging thousands of feet down the Auyán-Tepui, often shrouded in mist",
                "expeditions by boat and foot through the remote wilderness to reach the base of Angel Falls",
                "the sheer scale and pristine, untouched beauty of Angel Falls, a natural wonder"
            ]
        },
        "salar_de_uyuni": {
            "name": "Salar de Uyuni",
            "aliases": ["Uyuni Salt Flat"],
            "location": "Potosí, Bolivia",
            "prompts": [
                "vast, seemingly endless white expanse of the Salar de Uyuni salt flat in Bolivia, the world's largest salt desert, creating a surreal and minimalist landscape",
                "mirror-like reflections on Salar de Uyuni during the rainy season (December-April), transforming the salt flat into the world's largest natural mirror, blurring the horizon between sky and ground",
                "Isla Incahuasi (Fish Island) with its giant ancient cacti standing starkly against the white salt crust of Salar de Uyuni",
                "geometric patterns of salt polygons on the dry Salar de Uyuni",
                "creative forced perspective photographs taken by tourists on the Salar de Uyuni"
            ]
        }
    },

    # 中東/非洲地標
    "middle_east_africa": {
        "pyramids_of_giza": {
            "name": "Pyramids of Giza",
            "aliases": ["Great Pyramids", "Egyptian Pyramids", "Giza Necropolis"],
            "location": "Giza, Egypt (near Cairo)",
            "prompts": [
                "a photo of the Pyramids of Giza in Egypt, the ancient wonder of the world, with the Great Pyramid, Pyramid of Khafre, and Pyramid of Menkaure",
                "the ancient Egyptian pyramids on the Giza plateau, with the enigmatic Great Sphinx guarding them, against a desert backdrop or the Cairo skyline",
                "Great Pyramid of Giza, the largest of the three, with the smaller Queen's Pyramids nearby",
                "camels and horses carrying tourists around the Giza pyramid complex",
                "sunset or sunrise over the Pyramids of Giza, casting long shadows"
            ]
        },
        "burj_khalifa": {
            "name": "Burj Khalifa",
            "aliases": ["Khalifa Tower", "Dubai Tower", "World's Tallest Building"],
            "location": "Dubai, UAE",
            "prompts": [
                "a photo of Burj Khalifa in Dubai, the world's tallest building, a sleek, tapering skyscraper piercing the sky",
                "the ultra-modern skyscraper Burj Khalifa dominating the Dubai skyline with its impressive height and futuristic design",
                "Burj Khalifa skyscraper rising above the Dubai Fountain and surrounding modern architecture",
                "view from the observation deck ('At the Top') of Burj Khalifa, offering panoramic views of Dubai and the desert",
                "Burj Khalifa illuminated with spectacular light shows at night"
            ]
        },
        "petra_jordan": { # Added Jordan to differentiate from any other Petra
            "name": "Petra",
            "aliases": ["Rose City", "Lost City of Petra", "Al-Khazneh Petra"],
            "location": "Ma'an Governorate, Jordan",
            "prompts": [
                "a photo of Petra in Jordan, the ancient Nabataean city carved into rose-red sandstone cliffs, a UNESCO World Heritage site",
                "the iconic Treasury (Al-Khazneh) in Petra, with its ornate facade intricately carved into a sandstone rock face, revealed at the end of the Siq (narrow gorge)",
                "ancient city of Petra with its rock-cut architecture, including tombs, temples (like the Monastery, Ad Deir), and colonnaded streets",
                "the Siq, a narrow, winding gorge that serves as the dramatic main entrance to the city of Petra",
                "Bedouins with camels or donkeys in the ancient city of Petra"
            ]
        },
        "table_mountain": {
            "name": "Table Mountain",
            "aliases": ["Tafelberg", "Cape Town Table Mountain"],
            "location": "Cape Town, South Africa",
            "prompts": [
                "a photo of Table Mountain in Cape Town, the iconic flat-topped mountain majestically overlooking the city, Table Bay, and the Atlantic Ocean",
                "the flat-topped Table Mountain, often covered by a 'tablecloth' of clouds, with Devil's Peak and Lion's Head adjacent to it",
                "view from the rotating cable car ascending Table Mountain, or panoramic views from its summit",
                "Cape Town city nestled at the foot of Table Mountain, with the V&A Waterfront visible",
                "unique fynbos vegetation found on Table Mountain, part of the Cape Floral Kingdom"
            ]
        },
        "sheikh_zayed_grand_mosque": {
            "name": "Sheikh Zayed Grand Mosque",
            "aliases": ["Grand Mosque Abu Dhabi"],
            "location": "Abu Dhabi, UAE",
            "prompts": [
                "the stunning, pristine white marble Sheikh Zayed Grand Mosque in Abu Dhabi, with its numerous domes (82 of them) and four towering minarets",
                "intricate floral designs inlaid with semi-precious stones, gold accents, and massive reflective pools surrounding the Sheikh Zayed Grand Mosque",
                "the vast main prayer hall of Sheikh Zayed Grand Mosque, featuring the world's largest hand-knotted carpet and one of the world's largest Swarovski crystal chandeliers",
                "the gleaming white exterior and symmetrical courtyards of Sheikh Zayed Grand Mosque, a masterpiece of modern Islamic architecture",
                "Sheikh Zayed Grand Mosque illuminated at night with a unique lunar lighting system that changes with the phases of the moon"
            ]
        },
        "masai_mara_national_reserve": {
            "name": "Masai Mara National Reserve",
            "aliases": ["Maasai Mara", "The Mara"],
            "location": "Kenya",
            "prompts": [
                "vast, open savannas of the Masai Mara National Reserve in Kenya, teeming with iconic African wildlife like lions, elephants, giraffes, zebras, and wildebeest",
                "the Great Migration of millions of wildebeest and zebras crossing the Mara River, often facing crocodiles, during their annual journey (July-October)",
                "Masai people in their traditional vibrant red shuka robes, often seen near their villages or as safari guides in the Masai Mara",
                "hot air balloons drifting over the plains of the Masai Mara at sunrise, offering a unique perspective on the landscape and wildlife",
                "acacia trees silhouetted against a dramatic African sunset in the Masai Mara"
            ]
        },
        "victoria_falls": {
            "name": "Victoria Falls",
            "aliases": ["Mosi-oa-Tunya", "The Smoke that Thunders"],
            "location": "Livingstone, Zambia / Victoria Falls, Zimbabwe",
            "prompts": [
                "the spectacular, vast curtain of falling water at Victoria Falls, one of the largest waterfalls in the world by combined width and height, on the Zambezi River",
                "a plume of mist (the 'smoke') rising high above Victoria Falls, visible for miles, and frequent rainbows forming in the spray",
                "views of Victoria Falls from various viewpoints like the Knife-Edge Bridge, Danger Point, or from rainforest trails along the gorge",
                "adventure activities at Victoria Falls, such as bungee jumping from the Victoria Falls Bridge or white-water rafting on the Zambezi",
                "the Zambezi River plunging into the deep Batoka Gorge at Victoria Falls"
            ]
        },
        "kilimanjaro": {
            "name": "Mount Kilimanjaro",
            "aliases": ["Uhuru Peak", "Kibo Kilimanjaro", "Africa's Highest Mountain"],
            "location": "Tanzania",
            "prompts": [
                "the snow-capped, iconic peak of Mount Kilimanjaro, Africa's highest mountain and the world's tallest freestanding mountain, rising majestically from the plains of Tanzania",
                "the distinctive flat-topped dormant volcano, Kilimanjaro, with its three volcanic cones (Kibo, Mawenzi, and Shira), a popular and challenging climbing destination",
                "diverse ecosystems on the slopes of Kilimanjaro, transitioning from rainforest and moorland to alpine desert and arctic summit zones",
                "porters and climbers on one of the routes (e.g., Machame, Lemosho) ascending Mount Kilimanjaro",
                "Mount Kilimanjaro at sunrise or sunset, with its glaciers and snowfields gleaming"
            ]
        },
        "dead_sea": {
            "name": "Dead Sea",
            "aliases": ["Salt Sea"],
            "location": "Jordan / Israel / Palestine",
            "prompts": [
                "people floating effortlessly and buoyantly in the hyper-saline, turquoise waters of the Dead Sea, the lowest point on Earth's land surface",
                "mineral-rich black mud from the Dead Sea being applied to the skin for therapeutic benefits, with salt formations along the shores",
                "unique landscape of the Dead Sea, with its calm, dense waters reflecting the arid surrounding mountains and desert",
                "evaporation ponds for mineral extraction at the southern end of the Dead Sea",
                "salt crystals encrusting rocks and branches along the coastline of the Dead Sea"
            ]
        },
        "dome_of_the_rock": {
            "name": "Dome of the Rock",
            "aliases": ["Qubbat as-Sakhrah", "Temple Mount Jerusalem"],
            "location": "Old City of Jerusalem",
            "prompts": [
                "the iconic, gleaming golden dome of the Dome of the Rock, an Islamic shrine located on the Temple Mount (Haram al-Sharif) in the Old City of Jerusalem",
                "the octagonal structure of the Dome of the Rock, adorned with intricate blue and turquoise ceramic tilework, calligraphy, and mosaics",
                "iconic religious landmark, the Dome of the Rock, a site of great significance in Islam, Judaism, and Christianity, within the historic walls of Jerusalem",
                "view of the Dome of the Rock from the Mount of Olives, with the Old City in the background",
                "the interior of the Dome of the Rock (if permissible to depict), showing the Foundation Stone"
            ]
        }
    },

    # 大洋洲地標
    "oceania": {
        "sydney_opera_house": {
            "name": "Sydney Opera House",
            "aliases": ["Opera House Sydney", "Sydney Landmark"],
            "location": "Sydney, New South Wales, Australia",
            "prompts": [
                "a photo of the Sydney Opera House in Australia, its iconic white sail-shaped shells creating a distinctive silhouette on Sydney Harbour",
                "the multi-venue performing arts centre, Sydney Opera House, with its unique shell-like roof structures, a masterpiece of modern architecture",
                "Sydney Opera House with the Sydney Harbour Bridge in the background, a classic view of Sydney's landmarks",
                "the Sydney Opera House illuminated at night, often with colorful projections for events like Vivid Sydney",
                "close-up of the textured, chevron-patterned tiles covering the shells of the Sydney Opera House"
            ]
        },
        "uluru": {
            "name": "Uluru",
            "aliases": ["Ayers Rock", "The Rock", "Uluru-Kata Tjuta National Park"],
            "location": "Northern Territory, Australia",
            "prompts": [
                "a photo of Uluru (Ayers Rock) in Australia, the massive, sacred sandstone monolith rising from the flat desert landscape of the Red Centre",
                "the immense sandstone monolith of Uluru changing colors dramatically at sunrise or sunset, glowing in shades of red, orange, and purple",
                "Uluru in the Red Centre of Australia, a significant spiritual site for Indigenous Anangu people, with visible rock caves and ancient rock art",
                "the distinct shape and texture of Uluru, showing its weathered surface and gullies",
                "Kata Tjuta (The Olgas) formations visible in the distance from Uluru"
            ]
        },
        "great_barrier_reef": {
            "name": "Great Barrier Reef",
            "aliases": ["GBR", "World's Largest Coral Reef System"],
            "location": "Queensland, Australia",
            "prompts": [
                "an underwater photo of the Great Barrier Reef, showcasing its vibrant and diverse coral formations, colorful fish, and other marine life like sea turtles or manta rays",
                "colorful hard and soft coral gardens thriving in the clear turquoise waters of the Great Barrier Reef, the world's largest coral reef system",
                "aerial view of the Great Barrier Reef, revealing the intricate patterns of reefs, islands, and cays stretching along the Queensland coast",
                "scuba divers or snorkelers exploring the rich biodiversity of the Great Barrier Reef",
                "Heart Reef, a naturally formed heart-shaped coral formation in the Great Barrier Reef (often seen from the air)"
            ]
        },
        "hobbiton_movie_set": {
            "name": "Hobbiton Movie Set",
            "aliases": ["The Shire Tour", "Lord of the Rings Set NZ", "Hobbiton New Zealand"],
            "location": "Matamata, Waikato, New Zealand",
            "prompts": [
                "a photo of the Hobbiton Movie Set in New Zealand, the picturesque movie set for The Shire from 'The Lord of the Rings' and 'The Hobbit' trilogies",
                "charming, brightly colored circular Hobbit hole doors built into rolling green hills at the Hobbiton Movie Set, with meticulously tended gardens",
                "the Green Dragon Inn at Hobbiton, a faithfully reconstructed pub where visitors can enjoy a drink",
                "the Party Tree and Mill at Hobbiton, iconic locations from the movies, set within a lush pastoral landscape",
                "guided tour groups exploring the whimsical and detailed Hobbiton Movie Set"
            ]
        },
        "sydney_harbour_bridge": {
            "name": "Sydney Harbour Bridge",
            "aliases": ["The Coathanger Sydney"],
            "location": "Sydney, New South Wales, Australia",
            "prompts": [
                "the iconic steel through arch of the Sydney Harbour Bridge, affectionately known as 'The Coathanger', spanning Sydney Harbour",
                "people participating in the Sydney Harbour BridgeClimb, ascending the arch for panoramic views of the city and harbour",
                "view of Sydney Harbour featuring both the Opera House and the Harbour Bridge together, a quintessential Sydney scene",
                "ferries and sailboats passing under the massive Sydney Harbour Bridge",
                "fireworks display over the Sydney Harbour Bridge during New Year's Eve celebrations"
            ]
        },
        "fiordland_national_park_milford_sound": { # Specified Milford Sound
            "name": "Fiordland National Park (Milford Sound / Doubtful Sound)",
            "aliases": ["Milford Sound New Zealand", "Doubtful Sound Fiordland"],
            "location": "South Island, New Zealand",
            "prompts": [
                "dramatic fiords with sheer, towering cliffs and cascading waterfalls (like Stirling Falls or Bowen Falls) in Fiordland National Park, New Zealand, particularly Milford Sound",
                "the iconic Mitre Peak rising sharply from the dark, reflective waters of Milford Sound in Fiordland, often shrouded in mist",
                "boat cruises navigating through the stunning natural scenery of Milford Sound or the more remote Doubtful Sound, with seals, dolphins, or penguins sometimes spotted",
                "lush rainforest clinging to the steep mountain sides of Fiordland National Park",
                "the dramatic, moody atmosphere of Fiordland, known for its high rainfall and untouched wilderness"
            ]
        },
        "bondi_beach": {
            "name": "Bondi Beach",
            "aliases": ["Bondi Sydney"],
            "location": "Sydney, New South Wales, Australia",
            "prompts": [
                "the famous crescent-shaped Bondi Beach in Sydney, a popular destination with golden sand, turquoise waves, surfers, and sunbathers",
                "Bondi Icebergs Club swimming pool, an ocean pool with waves crashing into it, located at the southern end of Bondi Beach",
                "vibrant beach culture, surf lifesavers in their distinctive red and yellow uniforms, and bustling cafes along the promenade at Bondi Beach",
                "the Bondi to Coogee coastal walk, offering stunning ocean views starting from Bondi Beach",
                "aerial view of Bondi Beach, showcasing its iconic shape and lively atmosphere"
            ]
        },
        "aoraki_mount_cook_national_park": {
            "name": "Aoraki / Mount Cook National Park",
            "aliases": ["Mount Cook New Zealand", "Lake Pukaki Mount Cook", "Hooker Valley Track"],
            "location": "South Island, New Zealand",
            "prompts": [
                "the majestic, snow-capped pyramid peak of Aoraki / Mount Cook, New Zealand's highest mountain, dominating the Southern Alps",
                "glaciers like Tasman Glacier, alpine lakes with milky turquoise water (such as Lake Pukaki or Lake Tekapo often framed by colorful lupins in summer), and rugged mountain scenery in Aoraki / Mount Cook National Park",
                "stargazing in the Aoraki Mackenzie International Dark Sky Reserve, with the silhouette of Aoraki / Mount Cook against a starry night sky",
                "hiking trails like the Hooker Valley Track, offering spectacular views of Aoraki / Mount Cook, glaciers, and icebergs in Hooker Lake",
                "the Hermitage Hotel or other alpine lodges with views of Aoraki / Mount Cook"
            ]
        },
        "twelve_apostles_great_ocean_road": { # Specified Great Ocean Road
            "name": "The Twelve Apostles, Great Ocean Road",
            "aliases": ["Twelve Apostles Victoria", "Great Ocean Road Australia"],
            "location": "Victoria, Australia",
            "prompts": [
                "limestone stacks known as The Twelve Apostles (though fewer than twelve remain) rising dramatically from the Southern Ocean along the scenic Great Ocean Road in Victoria, Australia",
                "coastal scenery with rugged cliffs, powerful wave erosion, and the iconic sea stacks of The Twelve Apostles site, especially at sunrise or sunset",
                "sunset or sunrise over The Twelve Apostles rock formations, casting golden light and long shadows",
                "viewing platforms offering panoramic vistas of The Twelve Apostles and the surrounding coastline",
                "other nearby rock formations along the Great Ocean Road, like Loch Ard Gorge or London Arch (formerly London Bridge)"
            ]
        },
         "easter_island_moai": {
            "name": "Moai Statues, Easter Island",
            "aliases": [
                "Easter Island Heads",
                "Rapa Nui Moai",
                "復活節島摩艾石像",
                "拉帕努伊島石像",
                "摩艾"
            ],
            "location": "Easter Island (Rapa Nui), Chile",
            "prompts": [
                "colossal monolithic human figures known as moai, carved from volcanic rock, standing on Easter Island (Rapa Nui)",
                "iconic giant stone heads and torsos of the moai statues with long ears and stoic expressions, dotting the coastal and inland landscapes of Easter Island",
                "the mysterious moai statues of Rapa Nui, some with red scoria topknots (pukao), on ceremonial platforms called ahu",
                "Ahu Tongariki featuring a line-up of fifteen imposing moai statues against the Pacific Ocean, Easter Island",
                "moai statues in the Rano Raraku quarry, where they were carved, some partially buried or appearing to emerge from the hillside",
                "the unique archaeological site of Easter Island, showcasing hundreds of ancient monolithic moai, a testament to Rapa Nui culture",
                "weathered stone giants of Easter Island, their distinct profiles silhouetted against the sky or ocean, conveying a sense of ancient mystery",
                "close-up of a moai's characteristic features: heavy brow, elongated nose, thin lips, and often deep eye sockets, Easter Island",
                "a line of massive moai statues on an ahu platform overlooking the sea, representing deified ancestors of Rapa Nui",
                "the remote and windswept landscape of Easter Island, punctuated by the enigmatic presence of its ancient moai statues"
            ]
        }
    }
}

# 方便直接查詢所有地標
ALL_LANDMARKS = {}
for region, landmarks in LANDMARK_DATA.items():
    for landmark_id, landmark_info in landmarks.items():
        ALL_LANDMARKS[landmark_id] = landmark_info

# 獲取所有地標提示列表（用於CLIP分析）
def get_all_landmark_prompts():
    """
    返回所有地標的提示列表，用於CLIP分析

    Returns:
        list: 提示列表
    """
    prompts = []
    for landmark_id, landmark_info in ALL_LANDMARKS.items():
        # 使用第一個提示作為主要提示
        prompts.append(landmark_info["prompts"][0])
    return prompts

# 獲取地標名稱到ID的映射
def get_landmark_name_to_id_map():
    """
    返回地標名稱到ID的映射

    Returns:
        dict: {地標名稱: 地標ID}
    """
    name_to_id = {}
    for landmark_id, landmark_info in ALL_LANDMARKS.items():
        name_to_id[landmark_info["name"]] = landmark_id
        # 也添加所有別名
        for alias in landmark_info["aliases"]:
            name_to_id[alias] = landmark_id
    return name_to_id

# 獲取某個區域的所有地標ID
def get_landmarks_by_region(region):
    """
    返回指定區域的所有地標ID

    Args:
        region (str): 區域名稱

    Returns:
        list: 地標ID列表
    """
    if region not in LANDMARK_DATA:
        return []
    return list(LANDMARK_DATA[region].keys())

# 獲取每個地標的代表性提示
def get_landmark_prompt(landmark_id):
    """
    返回指定地標的代表性提示

    Args:
        landmark_id (str): 地標ID

    Returns:
        str: 提示文本
    """
    if landmark_id not in ALL_LANDMARKS:
        return None
    return ALL_LANDMARKS[landmark_id]["prompts"][0]
