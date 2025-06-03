
"""
Activity suggestions for specific landmarks.
This module provides custom activity recommendations for recognized landmarks.
"""

LANDMARK_ACTIVITIES = {
    # 亞洲地標 (Asia)
    "taipei_101": [
        "Visiting the observation deck (89F, 91F, 101F) for panoramic city and mountain views",
        "Shopping at the luxury mall at the base (Taipei 101 Mall)",
        "Photographing the cityscape, especially at sunset or during the New Year's Eve fireworks display",
        "Dining at high-altitude restaurants within the tower (e.g., Din Tai Fung, Starbucks on 35F)",
        "Learning about the engineering marvels, including the tuned mass damper, through exhibits",
        "Admiring the public art installations around the building and mall"
    ],
    "taroko_gorge": [
        "Hiking scenic trails like Shakadang Trail (Mysterious Valley Trail), Baiyang Waterfall Trail (Water Curtain Cave), or Lushui Trail",
        "Photographing the sheer marble cliffs, tunnels, and the turquoise Liwu River",
        "Visiting the Eternal Spring Shrine (Changchun Shrine) and Bell Tower",
        "Exploring Swallow Grotto (Yanzikou) and the Tunnel of Nine Turns (Jiuqudong) for close-up gorge views",
        "Learning about the geology, ecology, and indigenous Truku culture at the visitor center",
        "Cycling along parts of the gorge road (with extreme caution due to traffic and tunnels)"
    ],
    "sun_moon_lake": [
        "Taking a boat tour across the lake, visiting Lalu Island, Xuanguang Temple, and Ita Thao Pier",
        "Cycling or walking on the dedicated bike paths encircling the lake (e.g., Xiangshan to Shuishe section)",
        "Visiting Wenwu Temple and Ci'en Pagoda for stunning lake views and cultural insights",
        "Riding the Sun Moon Lake Ropeway (cable car) for aerial views and access to the Formosan Aboriginal Culture Village",
        "Enjoying local Thao aboriginal cuisine and street food at Ita Thao village",
        "Photographing the serene lake landscapes, especially at sunrise, sunset, or when mist-covered"
    ],
    "jiufen_old_street": [
        "Wandering through the narrow, lantern-lined alleyways of Jiufen Old Street, especially Shuqi Road",
        "Sampling local Taiwanese snacks like taro balls, fish balls, and peanut ice cream rolls",
        "Visiting traditional teahouses (e.g., A-Mei Tea House) for tea and panoramic coastal views",
        "Photographing the nostalgic atmosphere, red lanterns, and views of Keelung Mountain and the coastline",
        "Shopping for local crafts, souvenirs, and ocarinas",
        "Learning about Jiufen's gold mining history at the Gold Museum in nearby Jinguashi (optional day trip extension)"
    ],
    "kenting_national_park": [
        "Relaxing, swimming, or surfing at popular beaches like Nanwan (South Bay), Baishawan (White Sand Bay), or Little Bay (Xiaowan)",
        "Visiting Eluanbi Park to see the iconic Eluanbi Lighthouse, Taiwan's southernmost point",
        "Exploring unique geological formations like Sail Rock (Chuanfanshi), Longpan Park's limestone cliffs, and Maobitou's coastal terrain",
        "Snorkeling or scuba diving in the coral-rich waters (e.g., Houbihu Marine Protected Area)",
        "Hiking trails in Sheding Nature Park or Kenting Forest Recreation Area",
        "Enjoying the vibrant atmosphere and street food at Kenting Night Market on Kenting Street"
    ],
    "national_palace_museum_tw": [
        "Viewing renowned masterpieces like the Jadeite Cabbage, Meat-shaped Stone, and Mao Gong Ding",
        "Exploring extensive collections of Chinese imperial artifacts, calligraphy, paintings, ceramics, and bronzes",
        "Taking a guided tour or using an audio guide to understand the historical and cultural context of the exhibits",
        "Admiring the classical Chinese palace-style architecture of the museum building",
        "Visiting the serene Zhishan Garden, a traditional Chinese garden adjacent to the museum",
        "Attending special exhibitions and cultural events hosted by the museum"
    ],
    "alishan_national_scenic_area": [
        "Watching the famous sunrise over a sea of clouds from Chushan or Ogasawara Mountain viewing platforms",
        "Riding the Alishan Forest Railway on one of its scenic mountain routes (e.g., to Shenmu 'Sacred Tree' Station)",
        "Hiking through misty forests of ancient giant trees (Taiwan red cypress and cedar) along trails like the Giant Tree Plank Trail",
        "Visiting the Sister Ponds (Jiemei Tan) and Shouzhen Temple",
        "Learning about Alishan's tea culture and sampling high-mountain oolong tea at local plantations",
        "Photographing the cherry blossoms in spring or fireflies in summer"
    ],
    "shilin_night_market": [
        "Sampling a wide variety of iconic Taiwanese street foods like oyster omelets, stinky tofu, giant fried chicken cutlets, and bubble tea",
        "Exploring the bustling maze of food stalls, clothing shops, and souvenir vendors in the outdoor and underground sections",
        "Playing traditional night market games like claw machines, balloon darts, and shrimp fishing",
        "Shopping for trendy apparel, accessories, and electronics at affordable prices",
        "Experiencing the vibrant and energetic atmosphere of one of Taipei's largest and most famous night markets",
        "Trying unique snacks like flame-torched beef cubes or cheese-filled potatoes"
    ],
    "tokyo_tower": [
        "Ascending to the main (150m) and top (250m) observation decks for panoramic views of Tokyo, including Mount Fuji on clear days",
        "Photographing the tower, especially when illuminated with its iconic orange lights or special seasonal displays",
        "Visiting the official 'FootTown' at the base, featuring souvenir shops, cafes, an aquarium, and an e-sports park",
        "Enjoying refreshments or a meal at the tower's cafes or restaurants with city views",
        "Learning about the tower's history, construction, and its role as a broadcasting tower",
        "Visiting nearby Zojoji Temple for a cultural contrast and photo opportunities with the tower in the background"
    ],
    "mount_fuji": [
        "Climbing to the summit during the official climbing season (typically early July to early September) via one of the four main trails",
        "Photographing the iconic snow-capped conical peak from various viewpoints like the Fuji Five Lakes (especially Lake Kawaguchiko or Lake Yamanakako), Chureito Pagoda, or Hakone",
        "Visiting the Fuji Five Lakes region for activities like boating, fishing, hot springs (onsen), and museums (e.g., Itchiku Kubota Art Museum)",
        "Hiking or nature walking around the lower slopes, Aokigahara forest (Jukai), or Oshino Hakkai (traditional village with eight spring ponds)",
        "Visiting Fuji-Q Highland amusement park for thrill rides with views of Mount Fuji",
        "Learning about its volcanic geology, cultural significance (UNESCO World Heritage site), and spiritual importance in Shintoism and Buddhism"
    ],
    "kinkaku_ji": [
        "Admiring and photographing the stunning Golden Pavilion (Shariden) meticulously covered in gold leaf, reflected in the Mirror Pond (Kyōko-chi)",
        "Strolling through the meticulously maintained Japanese Muromachi period stroll garden (kaiyū-shiki teien)",
        "Learning about the Zen Buddhist history of the temple, originally a retirement villa for Shogun Ashikaga Yoshimitsu",
        "Observing the distinct architectural styles of each of the three floors of the pavilion",
        "Making a wish by tossing coins at designated spots for good luck",
        "Enjoying matcha tea and traditional sweets at the Sekkatei Teahouse or nearby tea houses (check availability)"
    ],
    "fushimi_inari_shrine": [
        "Walking through the thousands of vibrant vermilion (shu-iro) torii gates that form tunnels along the mountain trails",
        "Hiking the entire 4km (2-3 hour) trail up the sacred Mount Inari, exploring the network of paths and smaller shrines",
        "Photographing the iconic, seemingly endless tunnels of torii gates, a symbol of prosperity and good fortune",
        "Visiting the main shrine buildings (Go-Honden) at the base and offering prayers to Inari, the Shinto god of rice, sake, and business",
        "Exploring smaller sub-shrines (otsuka) and atmospheric graveyards along the mountain trails",
        "Looking for numerous fox statues (kitsune), considered messengers of Inari, often holding keys or jewels in their mouths"
    ],
    "shibuya_crossing": [
        "Experiencing the 'scramble' by walking across the multi-directional intersection with hundreds of other pedestrians",
        "Photographing or video recording the iconic crossing from a high vantage point, such as the Starbucks in the Tsutaya building or Mag's Park rooftop at Magnet by Shibuya109",
        "People-watching and soaking in the vibrant, energetic atmosphere of modern Tokyo",
        "Visiting the Hachiko statue, a famous meeting spot dedicated to the loyal Akita dog, located outside Shibuya Station",
        "Shopping at the numerous department stores (e.g., Shibuya 109, Shibuya Sky) and trendy boutiques in the surrounding area",
        "Exploring nearby entertainment venues, restaurants, and nightlife options"
    ],
    "tokyo_skytree": [
        "Ascending to the Tembo Deck (350m) and Tembo Galleria (450m) for breathtaking panoramic views of Tokyo and beyond, including Mount Fuji on clear days",
        "Photographing the modern architectural marvel, especially when illuminated with its signature 'Iki' (sky blue) or 'Miyabi' (purple) lights",
        "Walking on the glass floor section of the Tembo Deck for a thrilling view downwards",
        "Shopping and dining at Tokyo Solamachi, the large entertainment complex at the base of the Skytree, which includes an aquarium and planetarium",
        "Learning about the tower's earthquake-resistant design and construction",
        "Visiting nearby Sumida River for a different perspective of the tower and leisurely walks"
    ],
    "senso_ji_temple": [
        "Entering through the Kaminarimon (Thunder Gate) with its giant red lantern and statues of Raijin and Fujin",
        "Walking along Nakamise-dori, the bustling market street leading to the temple, lined with stalls selling traditional snacks, crafts, and souvenirs",
        "Visiting the main hall (Hondo) dedicated to Kannon Bosatsu (Bodhisattva of Compassion) and the five-story pagoda",
        "Wafting incense smoke from the large cauldron (jokoro) over oneself for good health and luck",
        "Photographing the vibrant temple complex, its traditional architecture, and the lively atmosphere",
        "Exploring the quieter Asakusa Shrine (a Shinto shrine) located next to Senso-ji"
    ],
    "osaka_castle": [
        "Exploring the interior of the reconstructed castle keep (tenshukaku), which now serves as a museum detailing the castle's history and Toyotomi Hideyoshi",
        "Ascending to the observation deck on the top floor of the castle keep for panoramic views of Osaka city and Osaka Castle Park",
        "Strolling through Osaka Castle Park, enjoying the vast green spaces, moats, stone walls, and seasonal blooms (especially cherry blossoms in spring and plum blossoms in late winter)",
        "Photographing the majestic castle, its golden embellishments, and its reflection in the moat",
        "Visiting Nishinomaru Garden within the park for beautiful views of the castle (especially during cherry blossom season, requires separate admission)",
        "Learning about the historical sieges and significance of Osaka Castle in Japanese history"
    ],
    "dotonbori": [
        "Taking iconic photos with the Glico Running Man billboard and other extravagant 3D signs (e.g., Kani Doraku crab, Zubora-ya pufferfish)",
        "Strolling along the Dotonbori canal and its vibrant pedestrianized streets, especially at night when the neon lights are dazzling",
        "Sampling famous Osakan street food (kuidaore) like takoyaki, okonomiyaki, kushikatsu, and ramen from numerous stalls and restaurants",
        "Taking a Tonbori River Cruise for a different perspective of the district from the water",
        "Shopping for souvenirs, fashion, and unique Japanese goods in the area",
        "Experiencing the lively entertainment, including arcades, theaters (like Shochikuza for kabuki), and karaoke bars"
    ],
    "arashiyama_bamboo_grove": [
        "Walking or cycling along the enchanting pathway through the towering stalks of the Sagano Bamboo Forest",
        "Listening to the rustling sound of the bamboo leaves in the wind, a designated 'Soundscape of Japan'",
        "Photographing the magical light filtering through the dense green bamboo canopy",
        "Visiting nearby Tenryu-ji Temple (a UNESCO World Heritage site) with its beautiful garden that backs onto the bamboo grove",
        "Crossing the Togetsukyo Bridge ('Moon Crossing Bridge') over the Hozugawa River for scenic views of Arashiyama",
        "Renting a rowboat on the Hozugawa River or taking the Sagano Romantic Train for more scenic views of the area"
    ],
    "itsukushima_shrine": [
        "Photographing the iconic 'floating' vermilion O-Torii Gate, especially during high tide when it appears to float on the water, and at sunset",
        "Exploring the Itsukushima Shrine complex (a UNESCO World Heritage site), built on stilts over the water, with its distinctive red-lacquered corridors and halls",
        "Walking out to the O-Torii Gate during low tide to see it up close (check tide schedules)",
        "Interacting with the friendly wild sika deer that roam freely on Miyajima Island",
        "Taking the Miyajima Ropeway up Mount Misen for panoramic views of the Seto Inland Sea and hiking its trails",
        "Sampling local Miyajima delicacies like grilled oysters and momiji manju (maple leaf-shaped cakes)"
    ],
    "gyeongbokgung_palace": [
        "Exploring the main halls like Geunjeongjeon (Throne Hall) and pavilions like Gyeonghoeru (Royal Banquet Hall on a pond)",
        "Watching the impressive Royal Guard Changing Ceremony (Sumunjang Gyedaeui) held several times a day at Gwanghwamun and Heungnyemun Gates",
        "Renting a Hanbok (traditional Korean attire) from nearby shops to enter the palace for free and for an immersive photo experience",
        "Visiting the National Folk Museum of Korea and the National Palace Museum of Korea, both located within the palace grounds",
        "Taking a guided tour (available in multiple languages) to learn about the Joseon Dynasty's history and palace architecture",
        "Photographing the intricate Dancheong (colorful painted patterns) on the wooden structures and the serene Hyangwonjeong Pavilion"
    ],
    "n_seoul_tower": [
        "Taking a scenic cable car ride up Namsan Mountain to reach the N Seoul Tower",
        "Ascending to the observatory (digital and analog) for breathtaking 360-degree panoramic views of Seoul's cityscape, day and night",
        "Attaching a 'love lock' with a personal message to the famous railings and tree-like structures on the tower's outdoor terrace",
        "Dining at the revolving N.Grill restaurant or other cafes and eateries within the tower complex, enjoying the views",
        "Photographing the city skyline, especially during sunset or when the city lights twinkle at night; the tower itself is also beautifully illuminated",
        "Exploring Namsan Park surrounding the tower, enjoying its walking trails, botanical gardens, and the Namsan Beacon Mounds"
    ],
    "bukchon_hanok_village": [
        "Walking respectfully through the narrow, hilly alleyways lined with hundreds of well-preserved traditional Korean houses (Hanoks)",
        "Photographing the beautiful tiled roofs, wooden beams, and stone walls of the Hanoks, often with views of modern Seoul in the background",
        "Visiting small, private museums, cultural centers, and craft workshops (e.g., for Gahoe Museum, Donglim Knot Museum, or embroidery workshops) within the village",
        "Renting a Hanbok to enhance the experience and take memorable photos in the traditional setting",
        "Observing the 'Eight Scenic Spots of Bukchon' for the best photo opportunities and views (look for photo spot markers)",
        "Enjoying tea at a traditional Hanok teahouse or Browse unique artisan shops while being mindful of it being a residential area"
    ],
    "myeongdong_shopping_street": [
        "Shopping for Korean cosmetics and skincare products from numerous flagship stores and local brands",
        "Exploring trendy fashion boutiques, shoe stores, and accessory shops catering to K-fashion enthusiasts",
        "Indulging in a wide variety of delicious Korean street food from bustling stalls (e.g., tteokbokki, gyeranppang, tornado potato, grilled cheese lobster)",
        "Visiting large department stores like Lotte Department Store and Shinsegae Department Store for luxury goods and diverse shopping",
        "Catching K-pop related events, Browse K-pop merchandise stores, or spotting advertisements featuring K-pop idols",
        "Visiting Myeongdong Cathedral, a historic Gothic-style Catholic church, as a peaceful contrast to the shopping frenzy"
    ],
    "dmz_korea": [
        "Taking a guided tour to the Korean Demilitarized Zone (DMZ), including the Joint Security Area (JSA/Panmunjom) if accessible",
        "Visiting the Third Infiltration Tunnel, one of several tunnels dug by North Korea under the DMZ",
        "Looking into North Korea from observation posts like Dora Observatory",
        "Learning about the history of the Korean War, the division of Korea, and ongoing inter-Korean relations at the DMZ Exhibition Hall or Imjingak Park",
        "Seeing the Bridge of No Return and the Freedom House (within JSA)",
        "Reflecting on the poignant reminders of a divided nation and hopes for reunification"
    ],
    "busan_gamcheon_culture_village": [
        "Exploring the maze-like alleyways adorned with vibrant street art, colorful murals, and whimsical art installations",
        "Photographing the brightly painted terraced houses cascading down the hillside, often called the 'Machu Picchu of Busan' or 'Santorini of Korea'",
        "Following a stamp tour map to discover hidden artworks and viewpoints throughout the village",
        "Visiting small art galleries, quirky shops, and charming cafes run by local artists and residents",
        "Taking photos with iconic sculptures like 'The Little Prince and the Desert Fox' overlooking the village",
        "Enjoying panoramic views of the village and the port of Busan from various observation decks"
    ],
    "jeju_island": [
        "Hiking Hallasan, South Korea's highest mountain and a dormant volcano, or exploring its surrounding national park trails",
        "Visiting Seongsan Ilchulbong (Sunrise Peak), a UNESCO World Heritage tuff cone, especially for sunrise views",
        "Exploring Manjanggul Cave, one of the longest lava tubes in the world (UNESCO site)",
        "Relaxing on beautiful beaches like Hyeopjae Beach (with views of Biyangdo Island) or Jungmun Saekdal Beach (popular for surfing)",
        "Chasing waterfalls such as Cheonjeyeon Falls ('Pond of God') or Jeongbang Falls (falls directly into the ocean)",
        "Discovering unique geological formations like Jusangjeolli Cliffs (hexagonal lava pillars) and Yongduam Rock (Dragon Head Rock)"
    ],
    "changdeokgung_palace_secret_garden": [
        "Taking a mandatory guided tour of the Huwon (Secret Garden), a stunningly beautiful and expansive rear garden of Changdeokgung Palace (UNESCO World Heritage site)",
        "Admiring the harmonious blend of traditional Korean palace architecture (pavilions, halls) with the natural landscape (ponds, streams, ancient trees) within the Secret Garden",
        "Exploring the main palace buildings of Changdeokgung, such as Injeongjeon (main throne hall) and Donhwamun Gate",
        "Photographing the serene Buyongji Pond with Buyongjeong Pavilion and Juhamnu Pavilion in the Secret Garden",
        "Learning about the lives of the Joseon Dynasty royal family who used this palace and garden for leisure and study",
        "Appreciating the palace's design, which is considered more integrated with its natural surroundings than other Seoul palaces"
    ],
    "great_wall": [
        "Hiking or walking along various restored sections like Mutianyu (family-friendly, cable car/chairlift options), Badaling (most famous, can be crowded), or Jinshanling and Simatai (more rugged, great for photography, partially restored)",
        "Taking a cable car, chairlift, or toboggan ride at sections like Mutianyu for easier access and fun descent",
        "Photographing the vast structure snaking across diverse mountainous landscapes, with its watchtowers and fortifications",
        "Learning about the history of its construction (spanning many dynasties), its purpose as a defensive barrier, and the legends associated with it",
        "Participating in guided tours to understand the historical context and logistical details of visiting specific sections",
        "Considering a visit to less crowded, 'wild' sections (with caution and appropriate preparation) for a more adventurous experience"
    ],
    "forbidden_city": [
        "Exploring the vast imperial palace complex, walking from the Meridian Gate (Wumen) through numerous courtyards and halls to the Gate of Divine Might (Shenwumen)",
        "Admiring the magnificent traditional Chinese palatial architecture, intricate roof details, dragon motifs, and symbolic color schemes (yellow and red)",
        "Visiting key structures like the Hall of Supreme Harmony (Taihedian), Hall of Central Harmony (Zhonghedian), and Hall of Preserving Harmony (Baohedian) on the Outer Court",
        "Exploring the Inner Court, including the Palace of Heavenly Purity (Qianqinggong), Hall of Union (Jiaotaidian), and Palace of Earthly Tranquility (Kunninggong)",
        "Visiting the Palace Museum's extensive collections of imperial treasures, including ceramics, paintings, jade, and clocks, housed in various side halls",
        "Strolling through the Imperial Garden (Yuhuayuan) at the northern end of the complex"
    ],
    "terracotta_army": [
        "Viewing the thousands of life-sized terracotta warriors, archers, chariots, and horses arranged in battle formation in the three main excavation pits (Pit 1, Pit 2, Pit 3)",
        "Admiring the individual details of the soldiers, each with unique facial expressions, hairstyles, and armor, reflecting the incredible craftsmanship",
        "Visiting the Exhibition Hall of Bronze Chariots to see the two remarkably preserved and intricate bronze chariots and horses",
        "Learning about Emperor Qin Shi Huang, the first emperor of China, and his quest for immortality that led to the creation of this mausoleum army",
        "Taking a guided tour or using an audio guide for in-depth historical context and insights into the archaeological discoveries",
        "Photographing the impressive scale of the pits and the ranks of warriors (flash photography is usually prohibited)"
    ],
    "the_bund": [
        "Strolling along the wide waterfront promenade (Zhongshan East First Road) to admire the grand colonial-era buildings in various architectural styles (e.g., Gothic, Baroque, Art Deco) – the 'museum of international architecture'",
        "Photographing the spectacular modern skyline of Pudong across the Huangpu River, featuring the Oriental Pearl Tower, Shanghai Tower, and Shanghai World Financial Center, especially stunning at night",
        "Taking a Huangpu River cruise (day or night) for panoramic views of both the historic Bund and the futuristic Pudong skyline",
        "Visiting the interiors of some heritage buildings (many now house banks, hotels, or luxury shops) or admiring their detailed facades",
        "Dining at upscale restaurants or enjoying drinks at rooftop bars within the Bund buildings, offering magnificent views",
        "Observing locals engaging in morning tai chi, evening strolls, or public dancing along the promenade"
    ],
    "li_river_guilin": [
        "Taking a scenic boat cruise along the Li River from Guilin to Yangshuo (or vice versa) to admire the stunning karst mountain landscapes",
        "Photographing the picturesque limestone peaks, bamboo groves, rice paddies, and water buffalo along the riverbanks – scenes often depicted in traditional Chinese paintings",
        "Opting for a traditional bamboo raft ride (motorized) on sections of the Li River or the Yulong River (a tributary near Yangshuo) for a closer experience with the scenery",
        "Exploring the Reed Flute Cave or Seven Star Park in Guilin before or after the river cruise",
        "Visiting Xingping Town along the Li River, known for the landscape depicted on the 20 RMB banknote",
        "Cycling or hiking in the countryside around Yangshuo to further explore the karst scenery"
    ],
    "potala_palace": [
        "Exploring the majestic Potala Palace, a UNESCO World Heritage site, former winter residence of the Dalai Lamas, with its White Palace (administrative) and Red Palace (religious sections)",
        "Admiring the unique Tibetan architecture, including massive stone walls, golden roofs, intricate murals, thangkas, and numerous chapels and tombs of past Dalai Lamas",
        "Ascending the numerous staircases (visitors need to be acclimatized to the high altitude of Lhasa)",
        "Photographing the imposing palace from various viewpoints, such as Chakpori Hill (for sunrise/sunset views) or Potala Palace Square",
        "Learning about Tibetan Buddhism, the history of the Dalai Lamas, and the cultural significance of the palace",
        "Observing pilgrims performing kora (circumambulation) around the palace"
    ],
    "zhangjiajie_national_forest_park": [
        "Admiring the towering quartz-sandstone pillars and peaks, often shrouded in mist, that inspired the 'Hallelujah Mountains' in the movie Avatar (especially in the Yuanjiajie area)",
        "Riding the Bailong Elevator (Hundred Dragons Elevator), a massive glass elevator built onto the side of a cliff, for dramatic views",
        "Walking across the Zhangjiajie Grand Canyon Glass Bridge (requires separate entry/location from the main park but often combined in tours) for a thrilling experience",
        "Taking cable cars, like the one to Tianzi Mountain or Huangshizhai, for breathtaking panoramic vistas",
        "Hiking along scenic trails, such as the Golden Whip Stream, to enjoy the lush forests, clear streams, and unique rock formations from below",
        "Exploring other key areas like Tianmen Mountain (Heaven's Gate Mountain, often a separate visit) with its cable car, skywalk, and Tianmen Cave"
    ],
    "west_lake_hangzhou": [
        "Taking a leisurely boat ride on the serene West Lake to enjoy its scenic beauty and visit islands like Xiaoyingzhou (Three Ponds Mirroring the Moon)",
        "Walking or cycling along the Su Causeway (Sudi) and Bai Causeway (Baidi), lined with willow and peach trees",
        "Admiring iconic landmarks such as Leifeng Pagoda (rebuilt), Broken Bridge (Duan Qiao, famous in legend), and the Mid-Lake Pavilion",
        "Visiting traditional gardens and temples around the lake, like Lingyin Temple (Temple of the Soul's Retreat) and Guo's Villa",
        "Watching the 'Impression West Lake' outdoor musical performance on the lake at night (seasonal, created by Zhang Yimou)",
        "Photographing the picturesque landscapes, especially during different seasons (e.g., lotus blooms in summer, osmanthus in autumn)"
    ],
    "summer_palace_beijing": [
        "Strolling through the vast imperial garden, a masterpiece of Chinese landscape garden design, centered around Longevity Hill and Kunming Lake",
        "Taking a boat ride on Kunming Lake, especially a traditional dragon boat",
        "Walking along the Long Corridor (Chang Lang), a covered walkway decorated with thousands of paintings",
        "Visiting key structures like the Hall of Benevolence and Longevity (Renshoudian), Tower of Buddhist Incense (Foxiangge) on Longevity Hill, and the Marble Boat",
        "Exploring Suzhou Street, a recreated canal-side shopping street from imperial times",
        "Admiring the beautiful traditional Chinese architecture, bridges (like the Seventeen-Arch Bridge), and landscaped gardens"
    ],
    "petronas_towers": [
        "Visiting the Skybridge, the double-decker bridge connecting the two towers on the 41st and 42nd floors, for close-up views and a unique perspective (book tickets in advance)",
        "Ascending to the Observation Deck on the 86th floor of Tower 2 for breathtaking panoramic views of Kuala Lumpur's skyline",
        "Photographing the iconic twin skyscrapers, an architectural marvel, especially when brilliantly illuminated at night",
        "Shopping at the upscale Suria KLCC mall located at the base of the towers, featuring luxury brands and diverse retail options",
        "Relaxing or strolling in the beautifully landscaped KLCC Park, which includes a man-made lake (Lake Symphony) with water fountain shows, a jogging track, and a children's playground",
        "Visiting nearby attractions like Petrosains Discovery Centre (interactive science museum) or Aquaria KLCC (oceanarium)"
    ],
    "marina_bay_sands": [
        "Visiting the Sands SkyPark Observation Deck on the 57th floor for stunning 360-degree panoramic views of Singapore's skyline, Gardens by the Bay, and the Singapore Strait",
        "Swimming in the world-famous rooftop infinity pool (exclusive access for Marina Bay Sands hotel guests)",
        "Shopping at The Shoppes at Marina Bay Sands, a luxury mall featuring international brands, a canal with sampan rides, and the Digital Light Canvas",
        "Watching the 'Spectra – A Light & Water Show' in the evening, a spectacular outdoor multimedia presentation over Marina Bay",
        "Visiting the ArtScience Museum, with its iconic lotus-inspired design, showcasing art, science, culture, and technology exhibitions",
        "Dining at celebrity chef restaurants or exploring diverse culinary options within the integrated resort, and trying your luck at the casino"
    ],
    "gardens_by_the_bay": [
        "Exploring the two massive cooled conservatories: the Flower Dome (showcasing exotic plants from Mediterranean and semi-arid regions) and the Cloud Forest (featuring a stunning indoor waterfall and plants from tropical highlands)",
        "Walking along the OCBC Skyway, an aerial walkway suspended between two Supertrees, for panoramic views of the Supertree Grove and surrounding gardens",
        "Watching the 'Garden Rhapsody' light and sound show at Supertree Grove in the evening, where the Supertrees come alive with dazzling lights synchronized to music",
        "Strolling through the various outdoor themed gardens, such as the Heritage Gardens, Serene Garden, and the outdoor art sculptures",
        "Photographing the unique horticultural displays, futuristic Supertrees, and innovative sustainable architecture",
        "Letting children play at the Far East Organization Children's Garden with its water play areas and treehouses"
    ],
    "taj_mahal": [
        "Admiring the breathtaking beauty and perfect symmetry of the ivory-white marble mausoleum, a UNESCO World Heritage site and a symbol of eternal love",
        "Photographing the iconic structure from various angles, especially during sunrise or sunset when the light casts different hues on the marble, and its reflection in the central water channels",
        "Strolling through the formal Mughal gardens (Charbagh) that surround the mausoleum, with their pathways, fountains, and reflecting pools",
        "Learning about the poignant love story of Mughal emperor Shah Jahan and his wife Mumtaz Mahal, for whom the Taj Mahal was built",
        "Observing the intricate marble inlay work (pietra dura) featuring semi-precious stones, and the detailed calligraphy on the monument's facade",
        "Visiting the mosque and the guesthouse (jawab) that flank the main mausoleum, providing architectural balance"
    ],
    "angkor_wat": [
        "Exploring the vast temple complex of Angkor Wat, the world's largest religious monument, admiring its iconic five lotus-bud towers and intricate bas-reliefs",
        "Watching the sunrise over Angkor Wat, a magical experience as the temple silhouette emerges against the colorful sky (can be very crowded)",
        "Taking a guided tour to understand the history of the Khmer Empire, the Hindu and Buddhist symbolism, and the architectural features of the temple",
        "Examining the detailed stone carvings and extensive galleries depicting scenes from Hindu epics like the Ramayana and Mahabharata, and historical events",
        "Photographing the temple's reflection in the surrounding moat and its grandeur from different perspectives",
        "Visiting other nearby temples in the Angkor Archaeological Park, such as Angkor Thom (Bayon, Baphuon), Ta Prohm (Tomb Raider temple), and Banteay Srei"
    ],
    "ha_long_bay": [
        "Taking an overnight cruise or a day boat trip to explore the stunning seascape of thousands of limestone karsts and islets rising from the emerald green waters (UNESCO World Heritage site)",
        "Kayaking or taking a bamboo boat ride through hidden lagoons, caves, and around the towering rock formations for a closer experience",
        "Visiting impressive caves and grottoes, such as Thien Cung Cave (Heavenly Palace Cave) or Dau Go Cave (Wooden Stakes Cave), with their stalactites and stalagmites",
        "Swimming in the calm waters or relaxing on small, secluded beaches (e.g., Titop Island beach)",
        "Photographing the breathtaking and often misty scenery, especially at sunrise or sunset",
        "Learning about the local culture by visiting floating fishing villages or pearl farms"
    ],
    "mount_everest": [
        "Trekking to Everest Base Camp (South Base Camp in Nepal or North Base Camp in Tibet) for close-up views of Mount Everest and the surrounding Himalayan giants (requires significant preparation and acclimatization)",
        "Taking a scenic mountain flight from Kathmandu for breathtaking aerial views of Mount Everest and the Himalayan range (a popular option for non-trekkers)",
        "Photographing the world's highest peak, especially during sunrise or sunset when alpenglow illuminates the summit",
        "Learning about Sherpa culture and mountaineering history in villages like Namche Bazaar (on the Nepal EBC trek)",
        "Visiting monasteries like Tengboche Monastery (Nepal) for spiritual insights and stunning mountain backdrops",
        "For experienced mountaineers: attempting to summit Mount Everest (a highly challenging and expensive endeavor)"
    ],
    "bagan": [
        "Exploring the vast archaeological zone, home to thousands of ancient Buddhist temples, pagodas, and stupas dating from the 9th to 13th centuries (UNESCO World Heritage site)",
        "Renting an e-bike or horse cart to navigate the sandy tracks and discover both major temples (like Ananda, Shwezigon, Thatbyinnyu, Dhammayangyi) and smaller, hidden gems",
        "Watching the sunrise or sunset over the temple-strewn plains from a high vantage point (e.g., a designated viewing mound or a temple terrace where permitted)",
        "Taking a hot air balloon ride over Bagan at sunrise for an unforgettable panoramic view of the temples (seasonal, typically October to April)",
        "Photographing the unique landscape of ancient religious structures against the backdrop of the Irrawaddy River and distant hills",
        "Learning about the history of the Pagan Kingdom and Buddhist art and architecture through temple murals and local guides"
    ],
    "grand_palace_wat_phra_kaew": [
        "Exploring the magnificent Grand Palace complex, the former official residence of the Kings of Siam (Thailand), with its stunning traditional Thai architecture",
        "Visiting Wat Phra Kaew (Temple of the Emerald Buddha) within the palace grounds to see the highly revered Emerald Buddha statue, carved from a single jade stone",
        "Admiring the intricate details of the golden spires (chedis), colorful mosaics made of glass and porcelain, ornate gables, and mythical guardian statues (yakshas and kinnaras)",
        "Photographing the dazzling exteriors of the royal halls (e.g., Chakri Maha Prasat Hall), temples, and courtyards",
        "Learning about Thai history, royal traditions, and Buddhist art and iconography (dressing respectfully is required: no shorts, sleeveless shirts)",
        "Observing the Ramakien murals (Thai version of the Ramayana) that adorn the walls of the cloister surrounding Wat Phra Kaew"
    ],

    # 歐洲地標 (Europe)
    "eiffel_tower": [
        "Ascending to the different observation platforms (1st floor, 2nd floor, summit) for stunning panoramic views of Paris",
        "Enjoying a romantic meal or champagne at Le Jules Verne restaurant (2nd floor) or other tower eateries",
        "Picnicking on the Champ de Mars park with the Eiffel Tower as a magnificent backdrop",
        "Photographing the iconic structure day and night, especially during the hourly sparkling lights show after sunset",
        "Taking a Seine River cruise that offers unique perspectives of the tower from the water",
        "Learning about its history, engineering, and construction at the first-floor exhibition or through guided tours"
    ],
    "louvre_museum": [
        "Viewing iconic masterpieces like the Mona Lisa, Venus de Milo, Winged Victory of Samothrace, and Liberty Leading the People",
        "Exploring diverse and extensive art collections spanning from ancient civilizations (Egyptian, Greek, Roman) to 19th-century European painting and sculpture",
        "Taking a guided tour or using an audio guide to navigate the vast museum and focus on key exhibits or specific interests",
        "Admiring the architecture of the former royal palace (Louvre Palace) and the modern glass Louvre Pyramid designed by I. M. Pei",
        "Strolling through the Tuileries Garden (Jardin des Tuileries) adjacent to the museum, leading towards Place de la Concorde",
        "Photographing the impressive exterior, the pyramid, and select artworks (where permitted without flash)"
    ],
    "mont_saint_michel": [
        "Walking up the winding Grande Rue, lined with shops and restaurants, to reach the magnificent Benedictine Abbey at the summit",
        "Taking a guided tour (or audio guide) of the Mont Saint-Michel Abbey to explore its Romanesque and Gothic sections, cloister, and refectory",
        "Admiring the breathtaking views of the surrounding bay and tidal flats from the abbey terraces and ramparts",
        "Photographing the island commune at different times of the day and tides, especially during high tide when it's completely surrounded by water, or at sunrise/sunset",
        "Exploring the small museums, chapels, and historic houses within the village",
        "Witnessing the dramatic tidal changes (some of the highest in Europe) from a safe vantage point on the island or the mainland causeway"
    ],
    "arc_de_triomphe": [
        "Climbing the stairs (or taking the elevator part-way) to the rooftop observation deck for panoramic 360-degree views of Paris, including the Champs-Élysées, Eiffel Tower, Sacré-Cœur Basilica, and La Défense",
        "Admiring the intricate Neoclassical sculptures and reliefs on the arch's facade, depicting scenes from French military history, particularly Napoleonic victories",
        "Visiting the Tomb of the Unknown Soldier beneath the arch and witnessing the rekindling of the eternal flame each evening at 6:30 PM",
        "Photographing the monument, the twelve avenues radiating from Place Charles de Gaulle (formerly Place de l'Étoile), and the surrounding cityscape",
        "Learning about its historical significance and the events it commemorates through the small museum inside the arch",
        "Observing major events like the Bastille Day military parade (July 14th) which passes through the arch"
    ],
    "big_ben": [ # Elizabeth Tower
        "Photographing the iconic clock tower (officially Elizabeth Tower), the Great Bell (Big Ben), and the adjacent Houses of Parliament (Palace of Westminster) from Westminster Bridge or Parliament Square",
        "Hearing the famous chimes of Big Ben, especially the distinctive Westminster Quarters",
        "Taking a guided tour of the Houses of Parliament (when Parliament is not in session, book in advance) to see inside this historic building",
        "Walking along the South Bank of the River Thames for classic views of Big Ben and the Houses of Parliament, especially at dusk or night when illuminated",
        "Visiting nearby attractions such as Westminster Abbey, the Churchill War Rooms, and the London Eye",
        "Learning about its history, recent restoration, and the workings of the UK Parliament"
    ],
    "stonehenge": [
        "Walking the designated pathway around the prehistoric stone circle, observing the massive sarsen stones and smaller bluestones",
        "Listening to the informative audio guide (available in multiple languages) to learn about the history, construction phases, and various theories about Stonehenge's purpose (e.g., astronomical observatory, burial site, ceremonial center)",
        "Visiting the world-class exhibition and visitor centre to see archaeological finds, reconstructions of a Neolithic village, and interactive displays",
        "Photographing the mysterious ancient monument against the backdrop of Salisbury Plain, especially during sunrise or sunset for dramatic lighting (special access may be required for inner circle access during these times)",
        "Considering the astronomical alignments, particularly during the summer and winter solstices when special events are often held",
        "Exploring the surrounding landscape, which is rich in other Neolithic and Bronze Age sites, including Woodhenge and Durrington Walls"
    ],
    "tower_of_london": [
        "Taking a captivating tour led by a Yeoman Warder (Beefeater), who shares fascinating stories, historical anecdotes, and traditions of the Tower",
        "Viewing the spectacular Crown Jewels, a dazzling collection of royal regalia including crowns, orbs, and sceptres, housed securely in the Jewel House",
        "Exploring the White Tower, the oldest part of the castle (a Norman keep), which houses the Royal Armouries collection",
        "Walking the ancient ramparts and learning about the Tower's multifaceted history as a royal palace, fortress, prison, and place of execution",
        "Seeing the famous resident ravens and learning about the legend that the kingdom and Tower will fall if they leave",
        "Visiting sites of historical significance such as Traitors' Gate, the Bloody Tower, and the execution site on Tower Green"
    ],
    "buckingham_palace": [
        "Watching the iconic Changing of the Guard ceremony (check schedule and arrive early for a good view), a display of British pageantry with guards in red tunics and bearskin hats",
        "Photographing the grand facade of Buckingham Palace, the official London residence of the UK monarch, and the impressive Victoria Memorial in front",
        "Touring the magnificent State Rooms during the Summer Opening (typically July to September, when the King is not in residence, book tickets in advance)",
        "Visiting the Queen's Gallery to see rotating exhibitions of artworks and treasures from the Royal Collection",
        "Exploring the Royal Mews to see historic royal carriages and vehicles",
        "Strolling through St. James's Park or Green Park, adjacent to the palace, for pleasant walks and views"
    ],
    "colosseum": [
        "Taking a guided historical tour (or using an audio guide) to learn about the gladiatorial contests, wild animal hunts, public spectacles, and the lives of Roman citizens and emperors associated with the amphitheater",
        "Exploring the different levels of the ancient Flavian Amphitheatre, including the arena floor (if accessible with your ticket), the hypogeum (underground tunnels and chambers), and the upper tiers for panoramic views",
        "Photographing the imposing structure, a symbol of Imperial Rome, noting its architectural features like arches, columns, and the remaining sections of the outer wall",
        "Visiting the adjacent Roman Forum (Foro Romano) and Palatine Hill (Colle Palatino), which are usually included in a combined ticket, to explore the heart of ancient Roman life, politics, and mythology",
        "Learning about Roman engineering, architecture, and the social importance of the games held in the Colosseum",
        "Imagining the roar of the crowds and the dramatic events that once unfolded within its ancient walls, especially when illuminated at night"
    ],
    "leaning_tower_of_pisa": [
        "Climbing the spiral staircase of 294 steps to the top of the leaning bell tower (Torre Pendente di Pisa) for panoramic views of Pisa and the Piazza dei Miracoli (book tickets well in advance as numbers are limited)",
        "Taking the classic and fun forced-perspective photo where you appear to be 'holding up' or 'pushing over' the Leaning Tower",
        "Visiting the magnificent Pisa Cathedral (Duomo di Santa Maria Assunta) and the impressive Baptistery of St. John (Battistero di San Giovanni) located in the same Square of Miracles (Piazza dei Miracoli)",
        "Admiring the beautiful Romanesque architecture of white marble that characterizes the entire complex",
        "Strolling around the well-manicured lawns of the Piazza dei Miracoli and exploring the Camposanto Monumentale (monumental cemetery)",
        "Learning about the history of the tower's construction, the reasons for its unintended tilt, and the centuries of efforts to stabilize it"
    ],
    "trevi_fountain": [
        "Tossing a coin over your right shoulder with your left hand into the fountain – tradition holds that this ensures your return to Rome (toss two coins for a new romance, three for marriage)",
        "Admiring the spectacular Baroque sculptures depicting Oceanus (god of the sea) on a shell-shaped chariot pulled by sea horses and tritons, set against the backdrop of Palazzo Poli",
        "Photographing the magnificent fountain, one of the most famous in the world, especially beautiful when illuminated at night (be prepared for crowds)",
        "Enjoying delicious Italian gelato from a nearby gelateria while people-watching and soaking in the lively atmosphere",
        "Learning about the history and design of the fountain, completed by Nicola Salvi and Giuseppe Pannini, and its connection to ancient Roman aqueducts",
        "Visiting at different times of day, such as early morning, to experience it with fewer crowds and different lighting conditions"
    ],
    "st_peters_basilica": [
        "Exploring the vast and awe-inspiring interior of St. Peter's Basilica, the largest Christian church in the world and a masterpiece of Renaissance architecture",
        "Admiring Michelangelo's magnificent dome from the inside and considering climbing to the top (cupola) for breathtaking panoramic views of St. Peter's Square, Vatican City, and Rome (requires a ticket, involves stairs and an elevator option for part of the way)",
        "Viewing renowned masterpieces such as Michelangelo's Pietà (sculpture of Mary holding Christ's body), Bernini's bronze Baldachin over the papal altar, and various ornate chapels and papal tombs",
        "Visiting the Vatican Grottoes (Papal Tombs) located beneath the basilica, where many popes are interred",
        "Attending a Papal Mass, audience, or the Angelus prayer in St. Peter's Square if your visit coincides with these events (check Vatican schedules)",
        "Strolling through the immense St. Peter's Square (Piazza San Pietro), designed by Bernini, with its grand colonnades, central obelisk, and twin fountains"
    ],
    "sagrada_familia": [
        "Admiring the extraordinary and unique facades of Antoni Gaudí's unfinished masterpiece: the Nativity Facade (celebrating Christ's birth), the Passion Facade (depicting his suffering), and the still-under-construction Glory Facade",
        "Exploring the breathtaking interior, a forest of tree-like columns that branch out to support the roof, illuminated by vibrant stained-glass windows that create an ethereal play of light and color",
        "Taking a guided tour or using an audio guide (highly recommended) to understand Gaudí's visionary architectural concepts, complex symbolism, and the ongoing construction process (book tickets well in advance online)",
        "Ascending one of the towers (Nativity or Passion, requires separate ticket and booking) for close-up views of the spires and panoramic vistas of Barcelona",
        "Visiting the museum located underneath the basilica to learn about Gaudí's life, design models, construction techniques, and the history of the Sagrada Família",
        "Photographing the incredibly detailed architectural elements, both inside and out, reflecting Gaudí's deep connection with nature and spirituality"
    ],
    "alhambra": [
        "Exploring the Nasrid Palaces, the heart of the Alhambra, with their exquisite stucco work, intricate geometric tile mosaics (azulejos), delicate muqarnas (stalactite vaulting), and tranquil courtyards like the Court of the Lions and Court of the Myrtles (book tickets well in advance, as entry is timed and limited)",
        "Wandering through the serene Generalife gardens, the summer palace of the Nasrid rulers, with its beautiful patios, fountains, flowerbeds, and scenic pathways",
        "Visiting the Alcazaba, the oldest part of the Alhambra, a formidable fortress with towers offering panoramic views over the city of Granada and the surrounding landscape",
        "Admiring the Palace of Charles V, a contrasting Renaissance-style building within the Alhambra complex, now housing museums",
        "Photographing the stunning Moorish architecture, intricate details, and the interplay of light, water, and shadow throughout the complex",
        "Learning about the history of the last Muslim emirs in Spain, the Reconquista, and the subsequent Christian modifications to the palace"
    ],
    "brandenburg_gate": [
        "Walking through the iconic neoclassical triumphal arch, a potent symbol of German history, division, and reunification",
        "Photographing the gate, especially when illuminated at night, and the Quadriga statue (a chariot drawn by four horses, driven by Victoria, the Roman goddess of victory) that crowns it",
        "Visiting Pariser Platz, the grand square in front of the gate, which also houses embassies (e.g., US, French) and the Hotel Adlon",
        "Learning about its historical significance, from its construction under King Frederick William II of Prussia to its role during the Napoleonic Wars, the Nazi era, the Cold War (when it stood near the Berlin Wall), and German reunification",
        "Reflecting at nearby historical sites such as the Reichstag Building (German Parliament), the Memorial to the Murdered Jews of Europe, and the remnants of the Berlin Wall",
        "Observing the bustling atmosphere and often, street performers or public events taking place in Pariser Platz"
    ],
    "neuschwanstein_castle": [
        "Taking a mandatory guided tour of the fairytale-like interiors of King Ludwig II's dream castle, inspired by Richard Wagner's operas and medieval legends (tickets must be purchased in Hohenschwangau village and are timed; book online in advance to secure a spot)",
        "Photographing the castle from Marienbrücke (Mary's Bridge), a pedestrian bridge offering the classic, breathtaking postcard view of Neuschwanstein perched on its rugged hill with the Bavarian Alps in the background",
        "Hiking, taking a horse-drawn carriage, or riding a shuttle bus up the steep road from Hohenschwangau village to the castle entrance",
        "Admiring the scenic beauty of the Bavarian Alps, forests, and nearby lakes (Alpsee and Schwansee) surrounding the castle",
        "Learning about the eccentric 'Mad' King Ludwig II of Bavaria, his passion for art and architecture, and the romantic inspirations behind the castle's design",
        "Visiting the nearby Hohenschwangau Castle, King Ludwig II's childhood home, for more royal history and contrasting architectural style"
    ],
    "acropolis_of_athens": [
        "Exploring the Parthenon, the magnificent Doric temple dedicated to the goddess Athena Parthenos, an enduring symbol of Ancient Greece, democracy, and Western civilization",
        "Visiting other significant ancient structures on the Sacred Rock, such as the Erechtheion (with its iconic Porch of the Caryatids), the Propylaea (the monumental gateway), and the Temple of Athena Nike",
        "Walking to the ancient Theatre of Dionysus on the southern slope, considered the birthplace of Greek tragedy, and the well-preserved Odeon of Herodes Atticus (still used for performances)",
        "Enjoying breathtaking panoramic views of Athens, including the Plaka district, Mount Lycabettus, and the Saronic Gulf, from the summit of the Acropolis",
        "Visiting the Acropolis Museum, located at the foot of the Acropolis, to see original sculptures, friezes (including parts of the Parthenon Marbles), and artifacts found on the site, displayed in a modern architectural setting",
        "Photographing the ancient ruins against the backdrop of the modern city, especially beautiful during sunrise or sunset"
    ],
    "santorini_oia": [
        "Exploring the charming, narrow, winding pathways of Oia village, famous for its whitewashed cave houses (yposkafa), blue-domed churches, and vibrant bougainvillea",
        "Watching and photographing the world-renowned sunset over the caldera from Oia, typically from the ruins of the Venetian castle (Kastro) or designated sunset viewing spots (arrive very early to secure a good spot as it gets extremely crowded)",
        "Taking a boat tour to the volcanic islands of Nea Kameni (to hike on the crater) and Palea Kameni (to swim in the hot springs), and Thirassia island",
        "Relaxing on unique volcanic sand beaches around Santorini, such as Red Beach (Kokkini Paralia), Perissa Beach, or Kamari Beach (Oia itself is on cliffs, not a beach town)",
        "Hiking the scenic caldera trail between Fira (the capital) and Oia (approx. 3-4 hours) for stunning views",
        "Indulging in local Santorinian cuisine and distinctive Assyrtiko wines at cliffside restaurants with caldera views"
    ],
    "canals_of_venice": [
        "Taking a classic gondola ride through the narrow, picturesque canals, often accompanied by a serenading gondolier",
        "Exploring the city on foot, getting lost in the maze-like streets (calli) and discovering charming squares (campi) and countless bridges",
        "Cruising along the Grand Canal on a vaporetto (public water bus) to see famous landmarks like the Rialto Bridge, Doge's Palace, and Ca' d'Oro from the water",
        "Photographing the unique cityscape with its historic buildings seemingly rising from the water, colorful reflections, and bustling canal life",
        "Visiting iconic sights such as St. Mark's Square (Piazza San Marco), St. Mark's Basilica, and the Doge's Palace",
        "Enjoying cicchetti (Venetian tapas) and wine at traditional bacari (local bars)"
    ],
    "florence_cathedral_duomo": [ # Cattedrale di Santa Maria del Fiore
        "Admiring Brunelleschi's magnificent red-tiled dome, an architectural marvel of the Renaissance, and climbing to the top (463 steps, book well in advance) for breathtaking panoramic views of Florence",
        "Exploring the interior of the Florence Cathedral (Cattedrale di Santa Maria del Fiore), noting its Gothic architecture, stained glass windows, and Vasari's frescoes of the Last Judgment inside the dome",
        "Visiting Giotto's Campanile (bell tower) and climbing its 414 steps for another stunning perspective of the Duomo and the city",
        "Admiring the Florence Baptistery (Battistero di San Giovanni) with its famous bronze doors, especially Ghiberti's 'Gates of Paradise'",
        "Visiting the Museo dell'Opera del Duomo to see original artworks from the Duomo complex, including Ghiberti's original doors and Michelangelo's 'The Deposition' Pietà",
        "Photographing the intricate white, green, and pink marble facade of the Duomo, Campanile, and Baptistery in Piazza del Duomo"
    ],
    "anne_frank_house": [
        "Taking a poignant and reflective tour through the Secret Annex (Achterhuis) where Anne Frank and her family hid from Nazi persecution during World War II (book tickets online months in advance, as they sell out quickly)",
        "Viewing Anne Frank's original diary and other personal belongings displayed in the museum",
        "Learning about the lives of the people who hid in the annex, their helpers, and the historical context of the Holocaust and Jewish persecution in Amsterdam",
        "Reflecting on Anne's story, her writings, and the enduring messages of tolerance, human rights, and hope",
        "Visiting the museum exhibitions that provide further information about the persecution of Jews during the war and contemporary issues of discrimination",
        "Seeing the unassuming exterior of the canal house on Prinsengracht 263 and imagining the hidden life within"
    ],
    "canals_of_amsterdam": [
        "Taking a scenic canal cruise (day or evening) through the Grachtengordel (canal belt), a UNESCO World Heritage site, to admire the historic gabled canal houses, bridges, and houseboats",
        "Renting a private boat or pedal boat (canal bike) to explore the canals at your own pace",
        "Walking or cycling along the picturesque canals, such as Prinsengracht, Keizersgracht, and Herengracht, enjoying the charming atmosphere",
        "Photographing the iconic canal scenes, including narrow houses, beautiful bridges (like Magere Brug 'Skinny Bridge'), and reflections in the water",
        "Visiting canal house museums like Museum Willet-Holthuysen or Museum Van Loon to see how wealthy Amsterdammers lived in the Golden Age",
        "Enjoying a drink or meal at a waterside cafe or restaurant along the canals"
    ],
    "charles_bridge_prague": [
        "Walking across the historic Charles Bridge (Karlův most), a medieval stone arch bridge connecting the Old Town and Lesser Town (Malá Strana) over the Vltava River",
        "Admiring the 30 statues and statuaries of saints that line the bridge, including the famous statue of St. John of Nepomuk (touching the plaque is said to bring good luck or ensure a return to Prague)",
        "Enjoying panoramic views of Prague Castle, St. Vitus Cathedral, the Vltava River, and the surrounding city skyline from the bridge",
        "Photographing the bridge, its Gothic towers (Old Town Bridge Tower and Lesser Town Bridge Towers), and the bustling atmosphere, especially beautiful at dawn, dusk, or at night",
        "Observing street artists, musicians, and vendors who often line the bridge (can be very crowded during peak season)",
        "Climbing one of the bridge towers for a higher vantage point and stunning photos"
    ],
    "red_square_st_basils_cathedral": [
        "Admiring the iconic, vibrantly colored onion domes of St. Basil's Cathedral (Cathedral of Vasily the Blessed), a unique masterpiece of Russian architecture, located at the southern end of Red Square",
        "Exploring Red Square (Krasnaya Ploshchad), the historic central square of Moscow, and visiting other significant landmarks such as Lenin's Mausoleum, the fortified walls of the Moscow Kremlin, the State Historical Museum, and the GUM department store",
        "Taking a guided tour or visiting the interior of St. Basil's Cathedral (now a museum) to see its chapels and learn about its history",
        "Photographing the stunning ensemble of Red Square, especially the contrast between St. Basil's, the Kremlin towers, and the red brick of the Historical Museum",
        "Learning about the historical events that have taken place in Red Square, a UNESCO World Heritage site and a focal point of Russian history and culture",
        "Visiting the GUM department store for its unique architecture, high-end shops, and traditional Russian ice cream"
    ],
    "edinburgh_castle": [
        "Exploring the historic fortress perched atop Castle Rock, dominating the skyline of Edinburgh",
        "Viewing the Honours of Scotland (the Scottish Crown Jewels) and the Stone of Destiny, ancient symbols of Scottish royalty",
        "Visiting St. Margaret's Chapel, the oldest surviving building in Edinburgh, and the Great Hall with its impressive timber roof and collection of arms and armour",
        "Witnessing the firing of the One O'Clock Gun (Monday to Saturday), a time-honored tradition",
        "Walking along the castle ramparts for panoramic views of Edinburgh city, including the Royal Mile, Arthur's Seat, and the Firth of Forth",
        "Learning about the castle's rich and often turbulent history, its role in Scottish wars, and its famous residents through exhibitions and guided tours"
    ],
    "matterhorn": [
        "Photographing the iconic pyramidal peak of the Matterhorn, one of the world's most recognizable mountains, from Zermatt (Switzerland) or Breuil-Cervinia (Italy)",
        "Taking a cable car or cogwheel train to viewpoints like Gornergrat (for stunning Matterhorn and glacier views), Klein Matterhorn (Matterhorn Glacier Paradise - Europe's highest cable car station), or Sunnegga/Rothorn for different perspectives",
        "Hiking or mountain biking on the numerous trails around Zermatt that offer spectacular views of the Matterhorn and surrounding Alpine scenery (e.g., Riffelsee lake trail for reflections)",
        "Skiing or snowboarding in the Zermatt-Cervinia ski area, which offers year-round skiing on the Theodul Glacier with the Matterhorn as a backdrop",
        "Learning about the history of the first ascent and mountaineering in the region at the Matterhorn Museum (Zermatlantis) in Zermatt",
        "Enjoying Alpine cuisine and the charming car-free village atmosphere of Zermatt"
    ],
    "palace_of_versailles": [
        "Exploring the opulent State Apartments of the Palace of Versailles, including the magnificent Hall of Mirrors (Galerie des Glaces), the King's Grand Apartment, and the Queen's Grand Apartment",
        "Wandering through the vast and meticulously designed Gardens of Versailles (Jardins de Versailles) by André Le Nôtre, with their formal parterres, fountains (musical fountain shows are held on certain days), canals, and groves",
        "Visiting the Grand Trianon and Petit Trianon, smaller palaces within the estate, offering a more intimate glimpse into royal life, and Marie Antoinette's Hamlet (Le Hameau de la Reine), a rustic retreat",
        "Taking a guided tour or using an audio guide to learn about the history of the French monarchy (especially Louis XIV, Louis XV, and Louis XVI), courtly life, and the significant historical events that took place at Versailles (like the signing of the Treaty of Versailles)",
        "Renting a boat on the Grand Canal or exploring the gardens by bicycle or electric golf cart",
        "Photographing the lavish Baroque architecture, gilded interiors, impressive sculptures, and expansive landscapes"
    ],

    # 北美地標 (North America)
    "statue_of_liberty": [
        "Taking a ferry from Battery Park (Manhattan) or Liberty State Park (New Jersey) to Liberty Island",
        "Visiting the Statue of Liberty Museum to learn about its history, construction, symbolism, and Frederic Auguste Bartholdi's design",
        "Accessing the pedestal for closer views of the statue and panoramic views of New York Harbor and the Manhattan skyline (requires advance booking)",
        "Climbing to the crown for a unique, albeit small, viewing experience (requires very limited, advance booking months ahead)",
        "Combining the visit with a trip to Ellis Island and the National Museum of Immigration, which is usually included in the same ferry ticket",
        "Walking around the perimeter of Liberty Island for different photo opportunities of Lady Liberty and the surrounding views"
    ],
    "golden_gate_bridge": [
        "Walking or biking across the bridge on the dedicated pedestrian and bicycle paths (check access times for each side)",
        "Photographing the iconic Art Deco suspension bridge, painted in 'International Orange,' from various renowned viewpoints like Battery Spencer (Marin Headlands), Vista Point (north end), Fort Point National Historic Site (south end, underneath the bridge), or Baker Beach",
        "Visiting the Golden Gate Bridge Welcome Center (south end) to learn about the bridge's history, engineering marvels, and see exhibits",
        "Driving across the bridge for a classic experience (toll applies southbound into San Francisco)",
        "Exploring the Golden Gate National Recreation Area, which encompasses areas on both sides of the bridge offering trails and views",
        "Taking a bay cruise that sails under the Golden Gate Bridge and around Alcatraz Island for different perspectives"
    ],
    "grand_canyon": [
        "Hiking along the Rim Trail or venturing down into the canyon on trails like Bright Angel Trail or South Kaibab Trail (South Rim) or North Kaibab Trail (North Rim) – be aware of difficulty, heat, and need for water",
        "Taking a mule ride along the rim or into the canyon (book well in advance as these are very popular)",
        "Watching and photographing the spectacular sunrise or sunset over the canyon from popular viewpoints like Mather Point, Yavapai Point, Hopi Point (South Rim), or Bright Angel Point (North Rim) as the canyon walls change colors dramatically",
        "Visiting various viewpoints along the South Rim Drive (using the free park shuttle buses for much of the year) or the North Rim Scenic Drives",
        "Taking a helicopter or fixed-wing aircraft tour for a breathtaking aerial perspective of the vastness of the Grand Canyon (optional, can be costly)",
        "Learning about the geology, ecology, and human history (including Native American cultures) at visitor centers and ranger programs"
    ],
    "hollywood_sign": [
        "Hiking to viewpoints for relatively close-up views of the sign, such as from trails in Griffith Park (e.g., Hollyridge Trail, Wonder View Trail, or near Griffith Observatory) or from Lake Hollywood Park (great for ground-level photos with the sign)",
        "Photographing the iconic white capital letters on Mount Lee from various locations throughout Hollywood and Los Angeles, including the Hollywood & Highland Center observation decks",
        "Visiting Griffith Observatory for excellent panoramic views of the Hollywood Sign, the Los Angeles basin, and the Pacific Ocean (on clear days), as well as its science exhibits",
        "Learning about the history of the sign (originally 'HOLLYWOODLAND') and its significance as a symbol of the American film industry",
        "Driving through the Hollywood Hills for different perspectives (be mindful of residential areas and parking restrictions)",
        "Taking a guided Hollywood tour that often includes stops at optimal sign viewing locations"
    ],
    "white_house": [
        "Viewing and photographing the iconic exterior of the White House from the North Side (Pennsylvania Avenue, in front of Lafayette Square) or the South Side (from the Ellipse)",
        "Taking a public tour of select rooms in the East Wing (requires advance request through a U.S. Member of Congress for citizens, or through one's embassy for foreign nationals, months in advance; security is very strict)",
        "Visiting the White House Visitor Center (located nearby) for exhibits on the history of the White House, its architecture, furnishings, first families, and presidential life",
        "Exploring President's Park, which includes Lafayette Square to the north (with statues of revolutionary war heroes) and the Ellipse to the south",
        "Learning about the history of the U.S. presidency and the symbolic importance of the White House as the executive mansion",
        "Observing any official motorcades or security measures, which can be a unique part of the D.C. experience"
    ],
    "mount_rushmore": [
        "Viewing and photographing the colossal carved faces of U.S. Presidents George Washington, Thomas Jefferson, Theodore Roosevelt, and Abraham Lincoln in the granite of Mount Rushmore",
        "Walking the Presidential Trail, a boardwalk loop that provides closer views and different angles of the sculptures",
        "Visiting the Lincoln Borglum Visitor Center and Museum to learn about the history of the carving, the sculptor Gutzon Borglum, the tools and techniques used, and the depicted presidents",
        "Attending the evening lighting ceremony (held seasonally, typically May to September), which includes a ranger talk and the illumination of the monument",
        "Exploring the Sculptor's Studio to see original plaster models and tools used in the carving process",
        "Learning about the significance of each president chosen and the broader history of the United States represented by the memorial"
    ],
    "times_square": [
        "Experiencing the dazzling and overwhelming display of massive digital billboards, flashing neon lights, and advertisements that illuminate the square 24/7",
        "People-watching in one of the world's busiest and most famous pedestrian intersections, soaking in the vibrant and energetic atmosphere",
        "Attending a Broadway show in the renowned Theater District, which is centered around Times Square (purchase tickets in advance or try the TKTS booth for same-day discounts)",
        "Shopping at flagship stores of international brands, unique boutiques, and souvenir shops that line the square",
        "Dining at diverse restaurants, from themed eateries and fast-food chains to upscale dining options",
        "Photographing the iconic scenery, costumed characters (be aware they may expect tips), and the general buzz, especially stunning at night or during the New Year's Eve ball drop"
    ],
    "cn_tower": [
        "Ascending via high-speed glass-fronted elevators to the main observation levels: the LookOut Level (with floor-to-ceiling panoramic window walls) and the famous Glass Floor (for a thrilling view 342m/1,122ft straight down)",
        "Going even higher to the SkyPod, one of the highest observation platforms in the world (an additional 33 storeys above the main observation level, requires separate ticket)",
        "Experiencing the EdgeWalk (seasonal, weather permitting), the world’s highest full-circle, hands-free walk on a 1.5m (5ft) wide ledge encircling the top of the Tower’s main pod, 356m/1,168ft above the ground (for thrill-seekers, book well in advance)",
        "Dining at the award-winning 360 Restaurant, a revolving restaurant that completes a full rotation approximately every 72 minutes, offering stunning views and a cellar in the sky",
        "Photographing the panoramic cityscape of Toronto, Lake Ontario, and on clear days, even Niagara Falls or New York State",
        "Learning about the tower's construction, engineering, and its role as a communications tower through exhibits and informational displays"
    ],
    "chichen_itza": [
        "Exploring the magnificent El Castillo (Pyramid of Kukulcán), the iconic step-pyramid that dominates the site, and learning about its Mayan astronomical alignments (e.g., the serpent shadow effect during the spring and autumn equinoxes)",
        "Visiting other significant structures within the ancient Mayan city, such as the Great Ball Court (the largest in Mesoamerica), the Temple of the Warriors (with its Chac Mool figure and colonnades), the Group of a Thousand Columns, and the Observatory (El Caracol)",
        "Observing the intricate stone carvings, Mayan hieroglyphs, and depictions of deities like Kukulcán (the feathered serpent) and Chaac (the rain god) on the buildings",
        "Swimming in a nearby sacred cenote (natural sinkhole), such as Ik Kil or Yokdzonot, for a refreshing break after exploring the hot ruins (often combined with Chichen Itza tours)",
        "Taking a guided tour (highly recommended) to understand the rich history, culture, rituals, and astronomical knowledge of the Mayan civilization that flourished at Chichen Itza (a UNESCO World Heritage site)",
        "Photographing the impressive pyramids, ancient temples, and unique architectural features of this important archaeological site"
    ],
    "niagara_falls": [
        "Taking the 'Maid of the Mist' (USA side) or 'Hornblower Niagara Cruises/Niagara City Cruises' (Canada side) boat tour for an up-close and powerful experience of the mist and roar of the Horseshoe Falls, American Falls, and Bridal Veil Falls",
        "Experiencing the 'Cave of the Winds' (USA side), where you walk on wooden walkways near the base of the Bridal Veil Falls and get drenched by the spray",
        "Walking along the Niagara Parkway (Canada) or within Niagara Falls State Park (USA) for various viewpoints of the falls, including Terrapin Point (USA) and Table Rock Centre (Canada)",
        "Viewing the falls illuminated in vibrant colors at night, and enjoying seasonal fireworks displays over the falls",
        "Visiting Journey Behind the Falls (Canada) to descend and walk through tunnels behind the Horseshoe Falls",
        "Exploring other attractions in the Niagara Parks area, such as Queen Victoria Park, Clifton Hill (Canada side for entertainment), or taking a helicopter tour for an aerial view"
    ],
    "central_park": [
        "Strolling or biking through the vast network of paths, exploring iconic areas like The Mall and Literary Walk, Bethesda Terrace and Fountain, Strawberry Fields (John Lennon memorial), and Conservatory Water (for model sailboat racing)",
        "Having a picnic on the Great Lawn or Sheep Meadow with views of the surrounding Manhattan skyline",
        "Renting a rowboat or taking a gondola ride on The Lake, passing under Bow Bridge",
        "Visiting Central Park Zoo, the Central Park Carousel, or Wollman Rink (ice skating in winter, other activities in summer)",
        "Exploring the more rugged northern section with the North Woods and Harlem Meer, or the formal Conservatory Garden",
        "Attending free events like Shakespeare in the Park (summer), concerts by the New York Philharmonic, or SummerStage performances"
    ],
    "las_vegas_strip": [
        "Walking along Las Vegas Boulevard South to see the extravagant themed mega-resorts and casinos, such as Bellagio (with its famous fountains), Venetian (with canals and gondola rides), Caesars Palace, Luxor (pyramid and sphinx), and Paris Las Vegas (Eiffel Tower replica)",
        "Watching free street-side shows like the Fountains of Bellagio water ballet or the Mirage Volcano eruption",
        "Trying your luck at casino games (responsibly, if you choose to gamble)",
        "Attending a world-class show, concert, or residency featuring top entertainers",
        "Shopping at high-end retail promenades like The Forum Shops at Caesars, Grand Canal Shoppes at The Venetian, or Crystals at CityCenter",
        "Dining at celebrity chef restaurants or enjoying diverse culinary experiences, and experiencing the vibrant nightlife with clubs and bars"
    ],
    "yellowstone_national_park": [
        "Watching the eruption of Old Faithful geyser and exploring the Upper Geyser Basin, home to the world's largest concentration of geysers",
        "Admiring the vibrant colors of Grand Prismatic Spring in the Midway Geyser Basin and other geothermal features like mudpots and fumaroles at Mammoth Hot Springs' travertine terraces",
        "Wildlife viewing: looking for bison, elk, grizzly bears, wolves, pronghorn, and bighorn sheep in areas like Lamar Valley (often called 'America's Serengeti') and Hayden Valley",
        "Visiting the Grand Canyon of the Yellowstone to see the impressive Upper and Lower Falls of the Yellowstone River from viewpoints like Artist Point",
        "Driving the scenic Grand Loop Road to access various park attractions and enjoying hikes on numerous trails of varying difficulty",
        "Boating, fishing, or simply enjoying the views at Yellowstone Lake, North America's largest high-elevation lake"
    ],
    "banff_national_park_lake_louise": [ # and Moraine Lake
        "Photographing the stunning turquoise waters of Lake Louise with the Victoria Glacier and the iconic Fairmont Chateau Lake Louise in the background; also visit the equally breathtaking Moraine Lake with its vivid blue water framed by the Valley of the Ten Peaks (access to Moraine Lake is restricted, plan ahead with shuttle or tour)",
        "Canoeing or kayaking on Lake Louise or Moraine Lake for an unforgettable experience amidst the majestic mountain scenery",
        "Hiking popular trails such as the Lake Agnes Teahouse Trail, Plain of Six Glaciers Trail (from Lake Louise), or the Rockpile Trail and Lakeshore Trail (at Moraine Lake)",
        "Taking the Lake Louise Gondola (or Banff Gondola near Banff town) for panoramic views of the Canadian Rockies, glaciers, and valleys",
        "Wildlife viewing for animals like elk, deer, bighorn sheep, and occasionally bears (maintain a safe distance)",
        "Exploring the charming town of Banff, Johnston Canyon, or driving the scenic Icefields Parkway towards Jasper National Park"
    ],
    "space_needle_seattle": [
        "Taking a 41-second elevator ride to the 520-foot high observation deck for 360-degree panoramic views of Seattle's skyline, Elliott Bay, Puget Sound, the Olympic Mountains, Cascade Mountains, and Mount Rainier on clear days",
        "Stepping out onto 'The Loupe,' the world's first and only revolving glass floor, offering thrilling downward views of the city and the Needle's structure",
        "Enjoying a meal or drinks at 'The Loupe Lounge' (revolving glass floor level) or other dining options within the Space Needle",
        "Photographing the iconic futuristic structure, a symbol of Seattle and the 1962 World's Fair, from various angles and at different times of day (especially sunset or when illuminated at night)",
        "Learning about the history of the Space Needle and Seattle through interactive exhibits on the observation deck",
        "Visiting nearby attractions at Seattle Center, such as the Chihuly Garden and Glass, Museum of Pop Culture (MoPOP), or Pacific Science Center"
    ],

    # 南美地標 (South America)
    "machu_picchu": [
        "Exploring the remarkably preserved ancient Inca citadel, including the Temple of the Sun, Intihuatana stone, Royal Tomb, residential areas, and agricultural terraces",
        "Hiking Huayna Picchu or Machu Picchu Mountain for breathtaking panoramic views of the ruins and surrounding Andes (requires separate, pre-booked tickets well in advance as permits are limited)",
        "Taking a guided tour (highly recommended) to learn about Inca history, cosmology, architecture, and the possible purposes of this enigmatic 'Lost City of the Incas' (a UNESCO World Heritage site)",
        "Photographing the stunning landscape of stone ruins nestled dramatically in the cloud-forested mountains, especially during sunrise or late afternoon for optimal lighting",
        "Watching the sunrise over Machu Picchu (if staying in Aguas Calientes or taking an early bus/train to arrive before dawn)",
        "Visiting the Sun Gate (Inti Punku) for a classic first view of Machu Picchu if hiking a portion of the Inca Trail, or the Inca Bridge for a glimpse of Inca engineering"
    ],
    "christ_the_redeemer": [
        "Taking the historic cog train (Trem do Corcovado) or an official van up Corcovado Mountain through Tijuca National Park to reach the statue",
        "Admiring the colossal Art Deco statue of Jesus Christ with outstretched arms, a global icon of Rio de Janeiro and Brazil, and one of the New Seven Wonders of the World",
        "Enjoying breathtaking 360-degree panoramic views of Rio de Janeiro, including Sugarloaf Mountain, Copacabana and Ipanema beaches, Guanabara Bay, and the Rodrigo de Freitas Lagoon",
        "Photographing the statue from its base and the spectacular surrounding cityscape and landscape",
        "Visiting the small chapel located at the base of the statue, dedicated to Our Lady of Aparecida, the patron saint of Brazil",
        "Learning about the history, construction, and significance of the monument through informational plaques or guided tours"
    ],
    "iguazu_falls": [
        "Exploring the extensive network of walkways and viewpoints on both the Argentinian side (Parque Nacional Iguazú) and the Brazilian side (Parque Nacional do Iguaçu) for different perspectives of the hundreds of waterfalls",
        "Experiencing the immense power and roar of the Devil's Throat (Garganta del Diablo / Garganta do Diabo), the largest and most dramatic cataract, from viewing platforms that extend over the falls",
        "Taking a thrilling boat tour (e.g., Gran Aventura on the Argentinian side or Macuco Safari on the Brazilian side) that goes close to and even under some of the falls (expect to get soaked!)",
        "Walking the Upper Circuit and Lower Circuit trails on the Argentinian side for varied views and close encounters with different sections of the falls",
        "Photographing the spectacular waterfalls, lush subtropical rainforest, and frequent rainbows that form in the mist",
        "Wildlife spotting in the surrounding national parks, looking for coatis, toucans, monkeys, and colorful butterflies"
    ],
    "galapagos_islands": [
        "Taking a multi-day cruise on a small ship or yacht to visit various islands, as each island offers unique landscapes and wildlife viewing opportunities (the most common way to explore)",
        "Snorkeling or scuba diving in the rich marine environment to see sea lions, marine iguanas, Galapagos penguins, sea turtles, sharks (like hammerheads and Galapagos sharks), rays, and colorful fish",
        "Observing unique and fearless wildlife up close (maintaining a respectful distance as per park rules), such as giant tortoises in their natural habitat (e.g., at El Chato Tortoise Reserve on Santa Cruz Island), blue-footed boobies, frigatebirds with inflated red pouches, and land iguanas",
        "Hiking on volcanic landscapes, lava fields, and pristine beaches, learning about the geology and evolutionary history of the islands that inspired Charles Darwin",
        "Visiting the Charles Darwin Research Station on Santa Cruz Island to learn about conservation efforts and the tortoise breeding program",
        "Kayaking, birdwatching, and photographing the extraordinary biodiversity and unique natural beauty of this UNESCO World Heritage site"
    ],
    "torres_del_paine_national_park": [
        "Hiking famous multi-day treks like the 'W Trek' (typically 4-5 days) or the 'O Circuit' (full loop, 8-10 days) for stunning views of the granite Paine Towers, Cuernos del Paine (Horns), glaciers, lakes, and Patagonian landscapes",
        "Undertaking day hikes to iconic viewpoints, such as the Base of the Towers (Mirador Las Torres), Mirador Cuernos, or viewpoints overlooking Grey Glacier and Lake Pehoé",
        "Photographing the dramatic granite peaks, turquoise glacial lakes (like Pehoé, Nordenskjöld, Sarmiento), vast ice fields (Southern Patagonian Ice Field), and unique flora and fauna",
        "Wildlife viewing for animals such as guanacos, condors, foxes, and occasionally pumas (though elusive)",
        "Taking a boat trip on Lake Grey to get close to the face of Grey Glacier, or a catamaran across Lake Pehoé",
        "Experiencing the wild and unpredictable Patagonian weather, and enjoying the pristine, rugged beauty of this renowned national park (a UNESCO Biosphere Reserve)"
    ],
    "angel_falls": [
        "Taking a multi-day expedition (typically 3 days/2 nights) by motorized dugout canoe (curiara) up the Carrao and Churún rivers through Canaima National Park to reach the base of Angel Falls (Salto Ángel), the world's tallest uninterrupted waterfall",
        "Hiking through the rainforest from the river camp to a viewpoint near the base of the falls (Mirador Laime) to witness the spectacular cascade plunging from the summit of Auyán-Tepui",
        "Swimming in pools at the base of the falls or in the river (depending on conditions and guide advice)",
        "Spending nights in hammocks at rustic jungle camps, experiencing the sounds and atmosphere of the remote wilderness",
        "Taking a scenic overflight by small plane from Canaima or Ciudad Bolívar for a breathtaking aerial view of Angel Falls and the vast expanse of Auyán-Tepui (often done as part of a tour or separately)",
        "Learning about the indigenous Pemón culture and the unique geology of the tepuis (table-top mountains) in Canaima National Park (a UNESCO World Heritage site)"
    ],
    "salar_de_uyuni": [
        "Taking a multi-day 4x4 jeep tour (typically 1 to 4 days) across the vast Salar de Uyuni, the world's largest salt flat, and the surrounding Altiplano region",
        "During the dry season (May-November): Experiencing the seemingly endless white expanse of hexagonal salt patterns, visiting Incahuasi Island (Fish Island) with its giant ancient cacti and panoramic views, and taking creative forced-perspective photographs",
        "During the rainy season (December-April): Witnessing the breathtaking 'world's largest mirror' effect when a thin layer of water covers the salt flat, creating stunning reflections of the sky and clouds (access can be limited)",
        "Visiting the 'train cemetery' (Cementerio de Trenes) just outside Uyuni town, with its rusted antique steam locomotives",
        "Seeing the salt harvesting mounds and the original salt hotel (Palacio de Sal, now a museum), and learning about salt extraction",
        "Exploring the surrounding Altiplano highlights on longer tours, such as colorful lagoons (Laguna Colorada with flamingos, Laguna Verde), geysers (Sol de Mañana), hot springs (Termas de Polques), and dramatic volcanic landscapes"
    ],

    # 中東/非洲地標 (Middle East / Africa)
    "pyramids_of_giza": [
        "Exploring the exteriors of the three main pyramids: the Great Pyramid of Giza (Pyramid of Khufu), the Pyramid of Khafre, and the Pyramid of Menkaure",
        "Entering one of the pyramids (requires separate ticket, can be claustrophobic, limited access) to see the internal chambers",
        "Visiting the Great Sphinx of Giza, the colossal limestone statue with the body of a lion and the head of a human, and the adjacent Valley Temple of Khafre",
        "Taking a camel or horse ride across the Giza plateau for different perspectives and classic photo opportunities of the pyramids against the desert backdrop",
        "Visiting the Solar Boat Museum (Khufu Ship Museum) next to the Great Pyramid to see a remarkably preserved ancient Egyptian funerary boat",
        "Attending the Sound and Light Show in the evening, where the pyramids and Sphinx are illuminated with accompanying narration about ancient Egyptian history (optional)"
    ],
    "burj_khalifa": [
        "Ascending to one of the world's highest observation decks: 'At the Top' on levels 124 and 125, or the more exclusive 'At the Top, SKY' on level 148, for breathtaking panoramic views of Dubai's futuristic skyline, the desert, and the Arabian Gulf",
        "Photographing the stunning views, especially during sunset or at night when the city lights up",
        "Dining at At.mosphere, the restaurant and lounge on level 122, one of the world's highest restaurants",
        "Watching the spectacular Dubai Fountain show, a choreographed water and light display at the base of the Burj Khalifa on Burj Lake (best viewed from the Waterfront Promenade of The Dubai Mall or a nearby restaurant)",
        "Learning about the engineering, architecture, and construction of the world's tallest building through interactive exhibits at the observation deck levels",
        "Shopping at The Dubai Mall, one of the world's largest shopping malls, which is connected to the Burj Khalifa and also houses the Dubai Aquarium & Underwater Zoo"
    ],
    "petra_jordan": [ # (Was "petra")
        "Walking through the Siq, a narrow, winding sandstone gorge that serves as the dramatic main entrance to the ancient Nabataean city of Petra",
        "Emerging from the Siq to behold the iconic Al-Khazneh (The Treasury), an elaborately carved temple facade, one of the most famous archaeological sites in the world",
        "Exploring the vast archaeological site, including the Street of Facades, the Theatre, the Royal Tombs, the Colonnaded Street, and the Great Temple",
        "Hiking up the 800+ steps to Ad Deir (The Monastery), another magnificent rock-cut facade offering stunning views (a significant but rewarding climb)",
        "Photographing the 'Rose City' with its unique rock-cut architecture and dramatic desert landscape, noting how the sandstone changes color throughout the day",
        "Learning about the history of the Nabataean civilization, their ingenuity in water management, and their role as a trading hub, through a local guide or the Petra Museum"
    ],
    "table_mountain": [
        "Taking the rotating aerial cableway (Table Mountain Aerial Cableway) to the summit for spectacular 360-degree views of Cape Town, Table Bay, Robben Island, the Cape Peninsula, and the Atlantic Ocean",
        "Hiking up one of the many trails to the summit, such as Platteklip Gorge (the most direct but steep), Skeleton Gorge (more scenic, starting from Kirstenbosch Gardens), or India Venster (more challenging, for experienced hikers)",
        "Walking along the well-maintained pathways on the flat-topped summit, exploring different viewpoints, and enjoying the unique fynbos vegetation (part of the Cape Floral Kingdom, a UNESCO World Heritage site)",
        "Photographing the panoramic vistas, the city nestled below, and the dramatic cloud formations known as the 'tablecloth' that sometimes cover the mountain",
        "Abseiling from the top of Table Mountain (for adventure seekers, with registered operators)",
        "Enjoying refreshments or a meal at the café or restaurant on the summit while taking in the views"
    ],
    "sheikh_zayed_grand_mosque": [
        "Taking a guided tour (free, offered regularly) to learn about the mosque's stunning architecture, Islamic art, cultural significance, and design elements",
        "Admiring the pristine white Macedonian marble exterior, the 82 domes of various sizes, the four towering minarets, and the intricate floral designs inlaid with semi-precious stones",
        "Exploring the vast main prayer hall, which houses the world's largest hand-knotted carpet and one of the world's largest Swarovski crystal chandeliers",
        "Walking through the serene courtyards (sahan) with their reflective pools that beautifully mirror the mosque's columns and arches",
        "Photographing the breathtaking beauty of the mosque, both during the day (when the white marble gleams) and at night (when it is illuminated by a unique lunar lighting system that changes with the phases of the moon)",
        "Respecting the dress code (modest attire required for all visitors; abayas provided for women if needed) and behaving reverently within the sacred space"
    ],
    "masai_mara_national_reserve": [
        "Going on multiple game drives (early morning and late afternoon are often best) in a 4x4 safari vehicle with an experienced guide to spot the 'Big Five' (lion, leopard, elephant, rhino, buffalo) and other wildlife like cheetahs, giraffes, zebras, wildebeest, hyenas, and numerous bird species",
        "Witnessing the Great Migration (typically July to October), where millions of wildebeest and zebras cross the Mara River from Tanzania's Serengeti, facing crocodiles and predators (a dramatic natural spectacle)",
        "Taking a hot air balloon safari at sunrise over the Masai Mara plains for a breathtaking panoramic view of the landscape and wildlife, followed by a champagne breakfast in the bush",
        "Visiting a traditional Maasai village (manyatta) to learn about Maasai culture, customs, traditions, and their way of life (usually arranged through lodges or tour operators)",
        "Enjoying bush breakfasts, lunches, or sundowner drinks in scenic spots within the reserve",
        "Nature walks with armed rangers (where permitted) for a different perspective on the flora and fauna, or birdwatching (the Mara has a rich avian diversity)"
    ],
    "victoria_falls": [
        "Viewing the spectacular curtain of falling water from various viewpoints on both the Zambian side (e.g., Knife-Edge Bridge, Boiling Pot trail) and the Zimbabwean side (e.g., Main Falls, Rainbow Falls, Danger Point) of the Zambezi River",
        "Feeling the immense spray (the 'Smoke that Thunders' or Mosi-oa-Tunya) and hearing the roar of one of the world's largest waterfalls by combined width and height",
        "Taking a guided tour of the rainforest trails on the Zimbabwean side, which offer numerous perspectives of the falls",
        "Experiencing thrilling adventure activities such as white-water rafting or kayaking on the powerful Zambezi River below the falls (seasonal), bungee jumping from the Victoria Falls Bridge, or gorge swinging",
        "Taking a helicopter or microlight flight ('Flight of Angels') over Victoria Falls for breathtaking aerial views",
        "Enjoying a sunset cruise on the upper Zambezi River, spotting wildlife like hippos, crocodiles, and birds, or visiting Livingstone Island and swimming in Devil's Pool or Angel's Pool at the edge of the falls (seasonal, Zambian side only, requires guided tour)"
    ],
    "kilimanjaro": [
        "Attempting to climb Mount Kilimanjaro, Africa's highest peak and the world's tallest freestanding mountain, via one of several routes (e.g., Machame, Lemosho, Marangu, Rongai) with a licensed guide and porters (a challenging multi-day trek requiring good physical condition and acclimatization)",
        "Experiencing the diverse ecological zones on the ascent, from lush rainforest and moorland to alpine desert and the arctic summit zone with its glaciers and snowfields",
        "Reaching Uhuru Peak (5,895m / 19,341ft) on Kibo, the highest of Kilimanjaro's three volcanic cones, typically at sunrise for spectacular views",
        "Photographing the stunning mountain scenery, unique flora (like giant groundsels and lobelias), and the iconic snow-capped summit",
        "Learning about the geology of the dormant volcano and the local Chagga culture from guides and porters",
        "For those not climbing: Enjoying views of Kilimanjaro from nearby towns like Moshi or Arusha (Tanzania), or Amboseli National Park (Kenya) on clear days, or taking a scenic flight around the mountain"
    ],
    "dead_sea": [
        "Effortlessly floating in the hyper-saline waters of the Dead Sea, an experience unique due to the water's high salt and mineral concentration (avoid getting water in your eyes)",
        "Applying the mineral-rich black mud from the Dead Sea shores onto your skin, known for its therapeutic and cosmetic benefits, then rinsing off in the sea or showers",
        "Relaxing at one of the resorts or public beaches along the shores in Jordan or Israel, often equipped with facilities like showers, pools, and spas",
        "Photographing the unique landscape of the Dead Sea (the lowest point on Earth's land surface), with its calm, dense turquoise waters reflecting the arid surrounding mountains and desert, and distinctive salt formations",
        "Learning about the history, geology, and unique ecosystem of the Dead Sea, and the challenges it faces (e.g., receding water levels)",
        "Visiting nearby historical and natural sites, such as Masada and Ein Gedi (Israel side) or Mujib Biosphere Reserve and Ma'in Hot Springs (Jordan side)"
    ],
    "dome_of_the_rock": [
        "Admiring the iconic, gleaming golden dome of the Dome of the Rock (Qubbat as-Sakhrah), an Islamic shrine located on the Temple Mount (Haram al-Sharif) in the Old City of Jerusalem (access for non-Muslims to the Temple Mount plaza is restricted to specific hours and days, and entry into the Dome of the Rock itself is generally not permitted for non-Muslims)",
        "Observing the octagonal structure of the Dome of the Rock, adorned with intricate blue and turquoise ceramic tilework, Quranic calligraphy, and mosaics, from the Temple Mount plaza",
        "Photographing this significant religious landmark, a site of immense importance in Islam, Judaism (as the location of the First and Second Temples), and Christianity, from various viewpoints within and outside the Old City (e.g., from the Mount of Olives)",
        "Exploring the wider Temple Mount/Haram al-Sharif plaza, also home to Al-Aqsa Mosque (entry for non-Muslims generally not permitted into the mosque building) and other smaller Islamic structures, while respecting the site's sanctity and rules",
        "Learning about the history and religious significance of the Foundation Stone (Even HaShetiya in Hebrew, As-Sakhrah in Arabic) which the Dome of the Rock enshrines, believed to be the site where Abraham prepared to sacrifice his son, and from where Prophet Muhammad is said to have ascended to heaven",
        "Visiting other holy sites within the Old City of Jerusalem, such as the Western Wall, the Church of the Holy Sepulchre, and walking the Via Dolorosa"
    ],

    # 大洋洲地標 (Oceania)
    "sydney_opera_house": [
        "Taking a guided architectural tour to learn about its history, innovative design by Jørn Utzon, construction challenges, and its various performance venues (e.g., Concert Hall, Joan Sutherland Theatre)",
        "Attending a world-class performance, such as opera, ballet, classical music concert, theatre, or contemporary music show, inside one of its iconic halls (book tickets in advance)",
        "Photographing the iconic white sail-like shells from various angles and vantage points, such as from a ferry on Sydney Harbour, Mrs Macquarie's Chair, Circular Quay, or while walking across the Sydney Harbour Bridge",
        "Dining at one of the many restaurants or enjoying drinks at bars with stunning harbour views located at the Opera House, like Opera Bar or Portside Sydney",
        "Walking around the exterior promenade, admiring the building's unique form up close, and enjoying the bustling atmosphere of Circular Quay",
        "Visiting the gift shop for souvenirs, or attending free outdoor events and festivals that sometimes take place on the forecourt"
    ],
    "uluru": [
        "Walking the full Uluru base walk (approximately 10.6 km / 6.6 miles loop, takes about 3-4 hours) to experience the immense scale and diverse features of the monolith up close, including waterholes, rock art sites, and unique geological formations (best done early morning or late afternoon to avoid heat)",
        "Watching and photographing the spectacular color changes of Uluru at sunrise and sunset from designated viewing areas (Talinguru Nyakunytjaku for sunrise, Uluru sunset viewing area for sunset)",
        "Taking a guided tour led by an Anangu (local Aboriginal people) guide to learn about the cultural significance, Dreamtime stories (Tjukurpa), and traditional laws associated with Uluru (highly recommended for a deeper understanding)",
        "Visiting the Uluru-Kata Tjuta Cultural Centre to gain insights into Anangu culture, art, and land management practices",
        "Exploring the nearby Kata Tjuta (The Olgas) rock formations, particularly the Walpa Gorge walk or the Valley of the Winds walk, for different but equally stunning desert landscapes",
        "Stargazing in the clear desert night sky (Uluru is in a remote area with minimal light pollution), or participating in an Sounds of Silence or Field of Light Uluru experience (optional, requires booking)"
    ],
    "great_barrier_reef": [
        "Snorkeling or scuba diving from a boat tour or island resort to explore the vibrant coral reefs, encounter colorful tropical fish, sea turtles, manta rays, reef sharks, and other diverse marine life (many tour options cater to different skill levels)",
        "Taking a glass-bottom boat tour or a semi-submersible tour to view the coral and marine life without getting wet, suitable for all ages",
        "Going on a scenic helicopter or seaplane flight for a breathtaking aerial perspective of the reef's intricate patterns, turquoise waters, islands, and cays (e.g., seeing Heart Reef from the air)",
        "Visiting a pontoon stationed on the outer reef, which often includes facilities like underwater observatories, snorkeling platforms, and introductory dive options",
        "Learning about marine conservation, the ecology of the Great Barrier Reef (a UNESCO World Heritage site), and the threats it faces (e.g., climate change, coral bleaching) through onboard marine biologist talks or visitor centers",
        "Sailing or taking a catamaran cruise through the Whitsunday Islands or other reef areas, often including stops for snorkeling, swimming, and relaxing on pristine beaches like Whitehaven Beach"
    ],
    "hobbiton_movie_set": [
        "Taking a fully guided walking tour through the 12-acre movie set, visiting numerous Hobbit Holes with their distinctive round doors and charming details, as seen in 'The Lord of the Rings' and 'The Hobbit' trilogies",
        "Photographing iconic locations such as Bag End (Bilbo and Frodo Baggins' home), the Party Tree, the Mill, and the double-arched stone bridge",
        "Enjoying a complimentary, specially brewed beverage (ale, cider, or non-alcoholic ginger beer) at The Green Dragon Inn, a faithfully reconstructed pub from the movies",
        "Listening to fascinating behind-the-scenes stories and filmmaking secrets from your guide about how the Hobbiton set was created and used in the films",
        "Shopping for Middle-earth themed souvenirs, Weta Workshop collectibles, and other memorabilia at the Hobbiton Shire Store",
        "Immersing oneself in the picturesque, meticulously maintained landscape of The Shire, with its rolling green hills, gardens, and charming atmosphere, located on a working sheep farm"
    ],
    "sydney_harbour_bridge": [
        "Undertaking the BridgeClimb Sydney experience, climbing the steel arches of the bridge for breathtaking 360-degree panoramic views of Sydney Harbour, the Opera House, the city skyline, and beyond (various climb options available, book in advance)",
        "Walking or cycling across the bridge on the dedicated pedestrian walkway (eastern side) or cycleway (western side) for free, enjoying fantastic views and photo opportunities",
        "Visiting the Pylon Lookout on the southeastern pylon for historical exhibits about the bridge's construction and impressive views from its observation deck",
        "Photographing the iconic 'Coathanger' from various vantage points, such as Mrs Macquarie's Chair, Luna Park, McMahon's Point, or from a ferry on the harbour",
        "Taking a ferry ride under the bridge for a different perspective of its massive scale and engineering",
        "Learning about the history, design, and construction of this engineering marvel, which opened in 1932"
    ],
    "fiordland_national_park_milford_sound": [ # and Doubtful Sound
        "Taking a scenic boat cruise (day cruise or overnight cruise) on Milford Sound to experience its dramatic fiord scenery, with sheer cliffs (like Mitre Peak), cascading waterfalls (e.g., Stirling Falls, Bowen Falls), lush rainforest, and often, mist-shrouded peaks",
        "Kayaking on Milford Sound or Doubtful Sound for a more intimate and peaceful experience of the fiords, getting closer to the shoreline and wildlife",
        "Spotting wildlife such as New Zealand fur seals basking on rocks, dolphins, Fiordland crested penguins (seasonal), and various seabirds",
        "Exploring Doubtful Sound, a larger and more remote fiord, often via a tour involving a cruise across Lake Manapouri and a bus trip over Wilmot Pass",
        "Hiking parts of famous multi-day tracks like the Milford Track (requires advance booking months or even years ahead) or Kepler Track, or shorter day walks accessible from the Milford Road",
        "Photographing the awe-inspiring, moody, and pristine landscapes of Fiordland National Park (a UNESCO World Heritage site), known for its high rainfall and dramatic beauty"
    ],
    "bondi_beach": [
        "Swimming, surfing, or sunbathing on the famous crescent-shaped golden sands of Bondi Beach, one of Australia's most iconic beaches",
        "Learning to surf at one of the surf schools located at Bondi Beach",
        "Walking or jogging the scenic Bondi to Coogee Coastal Walk (approx. 6km), offering stunning ocean views, dramatic cliffs, and access to other beaches like Tamarama and Bronte",
        "Visiting the Bondi Icebergs Club, a historic swimming club with its famous ocean pool that waves often crash into, and enjoying a meal or drink at its restaurant with panoramic beach views",
        "Exploring the vibrant Campbell Parade, the street fronting the beach, lined with cafes, restaurants, surf shops, and fashion boutiques",
        "People-watching and soaking in the lively beach culture, or visiting the Bondi Markets (on weekends) for local crafts, fashion, and food"
    ],
    "aoraki_mount_cook_national_park": [
        "Hiking popular trails such as the Hooker Valley Track (relatively easy, 3-hour return) for spectacular views of Aoraki/Mount Cook, Mueller Glacier, Hooker Lake (with icebergs), and suspension bridges",
        "Admiring New Zealand's highest peak, Aoraki/Mount Cook, and the surrounding majestic Southern Alps, glaciers, and turquoise glacial lakes like Lake Pukaki and Lake Tekapo (often framed by colorful lupins in summer - Nov/Dec)",
        "Stargazing in the Aoraki Mackenzie International Dark Sky Reserve, one of the best places in the world for viewing the night sky due to its clear, dark conditions (guided stargazing tours available)",
        "Taking a scenic helicopter or ski plane flight for breathtaking aerial views of the mountains and glaciers, possibly including a snow landing on Tasman Glacier",
        "Visiting the Sir Edmund Hillary Alpine Centre at The Hermitage Hotel to learn about the region's mountaineering history, flora, fauna, and geology",
        "Experienced mountaineering or glacier hiking with qualified guides (for advanced adventurers)"
    ],
    "twelve_apostles_great_ocean_road": [
        "Viewing and photographing the iconic limestone stacks known as The Twelve Apostles (though fewer than twelve now remain due to erosion) rising dramatically from the Southern Ocean, from the designated clifftop viewing platforms",
        "Visiting at sunrise or sunset for the most spectacular lighting conditions, as the stacks change color and long shadows are cast (be prepared for crowds during these times)",
        "Walking along the boardwalks and pathways to different lookout points offering various perspectives of The Twelve Apostles and the rugged coastline",
        "Exploring other nearby dramatic rock formations and coastal features along this section of the Great Ocean Road, such as Loch Ard Gorge (site of a famous shipwreck), Gibson Steps (descend to the beach for a different view of some stacks), The Arch, London Bridge (now London Arch), and The Grotto",
        "Taking a scenic helicopter flight over The Twelve Apostles and the Shipwreck Coast for a breathtaking aerial view (optional)",
        "Learning about the geology of the limestone formations, the power of coastal erosion, and the maritime history of the area at the Twelve Apostles Visitor Centre"
    ],
    "easter_island_moai": [
        "Exploring Rano Raraku, the volcanic crater quarry where hundreds of moai were carved and many still stand in various stages of completion, some half-buried",
        "Visiting Ahu Tongariki to witness the impressive sight of fifteen restored moai standing on the largest ceremonial platform on the island, especially spectacular at sunrise",
        "Relaxing or swimming at Anakena Beach, a picturesque white coral sand beach with palm trees, featuring Ahu Nau Nau with its well-preserved moai (some with topknots) and Ahu Ature Huki",
        "Hiking to the rim of the Rano Kau volcano to see its stunning crater lake and visiting the nearby ceremonial village of Orongo, known for its Birdman cult petroglyphs and dramatic cliffside location",
        "Discovering Ahu Akivi, a unique inland platform with seven moai that face the ocean, aligned with the equinox",
        "Learning about Rapa Nui culture, history, and the mysteries of the moai at the Museo Antropológico Sebastián Englert in Hanga Roa",
        "Photographing the diverse moai sites across the island, such as Ahu Tahai (near Hanga Roa, good for sunset), Puna Pau (quarry for the red scoria pukao topknots), and Te Pito Kura (site of a large fallen moai and a magnetic stone)",
        "Attending a traditional Rapa Nui cultural performance (dance and music) or experiencing a Curanto (traditional underground feast)",
        "Stargazing in the exceptionally clear night skies due to the island's remote location and lack of light pollution",
        "Renting a car, scooter, or bicycle to independently explore the island's numerous archaeological sites and natural beauty, or taking a guided tour for deeper insights"
    ]
}
