
import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Optional
from decimal import Decimal
from breed_health_info import breed_health_info, default_health_note
from breed_noise_info import breed_noise_info
import numpy as np
import random


def create_table():
    conn = sqlite3.connect('animal_detector.db')
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS AnimalCatalog (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Species TEXT NOT NULL,
        Breed TEXT NOT NULL,
        Size_Category TEXT,
        Typical_Lifespan TEXT,
        Temperament TEXT,
        Care_Level TEXT,
        Good_With_Children BOOLEAN,
        Exercise_Needs TEXT,
        Grooming_Needs TEXT,
        Brief_Description TEXT
    )
    ''')

    conn.commit()
    cursor.close()
    conn.close()

# 創建表
create_table()

def insert_dog_data():
    conn = sqlite3.connect('animal_detector.db')
    cursor = conn.cursor()

    dog_data = [
        ('Dog', 'Afghan_Hound', 'Large', '12-18 years', 'Independent, dignified, aloof', 'High', True, 'High', 'High', 'Known for their long, silky coat and regal appearance, Afghan Hounds are ancient sighthounds with a unique, elegant presence.'),
        ('Dog', 'African_Hunting_Dog', 'Medium', '10-12 years', 'Social, intelligent, energetic', 'Very High', False, 'Very High', 'Low', 'Also called African Wild Dogs, these are not domestic dogs and are endangered in their native habitats.'),
        ('Dog', 'Airedale', 'Large', '10-12 years', 'Friendly, clever, courageous', 'High', True, 'High', 'High', 'Known as the "King of Terriers," Airedales are the largest of the terrier breeds and very versatile. Originally from Aire Valley of Yorkshire, they served in police and military roles while maintaining their hunting abilities.'),
        ('Dog', 'American_Staffordshire_Terrier', 'Medium', '12-16 years', 'Confident, good-natured, courageous', 'Moderate', True, 'High', 'Low', 'Often confused with Pit Bulls, Am Staffs are strong, muscular dogs with a gentle and loving nature.'),
        ('Dog', 'Appenzeller', 'Medium', '12-15 years', 'Reliable, fearless, lively', 'High', True, 'High', 'Moderate', 'Swiss mountain dogs known for their agility and enthusiasm, Appenzellers make excellent working and family dogs. Traditional cattle herders from the Swiss Alps, they combine working intelligence with devoted guardianship.'),
        ('Dog', 'Australian_Terrier', 'Small', '12-15 years', 'Courageous, spirited, alert', 'Moderate', True, 'Moderate', 'Moderate', 'Small, sturdy terriers originally bred to control rodents, known for their confident personality. First breed developed in Australia, combining the tenacity of old terrier breeds with adaptability to harsh conditions.'),
        ('Dog', 'Bedlington_Terrier', 'Medium', '11-16 years', 'Mild, gentle, lively', 'High', True, 'Moderate', 'High', 'Known for their lamb-like appearance, Bedlington Terriers are energetic and good with families. Despite their gentle look, they were originally miners\' dogs bred for hunting vermin and remain surprisingly fast and athletic.'),
        ('Dog', 'Bernese_Mountain_Dog', 'Large', '6-8 years', 'Good-natured, calm, strong', 'High', True, 'Moderate', 'High', 'Large, tri-colored Swiss working dogs known for their gentle nature and striking appearance. Despite their short lifespan, they are excellent family companions and draft dogs. Often called "Berners", they are patient with children but prone to heat sensitivity due to their thick coat.'),
        ('Dog', 'Blenheim_Spaniel', 'Small', '12-14 years', 'Affectionate, gentle, lively', 'Moderate', True, 'Moderate', 'High', 'A color variety of the Cavalier King Charles Spaniel, known for their red and white coat and friendly nature.'),
        ('Dog', 'Border_Collie', 'Medium', '12-15 years', 'Intelligent, energetic, alert', 'High', True, 'Very High', 'Moderate', 'Often considered the most intelligent dog breed, Border Collies are tireless workers with intense herding instincts. Known for their "eye" - a distinctive herding gaze, they excel in dog sports and need constant mental stimulation to prevent boredom.'),
        ('Dog', 'Border_Terrier', 'Small', '12-15 years', 'Affectionate, intelligent, even-tempered', 'Moderate', True, 'High', 'Low', 'Small but tough terriers with an otter-like head, known for their friendly and adaptable nature.'),
        ('Dog', 'Boston_Bull', 'Small', '11-13 years', 'Friendly, lively, intelligent', 'Moderate', True, 'Moderate', 'Low', 'Also known as Boston Terriers, these "American Gentlemen" are friendly and adaptable. First bred in Boston as fighting dogs, they evolved into beloved companions known for their tuxedo-like coat pattern and gentle disposition.'),
        ('Dog', 'Bouvier_Des_Flandres', 'Large', '10-12 years', 'Gentle, loyal, rational', 'High', True, 'High', 'High', 'Large, powerful herding dogs with a tousled coat, known for their versatility and even temperament. Originally bred as cattle drivers in Flanders, they excel in various roles from farm work to family protection.'),
        ('Dog', 'Brabancon_Griffon', 'Small', '12-15 years', 'Self-important, sensitive, affectionate', 'Moderate', False, 'Moderate', 'Low', 'Also known as the Brussels Griffon, these small dogs have a distinctive beard and mustache, giving them an almost human-like expression.'),
        ('Dog', 'Brittany_Spaniel', 'Medium', '12-14 years', 'Bright, fun-loving, upbeat', 'High', True, 'High', 'Moderate', 'Versatile hunting dogs and active companions, Brittanys are known for their energy and intelligence. Originally from France, these agile bird dogs excel in both hunting and family life with their eager and athletic nature.'),
        ('Dog', 'Cardigan', 'Small', '12-15 years', 'Affectionate, loyal, intelligent', 'Moderate', True, 'Moderate', 'Moderate', 'Distinguished from Pembroke Welsh Corgis by their long tail, Cardigans are intelligent herding dogs with a fox-like appearance.'),
        ('Dog', 'Chesapeake_Bay_Retriever', 'Large', '10-13 years', 'Bright, sensitive, affectionate', 'High', True, 'High', 'Moderate', 'Known for their waterproof coat, Chessies are strong swimmers and excellent retrievers. Developed in the American Chesapeake Bay region, they are famous for their ability to work in icy waters and their distinctive wavy coat.'),
        ('Dog', 'Chihuahua', 'Small', '12-20 years', 'Charming, graceful, sassy', 'Moderate', False, 'Low', 'Low', 'One of the smallest dog breeds, known for their big personalities and loyalty to their owners. Named after Mexico\'s largest state, they are ancient companions with terrier-like attitudes and remarkably long lifespans for dogs.'),
        ('Dog', 'Dandie_Dinmont', 'Small', '12-15 years', 'Independent, intelligent, dignified', 'Moderate', True, 'Moderate', 'Moderate', 'Recognizable by their long body and distinctive topknot of hair on their head. Named after a character in Sir Walter Scott novel, these unique terriers combine determination with dignity, making them distinctive companions.'),
        ('Dog', 'Doberman', 'Large', '10-12 years', 'Loyal, fearless, alert', 'High', True, 'High', 'Low', 'Sleek, athletic dogs known for their intelligence and loyalty, often used as guard dogs. Highly trainable and protective, they excel in both working roles and as family guardians.'),
        ('Dog', 'English_Foxhound', 'Medium', '10-13 years', 'Friendly, active, gentle', 'High', True, 'Very High', 'Low', 'Athletic, pack-oriented hounds originally bred for fox hunting in England. Known for their stamina, melodious voice, and strong hunting instincts.'),
        ('Dog', 'English_Setter', 'Large', '10-12 years', 'Gentle, friendly, placid', 'High', True, 'High', 'High', 'Known for their speckled coat or "belton" markings, English Setters are elegant bird dogs and affectionate companions.'),
        ('Dog', 'English_Springer', 'Medium', '12-14 years', 'Friendly, playful, obedient', 'High', True, 'High', 'High', 'Energetic and eager to please, Springers are excellent hunting dogs and loving family pets. Their name comes from their hunting style of "springing" at game birds, combining strong work ethics with a merry, affectionate family nature.'),
        ('Dog', 'EntleBucher', 'Medium', '11-13 years', 'Loyal, enthusiastic, intelligent', 'High', True, 'High', 'Low', 'The smallest of the Swiss Mountain Dogs, known for their agility and herding abilities. Originally from the Swiss valley of Entlebuch, they combine the strength of mountain dogs with remarkable agility and quick thinking.'),
        ('Dog', 'Eskimo_Dog', 'Large', '10-15 years', 'Alert, loyal, intelligent', 'High', True, 'High', 'High', 'Also known as the Canadian Eskimo Dog, these are strong, resilient working dogs adapted to Arctic conditions. Ancient breed used by Inuit people for hunting and transportation, known for their power and endurance.'),
        ('Dog', 'French_Bulldog', 'Small', '10-12 years', 'Playful, adaptable, smart', 'Moderate', True, 'Low', 'Low', 'French Bulldogs are small, muscular dogs with a smooth coat, short face, and bat-like ears. They are affectionate, playful, and well-suited for family living.'),
        ('Dog', 'German_Shepherd', 'Large', '10-13 years', 'Confident, courageous, smart', 'High', True, 'High', 'Moderate', 'Versatile working dogs, German Shepherds excel in various roles from police work to family protection.'),
        ('Dog', 'German_Short-Haired_Pointer', 'Large', '10-12 years', 'Friendly, intelligent, willing to please', 'High', True, 'Very High', 'Moderate', 'Versatile hunting dogs known for their pointer stance, these dogs excel in both water and land retrieving.'),
        ('Dog', 'Gordon_Setter', 'Large', '10-12 years', 'Confident, fearless, alert', 'High', True, 'High', 'High', 'The largest of the setter breeds, Gordon Setters are known for their black and tan coloring and loyal nature. Developed in Scotland, these noble bird dogs combine strong hunting abilities with devoted family loyalty.'),
        ('Dog', 'Great_Dane', 'Giant', '7-10 years', 'Friendly, patient, dependable', 'High', True, 'Moderate', 'Low', 'One of the largest dog breeds, Great Danes are known as gentle giants with a friendly disposition. Originally bred for hunting large game, these noble giants now excel as loving family companions despite their imposing size.'),
        ('Dog', 'Great_Pyrenees', 'Large', '10-12 years', 'Patient, calm, gentle', 'High', True, 'Moderate', 'High', 'Large, powerful dogs originally bred to guard livestock, known for their gentle and protective nature. These ancient guardians of the Pyrenees mountains combine strength and nobility with a deep devotion to family and flock.'),
        ('Dog', 'Greater_Swiss_Mountain_dog', 'Large', '8-11 years', 'Faithful, alert, vigilant', 'Moderate', True, 'Moderate', 'Low', 'Large, strong working dogs with a tricolor coat, Swissies are gentle giants with a calm temperament.'),
        ('Dog', 'Ibizan_Hound', 'Medium', '12-14 years', 'Even-tempered, loyal, independent', 'Moderate', True, 'High', 'Low', 'Sleek, athletic sighthounds known for their large, erect ears and red and white coats. Ancient breed from Balearic Islands, they can jump remarkable heights and were bred to hunt rabbits in difficult terrain.'),
        ('Dog', 'Irish_Setter', 'Large', '11-15 years', 'Outgoing, sweet-tempered, active', 'High', True, 'High', 'High', 'Recognizable by their rich red coat, Irish Setters are energetic and playful dogs that love family life.'),
        ('Dog', 'Irish_Terrier', 'Medium', '12-16 years', 'Bold, daring, intelligent', 'Moderate', True, 'High', 'Moderate', 'Known as the Daredevil of dogdom, Irish Terriers are courageous and loyal with a distinctive red coat. One of the oldest terrier breeds, they earned fame for their bravery as messenger dogs in World War I.'),
        ('Dog', 'Irish_Water_Spaniel', 'Large', '10-12 years', 'Playful, brave, intelligent', 'High', True, 'High', 'High', 'Largest of the spaniels, known for their curly, liver-colored coat and rat-like tail. These distinctive water retrievers combine the agility of a spaniel with the endurance of a retriever, featuring a water-repellent double coat.'),
        ('Dog', 'Irish_Wolfhound', 'Giant', '6-8 years', 'Gentle, patient, dignified', 'High', True, 'Moderate', 'Moderate', 'The tallest of all dog breeds, Irish Wolfhounds are gentle giants known for their calm and friendly nature. Ancient warriors of Ireland, they once hunted wolves but now serve as peaceful family companions.'),
        ('Dog', 'Italian_Greyhound', 'Small', '12-15 years', 'Sensitive, alert, playful', 'Moderate', False, 'Moderate', 'Low', 'Miniature sighthounds known for their elegant appearance and affectionate nature, Italian Greyhounds make excellent companion dogs.'),
        ('Dog', 'Japanese_Spaniel', 'Small', '10-12 years', 'Charming, noble, affectionate', 'Moderate', False, 'Low', 'High', 'Also known as the Japanese Chin, these small companion dogs have a distinctive flat face and were once favorites of Japanese nobility.'),
        ('Dog', 'Kerry_Blue_Terrier', 'Medium', '12-15 years', 'Alert, adaptable, people-oriented', 'High', True, 'High', 'High', 'Medium-sized terriers with a distinctive blue coat, known for their versatility and intelligence. Originally from County Kerry, Ireland, they were all-purpose farm dogs that evolved into capable working and companion animals.'),
        ('Dog', 'Labrador_Retriever', 'Large', '10-12 years', 'Friendly, outgoing, even-tempered', 'Moderate', True, 'High', 'Moderate', 'One of the most popular dog breeds, known for their friendly nature and excellent retrieving skills. Originally from Newfoundland, these versatile dogs excel as family companions, service dogs, and working retrievers.'),
        ('Dog', 'Lakeland_Terrier', 'Small', '12-16 years', 'Bold, friendly, confident', 'Moderate', True, 'High', 'High', 'Named after the Lake District in England, these terriers are sturdy and bold with a wiry coat. Developed to protect sheep from foxes, they remain confident and fearless while being adaptable family companions.'),
        ('Dog', 'Leonberg', 'Giant', '7-9 years', 'Gentle, friendly, intelligent', 'High', True, 'Moderate', 'High', 'Large, muscular dogs with a lion-like mane, known for their gentle nature and water rescue abilities. Created in the German town of Leonberg, these gentle giants combine strength with remarkable patience and grace.'),
        ('Dog', 'Lhasa', 'Small', '12-15 years', 'Confident, smart, comical', 'High', False, 'Low', 'High', 'Lhasa Apsos are small but sturdy dogs with a long, flowing coat. They were originally bred as indoor sentinel dogs in Buddhist monasteries.'),
        ('Dog', 'Maltese_Dog', 'Small', '12-15 years', 'Gentle, playful, charming', 'High', False, 'Low', 'High', 'Small, elegant dogs with long, silky white coats, known for their sweet and affectionate nature. Ancient breed of Mediterranean origin, they were cherished by nobles for centuries and remain adaptable, gentle companions.'),
        ('Dog', 'Mexican_Hairless', 'Varies', '12-15 years', 'Loyal, alert, cheerful', 'Moderate', True, 'Moderate', 'Low', 'Also known as the Xoloitzcuintli, these dogs come in three sizes and can be either hairless or coated, known for their ancient history in Mexico.'),
        ('Dog', 'Newfoundland', 'Giant', '8-10 years', 'Sweet, patient, devoted', 'High', True, 'Moderate', 'High', 'Large, strong dogs known for their water rescue abilities and gentle nature, especially with children. These powerful swimmers have a natural lifesaving instinct and are famous for their calm, noble temperament.'),
        ('Dog', 'Norfolk_Terrier', 'Small', '12-15 years', 'Fearless, spirited, companionable', 'Moderate', True, 'Moderate', 'Moderate', 'Small, sturdy terriers with a wiry coat, known for their playful and affectionate nature.'),
        ('Dog', 'Norwegian_Elkhound', 'Medium', '12-15 years', 'Bold, playful, loyal', 'High', True, 'High', 'High', 'Ancient Nordic breed known for their silver-gray coat and curled tail, originally used for hunting moose and other large game.'),
        ('Dog', 'Norwich_Terrier', 'Small', '12-15 years', 'Fearless, loyal, affectionate', 'Moderate', True, 'Moderate', 'Moderate', 'One of the smallest terriers, Norwich Terriers are hardy, fearless, and affectionate companions.'),
        ('Dog', 'Old_English_Sheepdog', 'Large', '10-12 years', 'Adaptable, gentle, intelligent', 'High', True, 'Moderate', 'High', 'Recognizable by their shaggy coat, Old English Sheepdogs are adaptable and good-natured. Once droving dogs of western England, they combine herding ability with a playful, protective nature toward their families.'),
        ('Dog', 'Pekinese', 'Small', '12-14 years', 'Affectionate, loyal, regal in manner', 'Moderate', False, 'Low', 'High', 'Also spelled Pekingese, these small dogs with flat faces and long coats were once sacred to Chinese royalty.'),
        ('Dog', 'Pembroke', 'Small', '12-15 years', 'Affectionate, intelligent, outgoing', 'Moderate', True, 'Moderate', 'Moderate', 'Known for their short legs and long bodies, Pembroke Welsh Corgis are herding dogs favored by the British royal family.'),
        ('Dog', 'Pomeranian', 'Small', '12-16 years', 'Lively, bold, inquisitive', 'Moderate', False, 'Low', 'High', 'Small, fluffy dogs with fox-like faces, known for their vivacious personalities and luxurious coats. Once larger sled dogs, these bred-down companions retain their bold spirit and were favored by royalty including Queen Victoria.'),
        ('Dog', 'Rhodesian_Ridgeback', 'Large', '10-12 years', 'Dignified, intelligent, strong-willed', 'Moderate', True, 'High', 'Low', 'Large, muscular dogs known for the ridge of hair along their backs, originally bred to hunt lions in Africa.'),
        ('Dog', 'Rottweiler', 'Large', '8-10 years', 'Loyal, loving, confident guardian', 'High', True, 'High', 'Low', 'Powerful and protective, Rottweilers are excellent guard dogs but also loving family companions when well-trained.'),
        ('Dog', 'Saint_Bernard', 'Giant', '8-10 years', 'Gentle, patient, friendly', 'High', True, 'Moderate', 'High', 'Known for their massive size and gentle nature, Saint Bernards were originally bred for rescue work in the Swiss Alps.'),
        ('Dog', 'Saluki', 'Large', '12-14 years', 'Gentle, dignified, independent-minded', 'High', True, 'High', 'Low', 'Ancient sighthounds known for their grace and speed, Salukis have a distinctive feathered coat and ears.'),
        ('Dog', 'Samoyed', 'Medium', '12-14 years', 'Friendly, gentle, adaptable', 'High', True, 'High', 'High', 'Beautiful white Arctic dogs known for their "smiling" expression and thick, fluffy coat. Originally bred for sledding and herding reindeer, they combine working dog capability with a warm, family-friendly nature.'),
        ('Dog', 'Scotch_Terrier', 'Small', '11-13 years', 'Independent, confident, spirited', 'Moderate', True, 'Moderate', 'High', 'Also known as the Scottish Terrier, these distinctive dogs with beards and eyebrows are known for their dignified, almost human-like personality.'),
        ('Dog', 'Scottish_Deerhound', 'Large', '8-11 years', 'Gentle, dignified, polite', 'High', True, 'High', 'Moderate', 'Large, wiry-coated sighthounds resembling Greyhounds, known for their gentle nature and hunting ability.'),
        ('Dog', 'Sealyham_Terrier', 'Small', '12-14 years', 'Alert, outgoing, calm', 'Moderate', True, 'Moderate', 'High', 'Originally bred for hunting, Sealyhams are now rare but make charming and sturdy companions. Developed in Wales to hunt badgers and otters, they combine terrier tenacity with a surprisingly calm demeanor.'),
        ('Dog', 'Shetland_Sheepdog', 'Small', '12-14 years', 'Playful, energetic, intelligent', 'High', True, 'High', 'High', 'Small herding dogs resembling miniature Collies, known for their intelligence and agility. Originally from the Shetland Islands, these "Shelties" excel in obedience, herding, and agility competitions while being devoted family companions.'),
        ('Dog', 'Shih-Tzu', 'Small', '10-16 years', 'Affectionate, playful, outgoing', 'High', True, 'Low', 'High', 'Small, affectionate companion dogs known for their long, silky coat and sweet personality. Originally bred for Chinese royalty, they are excellent lap dogs and adapt well to both city and suburban life.'),
        ('Dog', 'Siberian_Husky', 'Medium', '12-14 years', 'Outgoing, mischievous, loyal', 'High', True, 'Very High', 'Moderate', 'Beautiful sled dogs known for their striking blue eyes, thick coats, and wolf-like appearance. Originally bred by the Chukchi people of northeastern Asia, they combine endurance with a friendly, adventurous spirit.'),
        ('Dog', 'Staffordshire_Bullterrier', 'Medium', '12-14 years', 'Courageous, intelligent, loyal', 'Moderate', True, 'High', 'Low', 'Strong, muscular terriers known for their courage and affectionate nature, especially with children.'),
        ('Dog', 'Sussex_Spaniel', 'Medium', '11-13 years', 'Calm, friendly, merry', 'Moderate', True, 'Moderate', 'Moderate', 'Rare breed of spaniel known for their golden-liver coat and low-set body, originally bred for hunting.'),
        ('Dog', 'Tibetan_Mastiff', 'Large', '10-12 years', 'Independent, reserved, intelligent', 'High', False, 'Moderate', 'High', 'Ancient guardian breed known for their massive size and thick coat, Tibetan Mastiffs are independent and protective.'),
        ('Dog', 'Tibetan_Terrier', 'Medium', '12-15 years', 'Affectionate, sensitive, clever', 'High', True, 'Moderate', 'High', 'Not actually terriers, these dogs were bred in Tibet and are known for their profuse, long coat.'),
        ('Dog', 'Walker_Hound', 'Large', '12-13 years', 'Smart, brave, friendly', 'Moderate', True, 'High', 'Low', 'Also known as the Treeing Walker Coonhound, these dogs are excellent hunters with a distinctive bark. Developed in Kentucky from Virginia Hounds, they are renowned for their speed, endurance, and melodious voice.'),
        ('Dog', 'Weimaraner', 'Large', '10-13 years', 'Friendly, fearless, obedient', 'High', True, 'High', 'Low', 'Known as the "Gray Ghost," Weimaraners are athletic and intelligent dogs with a distinctive silver-gray coat.'),
        ('Dog', 'Welsh_Springer_Spaniel', 'Medium', '12-15 years', 'Active, loyal, affectionate', 'High', True, 'High', 'Moderate', 'Similar to English Springers but with a distinctive red and white coat, Welsh Springers are devoted and energetic.'),
        ('Dog', 'West_Highland_White_Terrier', 'Small', '12-16 years', 'Friendly, hardy, confident', 'Moderate', True, 'Moderate', 'High', 'Commonly known as "Westies," these white terriers are friendly and sturdy with a bright personality.'),
        ('Dog', 'Yorkshire_Terrier', 'Small', '13-16 years', 'Affectionate, sprightly, tomboyish', 'High', False, 'Moderate', 'High', 'Popular toy breed known for their long silky coat and feisty personality. Despite their small size, they maintain a brave terrier spirit and were originally bred as ratters in Yorkshire mills.'),
        ('Dog', 'Affenpinscher', 'Small', '12-15 years', 'Confident, amusing, stubborn', 'Moderate', False, 'Moderate', 'Moderate', 'Small terrier-like toys known as "monkey dogs" due to their distinctive facial appearance.'),
        ('Dog', 'Basenji', 'Small', '12-16 years', 'Independent, smart, poised', 'Moderate', False, 'High', 'Low', 'Ancient African breed known for their inability to bark, instead making a unique yodel-like sound. Called "the barkless dog," they are intelligent hunters with cat-like cleanliness and independent nature.'),
        ('Dog', 'Basset', 'Medium', '10-12 years', 'Patient, low-key, charming', 'Moderate', True, 'Low', 'Moderate', 'Short-legged, long-bodied hounds known for their excellent sense of smell and gentle dispositions. Second only to Bloodhounds in scenting ability, these French-origin dogs combine persistence with a sweet, patient nature.'),
        ('Dog', 'Beagle', 'Small', '12-15 years', 'Merry, friendly, curious', 'Moderate', True, 'High', 'Low', 'Small hound dogs known for their excellent sense of smell and friendly, outgoing personalities. Popular family pets and skilled scent hunters, famous for their melodious bay and pack mentality.'),
        ('Dog', 'Black-and-Tan_Coonhound', 'Large', '10-12 years', 'Even-tempered, easygoing, friendly', 'Moderate', True, 'High', 'Low', 'Large, powerful scent hounds known for their distinctive black and tan coloration and melodious bay.'),
        ('Dog', 'Bloodhound', 'Large', '10-12 years', 'Gentle, patient, stubborn', 'High', True, 'Moderate', 'Moderate', 'Known for their exceptional sense of smell, Bloodhounds are large, gentle dogs often used in tracking.'),
        ('Dog', 'Bluetick', 'Large', '11-12 years', 'Friendly, intelligent, active', 'Moderate', True, 'High', 'Low', 'Known for their mottled blue coat, Bluetick Coonhounds are skilled hunting dogs with a keen sense of smell and a melodious howl.'),
        ('Dog', 'Borzoi', 'Large', '10-12 years', 'Quiet, gentle, athletic', 'High', True, 'Moderate', 'High', 'Also known as Russian Wolfhounds, Borzois are elegant sighthounds known for their silky coat and graceful demeanor.'),
        ('Dog', 'Boxer', 'Large', '10-12 years', 'Fun-loving, bright, active', 'Moderate', True, 'High', 'Low', 'Playful and energetic, Boxers are known for their patient and protective nature with children. Originally developed in Germany as working dogs, they combine strength with a uniquely playful and clownish personality.'),
        ('Dog', 'Briard', 'Large', '10-12 years', 'Confident, smart, loyal', 'High', True, 'High', 'High', 'Large French herding dogs with a distinctive long, wavy coat, Briards are loyal and protective. Known as "hearts wrapped in fur," they served as WWI sentries and now excel as both working dogs and devoted family guardians.'),
        ('Dog', 'Bull_mastiff', 'Large', '8-10 years', 'Affectionate, loyal, quiet', 'Moderate', True, 'Moderate', 'Low', 'Large, powerful dogs originally bred to guard estates, Bullmastiffs are gentle giants with a calm demeanor.'),
        ('Dog', 'Cairn', 'Small', '13-15 years', 'Alert, cheerful, busy', 'Moderate', True, 'Moderate', 'Moderate', 'Small, rugged terriers known for their shaggy coat and lively personality. Originally bred to hunt in the Scottish Highlands, these hardy dogs are intelligent and make excellent watchdogs despite their small size.'),
        ('Dog', 'Chow', 'Medium', '8-12 years', 'Aloof, loyal, quiet', 'High', False, 'Low', 'High', 'Ancient Chinese breed known for their lion-like mane and blue-black tongues. Independent and dignified, they make excellent watchdogs but require early socialization.'),
        ('Dog', 'Clumber', 'Large', '10-12 years', 'Gentle, loyal, thoughtful', 'Moderate', True, 'Moderate', 'High', 'The largest of the spaniels, Clumbers are known for their distinctive white coat and calm demeanor. Developed in France and England, these dignified hunters combine power with a methodical hunting style and gentle nature.'),
        ('Dog', 'Cocker_Spaniel', 'Small', '10-14 years', 'Gentle, smart, happy', 'High', True, 'Moderate', 'High', 'Known for their long, silky ears and expressive eyes, Cockers are popular family dogs with a merry disposition.'),
        ('Dog', 'Collie', 'Large', '10-14 years', 'Devoted, graceful, proud', 'High', True, 'High', 'High', 'Made famous by "Lassie," Collies are intelligent herding dogs known for their loyalty and grace. Their remarkable intuition and gentle nature make them exceptional family guardians, especially with children.'),
        ('Dog', 'Curly-Coated_Retriever', 'Large', '10-12 years', 'Confident, independent, intelligent', 'Moderate', True, 'High', 'Low', 'Sporting dogs with a distinctive curly coat, known for their excellent swimming and retrieving abilities.'),
        ('Dog', 'Dhole', 'Medium', '10-13 years', 'Social, intelligent, athletic', 'High', False, 'High', 'Low', 'Also known as the Asiatic wild dog, Dholes are not typically kept as pets but are important in Asian ecosystems.'),
        ('Dog', 'Dingo', 'Medium', '10-13 years', 'Independent, intelligent, alert', 'High', False, 'High', 'Low', 'Native wild dogs of Australia, dingoes are not typically kept as pets and are important to the Australian ecosystem.'),
        ('Dog', 'Flat-Coated_Retriever', 'Large', '8-10 years', 'Optimistic, good-humored, outgoing', 'High', True, 'Very High', 'Moderate', 'Known for their shiny black or liver-colored coat, Flat-coated Retrievers are energetic and playful, excelling in both hunting and family life.'),
        ('Dog', 'Giant_Schnauzer', 'Large', '10-12 years', 'Loyal, intelligent, powerful', 'High', True, 'High', 'High', 'Large and powerful, Giant Schnauzers were originally bred as working dogs and require plenty of exercise.'),
        ('Dog', 'Golden_Retriever', 'Large', '10-12 years', 'Intelligent, friendly, devoted', 'High', True, 'High', 'High', 'Beautiful, golden-coated dogs known for their gentle nature and excellence in various roles. Popular as family companions, therapy dogs, and service animals, they excel in both work and companionship with their eager-to-please attitude.'),
        ('Dog', 'Groenendael', 'Large', '10-12 years', 'Intelligent, protective, loyal', 'High', True, 'High', 'High', 'The black variety of Belgian Shepherd, Groenendaels are intelligent working dogs with a long, black coat. Named after their village of origin, they excel in police work, herding, and as vigilant family guardians.'),
        ('Dog', 'Keeshond', 'Medium', '12-15 years', 'Friendly, lively, outgoing', 'Moderate', True, 'Moderate', 'High', 'Distinctive "spectacles" marking around their eyes, Keeshonds are fluffy, fox-like dogs known for their friendly and affectionate nature.'),
        ('Dog', 'Kelpie', 'Medium', '10-13 years', 'Intelligent, energetic, loyal', 'High', True, 'Very High', 'Low', 'Australian herding dogs known for their incredible work ethic and agility. Developed to work in harsh outback conditions, they are renowned for their ability to herd from above, often running across the backs of sheep in large flocks.'),
        ('Dog', 'Komondor', 'Large', '10-12 years', 'Steady, fearless, affectionate', 'High', True, 'Moderate', 'High', 'Large Hungarian sheepdogs known for their distinctive corded white coat, resembling dreadlocks. Their unique coat once helped them blend in with sheep flocks while protecting them from wolves.'),
        ('Dog', 'Kuvasz', 'Large', '10-12 years', 'Protective, loyal, patient', 'High', True, 'Moderate', 'High', 'Large, white guardian dogs from Hungary, Kuvaszok are protective of their families and independent. Once royal guards of Hungarian nobility, they combine impressive strength with natural protective instincts.'),
        ('Dog', 'Malamute', 'Large', '10-12 years', 'Affectionate, loyal, playful', 'High', True, 'Very High', 'High', 'Large, powerful sled dogs with thick coats, known for their strength and endurance. Originally bred by the Mahlemut tribe for hauling heavy loads in arctic conditions, they combine impressive power with a friendly, family-oriented nature.'),
        ('Dog', 'Malinois', 'Medium', '12-14 years', 'Confident, smart, hardworking', 'High', True, 'High', 'Moderate', 'One of four varieties of Belgian Shepherd, known for their intelligence and use in police and military work.'),
        ('Dog', 'Miniature_Pinscher', 'Small', '12-16 years', 'Fearless, energetic, alert', 'Moderate', False, 'Moderate', 'Low', 'Often called King of Toys, Miniature Pinschers are small, energetic dogs with a big personality. Despite their small size, these fearless dogs possess proud carriage and spirited animation.'),
        ('Dog', 'Miniature_Poodle', 'Small', '12-15 years', 'Intelligent, active, alert', 'High', True, 'Moderate', 'High', 'Smaller version of the Standard Poodle, known for their intelligence and hypoallergenic coat. Popular show dogs and companions, they retain their larger relatives high intelligence while being more adaptable to city living.'),
        ('Dog', 'Miniature_Schnauzer', 'Small', '12-15 years', 'Friendly, smart, obedient', 'Moderate', True, 'Moderate', 'High', 'The smallest of the Schnauzer breeds, known for their distinctive beard and eyebrows. Originally ratters and farm dogs, they combine intelligence with a spunky personality, making excellent watchdogs and family companions.'),
        ('Dog', 'Otterhound', 'Large', '10-13 years', 'Friendly, boisterous, even-tempered', 'High', True, 'High', 'High', 'Large, shaggy-coated hounds originally bred for hunting otters, now a rare breed. Known for their strong swimming ability and powerful nose, with less than 1000 remaining worldwide, making them rarer than giant pandas.'),
        ('Dog', 'Papillon', 'Small', '13-15 years', 'Happy, alert, friendly', 'Moderate', True, 'Moderate', 'Moderate', 'Small, elegant dogs known for their butterfly-like ears and lively personalities. Despite their delicate appearance, they are surprisingly athletic and intelligent, ranking among the top 10 smartest dog breeds. Also called the Continental Toy Spaniel.'),
        ('Dog', 'Pug', 'Small', '12-15 years', 'Charming, mischievous, loving', 'Moderate', True, 'Low', 'Moderate', 'Small, wrinkly-faced dogs known for their charming personality and comical expression. Once favored by Chinese emperors, these "multum in parvo" (much in little) dogs are excellent companions but need attention to their breathing and temperature regulation.'),
        ('Dog', 'Redbone', 'Large', '10-12 years', 'Even-tempered, amiable, eager to please', 'Moderate', True, 'High', 'Low', 'Known for their solid red coat, Redbone Coonhounds are athletic, warm-hearted dogs originally bred for hunting. Developed in the American South, they excel at tracking and treeing with remarkable stamina and a melodious voice.'),
        ('Dog', 'Schipperke', 'Small', '13-15 years', 'Confident, alert, curious', 'Moderate', True, 'Moderate', 'Moderate', 'Small, black dogs with a fox-like face, Schipperkes are known for their distinctive ruff and small, pointed ears. Originally Belgian barge dogs, these little captains earned their name as boat watchdogs and ratters.'),
        ('Dog', 'Silky_terrier', 'Small', '12-15 years', 'Friendly, quick, alert', 'Moderate', False, 'Moderate', 'High', 'Similar to Yorkshire Terriers but larger, Silky Terriers are playful and enjoy being part of family activities. Developed in Australia, they combine the refinement of toy dogs with the sturdy nature of working terriers.'),
        ('Dog', 'Soft-Coated_Wheaten_Terrier', 'Medium', '12-14 years', 'Happy, steady, self-confident', 'High', True, 'High', 'High', 'Known for their soft, wheat-colored coat and friendly demeanor, they make great family dogs. Developed in Ireland as farm dogs, they combined versatility in herding and hunting with a uniquely soft coat unlike other terriers.'),
        ('Dog', 'Standard_Poodle', 'Large', '10-18 years', 'Intelligent, active, dignified', 'High', True, 'High', 'High', 'Highly intelligent and elegant dogs, known for their hypoallergenic coat and versatility in various dog sports.'),
        ('Dog', 'Standard_Schnauzer', 'Medium', '13-16 years', 'Friendly, intelligent, obedient', 'High', True, 'High', 'High', 'The original Schnauzer breed, known for their distinctive beard and eyebrows and versatile working abilities.'),
        ('Dog', 'Toy_Poodle', 'Small', '12-18 years', 'Intelligent, lively, playful', 'High', True, 'Moderate', 'High', 'The smallest variety of Poodle, known for their intelligence, agility, and hypoallergenic coat. Despite their diminutive size, they retain the intelligence and athletic ability of their larger relatives.'),
        ('Dog', 'Toy_Terrier', 'Small', '12-16 years', 'Lively, bold, intelligent', 'Moderate', False, 'Moderate', 'Low', 'A general term for small terrier breeds, often referring to breeds like the English Toy Terrier or Toy Fox Terrier.'),
        ('Dog', 'Vizsla', 'Medium', '10-14 years', 'Affectionate, energetic, gentle', 'High', True, 'High', 'Low', 'Known for their golden-rust coat, Vizslas are versatile hunters and loving family companions. These Hungarian pointers are often called "velcro dogs" for their strong desire to stay close to their owners.'),
        ('Dog', 'Whippet', 'Medium', '12-15 years', 'Gentle, affectionate, quiet', 'Low', True, 'High', 'Low', 'Slender, athletic sighthounds known for their speed - capable of reaching 35mph. Despite high exercise needs, they are calm indoor companions and excellent apartment dogs.'),
        ('Dog', 'Wire-Haired_Fox_Terrier', 'Small', '12-15 years', 'Alert, confident, gregarious', 'High', True, 'High', 'High', 'Energetic and wire-coated, these terriers were originally bred for fox hunting. Their tough, dense coat and fearless nature made them ideal for flushing foxes from their dens, and they remain bold and spirited companions.'),
    ]

    cursor.executemany('''
    INSERT INTO AnimalCatalog (Species, Breed, Size_Category, Typical_Lifespan, Temperament, Care_Level, Good_With_Children, Exercise_Needs, Grooming_Needs, Brief_Description)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', dog_data)

    conn.commit()
    cursor.close()
    conn.close()

def get_dog_description(breed):
    try:
        conn = sqlite3.connect('animal_detector.db')
        cursor = conn.cursor()

        breed_name = breed.split('(')[0].strip()

        cursor.execute("""
            SELECT * FROM AnimalCatalog
            WHERE Breed = ? OR Breed LIKE ? OR Breed LIKE ?
        """, (breed_name, f"{breed_name}%", f"%{breed_name}"))

        result = cursor.fetchone()

        cursor.close()
        conn.close()

        if result:
            # 標準化運動需求值
            exercise_needs = result[8]
            normalized_exercise = exercise_needs.strip().title()
            if normalized_exercise not in ["Very High", "High", "Moderate", "Low"]:
                normalized_exercise = "High"  # 預設值

            description = {
                "Breed": result[2],
                "Size": result[3],
                "Lifespan": result[4],
                "Temperament": result[5],
                "Care Level": result[6],
                "Good with Children": "Yes" if result[7] else "No",
                "Exercise Needs": normalized_exercise,
                "Grooming Needs": result[9],
                "Description": result[10]
            }
            return description
        else:
            print(f"No data found for breed: {breed_name}")
            return None

    except Exception as e:
        print(f"Error in get_dog_description: {str(e)}")
        return None

insert_dog_data()


@dataclass
class UserPreferences:
    """使用者偏好設定的資料結構"""
    living_space: str  # "apartment", "house_small", "house_large"
    exercise_time: int  # minutes per day
    grooming_commitment: str  # "low", "medium", "high"
    experience_level: str  # "beginner", "intermediate", "advanced"
    has_children: bool
    noise_tolerance: str  # "low", "medium", "high"
    space_for_play: bool
    other_pets: bool
    climate: str  # "cold", "moderate", "hot"
    health_sensitivity: str = "medium"  # 設置默認值
    barking_acceptance: str = None  

    def __post_init__(self):
        """在初始化後運行，用於設置派生值"""
        if self.barking_acceptance is None:
            self.barking_acceptance = self.noise_tolerance


def calculate_compatibility_score(breed_info: dict, user_prefs: UserPreferences) -> dict:
    """計算品種與使用者條件的相容性分數"""
    scores = {}
    try:
        # 1. 空間相容性計算
        def calculate_space_score(size, living_space, has_yard):
            base_scores = {
                "Small": {"apartment": 0.95, "house_small": 1.0, "house_large": 0.90},
                "Medium": {"apartment": 0.65, "house_small": 0.90, "house_large": 1.0},
                "Large": {"apartment": 0.35, "house_small": 0.75, "house_large": 1.0},
                "Giant": {"apartment": 0.15, "house_small": 0.55, "house_large": 1.0}
            }

            base_score = base_scores.get(size, base_scores["Medium"])[living_space]
            adjustments = 0

            # 特殊情況調整
            if living_space == "apartment":
                if size == "Small":
                    adjustments += 0.05
                elif size in ["Large", "Giant"]:
                    adjustments -= 0.15

            if has_yard and living_space in ["house_small", "house_large"]:
                adjustments += 0.05

            return min(1.0, max(0, base_score + adjustments))

        # 2. 運動相容性計算
        def calculate_exercise_score(breed_exercise_needs, user_exercise_time):
            exercise_needs = {
                'VERY HIGH': 120,
                'HIGH': 90,
                'MODERATE': 60,
                'LOW': 30,
                'VARIES': 60
            }

            breed_need = exercise_needs.get(breed_exercise_needs.strip().upper(), 60)
            difference = abs(user_exercise_time - breed_need) / breed_need

            if difference == 0:
                return 1.0
            elif difference <= 0.2:
                return 0.95
            elif difference <= 0.4:
                return 0.85
            elif difference <= 0.6:
                return 0.70
            elif difference <= 0.8:
                return 0.50
            else:
                return 0.30

        # 3. 美容需求計算
        def calculate_grooming_score(breed_grooming_needs, user_commitment, breed_size):
            base_scores = {
                "High": {"low": 0.3, "medium": 0.7, "high": 1.0},
                "Moderate": {"low": 0.5, "medium": 0.9, "high": 1.0},
                "Low": {"low": 1.0, "medium": 0.95, "high": 0.9}
            }

            base_score = base_scores.get(breed_grooming_needs, base_scores["Moderate"])[user_commitment]

            if breed_size == "Large" and user_commitment == "low":
                base_score *= 0.80
            elif breed_size == "Giant" and user_commitment == "low":
                base_score *= 0.70

            return base_score

        # 4. 經驗等級計算
        def calculate_experience_score(care_level, user_experience, temperament):
            base_scores = {
                "High": {"beginner": 0.3, "intermediate": 0.7, "advanced": 1.0},
                "Moderate": {"beginner": 0.6, "intermediate": 0.9, "advanced": 1.0},
                "Low": {"beginner": 0.9, "intermediate": 1.0, "advanced": 1.0}
            }

            score = base_scores.get(care_level, base_scores["Moderate"])[user_experience]

            temperament_lower = temperament.lower()
            if user_experience == "beginner":
                if any(trait in temperament_lower for trait in ['stubborn', 'independent', 'intelligent']):
                    score *= 0.80
                if any(trait in temperament_lower for trait in ['easy', 'gentle', 'friendly']):
                    score *= 1.15

            return min(1.0, score)

        def calculate_health_score(breed_name: str) -> float:
            if breed_name not in breed_health_info:
                return 0.5

            health_notes = breed_health_info[breed_name]['health_notes'].lower()

            # 嚴重健康問題
            severe_conditions = [
                'cancer', 'cardiomyopathy', 'epilepsy', 'dysplasia',
                'bloat', 'progressive', 'syndrome'
            ]

            # 中等健康問題
            moderate_conditions = [
                'allergies', 'infections', 'thyroid', 'luxation',
                'skin problems', 'ear'
            ]

            severe_count = sum(1 for condition in severe_conditions if condition in health_notes)
            moderate_count = sum(1 for condition in moderate_conditions if condition in health_notes)

            health_score = 1.0
            health_score -= (severe_count * 0.1)
            health_score -= (moderate_count * 0.05)

            # 特殊條件調整
            if user_prefs.has_children:
                if 'requires frequent' in health_notes or 'regular monitoring' in health_notes:
                    health_score *= 0.9

            if user_prefs.experience_level == 'beginner':
                if 'requires frequent' in health_notes or 'requires experienced' in health_notes:
                    health_score *= 0.8

            return max(0.3, min(1.0, health_score))

        def calculate_noise_score(breed_name: str, user_noise_tolerance: str) -> float:
            if breed_name not in breed_noise_info:
                return 0.5

            noise_info = breed_noise_info[breed_name]
            noise_level = noise_info['noise_level'].lower()


            # 基礎噪音分數矩陣
            noise_matrix = {
                'low': {'low': 1.0, 'medium': 0.8, 'high': 0.6},
                'medium': {'low': 0.7, 'medium': 1.0, 'high': 0.8},
                'high': {'low': 0.4, 'medium': 0.7, 'high': 1.0}
            }

            # 從噪音矩陣獲取基礎分數
            base_score = noise_matrix.get(noise_level, {'low': 0.7, 'medium': 0.7, 'high': 0.7})[user_noise_tolerance]

            # 特殊情況調整
            special_adjustments = 0
            if user_prefs.has_children and noise_level == 'high':
                special_adjustments -= 0.1
            if user_prefs.living_space == 'apartment':
                if noise_level == 'high':
                    special_adjustments -= 0.15
                elif noise_level == 'medium':
                    special_adjustments -= 0.05

            final_score = base_score + special_adjustments
            return max(0.3, min(1.0, final_score))

        # 計算所有基礎分數
        scores = {
            'space': calculate_space_score(breed_info['Size'], user_prefs.living_space, user_prefs.space_for_play),
            'exercise': calculate_exercise_score(breed_info.get('Exercise Needs', 'Moderate'), user_prefs.exercise_time),
            'grooming': calculate_grooming_score(breed_info.get('Grooming Needs', 'Moderate'), user_prefs.grooming_commitment.lower(), breed_info['Size']),
            'experience': calculate_experience_score(breed_info.get('Care Level', 'Moderate'), user_prefs.experience_level, breed_info.get('Temperament', '')),
            'health': calculate_health_score(breed_info.get('Breed', '')),
            'noise': calculate_noise_score(breed_info.get('Breed', ''), user_prefs.noise_tolerance)
        }

        # 更新權重配置
        weights = {
            'space': 0.20,
            'exercise': 0.20,
            'grooming': 0.15,
            'experience': 0.15,
            'health': 0.15,
            'noise': 0.15
        }

        # 基礎分數計算
        base_score = sum(score * weights[category]
                        for category, score in scores.items()
                        if category != 'overall')

        # 額外調整
        adjustments = 0

        # 1. 適應性加分
        if breed_info.get('Adaptability', 'Medium') == 'High':
            adjustments += 0.02

        # 2. 氣候相容性
        if user_prefs.climate in breed_info.get('Suitable Climate', '').split(','):
            adjustments += 0.02

        # 3. 其他寵物相容性
        if user_prefs.other_pets and breed_info.get('Good with Other Pets') == 'Yes':
            adjustments += 0.02

        final_score = min(1.0, max(0, base_score + adjustments))
        scores['overall'] = round(final_score, 4)

        # 四捨五入所有分數
        for key in scores:
            scores[key] = round(scores[key], 4)

        return scores

    except Exception as e:
        print(f"Error in calculate_compatibility_score: {str(e)}")
        return {k: 0.5 for k in ['space', 'exercise', 'grooming', 'experience', 'health', 'noise', 'overall']}


def get_breed_recommendations(user_prefs: UserPreferences, top_n: int = 10) -> List[Dict]:
    """基於使用者偏好推薦狗品種，確保正確的分數排序"""
    print("Starting get_breed_recommendations")
    recommendations = []
    seen_breeds = set()

    try:
        # 獲取所有品種
        conn = sqlite3.connect('animal_detector.db')
        cursor = conn.cursor()  
        cursor.execute("SELECT Breed FROM AnimalCatalog")
        all_breeds = cursor.fetchall()
        conn.close()

        # 收集所有品種的分數
        for breed_tuple in all_breeds:
            breed = breed_tuple[0]
            base_breed = breed.split('(')[0].strip()

            if base_breed in seen_breeds:
                continue
            seen_breeds.add(base_breed)

            # 獲取品種資訊
            breed_info = get_dog_description(breed)
            if not isinstance(breed_info, dict):
                continue

            # 獲取噪音資訊
            noise_info = breed_noise_info.get(breed, {
                "noise_notes": "Noise information not available",
                "noise_level": "Unknown",
                "source": "N/A"
            })

            # 將噪音資訊整合到品種資訊中
            breed_info['noise_info'] = noise_info

            # 計算基礎相容性分數
            compatibility_scores = calculate_compatibility_score(breed_info, user_prefs)

            # 計算品種特定加分
            breed_bonus = 0.0

            # 壽命加分
            try:
                lifespan = breed_info.get('Lifespan', '10-12 years')
                years = [int(x) for x in lifespan.split('-')[0].split()[0:1]]
                longevity_bonus = min(0.02, (max(years) - 10) * 0.005)
                breed_bonus += longevity_bonus
            except:
                pass

            # 性格特徵加分
            temperament = breed_info.get('Temperament', '').lower()
            positive_traits = ['friendly', 'gentle', 'affectionate', 'intelligent']
            negative_traits = ['aggressive', 'stubborn', 'dominant']

            breed_bonus += sum(0.01 for trait in positive_traits if trait in temperament)
            breed_bonus -= sum(0.01 for trait in negative_traits if trait in temperament)

            # 與孩童相容性加分
            if user_prefs.has_children:
                if breed_info.get('Good with Children') == 'Yes':
                    breed_bonus += 0.02
                elif breed_info.get('Good with Children') == 'No':
                    breed_bonus -= 0.03

            # 噪音相關加分
            if user_prefs.noise_tolerance == 'low':
                if noise_info['noise_level'].lower() == 'high':
                    breed_bonus -= 0.03
                elif noise_info['noise_level'].lower() == 'low':
                    breed_bonus += 0.02
            elif user_prefs.noise_tolerance == 'high':
                if noise_info['noise_level'].lower() == 'high':
                    breed_bonus += 0.01

            # 計算最終分數
            breed_bonus = round(breed_bonus, 4)
            final_score = round(compatibility_scores['overall'] + breed_bonus, 4)

            recommendations.append({
                'breed': breed,
                'base_score': round(compatibility_scores['overall'], 4),
                'bonus_score': round(breed_bonus, 4),
                'final_score': final_score,
                'scores': compatibility_scores,
                'info': breed_info,
                'noise_info': noise_info  # 添加噪音資訊到推薦結果
            })
        # 嚴格按照 final_score 排序
        recommendations.sort(key=lambda x: (round(-x['final_score'], 4), x['breed'] ))  # 負號使其降序排列，並確保4位小數

        # 選擇前N名並確保正確排序
        final_recommendations = []
        last_score = None
        rank = 1

        for rec in recommendations:
            if len(final_recommendations) >= top_n:
                break

            current_score = rec['final_score']

            # 確保分數遞減
            if last_score is not None and current_score > last_score:
                continue

            # 添加排名資訊
            rec['rank'] = rank
            final_recommendations.append(rec)

            last_score = current_score
            rank += 1

        # 驗證最終排序
        for i in range(len(final_recommendations)-1):
            current = final_recommendations[i]
            next_rec = final_recommendations[i+1]

            if current['final_score'] < next_rec['final_score']:
                print(f"Warning: Sorting error detected!")
                print(f"#{i+1} {current['breed']}: {current['final_score']}")
                print(f"#{i+2} {next_rec['breed']}: {next_rec['final_score']}")

                # 交換位置
                final_recommendations[i], final_recommendations[i+1] = \
                    final_recommendations[i+1], final_recommendations[i]

        # 打印最終結果以供驗證
        print("\nFinal Rankings:")
        for rec in final_recommendations:
            print(f"#{rec['rank']} {rec['breed']}")
            print(f"Base Score: {rec['base_score']:.4f}")
            print(f"Bonus: {rec['bonus_score']:.4f}")
            print(f"Final Score: {rec['final_score']:.4f}\n")

        return final_recommendations

    except Exception as e:
        print(f"Error in get_breed_recommendations: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return []



def calculate_breed_bonus(breed_info: dict, user_prefs: UserPreferences) -> float:
    """計算品種額外加分"""
    bonus = 0.0

    # 壽命加分
    try:
        lifespan = breed_info.get('Lifespan', '10-12 years')
        years = [int(x) for x in lifespan.split('-')[0].split()[0:1]]
        longevity_bonus = min(0.05, (max(years) - 10) * 0.01)
        bonus += longevity_bonus
    except:
        pass

    # 性格特徵加分
    temperament = breed_info.get('Temperament', '').lower()
    if user_prefs.has_children:
        if 'gentle' in temperament or 'patient' in temperament:
            bonus += 0.03

    # 適應性加分
    if breed_info.get('Size') == "Small" and user_prefs.living_space == "apartment":
        bonus += 0.02

    return bonus

def calculate_additional_factors(breed_info: dict, user_prefs: UserPreferences) -> dict:
    """計算額外的排序因素"""
    factors = {
        'versatility': 0.0,
        'health_score': 0.0,
        'adaptability': 0.0
    }

    # 計算多功能性分數
    temperament = breed_info.get('Temperament', '').lower()
    versatile_traits = ['intelligent', 'adaptable', 'versatile', 'trainable']
    factors['versatility'] = sum(trait in temperament for trait in versatile_traits) / len(versatile_traits)

    # 計算健康分數（基於預期壽命）
    lifespan = breed_info.get('Lifespan', '10-12 years')
    try:
        years = [int(x) for x in lifespan.split('-')[0].split()[0:1]]
        factors['health_score'] = min(1.0, max(years) / 15)  # 標準化到0-1範圍
    except:
        factors['health_score'] = 0.5  # 預設值

    # 計算適應性分數
    size = breed_info.get('Size', 'Medium')
    factors['adaptability'] = {
        'Small': 0.9,
        'Medium': 0.7,
        'Large': 0.5,
        'Giant': 0.3
    }.get(size, 0.5)

    return factors

def format_recommendation_html(recommendations: List[Dict]) -> str:
    """將推薦結果格式化為HTML"""
    html_content = "<div class='recommendations-container'>"

    for rec in recommendations:
        breed = rec['breed']
        scores = rec['scores']
        info = rec['info']
        rank = rec.get('rank', 0)
        final_score = rec.get('final_score', scores['overall'])
        bonus_score = rec.get('bonus_score', 0)

        health_info = breed_health_info.get(breed, {"health_notes": default_health_note})
        noise_info = breed_noise_info.get(breed, {
            "noise_notes": "Noise information not available",
            "noise_level": "Unknown",
            "source": "N/A"
        })

        # 解析噪音資訊
        noise_notes = noise_info.get('noise_notes', '').split('\n')
        noise_characteristics = []
        barking_triggers = []
        noise_level = ''

        current_section = None
        for line in noise_notes:
            line = line.strip()
            if 'Typical noise characteristics:' in line:
                current_section = 'characteristics'
            elif 'Noise level:' in line:
                noise_level = line.replace('Noise level:', '').strip()
            elif 'Barking triggers:' in line:
                current_section = 'triggers'
            elif line.startswith('•'):
                if current_section == 'characteristics':
                    noise_characteristics.append(line[1:].strip())
                elif current_section == 'triggers':
                    barking_triggers.append(line[1:].strip())

        # 生成特徵和觸發因素的HTML
        noise_characteristics_html = '\n'.join([f'<li>{item}</li>' for item in noise_characteristics])
        barking_triggers_html = '\n'.join([f'<li>{item}</li>' for item in barking_triggers])

        # 處理健康資訊
        health_notes = health_info.get('health_notes', '').split('\n')
        health_considerations = []
        health_screenings = []

        current_section = None
        for line in health_notes:
            line = line.strip()
            if 'Common breed-specific health considerations' in line:
                current_section = 'considerations'
            elif 'Recommended health screenings:' in line:
                current_section = 'screenings'
            elif line.startswith('•'):
                if current_section == 'considerations':
                    health_considerations.append(line[1:].strip())
                elif current_section == 'screenings':
                    health_screenings.append(line[1:].strip())

        health_considerations_html = '\n'.join([f'<li>{item}</li>' for item in health_considerations])
        health_screenings_html = '\n'.join([f'<li>{item}</li>' for item in health_screenings])

        # 獎勵原因計算
        bonus_reasons = []
        temperament = info.get('Temperament', '').lower()
        if any(trait in temperament for trait in ['friendly', 'gentle', 'affectionate']):
            bonus_reasons.append("Positive temperament traits")
        if info.get('Good with Children') == 'Yes':
            bonus_reasons.append("Excellent with children")
        try:
            lifespan = info.get('Lifespan', '10-12 years')
            years = int(lifespan.split('-')[0])
            if years > 12:
                bonus_reasons.append("Above-average lifespan")
        except:
            pass

        html_content += f"""
        <div class="dog-info-card recommendation-card">
            <div class="breed-info">
                <h2 class="section-title">
                    <span class="icon">🏆</span> #{rank} {breed.replace('_', ' ')}
                    <span class="score-badge">
                        Overall Match: {final_score*100:.1f}%
                    </span>
                </h2>
                <div class="compatibility-scores">
                    <div class="score-item">
                        <span class="label">Space Compatibility:</span>
                        <div class="progress-bar">
                            <div class="progress" style="width: {scores['space']*100}%"></div>
                        </div>
                        <span class="percentage">{scores['space']*100:.1f}%</span>
                    </div>
                    <div class="score-item">
                        <span class="label">Exercise Match:</span>
                        <div class="progress-bar">
                            <div class="progress" style="width: {scores['exercise']*100}%"></div>
                        </div>
                        <span class="percentage">{scores['exercise']*100:.1f}%</span>
                    </div>
                    <div class="score-item">
                        <span class="label">Grooming Match:</span>
                        <div class="progress-bar">
                            <div class="progress" style="width: {scores['grooming']*100}%"></div>
                        </div>
                        <span class="percentage">{scores['grooming']*100:.1f}%</span>
                    </div>
                    <div class="score-item">
                        <span class="label">Experience Match:</span>
                        <div class="progress-bar">
                            <div class="progress" style="width: {scores['experience']*100}%"></div>
                        </div>
                        <span class="percentage">{scores['experience']*100:.1f}%</span>
                    </div>
                    {f'''
                    <div class="score-item bonus-score">
                        <span class="label">
                            Breed Bonus:
                            <span class="tooltip">
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Breed Bonus Points:</strong><br>
                                    • {('<br>• '.join(bonus_reasons)) if bonus_reasons else 'No additional bonus points'}<br>
                                    <br>
                                    <strong>Bonus Factors Include:</strong><br>
                                    • Friendly temperament<br>
                                    • Child compatibility<br>
                                    • Longer lifespan<br>
                                    • Living space adaptability
                                </span>
                            </span>
                        </span>
                        <div class="progress-bar">
                            <div class="progress" style="width: {bonus_score*100}%"></div>
                        </div>
                        <span class="percentage">{bonus_score*100:.1f}%</span>
                    </div>
                    ''' if bonus_score > 0 else ''}
                </div>
                <div class="breed-details-section">
                    <h3 class="subsection-title">
                        <span class="icon">📋</span> Breed Details
                    </h3>
                    <div class="details-grid">
                        <div class="detail-item">
                            <span class="tooltip">
                                <span class="icon">📏</span>
                                <span class="label">Size:</span>
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Size Categories:</strong><br>
                                    • Small: Under 20 pounds<br>
                                    • Medium: 20-60 pounds<br>
                                    • Large: Over 60 pounds
                                </span>
                                <span class="value">{info['Size']}</span>
                            </span>
                        </div>
                        <div class="detail-item">
                            <span class="tooltip">
                                <span class="icon">🏃</span>
                                <span class="label">Exercise Needs:</span>
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Exercise Needs:</strong><br>
                                    • Low: Short walks<br>
                                    • Moderate: 1-2 hours daily<br>
                                    • High: 2+ hours daily<br>
                                    • Very High: Constant activity
                                </span>
                                <span class="value">{info['Exercise Needs']}</span>
                            </span>
                        </div>
                        <div class="detail-item">
                            <span class="tooltip">
                                <span class="icon">👨‍👩‍👧‍👦</span>
                                <span class="label">Good with Children:</span>
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Child Compatibility:</strong><br>
                                    • Yes: Excellent with kids<br>
                                    • Moderate: Good with older children<br>
                                    • No: Better for adult households
                                </span>
                                <span class="value">{info['Good with Children']}</span>
                            </span>
                        </div>
                        <div class="detail-item">
                            <span class="tooltip">
                                <span class="icon">⏳</span>
                                <span class="label">Lifespan:</span>
                                <span class="tooltip-icon">ⓘ</span>
                                <span class="tooltip-text">
                                    <strong>Average Lifespan:</strong><br>
                                    • Short: 6-8 years<br>
                                    • Average: 10-15 years<br>
                                    • Long: 12-20 years<br>
                                    • Varies by size: Larger breeds typically have shorter lifespans
                                </span>
                            </span>
                            <span class="value">{info['Lifespan']}</span>
                        </div>
                    </div>
                </div>
                <div class="description-section">
                    <h3 class="subsection-title">
                        <span class="icon">📝</span> Description
                    </h3>
                    <p class="description-text">{info.get('Description', '')}</p>
                </div>
                <div class="noise-section">
                    <h3 class="subsection-title">
                        <span class="icon">🔊</span> Noise Behavior
                    </h3>
                    <div class="noise-info">
                        <div class="noise-details">
                            <div class="characteristics-block">
                                <h4>Typical noise characteristics:</h4>
                                <ul>
                                    {noise_characteristics_html}
                                </ul>

                                <div class="noise-level-block">
                                    <h4>Noise level:</h4>
                                    <p>{noise_level}</p>
                                </div>

                                <h4>Barking triggers:</h4>
                                <ul>
                                    {barking_triggers_html}
                                </ul>
                            </div>
                        </div>
                        <div class="health-disclaimer">
                            <p class="source-text">› Source: Compiled from various breed behavior resources, 2024</p>
                            <p>› Individual dogs may vary in their vocalization patterns.</p>
                            <p>› Training can significantly influence barking behavior.</p>
                            <p>› Environmental factors may affect noise levels.</p>
                        </div>
                    </div>
                </div>

                <div class="health-section">
                    <h3 class="subsection-title">
                        <span class="icon">🏥</span> Health Insights
                        <span class="tooltip">
                            <span class="tooltip-icon">ⓘ</span>
                            <span class="tooltip-text">
                                Health information is compiled from multiple sources including veterinary resources, breed guides, and international canine health databases.
                                Each dog is unique and may vary from these general guidelines.
                            </span>
                        </span>
                    </h3>
                    <div class="health-info">
                        <div class="health-details">
                            <h4>Common breed-specific health considerations :</h4>
                            <ul>
                                <li>Patellar luxation</li>
                                <li>Progressive retinal atrophy</li>
                                <li>Von Willebrand's disease</li>
                                <li>Open fontanel</li>
                            </ul>
                            <h4>Recommended health screenings:</h4>
                            <ul>
                                <li>Patella evaluation</li>
                                <li>Eye examination</li>
                                <li>Blood clotting tests</li>
                                <li>Skull development monitoring</li>
                            </ul>
                        </div>
                    </div>
                    <div class="health-disclaimer">
                        <p class="source">› Source: Compiled from various veterinary and breed information resources, 2024</p>
                        <p>› This information is for reference only and based on breed tendencies.</p>
                        <p>› Each dog is unique and may not develop any or all of these conditions.</p>
                        <p>› Always consult with qualified veterinarians for professional advice.</p>
                    </div>
                </div>
                <div class="action-section">
                    <a href="https://www.akc.org/dog-breeds/{breed.lower().replace('_', '-')}/"
                       target="_blank"
                       class="akc-button">
                        <span class="icon">🌐</span>
                        Learn more about {breed.replace('_', ' ')} on AKC website
                    </a>
                </div>
            </div>
        </div>
        """

    html_content += "</div>"
    return html_content
