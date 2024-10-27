# 🐕 Dog Breed Classifier 🐾

## ✨ Project Overview:
* This project focuses on classifying **120 dog breeds** using the **Stanford Dog Dataset**, containing **20,580 images**. The model architecture utilizes **EfficientNetV2(M)**, enhanced with **Multi-Head Attention** for refined feature extraction. It also incorporates **Prototype Networks** for **Few-Shot Learning**, allowing new breeds to be added in the future without retraining the entire model.

* Future expansions include adding more animal species, such as **cats**, to transform the model into a multi-species classification system capable of recognizing various breeds from different animals.

* Advanced loss functions like **Focal Loss** and **Contrastive Loss** were used to address class imbalance and ensure effective feature separation, which is critical in handling a large number of dog breeds and potentially new species.

## ⚙️ Skills Used:
### 🐍 Python Libraries and Frameworks:
* PyTorch
* NumPy
* Matplotlib
* Seaborn
* PIL
* Scikit-learn
* TensorBoard
* SQLite

### 📊 Deep Learning Techniques:
* **EfficientNetV2(M)**: Backbone architecture pretrained on ImageNet for superior feature extraction.
* **Multi-Head Attention**: Enhances the model’s focus on relevant parts of the image, improving classification performance by effectively weighting image regions.
* **Prototype Networks**: Designed to support the future classification of not only new dog breeds but also new animal species, enabling the model to quickly learn prototypes for new classes.
* **Few-Shot Learning**: Although not fully realized yet, the model is prepared to handle new breeds with minimal training data in future updates.
* **Contrastive Loss**: Ensures that similar classes are grouped closer in the feature space while dissimilar ones are further apart, which is crucial for high-dimensional, multi-class classification.
* **Focal Loss**: Addresses class imbalance by focusing on misclassified or harder examples, allowing the model to better learn underrepresented breeds.
* **OneCycle Learning Rate Scheduler**: Optimizes the learning rate dynamically to improve convergence speed and performance.
* **Mixed Precision Training**: Applied **GradScaler** and **autocast** to speed up training and reduce memory usage.

## 🧠 Skills Detail:
### 🦴 Model Backbone:
* The model's base architecture is **EfficientNetV2(M)**, well-known for its efficiency and accuracy in feature extraction. By integrating **Multi-Head Attention**, the model can better focus on critical features of the input image, enhancing breed classification. This attention mechanism splits the feature space into multiple heads, allowing for the most relevant features to be extracted.
* **Prototype Networks** enable efficient classification by storing prototypes for each class. New data points are classified based on their similarity to these prototypes. This approach makes the model scalable for future additions of new breeds or even new animal species, such as cats, with minimal training data required.
* **torch.einsum** is used to efficiently compute attention across multiple heads, optimizing the computation process for the attention layers.

### 📈 Training and Optimization:
* **Data Augmentation**: Applied extensive data augmentation techniques during training to enhance model generalization. Techniques included **RandomResizedCrop**, **RandomHorizontalFlip**, **RandAugment**, **ColorJitter**, and **RandomErasing**. These augmentations help the model handle variations in image size, lighting, and occlusions, making it more robust to real world data.

* **Contrastive Loss**: Applied to maximize the feature separation between different classes while minimizing the distance between samples of the same class. This ensures that breeds with similar features are correctly classified while avoiding confusion between different breeds.
* **Focal Loss**: Adjusts the weight of hard-to-classify samples, especially those from underrepresented classes, making the model more robust against class imbalance.
* **OneCycleLR**: This learning rate scheduler adjusts the learning rate dynamically during training, helping the model converge faster and preventing overfitting in the later stages of training.
* **Progressive Unfreezing**: Gradually unfroze layers of the **EfficientNetV2(M)** backbone during training. This technique starts with only the final layers being trainable and progressively unfreezes deeper layers at set intervals, helping to stabilize training and prevent catastrophic forgetting while fine-tuning the model.

### 🔎 Dog Detection using YOLO
- In this project, **YOLO (You Only Look Once)** was used for detecting multiple dogs in an image. YOLO is known for its real-time object detection capabilities, and it was applied here for dog breed identification during the deployment phase, not during model training.
  
- The `detect_multiple_dogs` function uses a pre-trained YOLO model to detect multiple dogs in a given image. The function applies non-maximum suppression (NMS) to filter out overlapping bounding boxes, ensuring that each dog is detected only once.

- #### Key Parameters:
- **`conf_threshold`**: The confidence threshold for filtering weak detections. Detections below this threshold are ignored.
- **`iou_threshold`**: The Intersection over Union (IoU) threshold used during non-maximum suppression to remove overlapping bounding boxes.

### 🧪 Evaluation:
* The **EfficientNetV2(M)** backbone combined with **Multi-Head Attention** allowed the model to focus on critical features, while **Focal Loss and Contrastive Loss** helped address class imbalances and improve performance.
* The model is also ready to support **Few-Shot Learning**, allowing new breeds or animal species to be added efficiently by updating the prototypes.
* Although **Few-Shot Learning** is currently not implemented in practice, the model architecture is designed to support it in the future. Prototypes can be learned from a small number of examples, enabling quick adaptation to new breeds or species.

## 📊 Results:
* **F1 Score**: The model achieved an overall **F1 Score** of **81.22%** across all 120 breeds.

## 🎯 Conclusion:
This project demonstrates a sophisticated use of **EfficientNetV2(M)** and **Multi-Head Attention** for large-scale dog breed classification. With the integration of **Prototype Networks**, the system is ready for future expansions, allowing new breeds or even new species, such as **cats**, to be added with minimal data through **Few-Shot Learning**. The use of **Contrastive Loss** and **Focal Loss** enhances the model's capability to manage class imbalance and distinguish similar breeds effectively.

## 🚀 Potential Improvements:
* **Prototype Optimization**: Refining the prototype learning process will improve the system's ability to incorporate new breeds or animal species quickly.
* **Advanced Data Augmentation**: Further exploration of data augmentation techniques can make the model more resilient to variations in input images.
* **Attention Mechanism Tuning**: Fine-tuning the **Multi-Head Attention** parameters could improve the model's ability to focus on the most important features within images, leading to better classification accuracy.

## 🌱 Future Thoughts:
* **Multi-Species Expansion**: Exploring ways to extend the model to recognize other species, such as cats and birds, while maintaining accuracy and efficiency across diverse animal categories.
* **Real-Time Inference**: Optimizing the model for faster inference times, potentially enabling real-time breed classification in mobile or embedded devices.
* **Transfer Learning for Species**: Investigating the use of transfer learning to quickly adapt the model to entirely new species without extensive retraining on massive datasets.

## 🚀 Model Deployment and Features:
* The model is deployed on **Hugging Face Spaces**, allowing users to upload an image and receive the predicted dog breed. After classification, the system integrates with an **SQLite** database to provide brief information about the identified breed.
* In addition to returning breed information, the system offers **external links** for users to explore more comprehensive resources, giving them further insights into the breed, such as its history, care requirements, or notable facts.

### 🎯 Key Features

### 1. Breed Recommendation System
This intelligent recommendation system helps users find their perfect canine companion based on various lifestyle factors and preferences:

#### Input Parameters:
- Living Space (apartment / small house / large house)
- Available Exercise Time (minutes per day)
- Grooming Commitment (low / medium / high)
- Experience Level (beginner / intermediate / advanced)
- Presence of Children
- Noise Tolerance
- Space for Play
- Other Pets
- Climate Conditions

#### Scoring Components:

##### Base Score Calculation (70% of Total)
- **Space Compatibility (30%)**: Evaluates how well the breed suits your living space
  - Small breeds score higher for apartments
  - Large breeds receive bonuses for houses with yards
  
- **Exercise Match (25%)**: Compares your available exercise time with breed needs
  - Very High: 120+ minutes/day
  - High: 90 minutes/day
  - Moderate: 60 minutes/day
  - Low: 30 minutes/day

- **Grooming Compatibility (15%)**: Matches grooming needs with your commitment level
  - Considers coat type and maintenance requirements
  - Adjusts for breed size (larger dogs need more grooming time)

- **Experience Level Match (30%)**: Aligns breed difficulty with owner experience
  - Factors in temperament and trainability
  - Considers breed-specific challenges

##### Bonus Score System (Additional 30%)
Breeds can earn bonus points based on:
- **Longevity**: Breeds with above-average lifespan (+0.5% per year above 10)
- **Temperament Traits**:
  - Friendly/Gentle/Affectionate (+1% each)
  - Good with children (when applicable) (+2%)
- **Adaptability**: Special bonuses for breeds that excel in specific environments
- **Health Factors**: Consideration of breed-specific health predispositions

### 2. Comprehensive Breed Information
Each recommendation includes:
- Detailed breed characteristics
- Exercise requirements
- Grooming needs
- Temperament description
- Child compatibility
- Health insights and common medical considerations
- External resources and AKC links

### 3. Health Information System
Provides breed-specific health insights:
- Common health concerns
- Recommended health screenings
- Veterinary care requirements
- Lifespan expectations
- Preventive care recommendations

### 4. Breed Comparison Tool
Allows users to:
- Compare multiple breeds side-by-side
- Evaluate differences in care requirements
- Assess compatibility scores
- Review health considerations
- Compare maintenance needs

### 5. Interactive Visualization
- Progress bars for compatibility scores
- Visual representation of match percentages
- Tooltips with detailed information
- Mobile-responsive design

## 🌐 Try it Yourself:
You can test the model directly on [PawMatch AI](https://huggingface.co/spaces/DawnC/Dog_Breed_Classifier), where it’s live and ready to classify your dog images. Simply upload an image, the model will identify the breed and give you some information about it.

## 📚 Acknowledgments and References:
- [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/data)
- [EfficientNetV2: Smaller Models and Faster Traing](https://arxiv.org/pdf/2104.00298)
- [torch.einsum resource](https://blog.csdn.net/ViatorSun/article/details/122710515)
