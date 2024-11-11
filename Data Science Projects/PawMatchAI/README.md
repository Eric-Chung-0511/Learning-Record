# 🐾 PawMatchAI 🐾

## ✨ Project Overview:
* This project focuses on classifying **124 dog breeds** , based on the **Stanford Dog Dataset**, containing **21,000+ images**. the model was extended to include four additional popular breeds: **Shiba Inu**, **Dachshund**, **Bichon Frise**, and **Havanese**. These additions were made because they are commonly recognized breeds, broadening the model's applicability and making it more versatile for practical use cases.
  

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
* Gradio

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
* **F1 Score**: The model achieved an overall **F1 Score** of **82.30%** across all 124 breeds.

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
* The model is deployed on **Hugging Face Spaces**, allowing users to upload an image and receive an accurate prediction of the dog breed. After classification, the system integrates with an **SQLite** database to provide essential information about the identified breed.
* In addition to returning breed information, the system offers **external links** for users to explore more comprehensive resources, such as the breed's history, care requirements, and notable facts.

### 🎯 Key Features

#### 1. 🐶 **Breed Detection**
* Utilizing the advanced **EfficientNetV2(M)** and **Multi-Head Attention** models, the system accurately detects and classifies 120 distinct dog breeds. Users can simply upload an image, and the model will identify the breed, providing breed details and a link to more in-depth resources for further exploration.
  

#### 2. 🔍 **Breed Comparison Tool**
* Users can select and compare two different dog breeds side-by-side, evaluating their characteristics, such as:

- **Care Requirements**: Information on grooming, exercise, lifespan, and overall maintenance

- **Personality Traits**: Typical behaviors and temperament

- **Health Considerations**: Common health concerns and recommended screenings

- **Noise Behavior**: Typical vocalization levels and triggers


#### 3. 💡 **Breed Recommendation System**
Experience a smarter way to find your ideal dog breed through our dual-methodology recommendation system, blending lifestyle compatibility analysis with AI-powered breed matching.

##### 🎯 A. Criteria-Based Matching
Evaluates compatibility based on lifestyle factors and preferences through a comprehensive scoring system:

###### 💯 Base Score Calculation (70%)
- **Space Compatibility (30%)**
  - Evaluates living space suitability (apartment/house)
  - Assigns higher scores for size-appropriate matches

- **Exercise Match (25%)**
  - Very High: 120+ minutes/day
  - High: 90 minutes/day
  - Moderate: 60 minutes/day
  - Low: 30 minutes/day

- **Grooming Compatibility (15%)**
  - Considers coat maintenance requirements
  - Adjusts for breed size variations

- **Experience Level Match (30%)**
  - Aligns breed difficulty with owner experience
  - Factors in temperament and trainability

###### 🌟 Bonus Score System (30%)
- **Longevity**: +0.5% per year over 10 years
- **Temperament Traits**:
  - Friendly/Gentle/Affectionate: +1% each
  - Child-friendly: +2%
- **Adaptability & Health Factors**
  - Environment suitability
  - Health predisposition consideration

##### 🔄 B. Description-Based Matching
Leverages advanced NLP technology for nuanced breed matching:

#### 🧠 Technical Implementation
- **Embedding Model**: Sentence-BERT (all-mpnet-base-v2)
  - State-of-the-art transformer architecture
  - Optimized for semantic textual similarity
  - Generates high-dimensional vectors capturing breed characteristics
  - Caching embeddings for efficient similarity computation

#### 📊 Matching Components & Weights
- **Description Similarity (25%)**: Semantic matching of user preferences using Sentence-BERT embeddings and cosine similarity.
- **Temperament Compatibility (20%)**: Evaluates alignment of personality traits through cosine similarity of temperament embeddings.
- **Exercise Requirements (20%)**: Matches user activity level preferences (low, moderate, high) with breed exercise needs.
- **Health Factors (15%)**: Based on breed health conditions, considers severe and moderate health issues, adjusted by user health sensitivity.
- **Noise Level (15%)**: Matches user preferences on barking tendencies using predefined noise levels (Low, Moderate, High).
- **Size Compatibility (5%)**: Considers physical size compatibility (Small, Medium, Large) between user preferences and breed.

#### 💫 Scoring Mechanism
- **Cosine Similarity**: Utilized for text-based comparisons (description, temperament).
- **Categorical Attribute Scoring**: Specialized scoring for exercise requirements, size, health, and noise level.
- **Weighted Average**: Combines all weighted scores for a comprehensive matching score.

#### 📑 Comprehensive Output
Each recommendation includes:
- **Breed Characteristics & Requirements**: Detailed breed information with key attributes.
- **Compatibility Breakdown**: Visualized breakdown of each score component (Description, Temperament, Exercise, etc.).
- **Health Insights & Care Requirements**: Includes common health issues and suggested care tips.
- **Interactive Visualizations**:
  - Compatibility score bars for each feature
  - Match percentage displays
  - Feature-specific tooltips for detailed explanations
  - Mobile-responsive interface for seamless user experience
  - Side-by-side breed comparisons for easy decision-making

#### 🔍 Unique Features
- Dual matching approaches for enhanced accuracy
- NLP-powered semantic understanding
- Comprehensive health information system
- User-friendly visualization tools
 

## 🌐 Try it Yourself:
You can test the model directly on HuggingFace — I call it [PawMatch AI](https://huggingface.co/spaces/DawnC/PawMatchAI). It's live and ready to classify your dog images with just a simple upload! The model will not only identify the breed but also provide detailed information about it, including key traits and care tips.

## 📚 Acknowledgments and References:
- [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/data)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [EfficientNetV2: Smaller Models and Faster Traing](https://arxiv.org/pdf/2104.00298)
- [torch.einsum resource](https://blog.csdn.net/ViatorSun/article/details/122710515)
- [A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)](https://arxiv.org/pdf/2002.05709)
- [Prototypical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175)

