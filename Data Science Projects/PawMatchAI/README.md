# 🐾 PawMatchAI 🐾

## ✨ Project Overview
**PawMatchAI** is an advanced dog breed classification system that identifies **124 dog breeds**, built on the **Stanford Dog Dataset** with **21,000+ images**, and extends to include four additional popular breeds: **Shiba Inu**, **Dachshund**, **Bichon Frise**, and **Havanese**.

As an extension of its core classification capabilities, PawMatchAI includes a breed recommendation system and breed comparison to provide personalized breed suggestions based on user preferences. This makes the system not only accurate in identifying breeds but also versatile in helping users find their perfect canine companion.

---

## 🎯 Key Features

### 1. 🔍 **Breed Detection**
- **Photo Identification:** Upload a photo, and the model identifies the dog's breed from 124 possible options.
- **Detailed Information:** Provides essential breed details, including:
  - **Exercise Needs:** Typical activity requirements.
  - **Lifespan:** Average expected lifespan.
  - **Grooming Requirements:** Coat care and maintenance levels.
  - **Health Insights:** Common health issues and considerations.
  - **Noise Behavior:** Typical vocalization tendencies.

### 2. 📋 **Breed Comparison Tool**
- Compare two breeds side-by-side based on:
  - **Care Requirements:** Grooming, exercise, and maintenance needs.
  - **Personality Traits:** Typical behaviors and temperament.
  - **Health Considerations:** Lifespan and common health issues.
  - **Noise Behavior:** Vocalization levels and tendencies.

### 3. 💡 **Breed Recommendation System**
Offers two methods to suggest suitable breeds:
#### ✅ Criteria-Based Matching
- **Status:** Currently under development to improve accuracy and reliability in generating recommendations.
- **Core Matching:** Evaluates key lifestyle compatibility factors:
  1. **Space Compatibility (30%):** Matches breed size and activity needs to user living environments (e.g., apartments, large houses).
  2. **Exercise Match (25%):** Compares user exercise availability with breed activity levels.
  3. **Grooming Compatibility (15%):** Considers coat type and grooming effort.
  4. **Experience Level Match (30%):** Aligns breed care difficulty with user experience.

- **Bonus Adjustments:**
  - **Longevity Bonus:** Rewards long-lived breeds.
  - **Family-Friendly Bonus:** Highlights gentle, sociable breeds.
  - **Adaptability Bonus:** Adds scores for breeds suitable for varying climates or environments.
  - **Health Penalty:** Penalizes breeds prone to severe health risks.

---

## 📊 Results
- **F1 Score:** The system achieved an overall **F1 Score of 82.30%,** excelling at fine-grained classification across 124 breeds.
- **Few-Shot Ready:** The architecture is prepared to support new breeds or species with minimal training data, enabling easy scalability.

---

## 🧠 Technical Highlights

### 🦴 Model Backbone

1. **EfficientNetV2(M):**
   - **What It Does:** EfficientNetV2(M) is the backbone model for feature extraction. Pretrained on ImageNet, it provides a robust foundation for extracting meaningful patterns from high-resolution images, which is essential for distinguishing between visually similar breeds.
   - **Why It Matters:** EfficientNetV2(M) achieves a balance between computational efficiency and accuracy. Its architecture includes advanced layers like Fused-MBConv, which improve both inference speed and feature quality, making it ideal for fine-grained tasks like dog breed classification.
   - **Key Features:**
     - Scalable design to handle varying image sizes.
     - Optimized for both training speed and real-world inference performance.
     - Handles subtle visual differences, such as slight variations in coat texture or color patterns.

2. **Multi-Head Attention:**
   - **How It Works:** Multi-Head Attention is inspired by the Transformer architecture. It divides the extracted features into multiple "heads," where each head focuses on specific parts of the image. These heads work independently to analyze different regions, such as the dog's face, ears, or body.
   - **Technical Insight:** This mechanism is implemented efficiently using **torch.einsum**, which computes attention weights while minimizing memory usage and improving computational speed.
   - **Why It’s Effective:** Multi-Head Attention enhances the model’s ability to:
     - Focus on critical details, such as unique facial patterns.
     - Capture dependencies between different image regions, improving classification accuracy.
   - **Example Impact:** Helps differentiate breeds with similar appearances, like the Siberian Husky and Alaskan Malamute.

3. **Prototype Networks:**
   - **Definition:** A prototype represents the central feature for a specific breed. It is calculated as the average embedding vector for all training samples of that breed.
   - **Use Case:** During classification, a new image’s features are compared to these prototypes, and the closest match determines the predicted breed.
   - **Why It’s Important:** Prototype Networks simplify scalability:
     - Adding a new breed only requires computing its prototype, avoiding full model retraining.
     - Works effectively with limited data, supporting **Few-Shot Learning** for new species or rare breeds.
   - **Real-World Example:** Allows quick adaptation for classifying additional species like cats or birds by adding their prototypes.

---

### 📈 Training and Optimization

1. **Data Augmentation:**
   - **RandAugment:** 
     - Applies random combinations of transformations (e.g., rotation, scaling, brightness adjustment).
     - Encourages the model to generalize better by exposing it to diverse image variations.
     - Example: Simulating different lighting conditions to make the model robust against outdoor and indoor photos.
   - **RandomErasing:**
     - Randomly removes a portion of the image during training to simulate occlusions (e.g., parts of the dog being blocked).
     - Trains the model to rely on global patterns rather than overly focusing on one region.

2. **Loss Functions:**
   - **Contrastive Loss:**
     - **Ensures that images of the same breed are grouped closer in the feature space, while images of different breeds are pushed farther apart.**
     - Formula:
       $[
       L = y \cdot d^2 + (1 - y) \cdot \max(0, m - d)^2
       $]
       - $( y $): Label (1 for the same breed, 0 for different breeds).
       - $( d $): Distance between embeddings.
       - $( m $): Margin separating different breeds.
     - **Impact:** Makes it easier for the model to handle breeds with subtle differences, such as Labrador Retrievers and Golden Retrievers.
   - **Focal Loss:**
     - Adjusts the loss function to focus more on hard-to-classify samples, especially those from underrepresented breeds.
     - Formula:
       $[
       L = -\alpha \cdot (1 - p_t)^\gamma \cdot \log(p_t)
       $]
       - $( p_t $): Predicted probability for the true class.
       - $( \alpha, \gamma $): Hyperparameters to control the focus.
     - **Impact:** Addresses class imbalance, ensuring rarer breeds like the Shiba Inu are classified with the same confidence as more common breeds.

3. **Learning Strategies:**
   - **OneCycleLR Scheduler:**
     - Dynamically adjusts the learning rate during training. Starts with a small value, increases it rapidly, and then gradually reduces it again.
     - **Why It’s Effective:** Encourages faster convergence and reduces the risk of overfitting by focusing training in the most critical learning phase.
     - Example: Prevents the model from getting stuck in local minima.
   - **Progressive Unfreezing:**
     - During fine-tuning, initially keeps deeper layers of the backbone frozen and gradually unfreezes them as training progresses.
     - **Why It Works:** Stabilizes the early training phase by focusing on learning high-level features before fine-tuning deeper, more specialized layers.

4. **Mixed Precision Training:**
   - **Efficiency:** Uses lower precision (16-bit floats) for most calculations to save memory and speed up training, while retaining 32-bit precision for critical operations.
   - **Implementation:** Combines **GradScaler** and **autocast** in PyTorch for seamless integration.
   - **Impact:** Enables training on larger datasets and models without requiring high-end GPUs.
     
---

### 🔎 Dog Detection using YOLO
- **YOLOv8:** Integrated during the **deployment phase** for multi-dog detection in a single image. YOLO was not used during the model training process but was added later to enhance the system's ability to handle real-world use cases where multiple dogs may appear in a single image.
  - **Key Parameters:**
    - **`conf_threshold`:** Filters weak predictions to retain only confident detections.
    - **`iou_threshold`:** Removes overlapping bounding boxes to ensure clean and accurate detections.
  - **Why YOLO?:** Real-time object detection capabilities make it ideal for deployment scenarios where speed and efficiency are critical. This ensures that users can receive accurate results even with complex images featuring multiple dogs.

---

## 🌐 Model Deployment
The model is deployed on **Hugging Face Spaces**, providing users with an intuitive interface for:
1. **Breed Detection:** Upload an image for detailed classification results.
2. **Breed Comparison:** Explore side-by-side comparisons of two breeds.
3. **Breed Recommendation:** Receive personalized suggestions based on preferences.

> **Try it yourself**: [PawMatch AI](https://huggingface.co/spaces/DawnC/PawMatchAI)

---

## 🚀 Potential Improvements
1. **Feature Enhancement for Challenging Breeds:** Identify and focus on poorly performing breeds by enhancing feature representations, using techniques like targeted data augmentation or fine-tuning specific layers for these categories.
2. **Expanded Augmentation:** Introduce more complex data augmentations to cover edge cases.
3. **Dynamic Weight Adjustment:** Allow users to customize recommendation weightings, such as prioritizing exercise needs.
4. **Real-Time Inference:** Optimize the system for deployment on mobile or embedded devices.

---

## 🌱 Future Thoughts
1. **Multi-Species Expansion:** Extend support to other species like cats or birds while maintaining accuracy.
2. **Transfer Learning for Species:** Quickly adapt the model to classify new species with minimal retraining.
3. **Interactive Feedback:** Incorporate user feedback to refine recommendations dynamically.

---

## 📚 Acknowledgments and References
- [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/data)
- [EfficientNetV2](https://arxiv.org/pdf/2104.00298)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [Prototypical Networks](https://arxiv.org/pdf/1703.05175)
- [Focal Loss](https://arxiv.org/pdf/1708.02002)
- [SBERT](https://arxiv.org/pdf/1908.10084)

---
