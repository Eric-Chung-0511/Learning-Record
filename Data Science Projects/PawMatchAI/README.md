# 🐾 PawMatchAI 🐾

## ✨ Project Overview
**PawMatchAI** is an advanced dog breed classification system that identifies **124 dog breeds**, built on the **Stanford Dog Dataset** with **21,000+ images**, and extends to include four additional popular breeds: **Shiba Inu**, **Dachshund**, **Bichon Frise**, and **Havanese**.

As an extension of its core classification capabilities, PawMatchAI includes a **breed recommendation system and breed comparison** to provide personalized breed suggestions based on user preferences. This makes the system not only accurate in identifying breeds but also versatile in helping users find their perfect canine companion.

---

## 📊 Project Impact
**This project has achieved:**

> ![Visits](https://img.shields.io/badge/Total%20Visits-30k+-blue)
![Model Runs](https://img.shields.io/badge/Model%20Runs-12k+-green)
![Weekly Users](https://img.shields.io/badge/Weekly%20Users-6k+-orange)


## 🎯 Key Features

### 1. 🔍 **Breed Detection**
- **Photo Identification:** Upload a photo, and the model identifies the dog's breed from 124 possible options.
  - **Recognition Process:**
    - **Initial Detection:** **YOLOv8** model first detects and focuses specifically on dog subjects within the uploaded image
    - **Breed Classification:** Our specialized model analyzes the detected dog and processes breed identification
    - **Confidence-Based Results:** System provides breed information based on confidence levels:
      - High Confidence (≥40%): Directly presents the identified breed with detailed information
      - Medium Confidence (15%-39%): Offers top 3 most probable breed matches for consideration
      - Low Confidence (<15%): Indicates the dog's breed may not be included in our current dataset
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
The intelligent matching system evaluates compatibility between potential dog owners and breeds through a sophisticated algorithm. The system consists of:

- **Core Matching Process:**
  - Processes comprehensive compatibility analysis through multiple dimensions
  - Adapts scoring based on individual circumstances
  - Employs dynamic evaluation mechanisms
  
- **Primary Evaluation Categories (Base Score):**
  - **Space Compatibility (25%):**
    - Evaluates living environment suitability
    - Assesses breed size compatibility
    - Considers available outdoor space
  - **Exercise Requirements (25%):**
    - Matches daily activity needs
    - Evaluates exercise intensity compatibility
    - Assesses energy level alignment
  - **Experience Requirements (20%):**
    - Considers owner expertise level
    - Evaluates breed-specific training needs
    - Assesses special care requirements
  - **Grooming Considerations (10%):**
    - Evaluates maintenance needs
    - Assesses owner's time commitment
    - Considers professional grooming requirements
  - **Health Factors (10%):**
    - Reviews genetic health considerations
    - Evaluates longevity factors
    - Assesses care intensity requirements
  - **Noise Compatibility (10%):**
    - Considers barking tendencies
    - Evaluates environmental restrictions
    - Assesses owner tolerance levels

- **Enhancement Scoring System:**
  - **Longevity Bonuses (up to +5%):**
    - Above-average lifespan consideration
    - Health resilience evaluation
    - Quality of life assessment
  - **Personality Trait Bonuses (up to +15%):**
    - Friendly disposition (+5%)
    - Gentle nature (+5%)
    - Patient demeanor (+5%)
    - Intelligence (+4%)
    - Adaptability (+4%)
    - Affectionate behavior (+4%)
  - **Adaptability Bonuses (up to +10%):**
    - Apartment suitability (+5%)
    - Temperament flexibility (+5%)
    - Climate adaptation (+5%)
  - **Family Compatibility Bonuses (up to +10%):**
    - Child-friendly characteristics (+6%)
    - Patient temperament (+5%)
    - Gentle nature (+5%)
    - Tolerant personality (+4%)

- **Final Score Interpretation:**
  - **Outstanding Match (90-100):**
    - Exceptional compatibility with significant breed advantages
    - Represents top 4% of matches
  - **Excellent Fit (80-89):**
    - Strong overall compatibility with positive breed traits
    - Represents top 15% of matches
  - **Good Match (70-79):**
    - Solid fundamental compatibility
    - Represents 35% of matches
  - **Acceptable Match (60-69):**
    - Basic compatibility with some considerations
    - Represents 30% of matches
  - **Not Recommended (Below 60):**
    - Significant compatibility concerns
    - Represents 16% of matches

- **Special Considerations:**
  - **Age-Specific Adjustments:**
    - Enhanced evaluation for families with toddlers
    - Standard assessment for school-age children
    - Flexible criteria for teenagers
  - **Environmental Factors:**
    - Apartment living adaptability (+8%)
    - Climate-specific considerations (+8%)
    - General adaptability bonus (5-10%)
  - **Special Abilities:**
    - Working capabilities (+3%)
    - Herding aptitude (+3%)
    - Hunting proficiency (+3%)
    - Tracking ability (+3%)
    - Agility potential (+2%)
     
---

### 4. 🧭 Visualization Analysis

To enhance explainability and user experience, the recommendation system now includes **lifestyle-based radar charts** that visualize each breed’s characteristics across six dimensions:

- Space Requirements  
- Exercise Needs  
- Grooming Level  
- Owner Experience  
- Health Considerations  
- Noise Behavior  

These visualizations help users intuitively understand how a breed aligns with their lifestyle preferences.

When users compare two breeds, both **radar charts** and **bar charts** are displayed side-by-side, offering a clear contrast across all factors—ideal for thoughtful decision-making.

All visual elements are powered by the **same internal dataset and scoring logic** used by the recommendation engine, ensuring consistency between what the model computes and what the user sees.

---

## 📊 Results
- **F1 Score:** The system achieved an overall **F1 Score of 88.70%,** across 124 breeds.
- **Few-Shot Ready:** The architecture is prepared to support new breeds or species with minimal training data, enabling easy scalability.

---

## 🧠 Technical Highlights

### 🦴 Model Backbone

1. **ConvNeXtV2 Base:**
   - **What It Does:** ConvNeXtV2 Base serves as my chosen feature extraction backbone, representing an evolution of the original ConvNeXt architecture with significant improvements. It introduces:
     - Global Response Normalization (GRN) for enhanced feature calibration
     - Fully Convolutional Masked AutoEncoder (FCMAE) for self-supervised learning
     - Optimized sparse convolution processing
   - **Why It Matters:** The transition from ConvNeXt to ConvNeXtV2 represents a significant leap in architectural design. ConvNeXtV2 enhances the pure convolutional approach with innovative normalization and self-supervised learning techniques. Imagine it as an advanced image processing pipeline where each component is specifically designed to enhance different aspects of the feature learning process - from basic patterns to complex structures.
   
   - **Key Features:**
     - Implements Global Response Normalization (GRN) to enhance inter-channel feature competition and prevent feature collapse
     - Utilizes sparse convolution during pre-training for efficient processing of masked regions
     - Provides enhanced feature extraction through the FCMAE framework
     - Employs adaptive feature calibration through learnable parameters in GRN
     - Creates an ideal foundation for computer vision tasks

2. **Multi-Level Attention Architecture:**
   - **Innovation in Design:** I developed a multi-level feature processing approach that processes features at different abstraction levels:
     - **Lower-level:** Utilizes ConvNeXtV2's built-in GRN to process and normalize features across channels
     - **Higher-level:** Implements an additional Multi-Head Attention layer near the output to capture complex feature relationships
   - **Technical Implementation:**
     - **Base Level:** GRN processes features through three key steps:
       - Global feature aggregation using L2-norm
       - Feature normalization through divisive normalization
       - Feature calibration with learnable parameters
     - **High Level:** Custom Multi-Head Attention with 8 attention heads
       - Each head specializes in different aspects of breed characteristics
       - Enables parallel processing of multiple feature relationships
       - Efficiently implemented using torch.einsum for attention calculations
     - Dynamic feature aggregation ensuring balanced feature integration
     - Seamless integration with ConvNeXtV2 architecture
   - **Why It's Effective:** This innovative architecture enhances the model's capabilities by:
     - Processing features hierarchically: from basic visual elements to complex breed-specific patterns
     - Enabling comprehensive feature analysis through both GRN and attention mechanisms
     - Lower level GRN ensures feature diversity while higher level attention focuses on breed-specific feature combinations
     - Creating a more robust and adaptable feature extraction pipeline that can better distinguish subtle differences between similar breeds
     - Maintaining computational efficiency while significantly increasing model expressiveness through strategic feature processing

3. **Morphological Feature Analysis System**

- Inspired by how humans observe—from overall form down to fine details—this system breaks down morphological features in a structured way to mimic expert-like visual understanding.

- **How It Works:**
  The system integrates five dedicated analyzers:
  - **Body Proportion:** Captures overall size and structure
  - **Head Shape:** Focuses on facial and cranial traits
  - **Tail Structure:** Detects tail shape and positioning
  - **Fur Texture:** Analyzes coat patterns, length, and density
  - **Color Distribution:** Identifies distinctive markings

- **Behind the Scenes:**
  - Dynamically transforms feature maps for better spatial representation
  - Uses 8-head Multi-Head Attention to model intra-feature relationships
  - Applies adaptive pooling and residual connections for consistent and stable learning
  - Combines outputs hierarchically for robust feature fusion

- **Why It Matters:**
  This biomimetic approach helps the model move beyond surface-level patterns. By modeling how humans differentiate breeds—especially those with subtle differences, it creates more informative and interpretable feature embeddings without sacrificing computational efficiency.

> 💡 True recognition comes not from memorizing isolated features, but from understanding how those features interact and form meaningful structures.

4. **Prototype Networks (Reserved for Future Development):**
  - **Definition:** A prototype represents the central feature for a specific breed. It is calculated as the average embedding vector for all training samples of that breed.
  
  - **Current Status:** While fully implemented in the codebase, this feature is currently inactive but maintained for future scalability purposes.
  
  - **Why It's Important:** When activated, Prototype Networks will enhance scalability through:
    - Adding a new breed only requires computing its prototype, avoiding full model retraining
    - Works effectively with limited data, supporting **Few-Shot Learning** for new species or rare breeds
    
  - **Future Implementation:** The system preserves the prototype network infrastructure while currently using traditional classification methods. This design choice maintains extensibility for:
    - Rapid integration of new breeds through prototype computation
    - Support for scenarios with limited training data
    - Easy expansion to additional animal classifications
    
  - **Real-World Applications:** Upon activation, this feature will enable quick adaptation for classifying additional species like cats or birds by simply adding their prototypes, demonstrating the system's future adaptability potential.

---

### 📈 Training and Optimization

1. **Data Augmentation:**
   - **Advanced Training Pipeline:** 
     - Implements a carefully balanced combination of multiple augmentation techniques for optimal training
     - Uses RandomResizedCrop with controlled scale (0.3-1.0) and aspect ratio (0.85-1.15) to maintain proper breed proportions
     - Applies minimal color adjustments (≤10% variation) to prevent over-reliance on color features
   
   - **Multi-Strategy Approach:**
     - Combines RandAugment (magnitude: 7) with AutoAugment for comprehensive but controlled image variations
     - Uses moderate RandomErasing (2-15% area) to improve robustness against partial occlusions
     - Maintains breed-specific characteristics while introducing sufficient variability for generalization

   - **Key Insight:** Through experimentation, I discovered that excessive color augmentation led to misclassifications between breeds with similar colors but different physical characteristics (e.g., golden-colored Dachshunds being misclassified as Golden Retrievers). By reducing color jittering and focusing on structural features, the model achieved more reliable breed identification based on fundamental morphological traits rather than color patterns.

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
     - **Core Mechanism:** Implements a sophisticated three-phase learning rate strategy:
       1. Warmup Phase (35% of training): Learning rate progressively scales from initial_lr/10 to initial_lr*4, following a carefully planned trajectory that matches the unfreezing schedule. This   
       extended warmup period (35% vs traditional 20%) allows the model to adapt to higher learning rates while most layers remain frozen, creating a stable foundation for subsequent training phases.
       2. Peak Performance Phase: Maintains higher learning rates for optimal learning
       3. Fine-tuning Phase (65% of training): Decreases using cosine annealing
     - **Technical Details:**
       - pct_start=0.35: Optimized warmup period allocation
       - div_factor=10: Controls initial learning rate scaling
       - final_div_factor=150: Ensures effective final fine-tuning
     - **Why It's Effective:** Combines fast convergence with robust training stability:
       - Prevents early training instability through careful warmup
       - Higher learning rates act as implicit regularization
       - Gradual cooldown allows precise parameter optimization
       - Synergizes with progressive unfreezing strategy

    - **Progressive Unfreezing:**
  I adopted a five-stage unfreezing strategy to gradually release the ConvNeXtV2 backbone, giving the model time to adapt from high-level features to low-level patterns.

      - **Stage Plan:**
        - **Epoch 10:** Unfreeze last 2 layers
        - **Epoch 18:** Expand to 4 layers
        - **Epoch 27:** Advance to 6 layers
        - **Epoch 37:** Deepen to 8 layers
        - **Epoch 50:** Unfreeze full backbone

    - **Why I Did This:**
    Treating it like teaching—start from summaries, then go into details. I found that gradual unfreezing gave the model more stability and led to better learning of subtle features. While full unfreezing early on gives more flexibility, it also uses more memory and risks overfitting too soon. This staged approach gave me better balance and control.


4. **Mixed Precision Training:**
   - **Efficiency:** Uses lower precision (16-bit floats) for most calculations to save memory and speed up training, while retaining 32-bit precision for critical operations.
   - **Implementation:** Combines **GradScaler** and **autocast** in PyTorch for seamless integration.
   - **Impact:** Enables training on larger datasets and models without requiring high-end GPUs.

5. **ICARL and Knowledge Distillation Enhancement:**

   - To improve recognition accuracy for certain challenging breeds like Havanese and Toy Poodle, I explored a hybrid approach combining ICARL with knowledge distillation. The idea was to help the model learn new patterns without forgetting old ones, while also letting a stronger "teacher" guide the learning process.

      - Inspired by how human teachers adapt their instruction, I adjusted the distillation temperature across training—starting with precise focus and gradually shifting to broader generalization. I also added lightweight protections to prevent key features from being overwritten, especially for similar-looking breeds.

      - This helped the model become more consistent when recognizing small or less distinctive breeds, and offered better stability across different poses and environments. While the full mechanism is still evolving, the improvements were clear in both testing and real-world use.
            
---

### 🔎 Advanced Dog Detection System
- **YOLOv8:** Integrated during the **deployment phase** for multi-dog detection in a single image. YOLO was not used during the model training process but was added later to enhance the system's ability to handle real-world use cases where multiple dogs may appear in a single image.
  - **Key Parameters:**
    - **`conf_threshold`:** Filters weak predictions to retain only confident detections.
    - **`iou_threshold`:** Removes overlapping bounding boxes to ensure clean and accurate detections.
  - **Why YOLO?:** Real-time object detection capabilities make it ideal for deployment scenarios where speed and efficiency are critical. This ensures that users can receive accurate results even with complex images featuring multiple dogs.

- **Biomimetic Processing Pipeline:** Inspired by human visual cognition patterns, the system implements a sophisticated post-detection processing pipeline:
  - **Intelligent Frame Adjustment:** Similar to how a professional photographer frames their shots, the system dynamically adjusts detection frames based on:
    - Dog's pose (standing, sitting)
    - Spatial relationships (overlapping dogs)
    - Position in frame (edge compensation)
    - Size variations (abnormal size handling)
  
  - **Enhanced Preprocessing:** Implements specialized image processing techniques that mimic human visual attention:
    - Maintains aspect ratios for standing poses to preserve natural proportions
    - Provides additional context for overlapping dogs
    - Centers subjects optimally within the frame
    - Uses high-quality resampling for detail preservation
---

## 🌐 Model Deployment
The model is deployed on **Hugging Face Space**, providing users with an intuitive interface for:
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
3. **Interactive Feedback:** Incorporate user feedback to refine detection accuracy and recommendations dynamically.
4. **Recommendation based on user description:** Using SBERT or other NLP models to analyze user needs and recommend suitable breeds tailored to their requirements.

---

## 📚 Acknowledgments and References
- [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/data)
- [How does the brain solve visual object recognition](https://pmc.ncbi.nlm.nih.gov/articles/PMC3306444/)
- [A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545)
- [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/pdf/2301.00808)
- [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/pdf/1909.13719)
- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/pdf/2104.00298)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [Part-based R-CNNs for Fine-grained Category Detection](https://arxiv.org/pdf/1407.3867)
- [Learning Subject-Aware Cropping by Outpainting Professional Photos](https://arxiv.org/pdf/2312.12080)
- [Prototypical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175)
- [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709)
- [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002)
- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186)
- [A disciplined approach to neural network hyper-parameters: Part 1 - learning rate, batch size, momentum, and weight decay](https://arxiv.org/pdf/1803.09820)
- [mixup: BEYOND EMPIRICAL RISK MINIMIZATION](https://arxiv.org/pdf/1710.09412)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531)
- [American Kennel Club](https://www.akc.org/)
  
---

## 📜 License
![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-lightgray)

© 2025 Eric Chung. This project is licensed under the Apache License 2.0, a permissive open source license that enables broad usage while ensuring proper attribution to the original author.

For detailed terms and conditions, please refer to the [LICENSE](./LICENSE.md) file.
