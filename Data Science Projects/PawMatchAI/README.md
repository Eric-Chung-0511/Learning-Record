# 🐾 PawMatchAI 🐾

## ✨ Project Overview
**PawMatchAI** is an advanced dog breed classification system that identifies **124 dog breeds**, built on the **Stanford Dog Dataset** with **21,000+ images**, and extends to include four additional popular breeds: **Shiba Inu**, **Dachshund**, **Bichon Frise**, and **Havanese**.

As an extension of its core classification capabilities, PawMatchAI includes a breed recommendation system and breed comparison to provide personalized breed suggestions based on user preferences. This makes the system not only accurate in identifying breeds but also versatile in helping users find their perfect canine companion.

---

## 📊 Project Impact
**Currently featured in Hugging Face Spaces of the Week, this project has achieved:**

> ![Visits](https://img.shields.io/badge/Total%20Visits-22k+-blue)
![Model Runs](https://img.shields.io/badge/Model%20Runs-10k+-green)
![Weekly Users](https://img.shields.io/badge/Weekly%20Users-6k+-orange)


## 🎯 Key Features

### 1. 🔍 **Breed Detection**
- **Photo Identification:** Upload a photo, and the model identifies the dog's breed from 124 possible options.
  - **Recognition Process:**
    - **Initial Detection:** **YOLOv8** model first detects and focuses specifically on dog subjects within the uploaded image
    - **Breed Classification:** Our specialized model analyzes the detected dog and processes breed identification
    - **Confidence-Based Results:** System provides breed information based on confidence levels:
      - High Confidence (≥45%): Directly presents the identified breed with detailed information
      - Medium Confidence (20-45%): Offers top 3 most probable breed matches for consideration
      - Low Confidence (<20%): Indicates the dog's breed may not be included in our current dataset
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

## 📊 Results
- **F1 Score:** The system achieved an overall **F1 Score of 90.95%,** across 124 breeds.
- **Few-Shot Ready:** The architecture is prepared to support new breeds or species with minimal training data, enabling easy scalability.

---

## 🧠 Technical Highlights

### 🦴 Model Backbone

1. **ConvNeXt Base:**
   - **What It Does:** ConvNeXt Base functions as my chosen feature extraction backbone, representing a fascinating reverse-engineered approach that brings Transformer's architectural benefits into CNN design. It modernizes traditional CNN by incorporating key Transformer principles like:
   - Increased channel dimensions and larger kernel sizes for better feature capture
   - Simplified architecture with fewer unique components compared to traditional CNNs
   - Layer normalization and GELU activation, inspired by Transformer designs

   - **Why It Matters:** Observing how Vision Transformers (ViT) have dominated recent research, I found it intriguing that ConvNeXt Base demonstrates how CNN can be modernized by incorporating Transformer-inspired designs. This sparked my idea - since ConvNeXt already successfully adapts Transformer concepts into CNN, why not complete the circle by adding actual Transformer components? This led me to create a hybrid architecture combining it with Multi-Head Attention.
   
   - **Key Features:**
     - Adopts Transformer's design principles while maintaining CNN's computational efficiency
     - Forms the perfect foundation for my planned attention mechanism integration
     - Provides robust feature extraction for the breed classification task

2. **Multi-Head Attention:**
   - **How It Works:** Building upon ConvNeXt's transformer-inspired nature, I integrated Multi-Head Attention to further enhance the model's capabilities. This mechanism divides the extracted features into multiple "heads," each focusing on specific parts of the image. These heads work independently to analyze different regions, such as the dog's face, ears, or body.
   
   - **Insight:** I implemented this mechanism efficiently using **torch.einsum**, ensuring optimal computation of attention weights while keeping memory usage and processing speed in check. The choice of torch.einsum was crucial for maintaining the model's overall efficiency.
   
   - **Why It's Effective:** This integration enhances the model's ability to:
     - Focus on critical details, such as unique facial patterns
     - Capture dependencies between different image regions, improving classification accuracy
     - Complement ConvNeXt's local feature processing with global relationship modeling

3. **Prototype Networks (Reserved for Future Development):**
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
     - **Core Mechanism:** Implements a sophisticated three-phase learning rate strategy:
       1. Warmup Phase (30% of training): Learning rate progressively scales from initial_lr/10 to initial_lr*6, following a carefully planned trajectory that matches the unfreezing schedule. This   
       extended warmup period (30% vs traditional 20%) allows the model to adapt to higher learning rates while most layers remain frozen, creating a stable foundation for subsequent training phases.
       2. Peak Performance Phase: Maintains higher learning rates for optimal learning
       3. Fine-tuning Phase (70% of training): Decreases using cosine annealing
     - **Technical Details:**
       - pct_start=0.25: Optimized warmup period allocation
       - div_factor=8: Controls initial learning rate scaling
       - final_div_factor=50: Ensures effective final fine-tuning
     - **Why It's Effective:** Combines fast convergence with robust training stability:
       - Prevents early training instability through careful warmup
       - Higher learning rates act as implicit regularization
       - Gradual cooldown allows precise parameter optimization
       - Synergizes with progressive unfreezing strategy

   - **Progressive Unfreezing:**
     - **Implementation Strategy:** I implemented a five-stage unfreezing schedule to provide finer control over the training process:
       - **Stage 1 (Epoch 10 / epochs/15):** Unfreezes Last 2 Layers
       - **Details:**
         - Target: Last 2 layers for focused adaptation
         - Timing: Epoch 10 marks our starting point,  model starts with familiar high-level features
         - Purpose: Initiating high-level feature refinement
      
      - **Stage 2 (Epoch 20 / epochs/7.5):** Expands to 4 Layers
       - **Details:**
         - Target: Increase to 4 layers for broader learning
         - Timing: Strategic point at epoch 20
         - Purpose: Mid-level feature processing begins
      
      - **Stage 3 (Epoch 30 / epochs/5):** Advances to 6 Layers
       - **Details:**
         - Target: 6 layers now actively learning, unlocking deeper network potential
         - Timing: Calculated for epoch 30 sweet spot
         - Purpose: Accessing deeper network knowledge
      
      - **Stage 4 (Epoch 40 / epochs/3.75):** Deepens to 8 Layers
       - **Details:**
         - Target: 8 layers engaged in learning, fundamental features begin fine-tuning
         - Timing: Carefully placed at epoch 40
         - Purpose: Deep feature refinement phase
      
      - **Final Stage (Epoch 50 / epochs/3):** Complete Backbone Release
       - **Details:**
         - Target: Complete backbone unleashed, makes entire model joins the optimization
         - Timing: Final stage at epoch 50
         - Purpose: Comprehensive model refinement
        
      - **Why I Gradually Unfreeze - A Personal Approach:**
        - **The Learning Journey:**
           - Think of it like teaching a student - start with basics and go deeper
           - It's fascinating how each layer contributes uniquely to learning
           - Just like humans, models need time to grasp finer details
       
        - **My Practical Experience:**
           - Found that only unfreezing final layers is like learning just surface features
           - It's similar to only learning what something looks like without understanding why
           - Like teaching art but only focusing on outlines, missing all the subtle shading
       
        - **The Trade-offs I've Found:**
           - **Benefits of Early Complete Unfreezing:**
               - Model learns deeper, more nuanced features
               - Catches subtle patterns we might miss otherwise
               - Like giving the model full artistic freedom
           
           - **Real-world Considerations:**
               - It needs more GPU memory - like needing a bigger canvas
               - Training takes longer - but often worth the wait
               - Memory usage increases, but the results can be stunning

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
4. **Recommendation based on user description:** Using SBERT or other NLP models to analyze user needs and recommend suitable breeds tailored to their requirements.

---

## 📚 Acknowledgments and References
- [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/data)
- [A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545)
- [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/pdf/2301.00808)
- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/pdf/2104.00298)
- [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/pdf/1909.13719)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [Prototypical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175)
- [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709)
- [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002)
- [mixup: BEYOND EMPIRICAL RISK MINIMIZATION](https://arxiv.org/pdf/1710.09412)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084)
- [American Kennel Club](https://www.akc.org/)

---
