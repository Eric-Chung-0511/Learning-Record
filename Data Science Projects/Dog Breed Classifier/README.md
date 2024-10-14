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
### 🐾 Model Architecture:
* The model's base architecture is **EfficientNetV2(M)**, well-known for its efficiency and accuracy in feature extraction. By integrating **Multi-Head Attention**, the model can better focus on critical features of the input image, enhancing breed classification. This attention mechanism splits the feature space into multiple heads, allowing for the most relevant features to be extracted.
* **Prototype Networks** enable efficient classification by storing prototypes for each class. New data points are classified based on their similarity to these prototypes. This approach makes the model scalable for future additions of new breeds or even new animal species, such as cats, with minimal training data required.
* **torch.einsum** is used to efficiently compute attention across multiple heads, optimizing the computation process for the attention layers.

### 📈 Training and Optimization:
* **Data Augmentation**: Applied extensive data augmentation techniques during training to enhance model generalization. Techniques included **RandomResizedCrop**, **RandomHorizontalFlip**, **RandAugment**, **ColorJitter**, and **RandomErasing**. These augmentations help the model handle variations in image size, lighting, and occlusions, making it more robust to real world data.

* **Contrastive Loss**: Applied to maximize the feature separation between different classes while minimizing the distance between samples of the same class. This ensures that breeds with similar features are correctly classified while avoiding confusion between different breeds.
* **Focal Loss**: Adjusts the weight of hard-to-classify samples, especially those from underrepresented classes, making the model more robust against class imbalance.
* **OneCycleLR**: This learning rate scheduler adjusts the learning rate dynamically during training, helping the model converge faster and preventing overfitting in the later stages of training.
* **Progressive Unfreezing**: Gradually unfroze layers of the **EfficientNetV2(M)** backbone during training. This technique starts with only the final layers being trainable and progressively unfreezes deeper layers at set intervals, helping to stabilize training and prevent catastrophic forgetting while fine-tuning the model.

### 🧪 Evaluation:
* The overall **F1 Score** across the 120 breeds was **81.22%**.The **EfficientNetV2(M)** backbone combined with **Multi-Head Attention** allowed the model to focus on critical features, while **Focal Loss and Contrastive Loss** helped address class imbalances and improve performance.
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

## 🚀 Model Deployment:
* The model is deployed on **Hugging Face Spaces**, allowing users to upload an image and receive the predicted dog breed. After classification, the system integrates with an **SQLite** database to provide brief information about the identified breed.
* In addition to returning breed information, the system offers **external links** for users to explore more comprehensive resources, giving them further insights into the breed, such as its history, care requirements, or notable facts.

## 🌐 Try it Yourself:
You can test the model directly on [Hugging Face](https://huggingface.co/spaces/DawnC/Dog_Breed_Classifier), where it’s live and ready to classify your pet images. Simply upload an image, and see how well the model can predict the pet's breed!

## 📚 Acknowledgments and References:
- [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/data)
- [EfficientNetV2: Smaller Models and Faster Traing](https://arxiv.org/pdf/2104.00298)
