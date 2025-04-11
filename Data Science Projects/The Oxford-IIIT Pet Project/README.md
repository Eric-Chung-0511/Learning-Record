# 🐕🐈 🐾  Oxford Pet Classifier  👀

## ✨ Project Overview:
The goal of this project is to develop a model capable of accurately recognizing 37 different pet breeds, using the Oxford-IIIT Pet Dataset. The dataset contains 7,349 images used for training. 

## 🛠️ Key Technologies and Tools:
- PyTorch
- Python
- Matplotlib
- Pandas
- Gradio
- Transfer Learning (ResNet-50, EfficientNetB3)

---


## ⚠️ Important Notice
**Due to a current GitHub issue with rendering Jupyter notebooks (missing 'state' key in metadata.widgets), the notebook code and outputs may not display properly in this repository.**

For the complete notebook with all outputs and visualizations, please access the project via this Google Colab link:  
👉 [View Complete Project in Google Colab](https://colab.research.google.com/drive/1omaPJG1CTiJTNMuRsXyJ3YEKk56QXabC?usp=sharing)

The issue is being tracked by GitHub and will be resolved as soon as possible. Thank you for your understanding!

---

## 🚀 Project Details:

### 🧠 Model Exploration and Selection:
## **Model Testing with ResNet-50:**
- Utilizing the **ResNet-50 model** for its proven track record in image classification tasks, known for its ability to learn deep hierarchical features from images.
- **ResNet-50** addresses a common issue in deep networks: when networks become very deep, gradients can either explode or vanish during backpropagation, making training difficult. To solve this, ResNet-50 introduces a key innovation called the **Residual Block, which allows the network to "skip" certain layers, preserving information and stabilizing training in deep architectures.**
- ResNet-50 was chosen due to its balance between depth and computational efficiency, making it suitable for training on the Oxford Pet dataset while delivering strong classification performance.
- Transfer learning allowed us to fine-tune the pretrained ResNet-50 model on the Oxford Pet dataset, adapting its learned features to accurately classify 37 different pet breeds.

### 🖥️ Deployment:
**Final Model Deployment:**
- The fine-tuned ResNet-50 model was deployed on Hugging Face Spaces, making it accessible for real time predictions.
- **Users can upload images of pets to receive predictions about their breed, showcasing the practical application of transfer learning and real time image classification.**

### ⚙️  Technical Implementation:

#### Data Handling and Preprocessing:
- Pandas was used for managing and cleaning the dataset, while Matplotlib was utilized for visualizing image data distribution and model performance.
- The data preprocessing steps included resizing the images, normalizing pixel values, and augmenting the dataset to improve model generalization.
- These steps ensured that the model could handle diverse inputs and improve its robustness when predicting new images.

#### Transfer Learning:
- Transfer learning played a critical role in this project. By leveraging a pretrained ResNet-50 model, its capability can quickly build an effective image classifier by fine-tuning the model on a smaller dataset, significantly reducing training time while improving accuracy.
  
- **To fine-tune the model, I froze most of the ResNet-50 layers except for the last 4 layers (layer4 is the last Conv2d layer)**. This allows the model to retain its general image features while focusing on learning the specific features for pet classification. The following code was used to implement this fine-tuning strategy:

- In addition, **I adjusted the learning rate and optimizer to further improve the model's performance.** I used the **AdamW** optimizer with weight decay for regularization and the **OneCycleLR** scheduler to dynamically adjust the learning rate during training:

- This fine-tuning approach and learning rate adjustments helped to optimize the model's learning process and achieve a higher accuracy on the Oxford Pet dataset.


#### Interactive Interface with Gradio:
- Gradio was integrated to provide an intuitive and user friendly interface where users can upload images and instantly get breed predictions.
- This interface highlights the model's potential for real world applications, offering an accessible way to interact with machine learning models without needing to write code.

## 🎯 Conclusion:
**Oxford Pet Classifier** showcases the power of transfer learning in reducing the time and computational resources required for training deep learning models.
- The decision to use ResNet-50 underscores the importance of model selection in balancing performance and efficiency, particularly for real world applications.
- The deployment on Hugging Face Spaces with a Gradio interface makes the model accessible to the public, demonstrating how deep learning models can be utilized in real world scenarios.

## 🌐 Try it Yourself:
You can test the model directly on [Hugging Face](https://huggingface.co/spaces/DawnC/OxFord_Pet_Project), where it’s live and ready to classify your pet images. Simply upload an image, and see how well the model can predict the pet's breed!

## 📚 Acknowledgments and References:
- [Oxford-IIIT Pet Dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html#oxfordiiitpet)
- [ResNet-50 Model document](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)

## 📄 Viewing Jupyter Notebooks
* Sometimes there's bug on GitHub, if you encounter any problems displaying the Jupyter Notebooks directly on GitHub, you can view this project with the following link:
  [Oxford Pet classifier](https://nbviewer.org/github/Eric-Chung-0511/Learning-Record/blob/main/Data%20Science%20Projects/The%20Oxford-IIIT%20Pet%20Project/OxfordIIITPet_Project__Eric.ipynb)

  Thank you for your understanding!😊





