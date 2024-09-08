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

## 🚀 Project Details:

### 🧠 Model Exploration and Selection:
**Model Testing with ResNet-50:**
- Utilizing the **ResNet-50 model** for its proven track record in image classification tasks, known for its ability to learn deep hierarchical features from images.
- ResNet-50 was chosen due to its balance between depth and computational efficiency, making it suitable for training on the Oxford Pet dataset while delivering strong classification performance.
- Transfer learning allowed us to fine-tune the pretrained ResNet-50 model on the Oxford Pet dataset, adapting its learned features to accurately classify 37 different pet breeds.
- I also tested **EfficientNet-B3**, which achieved an **88% F1 score**. However, **ResNet-50** outperformed it with a **92% F1 score**, making it the final model selection for this project.

### 🖥️ Deployment:
**Final Model Deployment:**
- The fine-tuned ResNet-50 model was deployed on Hugging Face Spaces, making it accessible for real time predictions.
- **Users can upload images of pets to receive predictions about their breed, showcasing the practical application of transfer learning and real time image classification.**

### 🔧 Technical Implementation:

#### Data Handling and Preprocessing:
- Pandas was used for managing and cleaning the dataset, while Matplotlib was utilized for visualizing image data distribution and model performance.
- The data preprocessing steps included resizing the images, normalizing pixel values, and augmenting the dataset to improve model generalization.
- These steps ensured that the model could handle diverse inputs and improve its robustness when predicting new images.

#### Transfer Learning:
- Transfer learning played a critical role in this project. By leveraging a pretrained ResNet-50 model, Its capability can quickly build an effective image classifier by fine-tuning the model on a smaller dataset, significantly reducing training time while improving accuracy.

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





