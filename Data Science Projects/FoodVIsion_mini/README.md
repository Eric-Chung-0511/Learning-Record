# 🍕 FoodVision Mini 🍣

## ✨ Project Overview:
* The goal of this project is to develop a model capable of accurately recognizing three types of food: steak, pizza, and sushi, using a dataset of training images. **FoodVision Mini** illustrates the efficiency of leveraging pretrained models through transfer learning, allowing us to build an effective image classification system without needing to train from scratch.

## ⚙️ Skills Used:
### 🛠️ Key Technologies and Tools:
* PyTorch
* Python
* Matplotlib
* Pandas
* Gradio
* Transfer Learning

## 🚀 Project Details:
### 🧠 Model Exploration and Selection:
* **Model Testing with EfficientNetB2 and ViT**:
  - We began by experimenting with two cutting-edge models: **EfficientNetB2** and **Vision Transformer (ViT)**. These models are known for their strong performance in image classification tasks.
  - **EfficientNetB2** offers a balanced approach, providing high accuracy with relatively fast inference times. On the other hand, **ViT** is known for its ability to handle larger images and capture long-range dependencies within them.
  - After extensive evaluation, **EfficientNetB2** was chosen as the final model due to its superior speed in predictions. In machine learning operations (MLOps), balancing speed and accuracy is crucial, especially when deploying models in real-world applications. **When accuracy differences are minimal, opting for a faster model can significantly improve the user experience and reduce computational costs.**

### 🖥️ Deployment:
* **Final Model Deployment**:
  - The selected **EfficientNetB2** model was fine-tuned on our specific dataset to enhance its accuracy in identifying steak, pizza, and sushi.
  - Following this, the model was deployed on **Hugging Face**, providing an accessible platform for real-time predictions. This deployment demonstrates how a well-constructed machine learning model can be integrated into practical applications, enabling users to upload their images and receive instant predictions.

### 🔧 Technical Implementation:
* **Data Handling and Preprocessing**:
  - **Pandas** was utilized for data manipulation, ensuring the dataset was properly formatted and ready for model training. **Matplotlib** was employed to visualize the data, helping to understand the distribution and characteristics of the images before feeding them into the model.
  - The data preprocessing phase included resizing images, normalizing pixel values, and augmenting the dataset to increase its diversity, which is vital for improving the model's robustness.

* **Transfer Learning**:
  - Transfer learning was a key component in this project, allowing us to take advantage of models pretrained on large datasets and fine-tune them for our specific task. This not only reduced the time required for training but also improved the model's accuracy by utilizing the rich feature representations learned by the pretrained models.

* **Interactive Interface with Gradio**:
  - To make the model accessible and user-friendly, utilize **Gradio** to create an interactive web interface. This allows users to easily upload images and get predictions in real-time, showcasing the model's capabilities and potential for practical applications.

## 🎯 Conclusion:
* **FoodVision Mini** is a testament to the power and versatility of transfer learning in modern machine learning projects. By leveraging a combination of robust tools and techniques, the project was able to deliver a high-performing image classification model in a relatively short period of time.
  
* The decision to choose **EfficientNetB2** over **ViT** underscores the importance of considering both speed and accuracy in model selection. In MLOps, this tradeoff is essential for creating models that are not only accurate but also efficient and scalable.

* The project also highlights the importance of deployment and accessibility. By deploying the model on **Hugging Face** and integrating it with **Gradio**, so that the model is not just a theoretical success but also a practical tool that can be used in real-world scenarios.

## 🌐 Try it Yourself:
* You can test the model directly on [Hugging Face](https://huggingface.co/spaces/DawnC/Foodvision_mini), where it’s live and ready to classify your food images. Simply upload an image, and see how well the model can predict whether it’s steak, pizza, or sushi!

## 📚 Acknowledgments and References:
* [Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/09_pytorch_model_deployment/)

* [TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929)
