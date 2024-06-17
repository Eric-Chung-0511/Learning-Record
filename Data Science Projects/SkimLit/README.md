# 📄 SkimLit 🔍

## ✨ Project Overview:
* The primary purpose of this project is to develop a deep learning model capable of automatically classifying and summarizing biomedical literature. This helps researchers quickly access relevant information from vast amounts of scientific papers. 

## ⚙️ Skills Used:
### 🐍 Python Programming and Data Handling Skills:
* Pandas
* Numpy
* Matplotlib
* Scikit-learn (Tfidf, MultinomialNB, One Hot Encoding, Label Encoding)


### 🧠 Deep Learning:
* TensorFlow (TextVectorization, Embedding, EarlyStopping, ReduceLROnPlateau)
* TensorFlow Hub (Universal Sentence Encoder)
* Conv1D
* Bidirectional LSTM
* Functional API
* tf.data.Dataset

## 🤖 Skills Detail:
### 👓 Data Handling and Text Preprocessing:
* **Text Preprocessing**: Implemented a function `preprocess_text_with_line_numbers` to process abstracts by extracting lines, assigning labels, and creating a structured format of the text, crucial for subsequent text analysis and classification.

* **One Hot Encoding and Label Encoding**: Utilized one hot encoding for deep learning models and label encoding for evaluation metrics like F1 score, ensuring compatibility and accurate performance measurement.

### ⚒️ Data Preprocessing and Encoding:
* **Text Vectorization**: Employed TensorFlow's `TextVectorization` layer to convert text into numerical representations, which are then used as inputs for deep learning models. This layer helps in standardizing and tokenizing the input text.

* **Embedding Layer**: Created embedding layers to transform tokens into dense vectors, capturing the semantic meaning of words. This step is fundamental for models that leverage word embeddings for improved contextual understanding.

* **tf.data API**: Utilized `tf.data.Dataset` for efficient data pipeline creation, enabling batching, shuffling, and prefetching to optimize the training process.

### 🧬 Model Development and Evaluation:
* **Model 0: Baseline with TF-IDF and MultinomialNB**
  * Built a pipeline using TfidfVectorizer and MultinomialNB, achieving an F1 score of **70%**.

* **Model 1: Conv1D with Token Embeddings**
  * Developed a Conv1D model with token embeddings using the Functional API, achieving an F1 score of **81%**.

* **Model 2: Feature Extraction with Pretrained Token Embeddings**
  * Utilized TensorFlow Hub's Universal Sentence Encoder (USE4) for feature extraction, with an F1 score of **75%**.

* **Model 3: Conv1D with Character Embeddings**
  * Created a Conv1D model with character embeddings, leveraging GlobalMaxPooling to prevent overfitting, resulting in an F1 score of **68%**.

* **Model 4: Combining Pretrained Token Embeddings and Character Embeddings**
  * Developed a hybrid model combining token and character embeddings:
    1. **Token Inputs Model**: Processed token inputs through a pretrained embedding layer.
    2. **Character Inputs Model**: Tokenized character inputs and processed through a Bidirectional LSTM.
    3. **Concatenation**: Merged token and character embeddings, followed by dropout and dense layers, achieving an F1 score of **74%**.

* **Model 5: Transfer Learning with Pretrained Token, Character, and Positional Embeddings**
  * Extended Model 4 by adding positional embeddings:
    1. **Token Inputs**: Pretrained token embeddings with dense layers.
    2. **Character Inputs**: Character embeddings processed through Bidirectional LSTM.
    3. **Line Number and Total Lines Models**: Positional embeddings for sentence and abstract structure.
    4. **Hybrid Embedding and Output Layer**: Combined all embeddings and processed through dense and dropout layers, achieving an F1 score of **84%**.

### 🔍 Fine-Tuning:
* **Fine-tuned Model 5** by unfreezing all layers and retraining, which improved the F1 score to **87%**.

### 🧭 Model Evaluation:
* Evaluated models using metrics like accuracy, precision, recall, and F1 score. Conducted detailed analysis through precision-recall curves to optimize decision thresholds and balance recall and precision.

## 🎯 Conclusion:
* The SkimLit project demonstrated the effectiveness of combining various text processing and deep learning techniques to enhance the extraction of relevant information from medical papers. Fine-tuning and transfer learning improved the model's performance, making it a robust tool for automatic literature review.

* This project highlights the importance of preprocessing in handling text data and the potential of deep learning models in understanding and classifying complex text data, particularly in the context of medical literature. The final model achieved a high F1 score, demonstrating its capability to accurately extract and classify different sections of medical abstracts.

* ## 📚 Acknowledgments and References:
* I am grateful to the developers and researchers whose work has significantly influenced this project. Below are the resources that have been instrumental:

* **Udemy Course:** [TensorFlow for Deep Learning Bootcamp](https://www.udemy.com/course/tensorflow-developer-certificate-machine-learning-zero-to-mastery/?couponCode=KEEPLEARNING)

* [PubMed 200k RCT: a Dataset for Sequenctial Sentence Classification in Medical Abstracts](https://arxiv.org/pdf/1710.06071)

* [Neural Networks for Joint Sentence Classification in Medical Paper Abstracts](https://arxiv.org/pdf/1612.05251)
