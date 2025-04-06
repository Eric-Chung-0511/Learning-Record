# 💫 Data Science Project Highlights

<a name="top"></a>

## 📋 Table of Contents
- [Data Analysis](#data-analysis)
- [Supervised Machine Learning](#supervised-machine-learning)
- [Unsupervised Machine Learning](#unsupervised-machine-learning)
- [Deep Learning](#deep-learning)
- [Natural Language Processing](#natural-language-processing)
- [Future Directions](#future-directions)
- [My Project Workflow](#my-project-workflow)

<h2 id="data-analysis">📊 Data Analysis</h2>

I started my journey in data science by getting hands-on with statistics and exploratory data analysis. Before jumping into any modeling, I’ve always believed it’s worth spending time just looking at the data — figuring out what’s missing, what looks odd, and where the patterns might be hiding.

In practice, I often spend more time cleaning and shaping the data than building the model itself. I've come to realize that a well-prepared dataset usually leads to better results, even with simpler algorithms.

One area I’ve invested time in is **feature engineering**. Whether it's scaling variables, reducing dimensionality, or crafting new interactions, I try to shape the data in a way that makes sense to both the model and the business context. Some common techniques I use include:

- **Scaling** (StandardScaler, MinMaxScaler) to bring features to a similar range  
- **PCA** when dimensionality is high and interpretability is less of a concern  
- **Interaction features** when variable combinations seem to hold predictive power  

Working with real world data also means dealing with **class imbalance**, especially in cases like fraud detection. For this, I’ve used techniques like:

- **SMOTE** to generate synthetic minority class samples  
- **ADASYN**, which focuses more on harder-to-learn examples  
These methods often make a noticeable difference, especially in improving recall for rare classes.

I also pay close attention to **outliers**, which can silently skew both training and evaluation. Depending on the situation, I usually go with:

- **IQR-based filtering**, to identify and isolate extreme values.This method typically defines outliers as values falling below Q1 - 1.5 × IQR or above Q3 + 1.5 × IQR, where IQR = Q3 - Q1.   
- **Quantile trimming**, which is more aggressive but sometimes necessary  
- **Log transforms**, especially for right-skewed features  
- **Median imputation**, to preserve rows while softening the impact of anomalies  

I don’t apply these methods blindly, I tend to explore different approaches, compare how they affect the data, and choose the one that fits best. There’s no perfect recipe, but being deliberate at this stage usually pays off later.


**[⇧ back to top ⇧](#top)**

<h2 id="supervised-machine-learning">⚙️ Supervised Machine Learning</h2>

I started with the basics models like logistic regression and decision trees, just to get a feel for how algorithms make decisions. Over time, I moved toward more complex techniques, not just to chase accuracy, but to understand how different models "see" the same data in different ways.

One thing I’ve learned is that no single model captures everything. I’ve worked with a range of classifiers, including:

- **Logistic Regression** – straightforward and easy to explain. I still use it when interpretability matters, or as a quick baseline to understand feature relationships.

- **Decision Trees** – fast to train and surprisingly effective on tabular data. I sometime start with them to explore how the model is splitting and whether the features are doing their job.

- **Support Vector Machines** – powerful when the data is clean and the classes are well-separated. Tuning the kernel and C values can be tricky, but the results are often worth it when you get them right.

- **Random Forest** – a solid all-rounder. I like using it when I want some level of interpretability (through feature importance) but also want the robustness of an ensemble.


In many projects, I’ve leaned heavily on **gradient boosting**, especially **XGBoost**. I like how boosting doesn’t try to get everything right at once — instead, it builds on the mistakes of previous iterations. That incremental mindset feels very intuitive to me, and it’s often delivered strong results even on noisy datasets.

For time series tasks, I’ve used models like **ARIMA**, **SARIMA**, and **SARIMAX**. In the [Walmart Sales Prediction](https://github.com/Eric-Chung-0511/Learning-Record/tree/main/Data%20Science%20Projects/Walmart%20Sales%20Prediction) project, those helped me forecast sales by accounting for seasonality and promotional events across different stores. It reminded me how important it is to think beyond just the numbers, context like holidays or local events can make a big difference.

When it comes to model tuning, I’ve tried a lot of approaches, but these two are the ones I return to most often:

- **Grid Search** – brute-force, but dependable. I use it when the parameter space is small and I want to try all combinations. It's simple to implement, but gets expensive quickly as dimensions grow.  
  _(Behind the scenes, it’s basically testing every possibility. For smaller models, that’s fine — but on larger ones, it becomes painfully slow.)_

- **Bayesian Optimization** – smarter and faster in many cases. It uses probability models (like Gaussian processes) to estimate where the best parameters might be, then focuses search in those areas.  
  _(It doesn’t test everything — it *learns* where to look.) I usually use libraries like [Optuna](https://optuna.org/) or [scikit-optimize](https://scikit-optimize.github.io/) for this. They tend to find good results with far fewer iterations._ 

The latter has been especially useful in fine-tuning complex models where I want smarter exploration rather than brute force.

At this stage, I don’t just look at accuracy anymore, I think about the tradeoffs, the model’s interpretability, and how well it fits the problem. Sometimes a simple model that aligns with the business goal is more valuable than a complicated one that no one trusts.


**[⇧ back to top ⇧](#top)**

<h2 id="unsupervised-machine-learning">🔍 Unsupervised Machine Learning</h2>

Unsupervised learning has always felt a bit like exploration work, looking for hidden patterns without knowing exactly what to expect. There’s something rewarding about seeing structure emerge from raw data, especially when the results challenge your assumptions.

In my [E-Commerce project](https://github.com/Eric-Chung-0511/Learning-Record/tree/main/Data%20Science%20Projects/E-Commerce), I used clustering to segment customers based on their purchase behavior. I applied **RFM analysis** (Recency, Frequency, Monetary), which helped me frame customer activity into measurable dimensions. Even before modeling, that alone gave some intuitive insights.

To group similar customers, I started with **K-Means**, paired with **PCA** to simplify the feature space. K-Means isn’t perfect, but I like its simplicity — the cluster centers are easy to interpret, which makes it easier to communicate results to non-technical stakeholders like marketing teams.

Later on, I also explored **DBSCAN**, which takes a very different approach. Instead of assigning every point to a cluster, it allows for "noise" — points that don’t belong anywhere. That turned out to be helpful for spotting customers with very atypical behavior, the kind that might otherwise skew averages or confuse the model.  

Both methods gave me something useful, but in different ways. To choose between them, I looked at **silhouette scores** and also just plotted the clusters. I’ve learned that visual intuition still plays a big role, sometimes charts tell you more than metrics alone.

More broadly, I’ve found that **dimensionality reduction** techniques like PCA or t-SNE aren’t just for preprocessing, they’re great for understanding. Visualizing high-dimensional data in 2D or 3D often reveals relationships you wouldn’t catch from raw tables or correlations alone.


**[⇧ back to top ⇧](#top)**

<h2 id="deep-learning">🤖 Deep Learning</h2>

Deep learning has been the area I’ve spent the most time on, especially within computer vision. What keeps me interested is how these models can learn layered representations, starting from textures and edges, then building up to more abstract structures — all from raw pixel data.

My main project in this area is [PawMatchAI](https://github.com/Eric-Chung-0511/Learning-Record/tree/main/Data%20Science%20Projects/PawMatchAI), a dog breed classifier trained on over 21,000 images across 124 breeds. It reaches an **F1 score of 88.7%**, and is live on Hugging Face Spaces with features like breed comparison and personalized recommendations. This project has also been featured in Hugging Face’s “Spaces of the Week.”

What sets this project apart is the model architecture. Instead of using a standard CNN, I built a hybrid system combining a modern backbone (ConvNeXtV2) with **multi-head attention** and a custom-designed **Morphological Feature Extractor**. The idea came from how humans often distinguish dog breeds — not just by color or size, but by shapes: ear structure, snout length, coat texture.

The Morphological Feature Extractor works alongside convolutional and attention modules to emphasize these structural details. It doesn't replace conventional vision features, but complements them. I found this especially helpful when the model had to tell apart visually similar breeds like Shelties and Collies, or different coat variants of the same breed.

To support learning, I applied advanced data augmentation techniques like **RandAugment**, **ColorJitter**, and **RandomErasing**, along with training strategies such as **progressive layer unfreezing**, **Contrastive Loss**, **Focal Loss** and **OneCycleLR** scheduling. Together, these helped the model generalize better without overfitting.

Working on PawMatchAI gave me hands-on experience across the full deep learning pipeline - from designing the architecture and training strategy, to optimizing inference and deploying it as a public-facing demo on Hugging Face Spaces. It wasn’t just about getting high accuracy, but about building something people could actually use and interact with.  

Eventually, I’d like to expand this into a more complete application, potentially as a mobile app with real time camera input and offline capabilities. But before that, I want to test it further, refine the model’s edge cases, and explore additional features.


**[⇧ back to top ⇧](#top)**

<h2 id="natural-language-processing">📝 Natural Language Processing</h2>

While computer vision remains my primary focus, my curiosity has led me to explore natural language processing through several projects. The parallels between how models process visual and textual information – both involving hierarchical feature extraction and context awareness – have given me a broader perspective on deep learning as a whole.

In my [SkimLit](https://github.com/Eric-Chung-0511/Learning-Record/tree/main/Data%20Science%20Projects/SkimLit) project, I fine-tuned a **RoBERTa** model to classify and summarize sections of biomedical research papers. This application addresses a real world need: helping researchers quickly extract relevant information from vast amounts of scientific literature. Working with transformer architectures gave me appreciation for how attention mechanisms revolutionized NLP by capturing long-range dependencies in text.

I've also explored more traditional NLP approaches in my [MBTI Prediction](https://github.com/Eric-Chung-0511/Learning-Record/tree/main/Data%20Science%20Projects/MBTI%20Prediction) project, where I combined feature engineering methods like **TF-IDF** with gradient boosting models to predict personality types from text samples. This project highlighted the importance of thoughtful text preprocessing and feature extraction, even when working with sophisticated algorithms.

These explorations into NLP have broadened my machine learning toolkit and given me valuable experience with text data processing – skills that complement my computer vision expertise and open the door to potential multimodal applications in the future.

**[⇧ back to top ⇧](#top)**

<h2 id="future-directions">🚀 Future Directions</h2>

As I continue building my skill set in AI and machine learning, there are a few areas I'm especially drawn to — not just because they’re trending, but because they align with how I think about learning, adaptability, and cross-domain understanding.

- **Transfer Learning**  
  This has already changed how we approach deep learning problems. Instead of training from scratch, we build on existing knowledge. I'm particularly interested in more advanced fine-tuning methods that help large models adapt to smaller, specialized domains. The challenge is making them flexible without overwriting what’s already valuable — a balance that’s surprisingly tricky.

- **Large Language Models (LLMs)**  
  I’ve been following the progression from GPT-4 to Llama 4 with a lot of interest. Beyond generation tasks, I’m curious about how LLMs can be tailored for specific domains, or integrated into pipelines that involve reasoning, retrieval, or symbolic logic. There’s real potential here, but also important questions around alignment, efficiency, and trust.

- **Multimodal Learning**  
  This is the area I’m most excited about. Combining different types of input — such as text, images, and potentially audio. Unlocks possibilities that go well beyond what single-modality models can achieve. I’d love to build tools that can read a paper and not only understand the text, but also interpret the accompanying charts, tables, and images. It feels like a meaningful step toward making complex knowledge more accessible and better connected.


**[⇧ back to top ⇧](#top)**

<h2 id="my-project-workflow">🔄 My Project Workflow</h2>

Across different projects, I’ve gradually shaped a [data science workflow](https://github.com/Eric-Chung-0511/Learning-Record/blob/main/Data%20Science%20Projects/Data%20Science%20Project%20WorkFlow.md) that helps me stay organized — from framing the problem, to collecting and cleaning data, to building and evaluating models.

I like having a structure, but I also know things change as insights emerge. This workflow keeps my process consistent without getting in the way of iteration. By breaking big problems into smaller steps and setting clear goals along the way, I’ve found it easier to stay focused while still leaving room to explore and improve.


**[⇧ back to top ⇧](#top)**
