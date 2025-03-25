# Disaster Tweet Classification Project

## Overview
The **Disaster Tweet Classification** project is a **Natural Language Processing (NLP) model** that predicts whether a given tweet is related to a natural disaster or not. The model is trained on a dataset containing real and non-disaster tweets, enabling it to classify tweets effectively using deep learning techniques.

## Features
- **Binary Classification**: Predicts whether a tweet is about a disaster (1) or not (0).
- **Pretrained Word Embeddings**: Utilizes models like **Word2Vec, GloVe, TfidfVectorizer**.
- **Deep Learning-Based NLP**: Built using **TensorFlow and Keras**.
- **Data Preprocessing**: Includes tokenization, stopword removal, and padding.
- **Evaluation Metrics**: Uses accuracy, precision, recall, and F1-score.
- **Deployment Ready**: Can be integrated into real-world applications.

## Dataset
The dataset consists of labeled tweets, categorized as **disaster-related** or **non-disaster-related**. Preprocessing techniques such as **lemmatization, stemming, and stopword removal** are applied to improve classification accuracy.

## Model Architecture
The model is built using **TensorFlow** and consists of:
- **Embedding Layer**: Converts words into vector representations.
- **Recurrent Layers (LSTM/GRU)**: Captures sequential dependencies in tweets.
- **Dense Layers**: For final classification.
- **Sigmoid Activation**: Outputs probability scores for disaster classification.

## Training & Evaluation
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Training Platform**: Google Colab with GPU acceleration


The model achieves **high accuracy** in distinguishing disaster-related tweets from non-disaster tweets. Performance evaluation includes:
- **Confusion Matrix Analysis**
  ![image](https://github.com/user-attachments/assets/f9c212b6-649b-476e-afd9-2a689184f9e8)

- **Example Tweet Predictions**
  ![image](https://github.com/user-attachments/assets/ce24c1b4-fad0-4adb-87bd-ef997b068632)


## Future Improvements
- **Hyperparameter Tuning**: Optimize model performance.
- **BERT-based Model**: Improve classification with Transformer models.
- **Deployment as an API**: Serve predictions via a web API.
- **Multi-Class Classification**: Expand to classify different types of disasters.


---


