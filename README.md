# Fake News Classifier using LSTM

This repository contains a deep learning pipeline designed to classify news articles as real or fake based on their headlines. By leveraging natural language processing (NLP) techniques and a Long Short-Term Memory (LSTM) recurrent neural network, the model effectively captures sequential context within text to identify misinformation.

## Key Features & Workflow
* **Text Preprocessing:** Cleans and normalizes raw text using tokenization, regular expressions, lowercase conversion, and stopword removal.
* **Stemming:** Applies the Porter Stemmer to reduce words to their root forms, minimizing vocabulary redundancy.
* **Word Embedding:** Converts processed text into fixed-length dense vectors via a Keras Embedding layer using a 5,000-word vocabulary limit.
* **Deep Learning Architecture:** Utilizes a Sequential LSTM network to map long-term dependencies and context across headlines, concluding with a Sigmoid activation layer for binary classification.

## Performance & Metrics
The model achieves an overall classification accuracy of **~81%**. Below is the detailed classification report showcasing the performance across both classes:

```text
              precision    recall  f1-score   support

           0       0.62      0.58      0.60      1842
           1       0.87      0.88      0.87      5704

    accuracy                           0.81      7546
   macro avg       0.74      0.73      0.74      7546
weighted avg       0.81      0.81      0.81      7546
```


Technologies Used
1) Python
2) TensorFlow / Keras (Sequential, Embedding, LSTM, Dense)
3) NLTK (Natural Language Toolkit for Stopwords & Stemming)
4) Scikit-Learn (Train-Test Split, Classification Report)
5) NumPy & Pandas
