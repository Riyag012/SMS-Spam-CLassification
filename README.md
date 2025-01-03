# SMS Spam Classification

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Dataset](#dataset)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Model Performance](#model-performance)
8. [Future Enhancements](#future-enhancements)

---

## Overview
This project implements an **SMS Spam Classification** system to classify text messages as either **Spam** or **Ham (Not Spam)**. It leverages machine learning and
natural language processing (NLP) techniques to preprocess data and train predictive models. 

Two machine learning algorithms, **Decision Tree Classifier (DTC)** and **Naive Bayes**, were used to build the classifier. The project aims to automate spam detection
for better communication security and reduce manual message filtering.

---

## Features
- Preprocessing of raw text data:
  - Removal of punctuation and stop words.
  - Conversion of text to lowercase.
  - Tokenization and vectorization using **TF-IDF Vectorizer**.
- Implementation of two classification models:
  - **Decision Tree Classifier** for rule-based splitting.
  - **Naive Bayes** for probabilistic classification.
- Performance evaluation using metrics like **accuracy**, **precision**, **recall**, and **F1 score**.

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `scikit-learn` for model building and evaluation.
  - `pandas` for data handling and analysis.
  - `nltk` for text preprocessing.
  - `matplotlib` and `seaborn` for data visualization.

---

## Dataset
- The dataset contains SMS messages labeled as **Spam** or **Ham**.
- Used for training and testing the classification models.
- Source: Publicly available datasets like [Kaggle](https://www.kaggle.com).

---

## Installation

### Prerequisites
Ensure Python 3.x is installed along with the following packages:
```bash
pip install pandas scikit-learn nltk matplotlib seaborn
```

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd sms-spam-classification
   ```
3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Prepare the dataset with columns for text and labels.
2. Run the script to preprocess data and train the models:
   ```bash
   python train_model.py
   ```
3. Use the trained model for predictions:
   ```bash
   python predict.py --input "Your SMS message here"
   ```

---

## Model Performance
- **Naive Bayes**:
  - **Accuracy**: 95%
  - **Precision**: 95%
  - **Recall**: 95%
  - **F1 Score**: 95%
- **Decision Tree Classifier**:
  - **Accuracy**: 97%
  - **Precision**: 98%
  - **Recall**: 97%
  - **F1 Score**: 97%

Both models performed well, with the Decision Tree Classifier showing slightly better overall results in this implementation.

---

## Future Enhancements
- Incorporate additional machine learning models for comparison.
- Implement ensemble techniques like **Random Forest** or **Gradient Boosting** for better accuracy.
- Add support for real-time spam detection via a web application.
- Expand the dataset for broader coverage of spam messages.

---
