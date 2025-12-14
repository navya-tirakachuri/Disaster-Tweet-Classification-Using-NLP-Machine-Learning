# Disaster-Tweet-Classification-Using-NLP-Machine-Learning
Natural Language Processing project to classify disaster-related tweets using TF-IDF features and machine learning models (Multinomial Naïve Bayes, Logistic Regression, KNN), achieving ~81% accuracy through model evaluation and cross-validation.

## Problem Statement
During natural disasters, millions of tweets are posted in real time.  
Manually identifying which tweets indicate **real disaster events** is inefficient and error-prone.

The challenge was to:
1. Process unstructured tweet text
2. Convert text into numerical features
3. Build and evaluate machine learning models
4. Accurately classify disaster vs non-disaster tweets

## Tools & Technologies Used
1. **Python**
2. **Natural Language Toolkit (NLTK)**
3. **Scikit-learn**
4. **Pandas, NumPy**
5. **TF-IDF Vectorizer**
6. Jupyter Notebook
   
## Dataset Details
1. Total Tweets: **7,613**
2. Columns:
  a. `tweets` – Raw tweet text
  b. `target` – Classification label (0 = Non-Disaster, 1 = Disaster)
3. Dataset Type: Text / Social Media Data

## What I Did
1. Loaded and explored the tweet dataset
2. Performed data cleaning and null value handling
3. Applied NLP preprocessing:
  a. Tokenization
  b. Lowercasing
  c. Stopword removal
  d. Punctuation removal
  e. Stemming and Lemmatization
4. Converted text data into numerical form using **TF-IDF Vectorization**
5. Split data into training and testing sets
6. Built and evaluated multiple ML models:
  a. Multinomial Naïve Bayes
  b. Logistic Regression
  c. K-Nearest Neighbors (KNN)
7. Evaluated models using:
  a. Confusion Matrix
  b. Precision, Recall, F1-score
  c. Cross-validation
8. Selected the best-performing model based on accuracy

## Model Performance Summary

### Multinomial Naïve Bayes (Best Model)
1. Accuracy: **~80.7%**
2. Strong balance between precision and recall
3. Best cross-validation performance

### Logistic Regression
1. Accuracy: ~80%
2. Competitive performance with stable predictions

### KNN Classifier
1. Accuracy: ~66%
2. High class imbalance sensitivity

## Best Model
**Multinomial Naïve Bayes**  
Chosen based on:
1. Highest test accuracy
2. Better cross-validation score
3. Efficient handling of TF-IDF features

## Key Outcomes
1. Successfully classified disaster-related tweets with **~81% accuracy**
2. Demonstrated end-to-end NLP workflow from raw text to predictions
3. Highlighted the importance of feature engineering and model selection
4. Built a scalable foundation for real-time disaster monitoring systems

##  Business Impact
This solution can be used by:
1. Emergency response teams
2. Government agencies
3. News & media organizations
4. Social media monitoring platforms

to quickly identify disaster-related information and respond faster.

## Repository Structure
├── disaster tweets sentiment analysis.pdf
├── README.md

## Project Highlight
**From raw tweet text → NLP preprocessing → TF-IDF feature engineering → machine learning classification**

## Contact
If you’d like to discuss this project, provide feedback, or explore collaboration opportunities, feel free to connect.  tvnnavyasree123@gmail.com








