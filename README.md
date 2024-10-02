# **Calculate the Similarity Between Two Phrases Using BERT and Machine Learning**

## **Project Overview**
This project demonstrates the use of **BERT embeddings** to calculate the similarity between two phrases. We explore various similarity metrics such as **Cosine Similarity**, **Jaccard Similarity**, and **Euclidean Distance** to measure the semantic closeness between the phrases. Additionally, machine learning models are applied to predict phrase similarities, and we evaluate the model performance using **R-squared (R²)** and **Mean Squared Error (MSE)**.
## **Key Features**

### 1. **Phrase Embedding with BERT**
### 2. **Multiple Similarity Metrics**
### 3. **Machine Learning Models for Prediction**
### 4. **Evaluation with R-squared and MSE**

## **Project Workflow**

### 1. **Load the Dataset**:
   - A CSV dataset containing pairs of phrases is loaded, which forms the basis for similarity calculation and prediction tasks.

### 2. **BERT Embedding Generation**:
   - For each phrase in the dataset, embeddings are generated using the **BERT base model**. The `[CLS]` token’s embedding is used to represent each phrase in the semantic space.

### 3. **Calculate Similarity Metrics**:
   - **Cosine Similarity**: Calculated between each pair of embeddings.
   - **Euclidean Distance**: Computed between the phrase embeddings to measure geometric similarity.
   - **Jaccard Similarity**: Computed based on token overlap between the two phrases.

### 4. **Train Machine Learning Models**:
The calculated similarity metrics (Cosine, Jaccard, and Euclidean) are used as features to train machine learning models to predict similarity scores between phrases.


### 5. **Evaluation**:
   - The models are evaluated using **R-squared (R²)** to assess how well the model fits the data.
   - **MSE** is calculated to determine the accuracy of the model's predictions.

---

## **How to Run the Project**

1. **Install Dependencies**:  
   Ensure that the following libraries are installed:
   - `transformers`
   - `torch`
   - `scikit-learn`
   - `scipy`
   - `pandas`

2. **Load Dataset**:  
   Load the CSV files containing pairs of phrases.

3. **Run the Script**:  
   The script will process the phrases, generate embeddings using BERT, calculate similarity metrics, and train the machine learning models.

4. **Evaluate the Models**:  
   The script will output R² and MSE to evaluate the accuracy of the similarity predictions.
