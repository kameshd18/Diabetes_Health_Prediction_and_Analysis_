# Diabetes Health Prediction and Analysis 🎉

![Diabetes Health Prediction](https://miro.medium.com/v2/resize:fit:828/format:webp/1*KkQbSEI9sT44_yxR9vscJA.gif)

---

Welcome to the **Diabetes Health Prediction and Analysis** project! This repository contains a comprehensive pipeline for predicting diabetes diagnosis using various machine learning and deep learning models, along with an in-depth exploratory data analysis and feature engineering steps.

## 🚀 Project Overview

This project aims to provide a thorough analysis of diabetes-related health data, develop predictive models, and evaluate their performance. The key components of the project include:

- 📊 Data Preprocessing
- 🔍 Exploratory Data Analysis (EDA)
- 🛠️ Feature Engineering
- 🧠 Model Training
- 📈 Model Evaluation
- 📑 Comprehensive Reports

## 📂 Project Structure

Here's an overview of the project directory structure:


```plaintext
Diabetes_Health_Prediction_and_Analysis/
├── data/
│   ├── raw/
│   │   └── diabetes_data.csv
│   ├── processed/
│   │   ├── X_train.csv
│   │   ├── X_train_engineered.csv
│   │   ├── X_test.csv
│   │   ├── X_test_engineered.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
├── app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── styles.css
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── scripts/
│   ├── plots/
│   ├── reports/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── model_performance_report.py
├── tests/
│   ├── models/
│   ├── test_data_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_model_training.py
├── requirements.txt 
└── README.md
```

## 🔧 Setup and Installation

To get started with this project, follow the steps below:

1. **Clone the repository:**

    ```sh
    git clone https://github.com/ThecoderPinar/Diabetes_Health_Prediction_and_Analysis.git
    cd Diabetes_Health_Prediction_and_Analysis
    ```

2. **Create and activate a virtual environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Run the data preprocessing script:**

    ```sh
    python scripts/data_preprocessing.py
    ```

5. **Run the feature engineering script:**

    ```sh
    python scripts/feature_engineering.py
    ```

6. **Train the models:**

    ```sh
    python scripts/model_training.py
    ```

7. **Evaluate the models:**

    ```sh
    python scripts/model_evaluation.py
    ```

8. **Generate comprehensive model performance reports:**

    ```sh
    python script/comprehensive_model_report.py
    ```

## 🚀 Usage

- **Exploratory Data Analysis**: Check the `notebooks/exploratory_data_analysis.ipynb` notebook for detailed data analysis and visualizations.
- **Scripts**: All scripts for data preprocessing, feature engineering, model training, and evaluation are located in the `scripts/` directory.
- **Tests**: To ensure code quality and correctness, tests are included in the `tests/` directory. Run them with `pytest`.

## 📊 Models

The following models are trained and evaluated in this project:

---

### Logistic Regression

#### ROC Curve:
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/c6a46c80-0532-44ac-be7a-768600b7d72b" />


*The ROC curve illustrates the true positive rate (sensitivity) versus the false positive rate (1-specificity) for different threshold settings. A higher area under the curve (AUC) indicates better model performance.*

#### Confusion Matrix:
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/68aae032-36db-4c24-bd08-c1d3d45794fd" />


*The confusion matrix provides a summary of the prediction results on the classification problem. It shows the number of true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions.*

---

### Random Forest

#### ROC Curve:
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/d159a276-8b8f-42dd-aba2-1d6faf70b345" />


*The ROC curve illustrates the true positive rate (sensitivity) versus the false positive rate (1-specificity) for different threshold settings. A higher area under the curve (AUC) indicates better model performance.*

#### Confusion Matrix:
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/e91f543a-42c5-43f0-b5f0-01137ded2860" />


*The confusion matrix provides a summary of the prediction results on



## 🎯 Performance Metrics

The performance of the models is evaluated using the following metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC AUC Score**
- **Confusion Matrix**

### Logistic Regression

- **Accuracy (Doğruluk):** %78.99
- **Precision (Kesinlik):** %73.19
- **Recall (Duyarlılık):** %70.63
- **F1 Score:** %71.89
- **ROC AUC:** %83.86

**Confusion Matrix:**
```plaintext
[[196  37]
 [ 42 101]]
```
Model dosyası: 
```sh
models/logistic_regression.pkl
```

### Random Forest

- **Accuracy (Doğruluk):** %91.22
- **Precision (Kesinlik):** %94.35
- **Recall (Duyarlılık):** %81.82
- **F1 Score:** %87.64
- **ROC AUC:** %97.69

**Confusion Matrix:**
```plaintext
[[226   7]
 [ 26 117]]
```
Model dosyası: 
```sh
models/random_forest.pkl
```
##### Explanations:

1. [x] **_Accuracy:_** The ratio of correctly predicted instances to the total instances.
2. [x] **_Precision:**_ The ratio of true positive predictions to the total predicted positives. It measures the accuracy of positive predictions.
3. [x] **_Recall:_** The ratio of true positive predictions to the actual positives. It measures the model's ability to identify positive instances.
4.  [x] **_F1 Score:_** The harmonic mean of precision and recall. It provides a balance between precision and recall.
5.  [x] **_ROC AUC:_** The area under the ROC curve. It summarizes the model's ability to distinguish between classes.

**Confusion Matrix:**

* True Positive (TP): 117 - The number of actual positive cases correctly identified by the model.
* True Negative (TN): 226 - The number of actual negative cases correctly identified by the model.
* False Positive (FP): 7 - The number of actual negative cases incorrectly identified as positive by the model.
* False Negative (FN): 26 - The number of actual positive cases incorrectly identified as negative by the model.

##### Explanations:
1. [x] **_Accuracy:_** The ratio of correctly predicted instances to the total instances.
2. [x] **_Precision:_** The ratio of true positive predictions to the total predicted positives. It measures the accuracy of positive predictions.
3. [x] **_Recall:_** The ratio of true positive predictions to the actual positives. It measures the model's ability to identify positive instances.
4. [x] **_F1 Score:_** The harmonic mean of precision and recall. It provides a balance between precision and recall.
5. [x] **_ROC AUC:_** The area under the ROC curve. It summarizes the model's ability to distinguish between classes.

**Confusion Matrix:**

* True Positive (TP): 117 - The number of actual positive cases correctly identified by the model.
* True Negative (TN): 226 - The number of actual negative cases correctly identified by the model.
* False Positive (FP): 7 - The number of actual negative cases incorrectly identified as positive by the model.
* False Negative (FN): 26 - The number of actual positive cases incorrectly identified as negative by the model.

### XGBoost

- **Accuracy (Doğruluk):** %91.76
- **Precision (Kesinlik):** %93.08
- **Recall (Duyarlılık):** %84.62
- **F1 Score:** %88.64
- **ROC AUC:** %98.41

**Confusion Matrix:**
```plaintext
[[224   9]
 [ 22 121]]
```
Model dosyası: 
```sh
models/xgboost.pkl
```
##### Explanations:

1. [x] **_Accuracy_:** The ratio of correctly predicted instances to the total instances.
2. [x] **_Precision:_** The ratio of true positive predictions to the total predicted positives. It measures the accuracy of positive predictions.
3. [x] **_Recall:_** The ratio of true positive predictions to the actual positives. It measures the model's ability to identify positive instances.
4. [x] _**F1 Score:**_ The harmonic mean of precision and recall. It provides a balance between precision and recall.
5. [x] **_ROC AUC:_** The area under the ROC curve. It summarizes the model's ability to distinguish between classes.

**Confusion Matrix:**

* True Positive (TP): 121 - The number of actual positive cases correctly identified by the model.
* True Negative (TN): 224 - The number of actual negative cases correctly identified by the model.
* False Positive (FP): 9 - The number of actual negative cases incorrectly identified as positive by the model.
* False Negative (FN): 22 - The number of actual positive cases incorrectly identified as negative by the model.

## 📈 Results

Model performance reports and evaluation metrics are saved and displayed in the `comprehensive_model_report.py` script output.

