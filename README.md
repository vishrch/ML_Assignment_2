Bank Marketing Classification

1. Problem Statement

The objective of this project is to develop and compare machine learning classification models to predict whether a bank customer will subscribe to a term deposit based on information about the customer and the bank’s marketing campaign.

The project evaluates five classification algorithms and compares their performance using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

2. Dataset Description

The Bank Marketing dataset is used for this classification task.

* Dataset: Bank Marketing (bank-full.csv)
* Total instances: 45,211
* Total features: 16
* Target variable: y
* Target classes: yes and no
* Classification type: Binary classification

The target variable indicates whether the customer subscribed to a term deposit.

The dataset contains customer details, campaign-related information, and previous marketing contact information. The data was divided into training and testing sets using stratified splitting.

The test dataset contains 9,043 records and is saved as test_data.csv.

3. GitHub Repository Link

Project source code, trained models, test data, and supporting files are available in the GitHub repository:

GitHub Repository

4. Models Used

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Data Preprocessing

The preprocessing steps include:

* Standardization of numerical features using StandardScaler.
* One-hot encoding of categorical features using OneHotEncoder.
* Combining preprocessing steps using ColumnTransformer.
* Using stratified train-test splitting to preserve the target class distribution.

The preprocessing and classification steps were implemented using Scikit-learn pipelines.

5. Model Performance Comparison

The models were evaluated on the same test dataset using the following metrics.

Model	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.9012	0.9056	0.6445	0.3478	0.4518	0.4261
Decision Tree	0.8746	0.7015	0.4649	0.4754	0.4701	0.3990
K-Nearest Neighbors	0.8962	0.8277	0.5990	0.3403	0.4340	0.4001
Naive Bayes	0.8548	0.8101	0.4059	0.5198	0.4559	0.3774
Support Vector Machine	0.9049	0.9037	0.6875	0.3431	0.4578	0.4420

6. Model Observations

Logistic Regression

Logistic Regression achieved an accuracy of 90.12% and the highest AUC score of 0.9056. It showed strong discrimination between the two target classes. Its recall for the subscription class was comparatively low.

Decision Tree

The Decision Tree achieved the highest F1 Score of 0.4701. It provided a better balance between precision and recall than the other evaluated models. Its AUC score was lower than the other models.

K-Nearest Neighbors

K-Nearest Neighbors achieved an accuracy of 89.62%. Its precision was relatively good, but its recall and F1 Score were lower than those of the Decision Tree.

Naive Bayes

Naive Bayes achieved the highest recall of 0.5198. It identified a larger proportion of customers who subscribed to a term deposit. However, its precision and overall accuracy were lower than those of the other models.

Support Vector Machine

Support Vector Machine achieved the highest accuracy of 90.49%, precision of 0.6875, and MCC of 0.4420. It demonstrated strong overall classification performance, although its recall was relatively low.

7. Overall Winner

Based on the F1 Score, the Decision Tree is the best-performing model, achieving a score of 0.4701.

The Support Vector Machine achieved the highest accuracy and MCC, while Logistic Regression achieved the highest AUC. Therefore, the choice of the best model depends on the evaluation metric and the objective of the classification task.

For this assignment, the Decision Tree is selected as the overall winner based on its F1 Score.

8. Streamlit Application

A Streamlit application was developed to evaluate the trained models using uploaded test data.

The application provides:

* CSV file upload for test data.
* Selection of one of the five trained classification models.
* Calculation of Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
* Confusion Matrix visualization.
* Classification Report.

The application loads the previously trained models and evaluates the uploaded test data without retraining.

Live Streamlit Application: To be added after deployment.

9. Project Structure

ML_Assignment_2/
│
├── app.py
├── model_training.ipynb
├── requirements.txt
├── test_data.csv
├── bank-full.csv
├── README.md
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── k_nearest_neighbors.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib

10. Requirements

The project uses the following Python libraries:

* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Joblib

The required dependencies are listed in requirements.txt.