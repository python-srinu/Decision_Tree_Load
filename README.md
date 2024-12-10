# Loan Approval Prediction using Decision Tree Classifier

## Overview
This project demonstrates the use of a **Decision Tree Classifier** to predict loan approval status based on customer data. The implementation covers all aspects of a machine learning pipeline, including **data preprocessing**, **feature engineering**, **model training**, **evaluation**, and **visualization**. Additionally, it incorporates **hyperparameter tuning** and manual calculation of metrics like **Gini Index** and **Entropy** for deeper insights.

This project highlights technical proficiency in machine learning and data handling, making it an excellent addition to a developer's portfolio.

---

## Table of Contents
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Workflow](#workflow)
- [How to Run](#how-to-run)
- [Key Features](#key-features)
- [Further Learning](#further-learning)

---

## Dataset
The dataset, **loan.csv**, includes customer information such as:

### Numerical Features:
- **age**
- **income**
- **credit_score**

### Categorical Features:
- **gender**
- **occupation**
- **education_level**
- **marital_status**

### Target Variable:
- **loan_status** (Approved or Denied)

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `pandas`: For data manipulation and preprocessing.
  - `numpy`: For numerical computations.
  - `scikit-learn`: For machine learning algorithms, evaluation, and hyperparameter tuning.
  - `matplotlib`: For data visualization.

---

## Workflow

### 1. Data Preprocessing
- One-hot encoding of categorical features such as `gender`, `occupation`, and others.
- Scaling of numerical features (`age`, `income`, and `credit_score`) using `StandardScaler`.
- Merging scaled and encoded features into a unified dataset.

### 2. Model Training
- Splitting the dataset into training and testing subsets (80:20 split).
- Training a **Decision Tree Classifier** using the **Gini Index** criterion.

### 3. Visualization
- Plotting the trained **Decision Tree** for better interpretability.
- Displaying feature importances to understand the impact of each feature on the predictions.

### 4. Evaluation
- Generating a **Confusion Matrix** and **Classification Report**.
- Summarizing predictions in a DataFrame to compare actual and predicted values.

### 5. Custom Metrics
- Calculating **Gini Index** and **Entropy** for the `gender` feature to explore data splits.

### 6. Hyperparameter Tuning
- Using `GridSearchCV` to find the optimal combination of parameters for the Decision Tree Classifier.
- Evaluating the model with the best parameters for improved performance.

### 7. Prediction Results
- Generating a summary of correct and incorrect predictions.
- Adding a column to indicate the accuracy of individual predictions.

---

## How to Run

### Clone the Repository
git clone https://github.com/python-srinu/Decision_Tree_Load.git
cd Decision_Tree_Load
## Key Features
- **Comprehensive Preprocessing**: Combines one-hot encoding and feature scaling.
- **Model Training**: Implementation of a Decision Tree Classifier with Gini and Entropy criteria.
- **Hyperparameter Tuning**: `GridSearchCV` used to optimize the model for better predictions.
- **Custom Metrics**: Manual calculations for Gini Index and Entropy for selected features.
- **Visualizations**: Clear, interpretable visual representation of the decision-making process.

## Further Learning
Advanced methods include exploring **Ensemble Methods** such as **Random Forests** and **Boosting algorithms** like **AdaBoost** and **Gradient Boosting**. These techniques combine multiple decision trees to enhance model performance by reducing overfitting and improving predictive accuracy.

