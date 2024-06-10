# Logistic Regression for Breast Cancer Classification

This project demonstrates the implementation of a Logistic Regression model to classify breast cancer data. The steps below detail the process from importing libraries and data, to training the model and evaluating its performance.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Implementation](#implementation)
  - [Importing the Libraries](#importing-the-libraries)
  - [Importing the Dataset](#importing-the-dataset)
  - [Splitting the Dataset](#splitting-the-dataset)
  - [Training the Model](#training-the-model)
  - [Predicting Test Results](#predicting-test-results)
  - [Evaluating the Model](#evaluating-the-model)
- [Results](#results)

## Installation
Ensure you have Python and the following libraries installed:
- pandas
- scikit-learn

You can install the required libraries using pip:
```bash
pip install pandas scikit-learn
```

## Dataset
The dataset used is `breast_cancer.csv`. Make sure this file is in the same directory as your script or provide the correct path to the file.

## Implementation

### Importing the Libraries
```python
import pandas as pd
```
Pandas is used for data manipulation and analysis.

### Importing the Dataset
```python
dataset = pd.read_csv("breast_cancer.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
```
- The dataset is read from a CSV file.
- `X` contains the feature variables (all columns except the first and the last).
- `y` contains the target variable (the last column).

### Splitting the Dataset
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
- The dataset is split into training and testing sets.
- 20% of the data is used for testing, and the split is reproducible with `random_state=0`.

### Training the Model
```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
```
- A Logistic Regression classifier is instantiated and trained on the training set.

### Predicting Test Results
```python
y_pred = classifier.predict(X_test)
#print(y_pred)
```
- The model predicts the target variable for the test set.

### Evaluating the Model
#### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
```
- A confusion matrix is created to evaluate the model's performance.
- Output: 
  ```
  [[84  3]
   [ 3 47]]
  ```

#### Accuracy
```python
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
```
- The accuracy of the model is calculated.
- Output: `0.9562043795620438` (95.62%)

#### k-Fold Cross Validation
```python
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:0.2f} %".format(accuracies.mean() * 100))
print("Standard Deviation: {:0.2f} %".format(accuracies.std() * 100))
```
- k-Fold Cross Validation is used to evaluate the model more robustly.
- Output: 
  ```
  Accuracy: 96.70 %
  Standard Deviation: 1.97 %
  ```

## Results
This Logistic Regression model demonstrates a high accuracy of approximately 95.62% on the test set and an average cross-validation accuracy of 96.70% with a standard deviation of 1.97%, indicating a reliable model with consistent performance.
