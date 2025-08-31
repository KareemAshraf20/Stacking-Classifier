# Stacking Classifier Project for Boston Housing Dataset

## 📖 Project Overview
This project implements a **Stacking Ensemble Classifier** to predict categorized housing prices from the Boston Housing Dataset. The model combines multiple machine learning algorithms to improve prediction accuracy through a meta-learning approach.

## 📊 Dataset Information
The Boston Housing Dataset contains 506 samples with 13 feature variables and 1 target variable (medv - median value of owner-occupied homes in $1000s). The target variable was categorized into three classes for classification:
- Class 0: Homes valued under $20,000
- Class 1: Homes valued between $20,000-$35,000  
- Class 2: Homes valued over $35,000

## 🧠 Model Architecture

### Base Models (Level 0):
1. **Random Forest Classifier** (10 estimators)
2. **K-Nearest Neighbors Classifier** (5 neighbors)
3. **Gradient Boosting Classifier** (default parameters)
4. **Support Vector Classifier** (RBF kernel with probability estimates)

### Meta Model (Level 1):
- **Decision Tree Classifier** (max depth = 3)

## 🛠️ Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy

## 🚀 Implementation Details

### Code Explanation:

```python
# Import necessary libraries
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import pandas as pd
import numpy as np

# Load and explore the dataset
df = pd.read_csv('/content/Boston.csv')
df.head()  # Display first 5 rows

# Check for missing values
df.isna().sum()  # No missing values found

# Preprocessing: Remove index column and separate features/target
df.drop(['Unnamed: 0'], axis=1, inplace=True)  # Remove unnecessary index column
x = df.drop(['medv'], axis=1)  # Feature matrix (all columns except medv)
y = df['medv']  # Target variable

# Split data into training and testing sets (80/20 split)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert continuous target variable to categorical classes
bins = [0, 20, 35, np.inf]  # Define bin edges for categorization
labels = [0, 1, 2]  # Class labels
y_train = pd.cut(y_train, bins=bins, labels=labels)  # Convert training target
y_test = pd.cut(y_test, bins=bins, labels=labels)  # Convert testing target

# Define ensemble of base models
ems = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),  # Random Forest
    ('knn', KNeighborsClassifier(n_neighbors=5)),  # K-Nearest Neighbors
    ('gdc', GradientBoostingClassifier()),  # Gradient Boosting
    ('svc', SVC(kernel='rbf', probability=True))  # Support Vector Classifier
]

# Create and train stacking classifier
st = StackingClassifier(estimators=ems, final_estimator=DecisionTreeClassifier(max_depth=3))
st.fit(x_train, y_train)  # Train the model

# Make predictions and evaluate accuracy
y_pred = st.predict(x_test)
accuracy_score(y_test, y_pred)  # Calculate accuracy
```

## 📈 Results
The stacking ensemble achieved an accuracy of **82.35%** on the test set, demonstrating the effectiveness of combining multiple models for improved prediction performance.

## 💡 Key Concepts Demonstrated
- Ensemble Learning
- Stacking Classifier Architecture
- Data Preprocessing
- Model Evaluation
- Multi-class Classification
