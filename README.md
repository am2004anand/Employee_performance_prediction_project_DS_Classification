This project involves developing a machine
learning model to predict whether an
employee is eligible for promotion based on
their performance, years of experience,
training, and other organizational factors.
Using algorithms like Logistic Regression,
Random Forest, the goal is to classify
employees into "eligible" or "not eligible"
categories. This model can help HR
departments streamline the promotion
process and identify high-performing
individuals. Evaluation metrics such as
accuracy, precision, recall, and F1 score will
measure performance.



# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pickle

# Load your dataset
df = pd.read_csv("employee_promotion_data.csv")

# Drop unnecessary columns
df = df.drop(['name', 'employee_id'], axis=1)

# Define features and target
X = df.drop('promotion_eligibility', axis=1)
y = df['promotion_eligibility']

# Convert Yes/No → 1/0
y = y.map({'Yes': 1, 'No': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Separate columns
categorical_cols = ['department', 'job_role']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessor: Encode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Define full pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train model
model.fit(X_train, y_train)

#  Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

#  Save trained pipeline
pickle.dump(model, open('promotion_pipeline.pkl', 'wb'))
print("Model saved as 'promotion_pipeline.pkl'")
