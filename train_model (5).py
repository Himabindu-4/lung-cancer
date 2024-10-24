# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('credit_risk_dataset.csv')

# Data preprocessing (modify this based on your dataset)
# Assuming columns 'income', 'loan_amount', 'credit_score', 'employment_length', and 'loan_default'
data['debt_income_ratio'] = data['loan_amnt'] / data['person_income']

# Prepare feature and target variables
X = data[['person_income', 'debt_income_ratio', 'person_emp_length']]
y = data['cb_person_default_on_file']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model
joblib.dump(model, 'credit_risk_model.pkl')