# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
# Replace 'lung_cancer_dataset.csv' with your actual dataset file
data = pd.read_csv('lung_cancer_dataset.csv')

# Data preprocessing
# Assuming columns like 'age', 'smoking_history', 'family_history', 'chest_pain', 'shortness_of_breath', and 'lung_cancer' (as the target)
# Modify based on your actual dataset
# Example: Create a new feature 'smoking_years' based on age and smoking history
data['smoking_years'] = data['smoking_history'] * data['age'] / 100  # Adjust this as needed

# Prepare feature and target variables
# Assuming 'lung_cancer' is the target column and other features are relevant to the prediction
X = data[['age', 'smoking_history', 'family_history', 'chest_pain', 'shortness_of_breath', 'smoking_years']]
y = data['lung_cancer']  # This column should indicate whether the patient has lung cancer or not (binary 0 or 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model
joblib.dump(model, 'lung_cancer_model.pkl')


