import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model():
    df = pd.read_csv('processed_lung_cancer_data.csv')
    X = df.drop('cancer_stage', axis=1)  # Features
    y = df['cancer_stage']  # Target variable

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

    # Save the model
    joblib.dump(model, 'model.pkl')

if __name__ == '__main__':
    train_model()