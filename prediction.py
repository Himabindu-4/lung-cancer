import joblib
import pandas as pd

def make_predictions(input_data):
    model = joblib.load('model.pkl')
    predictions = model.predict(input_data)
    return predictions

if __name__ == '__main__':
    new_data = pd.DataFrame({
        'age': [55],
        'gender': [1],  # Male
        'smoking_status': [1],  # Smoker
        'tumor_size': [30]
    })
    print(make_predictions(new_data))