import joblib
import os
import numpy as np

def test_model_exists():
    # Check if the model file was actually created
    assert os.path.exists("models/iris_model.pkl")

def test_model_prediction():
    # Load the model and check if it can make a prediction
    model = joblib.load("models/iris_model.pkl")
    
    # Create a dummy input (4 features: sepal length/width, petal length/width)
    dummy_data = np.array([[5.1, 3.5, 1.4, 0.2]]) 
    prediction = model.predict(dummy_data)
    
    # Check if prediction is one of the 3 iris classes (0, 1, or 2)
    assert prediction[0] in [0, 1, 2]