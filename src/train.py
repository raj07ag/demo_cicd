import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model():
    # 1. Load Data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train Model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # 4. Save Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/iris_model.pkl')
    
    # Calculate accuracy for the log
    accuracy = model.score(X_test, y_test)
    print(f"Model trained with accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    train_model()