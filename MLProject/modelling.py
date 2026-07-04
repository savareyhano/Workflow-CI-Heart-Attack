import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import mlflow
import mlflow.sklearn
import warnings
import os

warnings.filterwarnings("ignore")

# Using MLflow autolog
mlflow.sklearn.autolog()

def load_and_preprocess_data():
    df = pd.read_csv("Heart Attack.csv")
    df = df.dropna().drop_duplicates()
    
    le = LabelEncoder()
    df['class'] = le.fit_transform(df['class'])
    
    X = df.drop('class', axis=1)
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # We don't set tracking URI here because MLflow project will use the environment variables
    # set by GitHub Actions (MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD)
    mlflow.set_experiment("Heart_Attack_Prediction_CI")
    
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy (CI Workflow): {acc:.4f}")
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
