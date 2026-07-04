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
    
    # When called via 'mlflow run', MLflow already creates an active run.
    # We just train and log directly — no need for set_experiment or start_run.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    
    # Save run_id to file so CI workflow can pick it up
    run_id = mlflow.active_run().info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    
    print(f"Model Accuracy (CI Workflow): {acc:.4f}")
    print(f"RUN_ID: {run_id}")

if __name__ == "__main__":
    main()
