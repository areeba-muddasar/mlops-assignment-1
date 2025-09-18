# src/train_model_mlflow.py

# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

# Step 1: Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize models
models = {
    "Logistic_Regression": LogisticRegression(max_iter=200),
    "Random_Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel='linear', probability=True)
}

# Step 4: Create folders
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Step 5: Set MLflow experiment
mlflow.set_experiment("iris_classification")

# Step 6: Train, log metrics & artifacts
results = []

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Log hyperparameters
        mlflow.log_params(model.get_params())
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Save metrics to list
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        })
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
        
        # Save trained model locally
        model_file = f"models/{name}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Log model with MLflow
        mlflow.sklearn.log_model(model, name)
        
        # Confusion matrix as artifact
        cm_display = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        cm_path = f"results/{name}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)
        
        print(f"{name} trained and logged to MLflow successfully.")

# Step 7: Save metrics to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('results/model_comparison.csv', index=False)
print("\nAll model metrics saved to results/model_comparison.csv")
