# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pandas as pd
import pickle

# Step 1: Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel='linear', probability=True)
}

# Step 4: Create folders if they don't exist
if not os.path.exists('results'):
    os.makedirs('results')

if not os.path.exists('models'):
    os.makedirs('models')

# Step 5: Train models, evaluate, and save metrics & models
results = []

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Save metrics
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })
    
    # Save trained model
    model_file = f"models/{name.replace(' ', '_')}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"{name} trained and saved successfully.")

# Step 6: Save metrics to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('results/model_comparison.csv', index=False)
print("\nAll model metrics saved to results/model_comparison.csv")
