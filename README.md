"# MLOps Assignment 1" 

## 1. Problem Statement

The objective of this assignment is to train and compare multiple machine learning models on the **Iris dataset**.  
The goal is to predict the type of Iris flower (**Setosa, Versicolor, Virginica**) based on its features: Sepal length, Sepal width, Petal length, and Petal width.  

Additionally, the assignment demonstrates **MLflow logging** and **model registration** in a structured MLOps workflow.

---

## 2. Dataset Description

**Dataset:** Iris dataset (built-in from scikit-learn)  

**Total samples:** 150  
**Features:** 4  
**Classes:** 3 (Setosa, Versicolor, Virginica)  

| Feature       | Description                |
|---------------|----------------------------|
| Sepal Length  | Length of sepal (cm)       |
| Sepal Width   | Width of sepal (cm)        |
| Petal Length  | Length of petal (cm)       |
| Petal Width   | Width of petal (cm)        |
| Target        | Iris flower class (0,1,2)  |

---

## 3. Model Selection & Comparison

**Models Trained:**

1. Logistic Regression  
2. Random Forest Classifier  
3. Support Vector Machine (SVM) with linear kernel  

**Evaluation Metrics:** Accuracy, Precision (weighted), Recall (weighted), F1-score (weighted)

**Metrics Table Example:**

| Model                 | Accuracy | Precision | Recall | F1-score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 1.0      | 1.0       | 1.0    | 1.0      |
| Random Forest         | 1.0      | 1.0       | 1.0    | 1.0      |
| SVM                   | 1.0      | 1.0       | 1.0    | 1.0      |

> *Metrics may vary depending on train-test split.*

---

## 4. MLflow Logging & Model Registration

- All experiments are tracked using **MLflow**.  
- Each run logs: model name, hyperparameters, and evaluation metrics.  
- Registered models can be used later for deployment or inference.  

## 5. Instructions to Run the Code

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/mlops-assignment-1.git
cd mlops-assignment-1
