# Experiment 16: Compare different Classification Algorithms
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(),
}
rows = []
for name, m in models.items():
    pipe = make_pipeline(StandardScaler(), m)
    p = pipe.fit(Xtr, ytr).predict(Xte)
    rows.append([name, accuracy_score(yte, p), precision_score(yte, p), recall_score(yte, p),
                 f1_score(yte, p), cross_val_score(pipe, X, y, cv=5).mean()])

res = pd.DataFrame(rows, columns=["Algorithm", "Accuracy", "Precision", "Recall", "F1", "CV-Acc"])
print("Dataset: Breast Cancer (569 samples, 30 features)\n")
print(res.sort_values("Accuracy", ascending=False).round(4).to_string(index=False))
print("\nBest model:", res.loc[res.Accuracy.idxmax(), "Algorithm"])
