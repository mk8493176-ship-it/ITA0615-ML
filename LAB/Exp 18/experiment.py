# Experiment 18: Perceptron based IRIS classification
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)
sc = StandardScaler().fit(Xtr)

model = Perceptron(max_iter=1000, eta0=0.1, random_state=1).fit(sc.transform(Xtr), ytr)
pred = model.predict(sc.transform(Xte))

print("Weights (one row per class):")
print(model.coef_.round(3))
print("Bias:", model.intercept_.round(3))
print(f"\nTraining Accuracy: {model.score(sc.transform(Xtr), ytr) * 100:.2f}%")
print(f"Test Accuracy    : {accuracy_score(yte, pred) * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(yte, pred))
