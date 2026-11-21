# Experiment 6: Naive Bayes with Confusion Matrix and Accuracy
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.25, random_state=42)

model = GaussianNB().fit(Xtr, ytr)
pred = model.predict(Xte)

print("Confusion Matrix:")
print(confusion_matrix(yte, pred))
print(f"\nAccuracy: {accuracy_score(yte, pred) * 100:.2f}%\n")
print(classification_report(yte, pred, target_names=data.target_names))
