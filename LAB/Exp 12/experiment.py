# Experiment 12: Iris Flower Classification using KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)
sc = StandardScaler().fit(Xtr)

knn = KNeighborsClassifier(n_neighbors=5).fit(sc.transform(Xtr), ytr)
pred = knn.predict(sc.transform(Xte))

print(f"Accuracy: {accuracy_score(yte, pred) * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(yte, pred))

sample = [[5.1, 3.5, 1.4, 0.2]]
print("\nNew sample", sample[0], "->", iris.target_names[knn.predict(sc.transform(sample))[0]])
