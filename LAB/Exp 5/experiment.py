# Experiment 5: K-Nearest Neighbours implemented from scratch
import numpy as np
from collections import Counter
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
X = StandardScaler().fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

def knn_predict(x, k):
    dist = np.sqrt(((Xtr - x) ** 2).sum(axis=1))      # Euclidean distance
    nearest = ytr[np.argsort(dist)[:k]]
    return Counter(nearest).most_common(1)[0][0]       # majority vote

print("Dataset: Wine  (178 samples, 13 features, 3 classes)\n")
for k in (1, 3, 5, 7, 9):
    pred = np.array([knn_predict(x, k) for x in Xte])
    print(f"K = {k}   Accuracy = {(pred == yte).mean() * 100:.2f}%")
