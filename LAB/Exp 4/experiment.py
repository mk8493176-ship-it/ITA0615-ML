# Experiment 4: Artificial Neural Network using Backpropagation (from scratch, NumPy)
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(1)
X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
Y = np.eye(3)[y]                                   # one-hot targets
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=y)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 4 inputs -> 8 hidden -> 3 outputs
W1 = np.random.randn(4, 8) * 0.5; b1 = np.zeros((1, 8))
W2 = np.random.randn(8, 3) * 0.5; b2 = np.zeros((1, 3))
lr, n = 0.5, len(Xtr)

for epoch in range(1, 1001):
    # forward pass
    h = sigmoid(Xtr @ W1 + b1)
    o = sigmoid(h @ W2 + b2)
    loss = np.mean((Ytr - o) ** 2)
    # backward pass
    d_o = (o - Ytr) * o * (1 - o)
    d_h = (d_o @ W2.T) * h * (1 - h)
    W2 -= lr * h.T @ d_o / n;   b2 -= lr * d_o.mean(0, keepdims=True)
    W1 -= lr * Xtr.T @ d_h / n; b1 -= lr * d_h.mean(0, keepdims=True)
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}  Loss: {loss:.5f}")

pred = sigmoid(sigmoid(Xte @ W1 + b1) @ W2 + b2).argmax(1)
acc = (pred == Yte.argmax(1)).mean()
print(f"\nTest Accuracy: {acc * 100:.2f}%")
