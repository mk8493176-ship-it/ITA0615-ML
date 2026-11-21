# Experiment 9: Compare Linear and Polynomial Regression
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(1)
x = rng.uniform(-3, 3, 150).reshape(-1, 1)
y = (0.5 * x ** 2 - x + 3).ravel() + rng.normal(0, 0.6, 150)   # non-linear data
xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=0)

print(f"{'Model':<24}{'MSE':>8}{'R2':>9}")
print("-" * 41)
for name, deg in [("Linear (degree 1)", 1), ("Polynomial (degree 2)", 2), ("Polynomial (degree 3)", 3), ("Polynomial (degree 8)", 8)]:
    m = make_pipeline(PolynomialFeatures(deg), LinearRegression()).fit(xtr, ytr)
    p = m.predict(xte)
    print(f"{name:<24}{mean_squared_error(yte, p):>8.3f}{r2_score(yte, p):>9.4f}")
print("\nConclusion: data is curved, so polynomial (degree 2) fits far better than a straight line.")
