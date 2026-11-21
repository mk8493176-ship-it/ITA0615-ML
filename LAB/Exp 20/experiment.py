# Experiment 20: Future Sales Prediction (advertising spend -> sales)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

rng = np.random.default_rng(9)
n = 200
df = pd.DataFrame({
    "TV": rng.uniform(0, 300, n).round(1),
    "Radio": rng.uniform(0, 50, n).round(1),
    "Newspaper": rng.uniform(0, 100, n).round(1),
})
df["Sales"] = (3 + 0.045 * df.TV + 0.19 * df.Radio + 0.002 * df.Newspaper + rng.normal(0, 1.2, n)).round(2)
df.to_csv("advertising_sales.csv", index=False)

X, y = df.drop(columns="Sales"), df["Sales"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(Xtr, ytr)
pred = model.predict(Xte)

print("Coefficients:", dict(zip(X.columns, model.coef_.round(4))))
print(f"R2 Score: {r2_score(yte, pred):.4f}    MAE: {mean_absolute_error(yte, pred):.3f}\n")

future = pd.DataFrame({"TV": [150, 220, 280], "Radio": [20, 30, 45], "Newspaper": [30, 40, 60]})
future["Predicted_Sales"] = model.predict(future).round(2)
print("Future sales forecast for planned ad budgets:")
print(future.to_string(index=False))
