# Experiment 11: Credit Score Classification (Poor / Standard / Good)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

rng = np.random.default_rng(7)
n = 1500
df = pd.DataFrame({
    "Age": rng.integers(18, 70, n),
    "Annual_Income": rng.integers(15000, 150000, n),
    "Num_Credit_Cards": rng.integers(1, 10, n),
    "Interest_Rate": rng.integers(2, 35, n),
    "Num_Delayed_Payments": rng.integers(0, 25, n),
    "Credit_Utilization": rng.uniform(10, 90, n).round(1),
    "Outstanding_Debt": rng.integers(0, 5000, n),
})
score = (df.Annual_Income / 20000 - 0.25 * df.Num_Delayed_Payments - 0.04 * df.Credit_Utilization
         - df.Outstanding_Debt / 2000 - 0.05 * df.Interest_Rate + rng.normal(0, 0.4, n))
df["Credit_Score"] = pd.qcut(score, [0, .3, .7, 1], labels=["Poor", "Standard", "Good"])
df.to_csv("credit_score.csv", index=False)

X, y = df.drop(columns="Credit_Score"), df["Credit_Score"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr)
pred = model.predict(Xte)

print(f"Accuracy: {accuracy_score(yte, pred) * 100:.2f}%\n")
print(classification_report(yte, pred))
print("Top features:")
print(pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False).head(3).round(3).to_string())
