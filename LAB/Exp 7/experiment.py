# Experiment 7: Logistic Regression (Hours studied -> Pass / Fail)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

rng = np.random.default_rng(0)
hours = rng.uniform(0, 10, 200)
passed = (rng.random(200) < 1 / (1 + np.exp(-(1.2 * hours - 6)))).astype(int)
df = pd.DataFrame({"Hours": hours.round(2), "Passed": passed})
df.to_csv("students.csv", index=False)

Xtr, Xte, ytr, yte = train_test_split(df[["Hours"]], df["Passed"], test_size=0.25, random_state=1)
model = LogisticRegression().fit(Xtr, ytr)
pred = model.predict(Xte)

print(f"Coefficient : {model.coef_[0][0]:.3f}")
print(f"Intercept   : {model.intercept_[0]:.3f}")
print(f"Accuracy    : {accuracy_score(yte, pred) * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(yte, pred))
for h in (2, 5, 8):
    p = model.predict_proba([[h]])[0][1]
    print(f"Study {h} hrs -> P(pass) = {p:.2f} -> {'Pass' if p > 0.5 else 'Fail'}")
