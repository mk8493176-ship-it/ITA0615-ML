# Experiment 19: Naive Bayes classification for Bank Loan Prediction
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

rng = np.random.default_rng(21)
n = 1000
df = pd.DataFrame({
    "Age": rng.integers(21, 65, n),
    "Applicant_Income": rng.integers(15000, 120000, n),
    "Loan_Amount": rng.integers(50, 600, n),
    "Credit_History": rng.integers(0, 2, n),
    "Married": rng.integers(0, 2, n),
    "Employed": rng.integers(0, 2, n),
})
z = (2.5 * df.Credit_History + df.Applicant_Income / 40000 - df.Loan_Amount / 200
     + 0.5 * df.Employed + 0.3 * df.Married - 2 + rng.normal(0, 0.8, n))
df["Loan_Status"] = (z > 0.8).astype(int)               # 1 = Approved, 0 = Rejected
df.to_csv("bank_loan.csv", index=False)

X, y = df.drop(columns="Loan_Status"), df["Loan_Status"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
nb = GaussianNB().fit(Xtr, ytr)
pred = nb.predict(Xte)

print(f"Accuracy: {accuracy_score(yte, pred) * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(yte, pred))
print("\n", classification_report(yte, pred, target_names=["Rejected", "Approved"]))
applicant = pd.DataFrame([[35, 60000, 150, 1, 1, 1]], columns=X.columns)
print("New applicant ->", "Approved" if nb.predict(applicant)[0] else "Rejected")
