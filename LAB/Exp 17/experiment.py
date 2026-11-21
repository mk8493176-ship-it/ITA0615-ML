# Experiment 17: Mobile Price Range Prediction
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

rng = np.random.default_rng(11)
n = 2000
df = pd.DataFrame({
    "battery_power": rng.integers(500, 2000, n),
    "ram": rng.integers(256, 4000, n),
    "int_memory": rng.integers(2, 64, n),
    "px_height": rng.integers(0, 1960, n),
    "px_width": rng.integers(500, 1998, n),
    "mobile_wt": rng.integers(80, 200, n),
    "n_cores": rng.integers(1, 9, n),
})
s = df.ram / 4000 * 3 + df.battery_power / 2000 + df.px_height / 1960 * 0.5 + df.int_memory / 64 * 0.5 + rng.normal(0, 0.15, n)
df["price_range"] = pd.qcut(s, 4, labels=[0, 1, 2, 3]).astype(int)    # 0=Low ... 3=Very High
df.to_csv("mobile_data.csv", index=False)

X, y = df.drop(columns="price_range"), df["price_range"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr)
pred = model.predict(Xte)

print(f"Accuracy: {accuracy_score(yte, pred) * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(yte, pred))
print("\nFeature importance:")
print(pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False).round(3).to_string())
