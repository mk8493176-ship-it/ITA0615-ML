# Experiment 13: Car Price Prediction
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

rng = np.random.default_rng(3)
n = 800
df = pd.DataFrame({
    "Year": rng.integers(2005, 2024, n),
    "Present_Price": rng.uniform(4, 40, n).round(2),       # lakhs
    "Kms_Driven": rng.integers(2000, 150000, n),
    "Fuel_Type": rng.integers(0, 2, n),                    # 0 = Petrol, 1 = Diesel
    "Owner": rng.integers(0, 3, n),
})
age = 2024 - df.Year
df["Selling_Price"] = (df.Present_Price * (0.92 - 0.045 * age) - df.Kms_Driven * 0.00004
                       + df.Fuel_Type * 0.8 - df.Owner * 0.6 + rng.normal(0, 0.5, n)).clip(0.3).round(2)
df.to_csv("car_data.csv", index=False)

X, y = df.drop(columns="Selling_Price"), df["Selling_Price"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

for name, m in [("Linear Regression", LinearRegression()), ("Random Forest", RandomForestRegressor(200, random_state=42))]:
    p = m.fit(Xtr, ytr).predict(Xte)
    print(f"{name:<18} R2 = {r2_score(yte, p):.4f}   MAE = {mean_absolute_error(yte, p):.3f} lakhs")

car = pd.DataFrame([[2018, 12.5, 40000, 0, 0]], columns=X.columns)
print("\nPredicted price for 2018 petrol car, 40,000 kms, 1st owner:",
      round(m.predict(car)[0], 2), "lakhs")
