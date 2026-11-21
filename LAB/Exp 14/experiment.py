# Experiment 14: House Price Prediction
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

rng = np.random.default_rng(5)
n = 1000
df = pd.DataFrame({
    "Area_sqft": rng.integers(500, 4000, n),
    "Bedrooms": rng.integers(1, 6, n),
    "Bathrooms": rng.integers(1, 4, n),
    "Age_years": rng.integers(0, 40, n),
    "Distance_to_city_km": rng.uniform(1, 30, n).round(1),
})
df["Price_lakhs"] = (df.Area_sqft * 0.045 + df.Bedrooms * 4 + df.Bathrooms * 3 - df.Age_years * 0.6
                     - df.Distance_to_city_km * 1.2 + 20 + rng.normal(0, 5, n)).round(2)
df.to_csv("house_data.csv", index=False)

X, y = df.drop(columns="Price_lakhs"), df["Price_lakhs"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(random_state=42).fit(Xtr, ytr)
pred = model.predict(Xte)

print("Algorithm: Gradient Boosting Regressor")
print(f"R2 Score : {r2_score(yte, pred):.4f}")
print(f"RMSE     : {np.sqrt(mean_squared_error(yte, pred)):.3f} lakhs")
house = pd.DataFrame([[1800, 3, 2, 5, 8.0]], columns=X.columns)
print("\nPredicted price for 1800 sqft, 3BHK, 2 bath, 5 yrs old, 8 km from city:",
      round(model.predict(house)[0], 2), "lakhs")
