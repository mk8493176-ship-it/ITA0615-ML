# Experiment 8: Linear Regression (Diabetes dataset)
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True, as_frame=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(Xtr, ytr)
pred = model.predict(Xte)

print("Intercept:", round(model.intercept_, 2))
print("Coefficients:")
print(pd.Series(model.coef_, index=X.columns).round(2).to_string())
print(f"\nMean Squared Error : {mean_squared_error(yte, pred):.2f}")
print(f"R2 Score           : {r2_score(yte, pred):.4f}")
print("\nActual vs Predicted (first 5):")
print(pd.DataFrame({"Actual": yte.values[:5], "Predicted": pred[:5].round(1)}).to_string(index=False))
