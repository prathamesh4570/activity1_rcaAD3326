# CIE-2 Unit 1: Multiple Linear Regression
# Run this file from the folder containing house_prices.csv.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

df = pd.read_csv("house_prices.csv")
print("First five rows:\n", df.head())
print("\nDataset information:")
print(df.info())

# 1) Small-sample manual calculation (simple illustrative example)
# For two predictors, the fitted equation is y = b0 + b1*x1 + b2*x2.
# The coefficients can be obtained from the normal equation:
# beta = inverse(X'X) X'y. This computes the hand-worked sample numerically.
small = df[["Area_sqft", "Bedrooms", "Price_lakh"]].head(5)
X_small = np.column_stack([np.ones(len(small)), small["Area_sqft"], small["Bedrooms"]])
y_small = small["Price_lakh"].to_numpy()
beta = np.linalg.pinv(X_small.T @ X_small) @ X_small.T @ y_small
print("\nSmall-sample manual normal-equation coefficients [b0, b1, b2]:")
print(beta)
print(f"Manual equation: Price = {beta[0]:.4f} + ({beta[1]:.4f} * Area) + ({beta[2]:.4f} * Bedrooms)")

# 2) Multiple-feature model
features = ["Area_sqft", "Bedrooms", "Age_years", "Distance_km"]
X = df[features]
y = df["Price_lakh"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nMultiple Linear Regression")
print("Intercept:", model.intercept_)
print("Coefficients:")
for name, coef in zip(features, model.coef_):
    print(f"{name}: {coef:.4f}")
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R2:", r2_score(y_test, pred))

n, p = len(y_test), len(features)
r2 = r2_score(y_test, pred)
adjusted_r2 = 1 - (1-r2)*(n-1)/(n-p-1) if n > p+1 else float("nan")
print("Adjusted R2:", adjusted_r2)

# 3) Single-feature comparison
single = LinearRegression()
X1_train, X1_test, y1_train, y1_test = train_test_split(
    df[["Area_sqft"]], y, test_size=0.25, random_state=42
)
single.fit(X1_train, y1_train)
single_pred = single.predict(X1_test)
print("\nSingle-feature model R2:", r2_score(y1_test, single_pred))
print("Single-feature MAE:", mean_absolute_error(y1_test, single_pred))

# 4) Multicollinearity check using VIF
X_vif = sm.add_constant(X)
vif = pd.DataFrame({
    "Feature": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
print("\nVIF (large values, often > 5 or > 10, can indicate multicollinearity):")
print(vif)

# 5) Residual check
residuals = y_test.to_numpy() - pred
plt.figure()
plt.scatter(pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted price (lakh)")
plt.ylabel("Residual")
plt.title("Residual Plot - Multiple Linear Regression")
plt.tight_layout()
plt.show()

# 6) Predict a new example
new_house = pd.DataFrame([{
    "Area_sqft": 1500, "Bedrooms": 3, "Age_years": 5, "Distance_km": 4
}])
print("\nPredicted price for new example (lakh):", model.predict(new_house)[0])
