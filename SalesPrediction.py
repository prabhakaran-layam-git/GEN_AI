import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Create the dataset from the image
data = {
    "TV_Budget": [230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 144.1, 111.6],
    "Radio_Budget": [37.8, 39.3, 45.9, 41.3, 10.8, 48.9, 32.8, 19.6, 16.0, 12.6],
    "Newspaper_Budget": [69.2, 45.1, 69.3, 58.5, 58.4, 75.0, 23.5, 11.6, 40.3, 37.9],
    "Sales": [22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 15.6, 12.2]
}

df = pd.DataFrame(data)

# Show dataset with aligned, column-wise formatting for readability
print("\nDataset (aligned columns):")
print(df.to_string(index=False, col_space=12, float_format=lambda x: f"{x:8.1f}"))

# 2. Separate features (X) and target variable (y)
X = df[["TV_Budget", "Radio_Budget", "Newspaper_Budget"]]
y = df["Sales"]

# 3. Initialize and fit the Multiple Linear Regression model
model = LinearRegression()
model.fit(X, y)

# 4. Display model parameters
print(f"Intercept (β0): {model.intercept_:.4f}")
for col, coef in zip(X.columns, model.coef_):
    print(f"Coefficient for {col}: {coef:.4f}")

# 5. Output the mathematical equation
print(f"\nRegression Equation:")
print(f"Sales = {model.intercept_:.4f} + ({model.coef_[0]:.4f} * TV) + ({model.coef_[1]:.4f} * Radio) + ({model.coef_[2]:.4f} * Newspaper)")

def prompt_and_predict():
    """Prompt the user for budget inputs and print the predicted sales."""
    try:
        tv = float(input("Enter TV budget: ").strip())
        radio = float(input("Enter Radio budget: ").strip())
        newspaper = float(input("Enter Newspaper budget: ").strip())
    except ValueError:
        print("Invalid input. Please enter numeric values for budgets.")
        return

    X_new = pd.DataFrame([[tv, radio, newspaper]], columns=X.columns)
    pred = model.predict(X_new)[0]
    print(f"Predicted Sales: {pred:.2f}")


if __name__ == "__main__":
    print("\nInteractive prediction mode. Provide budgets to predict Sales.")
    while True:
        prompt_and_predict()
        again = input("Predict again? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            print("Exiting interactive mode.")
            break
