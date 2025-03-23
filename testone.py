

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and Explore the Data
print("\n--- 1. Loading and Exploring the Data ---")

#  1.1: Load the Dataset
california = fetch_california_housing()
print("Dataset loaded successfully.")
print(california.DESCR)  # Print dataset description to understand features
# Convert the dataset to a pandas DataFrame for easier manipulation
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target

#  1.2: Scatter Plot
print("\nCreating Scatter Plot of MedInc vs. MedHouseVal...")
plt.figure(figsize=(10, 6))  # Adjust figure size for better readability
plt.scatter(df['MedInc'], df['MedHouseVal'], alpha=0.5)  # Alpha for transparency
plt.xlabel('Median Income (MedInc)')
plt.ylabel('Median House Value (MedHouseVal)')
plt.title('Scatter Plot: MedInc vs. MedHouseVal')
plt.grid(True)  # Add grid for easier visualization
plt.show()
print("Scatter plot created and displayed.")

#  1.3: Summary Statistics
print("\nCalculating Summary Statistics...")
medinc_mean = df['MedInc'].mean()
medinc_median = df['MedInc'].median()
medinc_std = df['MedInc'].std()

medhouseval_mean = df['MedHouseVal'].mean()
medhouseval_median = df['MedHouseVal'].median()
medhouseval_std = df['MedHouseVal'].std()

print(f"MedInc - Mean: {medinc_mean:.2f}, Median: {medinc_median:.2f}, Standard Deviation: {medinc_std:.2f}")
print(f"MedHouseVal - Mean: {medhouseval_mean:.2f}, Median: {medhouseval_median:.2f}, Standard Deviation: {medhouseval_std:.2f}")

# 2. Preprocess the Data
print("\n--- 2. Preprocessing the Data ---")

#  2.1: Split Data
X = df[['MedInc']]  # Feature (Median Income)
y = df['MedHouseVal']  # Target (Median House Value)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets (80/20).")


# 3. Build a Linear Regression Model
print("\n--- 3. Building the Linear Regression Model ---")

#  3.1: Create and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)
print("Linear Regression model trained.")

# 4. Make Predictions
print("\n--- 4. Making Predictions ---")

#  4.1: Predict on Test Set
y_pred = model.predict(X_test)
print("Predictions made on the test set.")

#  4.2: Predict for MedInc = 8.0
medinc_value = np.array([[8.0]])  # Create a 2D array with the desired value

if standardize:
  medinc_value_scaled = scaler.transform(medinc_value) #must transform to scaled value
  predicted_house_value = model.predict(medinc_value_scaled)

else:
  predicted_house_value = model.predict(medinc_value)
print(f"Predicted house value for MedInc = 8.0: {predicted_house_value[0]:.2f}")

# 5. Evaluate the Model
print("\n--- 5. Evaluating the Model ---")

#  5.1: Calculate Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# 6. Visualize the Results
print("\n--- 6. Visualizing the Results ---")

# 6.1: Plot Regression Line
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.5, label='Actual')  # Plot actual values
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')  # Plot regression line

plt.xlabel('Median Income (MedInc)')
plt.ylabel('Median House Value (MedHouseVal)')
plt.title('Linear Regression: Actual vs. Predicted')
plt.legend()  # Show legend
plt.grid(True)
plt.show()
print("Regression line plot created and displayed.")

# Print Regression equation

print("Regression Line Equation")
print(f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")
