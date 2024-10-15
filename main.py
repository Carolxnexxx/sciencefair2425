import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Cleaned_Master.csv')

# Check for missing values
print("Missing values in dataset:\n", df.isnull().sum())

# Split the dataset into features and target variable
X = df.iloc[:, 1:546]  # Features
y = df.iloc[:, 546]    # Target (continuous)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)

# Initialize HistGradientBoostingRegressor
model = HistGradientBoostingRegressor()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate and print regression metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Optional: Visualize actual vs predicted values

plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted')
plt.show()
