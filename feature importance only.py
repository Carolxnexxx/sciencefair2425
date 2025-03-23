import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, median_absolute_error

# Limit CPU usage for parallel computing
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Load dataset
df = pd.read_csv('Master Gene Expression + Drug.csv')
selected_df = pd.read_csv('selected.csv')

selected_drugs = selected_df.iloc[:, 0].tolist()

# Identify feature and target columns
end_feature_idx = df.columns.get_loc("Erlotinib")
feature_columns = df.columns[1:end_feature_idx]

# Ensure the selected drugs exist in the dataset
drug_columns = [drug for drug in selected_drugs if drug in df.columns]
if not drug_columns:
    raise ValueError("None of the specified drugs were found in the dataset.")

print(f"Number of feature columns: {len(feature_columns)}")
print(f"Drugs being analyzed: {', '.join(drug_columns)}")

# Handle missing values and standardize features
imputer = KNNImputer(n_neighbors=5)
scaler = StandardScaler()

X_imputed = pd.DataFrame(imputer.fit_transform(df[feature_columns].astype(np.float32)), columns=feature_columns)
X_normalized = pd.DataFrame(scaler.fit_transform(X_imputed), columns=feature_columns)

# Define the Random Forest model
rf_model = RandomForestRegressor(random_state=0, n_jobs=-1)

# Track results
average_results = {"MSE": [], "R2": []}

# Process selected drugs
for drug in tqdm(drug_columns, desc="Processing Selected Drugs"):
    print(f"\nEvaluating Random Forest for drug: {drug}")

    # Filter out missing target values
    df_drug = df.dropna(subset=[drug])
    X = X_normalized.loc[df_drug.index]
    y = df_drug[drug].astype(np.float32)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)

    print(f"  Training Random Forest...")
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    explained_var = explained_variance_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    std_dev_error = np.std(y_test - y_pred)
    median_ae = median_absolute_error(y_test, y_pred)

    # Store average metrics
    average_results["MSE"].append(mse)
    average_results["R2"].append(r2)

    print(f"    MSE: {mse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, Explained Var: {explained_var:.4f}, MAPE: {mape:.2f}%, Std Dev Error: {std_dev_error:.4f}, Median AE: {median_ae:.4f}")

    # Feature importance from Random Forest
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Optional: Print only the top 20 most important features
    top_n = 20
    print(f"\nTop {top_n} Features for Random Forest - Drug: {drug}")
    print(feature_importance_df.head(top_n).to_string(index=False))

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n))
    plt.title(f'Feature Importance for Random Forest - Drug: {drug}')
    plt.show()

# Compute and display average metrics
avg_mse = np.mean(average_results["MSE"])
avg_r2 = np.mean(average_results["R2"])
print(f"\nFinal Results - Random Forest: Average MSE: {avg_mse:.4f}, Average R2: {avg_r2:.4f}")

# Plot average MSE and R²
plt.figure(figsize=(6, 4))
plt.bar(["MSE", "R2"], [avg_mse, avg_r2], color=['blue', 'green'])
plt.title("Random Forest - Average MSE & R2 Across Selected Drugs")
plt.ylabel("Score")
plt.show()
