import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, median_absolute_error
import joblib
from tqdm import tqdm
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# DONE

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Load the dataset
df = pd.read_csv('Master Metabolome + Drug.csv')
print(df)

# Auto-detect the end of feature columns
end_feature_idx = df.columns.get_loc("Erlotinib")
feature_columns = df.columns[1:end_feature_idx]

# Print the number of columns being used as features
print(f"Number of columns used as features: {len(feature_columns)}")

# List of drugs (target columns) starting from "Erlotinib" onward
drug_columns = df.columns[end_feature_idx:]

# Impute missing values in X if necessary
imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(df[feature_columns].astype(np.float32)), columns=feature_columns)

# Apply Z-score normalization
scaler = StandardScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X_imputed), columns=feature_columns)

# Define the regression models to compare
models = {
    "Random Forest": RandomForestRegressor(random_state=0, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=0),
    "AdaBoost": AdaBoostRegressor(random_state=0),
    "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=0, n_jobs=-1),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=0),
    "Histogram-Based Gradient Boosting": HistGradientBoostingRegressor(random_state=0),
    "K-Nearest Neighbors": KNeighborsRegressor(n_jobs=-1),
    "Support Vector Regression": SVR()
    }

# Dictionary to store results for each drug
all_results = {}
average_results = {model_name: {"MSE": [], "R2": []} for model_name in models.keys()}
csv_data = []  # List to accumulate rows for CSV output

# Iterate over each drug column with a progress bar
for drug in tqdm(drug_columns, desc="Processing Drugs"):
    print(f"\nEvaluating models for drug: {drug}")
    
    # Drop rows where the target variable is missing
    df_drug = df.dropna(subset=[drug])
    
    # Separate features (X) and target variable (y)
    X = X_normalized.loc[df_drug.index]  # Align X with the rows in df_drug
    y = df_drug[drug].astype(np.float32)
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)
    
    # Dictionary to store results for this drug
    results = {}

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"  Training {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        explained_var = explained_variance_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # MAPE in percentage
        std_dev_error = np.std(y_test - y_pred)
        median_ae = median_absolute_error(y_test, y_pred)
        
        # Store metrics for this model and drug
        results[model_name] = {
            "MSE": mse,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "Explained Variance": explained_var,
            "MAPE": mape,
            "Error Std Dev": std_dev_error,
            "Median AE": median_ae
        }
                
        # Store metrics for averaging
        average_results[model_name]["MSE"].append(mse)
        average_results[model_name]["R2"].append(r2)
        
        # Append to CSV data list
        csv_data.append({
            "Algorithm": model_name,
            "Drug": drug,
            "R2": r2,
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "Explained Variance": explained_var,
            "MAPE": mape,
            "Error Std Dev": std_dev_error,
            "Median AE": median_ae
        })
        
        print(f"    {model_name} - MSE: {mse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, Explained Var: {explained_var:.4f}, MAPE: {mape:.2f}%, Std Dev Error: {std_dev_error:.4f}, Median AE: {median_ae:.4f}")

    # Store the results for this drug
    all_results[drug] = results

# Calculate average MSE and R² for each model across all drugs
final_results = {}
for model_name, metrics in average_results.items():
    avg_mse = np.mean(metrics["MSE"])
    avg_r2 = np.mean(metrics["R2"])
    final_results[model_name] = {"Average MSE": avg_mse, "Average R2": avg_r2}
    print(f"\nFinal Results for {model_name} - Average MSE: {avg_mse:.4f}, Average R2: {avg_r2:.4f}")

# Convert csv_data to a DataFrame and save as CSV
csv_df = pd.DataFrame(csv_data)
csv_df.to_csv("metabolome_results.csv", index=False)
print("\nR² results saved to metabolome_results.csv")

# Convert final_results to a DataFrame for easy plotting
final_results_df = pd.DataFrame(final_results).T

# Plot Average MSE for each model
final_results_df["Average MSE"].plot(kind='bar', title='Average MSE Across All Drugs for Each Model', ylabel='Average MSE')
plt.show()

# Plot Average R2 for each model
final_results_df["Average R2"].plot(kind='bar', title='Average R2 Score Across All Drugs for Each Model', ylabel='Average R-squared')
plt.show()
