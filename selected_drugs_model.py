import pandas as pd
from sklearn.model_selection import train_test_split
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
from sklearn.isotonic import IsotonicRegression

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

os.makedirs("saved_models", exist_ok=True)

# Load the dataset
df = pd.read_csv('Master Gene Expression + Drug.csv')
selected_df = pd.read_csv('selected.csv')

drug_list = selected_df.iloc[:, 0].tolist()
model_list = selected_df.iloc[:, 1].tolist()

drug_model_mapping = dict(zip(drug_list, model_list))

# Auto-detect the end of feature columns
end_feature_idx = df.columns.get_loc("Erlotinib")
feature_columns = df.columns[1:end_feature_idx]
drug_columns = df.columns[end_feature_idx:]  # Get all drug columns

matching_drugs = [drug for drug in drug_columns if drug in drug_list]

# Print the number of columns being used as features
print(f"Number of columns used as features: {len(feature_columns)}")
print(f"Number of matching drugs found: {len(matching_drugs)}")

# Remove rows with missing target values for each drug individually

drug_name = 'Ara-G'
df_cleaned = df.dropna(subset=[drug_name])  # Ensure no missing values for this drug column

# Separate features (X) and target variable (y)
X = df_cleaned[feature_columns]  # All the feature columns
y = df_cleaned[drug_name]

rf = RandomForestRegressor(random_state=0, n_jobs=-1)
rf.fit(X, y)

feature_importances = rf.feature_importances_
top_n_features_idx = np.argsort(feature_importances)[-100:]  # Select top 100 features

X_selected = X.iloc[:, top_n_features_idx]

# Apply Z-score normalization (standardization)
scaler = StandardScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X_selected), columns=feature_columns[top_n_features_idx])

joblib.dump(scaler, 'scaler.joblib')

# Reset index for both X_normalized and df_cleaned
X_normalized = X_normalized.reset_index(drop=True)
df_cleaned = df_cleaned.reset_index(drop=True)

# Define the regression models to compare
available_models = {
    "Random Forest": RandomForestRegressor(random_state=0, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=0),
    "AdaBoost": AdaBoostRegressor(random_state=0),
    "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=0, n_jobs=-1),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=0),
    "Histogram-Based Gradient Boosting": HistGradientBoostingRegressor(random_state=0),
    "K-Nearest Neighbors": KNeighborsRegressor(n_jobs=-1),
    "Support Vector Regression": SVR()
}

average_results = {model_name: {"MSE": [], "R2": []} for model_name in available_models.keys()}

# Dictionary to store results for each drug
all_results = {}
csv_data = []  # List to accumulate rows for CSV output

# List to accumulate raw predictions for each drug
raw_predictions_data = []

# Iterate over each drug column with a progress bar
for drug in tqdm(matching_drugs, desc="Processing Selected Drugs"):
    print(f"\nEvaluating models for drug: {drug}")
    
    # Drop rows where the target variable is missing for this specific drug
    df_drug = df_cleaned.dropna(subset=[drug])  # Only the rows with valid target data for this drug
    
    # Ensure alignment of indices between X_normalized and df_drug
    X_drug = X_normalized.loc[df_drug.index]  # Align X with the rows in df_drug
    y_drug = df_drug[drug].astype(np.float32)
    
    # Feature Selection using RandomForestRegressor to determine important features for this drug
    rf = RandomForestRegressor(random_state=0, n_jobs=-1)
    rf.fit(X_drug, y_drug)
    
    # Get feature importances and select top N features (e.g., top 100 features)
    feature_importances = rf.feature_importances_
    top_n_features_idx = np.argsort(feature_importances)[-100:]  # Select top 100 features
    X_drug_selected = X_drug.iloc[:, top_n_features_idx]  # Select only top N features
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_drug_selected, y_drug, random_state=11, test_size=0.2)
    
    # Dictionary to store results for this drug
    results = {}

    model_name = drug_model_mapping[drug]
    model = available_models[model_name]
    print(f"    Using model: {model_name}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Save raw predictions for this drug and model
    for i in range(len(y_pred)):
        raw_predictions_data.append({"Drug": drug, "Model": model_name, "True Value": y_test.iloc[i], "Predicted Value": y_pred[i]})

    model_filename = f"saved_models/{drug}_{model_name}.joblib"
    joblib.dump(model, model_filename)
    print(f"    Model for drug {drug} saved as {model_filename}")
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    explained_var = explained_variance_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
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

    all_results[drug] = results

# Save the raw predictions to a CSV file
raw_predictions_df = pd.DataFrame(raw_predictions_data)
raw_predictions_df.to_csv("raw_predictions.csv", index=False)
print("\nRaw predictions saved to raw_predictions.csv")

# Plotting the raw predictions for each drug and model
for drug in tqdm(matching_drugs, desc="Plotting Raw Predictions"):
    drug_predictions = raw_predictions_df[raw_predictions_df['Drug'] == drug]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(drug_predictions['True Value'], drug_predictions['Predicted Value'], color='black', alpha=0.5)
    plt.title(f'Raw Predictions for {drug}')
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.plot([drug_predictions['True Value'].min(), drug_predictions['True Value'].max()],
             [drug_predictions['True Value'].min(), drug_predictions['True Value'].max()], color='red', linestyle='--')  # Line of perfect prediction
    plt.show()

# Calculate average MSE and R² for each model across all drugs
final_results = {}
for model_name, metrics in average_results.items():
    avg_mse = np.mean(metrics["MSE"])
    avg_r2 = np.mean(metrics["R2"])
    final_results[model_name] = {"Average MSE": avg_mse, "Average R2": avg_r2}
    print(f"\nFinal Results for {model_name} - Average MSE: {avg_mse:.4f}, Average R2: {avg_r2:.4f}")

csv_df = pd.DataFrame(csv_data)
csv_df.to_csv("selected_results.csv", index=False)
print("\nR² results saved to selected_results.csv")

final_results_df = pd.DataFrame(final_results).T

final_results_df["Average MSE"].plot(kind='bar', title='Average MSE Across All Drugs for Each Model', ylabel='Average MSE')
plt.show()

final_results_df["Average R2"].plot(kind='bar', title='Average R2 Score Across All Drugs for Each Model', ylabel='Average R-squared')
plt.show()
