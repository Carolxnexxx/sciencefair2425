import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import numpy as np
from tqdm import tqdm
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Load dataset
df = pd.read_csv('Master Gene Expression + Drug.csv')
selected_df = pd.read_csv('selected2.csv')

drug_list = selected_df.iloc[:, 0].tolist()
model_list = selected_df.iloc[:, 1].tolist()
drug_model_mapping = dict(zip(drug_list, model_list))

# Find feature columns
end_feature_idx = df.columns.get_loc("Erlotinib")
feature_columns = df.columns[1:end_feature_idx]
drug_columns = df.columns[end_feature_idx:]

matching_drugs = [drug for drug in drug_columns if drug in drug_list]

print(f"Number of columns used as features: {len(feature_columns)}")
print(f"Number of matching drugs found: {len(matching_drugs)}")

cell_line = input("Select a cell line: ")

# Separate features and target variable
df_cleaned = df.dropna(subset=matching_drugs)
X = df_cleaned[feature_columns]  

# Standardization
scaler = StandardScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns)

X_normalized = X_normalized.reset_index(drop=True)
df_cleaned = df_cleaned.reset_index(drop=True)

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

raw_predictions_data = []

for drug in tqdm(matching_drugs, desc="Processing Selected Drugs"):
    print(f"\nEvaluating models for drug: {drug}")
    
    df_drug = df_cleaned.dropna(subset=[drug])
    X_drug = X_normalized.loc[df_drug.index] 
    y_drug = df_drug[drug].astype(np.float32)
    
    # Feature selection
    rf = RandomForestRegressor(random_state=0, n_jobs=-1)
    rf.fit(X_drug, y_drug)
    feature_importances = rf.feature_importances_
    top_n_features_idx = np.argsort(feature_importances)[-100:] 
    X_drug_selected = X_drug.iloc[:, top_n_features_idx] 

    # Define train and test sets
    X_test = X_drug_selected[df_drug["Cell_Line"] == cell_line]
    y_test = y_drug[df_drug["Cell_Line"] == cell_line]

    X_train = X_drug_selected[df_drug["Cell_Line"] != cell_line]
    y_train = y_drug[df_drug["Cell_Line"] != cell_line]

    # Identify the model to use for drugs
    model_name = drug_model_mapping[drug]
    model = available_models[model_name]
    print(f"    Using model: {model_name}")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)[0]

    raw_predictions_data.append({
        "Cell Line": cell_line,
        "Drug": drug,
        "Model": model_name,
        "True Value": y_test.values[0], 
        "Predicted Value": y_pred
    })

    print(f"    Prediction for {drug}: {y_pred}")

# Print in order from highest to lowest sensitivty
raw_predictions_data.sort(key=lambda x: x["Predicted Value"], reverse=True)

print("Highest Sensitivity:")
for entry in reversed(raw_predictions_data):
    print(f"Cell Line: {entry['Cell Line']}, Drug: {entry['Drug']}, Predicted Sensitivity: {entry['Predicted Value']}")
