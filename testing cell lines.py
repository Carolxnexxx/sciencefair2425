import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import numpy as np
from tqdm import tqdm
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

cell_line = input("Select a cell line: ")
cell_line_predictions = {}

df = pd.read_csv('Master Gene Expression + Drug.csv')

end_feature_idx = df.columns.get_loc("Erlotinib")
feature_columns = df.columns[1:end_feature_idx]

print(f"Number of columns used as features: {len(feature_columns)}")

drug_columns = df.columns[end_feature_idx:]

selected_df = pd.read_csv('selected.csv') 
selected_drugs = selected_df['Drug'].tolist()
selected_models = selected_df.set_index('Drug')['Model'].to_dict()

imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(df[feature_columns].astype(np.float32)), columns=feature_columns)

scaler = StandardScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X_imputed), columns=feature_columns)

models = {
    "Random Forest": RandomForestRegressor(random_state=0, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=0),
    "AdaBoost": AdaBoostRegressor(random_state=0),
    "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=0, n_jobs=-1),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=0),
    "Histogram-Based Gradient Boosting": HistGradientBoostingRegressor(random_state=0),
    "K-Nearest Neighbors": KNeighborsRegressor(n_jobs=-1),
    "Support Vector Regression": SVR(),
    "MLP Regressor": MLPRegressor(random_state=0)
}

print("Calculating feature importance for selected drugs using Random Forest...")

important_features = {}

for drug in tqdm(selected_drugs, desc="Calculating Feature Importance for Selected Drugs"):
    print(f"\nEvaluating feature importance for drug: {drug}")
    
    df_drug = df.dropna(subset=[drug])
    df_cell_line = df_drug[df_drug["Cell_Line"] == cell_line]

    if df_cell_line.empty:
        print(f"No data found for cell line {cell_line} in drug {drug}. Using full dataset for prediction.")
        df_cell_line = df_drug
    
    X = X_normalized.loc[df_drug.index]
    y = df_drug[drug].astype(np.float32)
    
    rf_model = RandomForestRegressor(random_state=0, n_jobs=-1)
    rf_model.fit(X, y)
    
    feature_importance = rf_model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    
    important_features[drug] = feature_importance_df.head(100)

    print(f"\nTraining and testing model for drug: {drug}")
    
    X = X_normalized.loc[df_drug.index]
    y = df_drug[drug].astype(np.float32)
    
    top_features = important_features[drug]['Feature'].tolist()
    X_selected = X[top_features]
    
    model_name = selected_models.get(drug, None)
    if not model_name:
        print(f"No model specified for {drug} in selected.csv. Skipping.")
        continue

    model = models.get(model_name, None)
    if not model:
        print(f"Model {model_name} is not recognized. Skipping.")
        continue
    
    print(f"  Training {model_name} model...")
    model.fit(X_selected, y)
    y_pred = model.predict(X_selected)

    cell_line_predictions[drug] = y_pred
    
    print(f"Predictions for {drug} in cell line {cell_line}: {y_pred}")
