import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Master Gene Expression + Drug.csv')
selected_df = pd.read_csv('selected.csv')

drug_list = selected_df.iloc[:, 0].tolist()
model_list = selected_df.iloc[:, 1].tolist()

drug_model_mapping = dict(zip(drug_list, model_list))

end_feature_idx = df.columns.get_loc("Erlotinib")
feature_columns = df.columns[1:end_feature_idx]
drug_columns = df.columns[end_feature_idx:] 

def train_and_predict(drug_name):
    print(f"\nTraining model for drug: {drug_name}")
    
    model_type = drug_model_mapping[drug_name]
    print(f"Model type for {drug_name}: {model_type}")
    
    df_cleaned = df.dropna(subset=[drug_name])
    
    X = df_cleaned[feature_columns]
    y = df_cleaned[drug_name]
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns)
    
    rf = RandomForestRegressor(random_state=0, n_jobs=-1)
    rf.fit(X_scaled, y)
    
    feature_importances = rf.feature_importances_
    top_n_features_idx = np.argsort(feature_importances)[-100:] 
    X_selected = X_scaled.iloc[:, top_n_features_idx] 
    
    if model_type == "Random Forest":
        model = RandomForestRegressor(random_state=0, n_jobs=-1)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=0)
    elif model_type == "AdaBoost":
        model = AdaBoostRegressor(random_state=0)
    elif model_type == "Histogram-Based Gradient Boosting":
        model = HistGradientBoostingRegressor(random_state=0)
    elif model_type == "XGBoost":
        model = XGBRegressor(objective="reg:squarederror", random_state=0, n_jobs=-1)
    elif model_type == "CatBoost":
        model = CatBoostRegressor(verbose=0, random_state=0)
    else:
        print(f"Error: Unknown model type '{model_type}' for drug '{drug_name}'.")
        return
    
    model.fit(X_selected, y)

    predictions = []

    for index, row in df.iterrows():
        input_values = row[feature_columns].values
        input_values_scaled = scaler.transform([input_values])
        
        input_values_scaled = input_values_scaled[:, top_n_features_idx]
        
        prediction = model.predict(input_values_scaled)

        if pd.notna(row[drug_name]):
            predictions.append((index, row[drug_name], prediction[0])) 
        else:
            predictions.append((index, None, prediction[0])) 

    prediction_df = pd.DataFrame(predictions, columns=['Row Index', 'True Value', 'Predicted Value'])
    prediction_df.to_csv(f'predictions_{drug_name}.csv', index=False)
    print(f"Predictions for {drug_name} saved to predictions_{drug_name}.csv")

    print(f"\nPredictions for {drug_name}:")
    for index, true_value, predicted_value in predictions:
        print(f"Row {index} (True Value: {true_value}, Predicted Value: {predicted_value})")

for drug in drug_list:
    train_and_predict(drug)
