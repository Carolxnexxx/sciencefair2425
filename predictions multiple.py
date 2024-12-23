import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('Master Gene Expression + Drug.csv')
selected_df = pd.read_csv('selected.csv')

# Extract the drug names from the first column and model types from the second column of selected.csv
drug_list = selected_df.iloc[:, 0].tolist()  # List of drugs
model_list = selected_df.iloc[:, 1].tolist()  # List of models corresponding to the drugs

# Create a mapping from drug name to model type
drug_model_mapping = dict(zip(drug_list, model_list))

# Auto-detect the end of feature columns
end_feature_idx = df.columns.get_loc("Erlotinib")
feature_columns = df.columns[1:end_feature_idx]
drug_columns = df.columns[end_feature_idx:]  # Get all drug columns

# Function to train the model and make predictions
def train_and_predict(drug_name):
    print(f"\nTraining model for drug: {drug_name}")
    
    # Get the model type for the current drug from the drug_model_mapping
    model_type = drug_model_mapping[drug_name]
    print(f"Model type for {drug_name}: {model_type}")
    
    # Filter rows where target values are not missing for training
    df_cleaned = df.dropna(subset=[drug_name])
    
    # Separate features and target variable
    X = df_cleaned[feature_columns]
    y = df_cleaned[drug_name]
    
    # Apply Z-score normalization (scaling)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns)
    
    # Select top 100 features using RandomForestRegressor
    rf = RandomForestRegressor(random_state=0, n_jobs=-1)
    rf.fit(X_scaled, y)
    
    feature_importances = rf.feature_importances_
    top_n_features_idx = np.argsort(feature_importances)[-100:]  # Select top 100 features
    X_selected = X_scaled.iloc[:, top_n_features_idx]  # Select only top N features
    
    # Dynamically select the model based on the model type in `model_type`
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
    
    # Train the selected model
    model.fit(X_selected, y)

    # List to store predictions (true value, predicted value, and row index)
    predictions = []

    # Iterate over all rows in the dataset (including NaN values)
    for index, row in df.iterrows():
        # Get the input features for the row
        input_values = row[feature_columns].values
        # Apply the same scaling to the input data as done during training
        input_values_scaled = scaler.transform([input_values])
        
        # Select only the top 100 features from the input values
        input_values_scaled = input_values_scaled[:, top_n_features_idx]
        
        # Make the prediction
        prediction = model.predict(input_values_scaled)

        # If the drug value is NaN, store the prediction but no true value
        if pd.notna(row[drug_name]):
            predictions.append((index, row[drug_name], prediction[0]))  # Store the row index, true value, and predicted value
        else:
            predictions.append((index, None, prediction[0]))  # Store the row index, NaN true value, and predicted value

    # Save predictions to a CSV file
    prediction_df = pd.DataFrame(predictions, columns=['Row Index', 'True Value', 'Predicted Value'])
    prediction_df.to_csv(f'predictions_{drug_name}.csv', index=False)
    print(f"Predictions for {drug_name} saved to predictions_{drug_name}.csv")

    # Optionally print the predictions
    print(f"\nPredictions for {drug_name}:")
    for index, true_value, predicted_value in predictions:
        print(f"Row {index} (True Value: {true_value}, Predicted Value: {predicted_value})")

# Loop through all the drugs and make predictions for each
for drug in drug_list:
    train_and_predict(drug)
