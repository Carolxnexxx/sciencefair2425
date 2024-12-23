import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

# Function to train the model and make predictions
def train_and_predict(drug_name):
    print(f"\nTraining model for drug: {drug_name}")
    
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
    
    # Train the XGBoost model
    model = XGBRegressor(objective="reg:squarederror", random_state=0, n_jobs=-1)
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

# Example usage: Predict for all rows for drug 'Ara-G'
train_and_predict('Ara-G')
