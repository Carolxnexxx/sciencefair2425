import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars

# Load the CSV file into a DataFrame
master_df = pd.read_csv('Updated_Master.csv')

# Define the threshold for deleting rows/columns (90% missing data)
row_threshold = 0.9 * master_df.shape[1]  # 90% of the columns
col_threshold = 0.9 * master_df.shape[0]  # 90% of the rows

# Create progress bar for row checking (cell lines)
print("Checking for rows (cell lines) with more than 90% missing data...")
for index, row in tqdm(master_df.iterrows(), total=master_df.shape[0], desc="Rows Progress"):
    if row.isna().sum() > row_threshold:
        master_df.drop(index, inplace=True)

# Reset the index after dropping rows
master_df.reset_index(drop=True, inplace=True)

# Create progress bar for column checking (proteins)
print("Checking for columns (proteins) with more than 90% missing data...")
for col in tqdm(master_df.columns, total=master_df.shape[1], desc="Columns Progress"):
    if master_df[col].isna().sum() > col_threshold:
        master_df.drop(col, axis=1, inplace=True)

# Save the cleaned DataFrame back to a CSV file
master_df.to_csv('Cleaned_Master.csv', index=False)

print("Rows and columns with more than 90% missing data have been deleted.")
