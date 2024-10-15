import pandas as pd
from tqdm import tqdm

master_df = pd.read_csv('Master.csv')
protein_df = pd.read_csv('Protein Data.csv')

protein_df.set_index('Gene_Symbol', inplace=True)

for index, row in tqdm(master_df.iterrows(), total=master_df.shape[0], desc="Updating Master DataFrame"):
    # Get the current cell line (column A in Excel)
    cell_line = row['ccl_name']  # Assuming 'ccl_name' is the equivalent of column A in Excel

    # Check if this cell line exists as a column in protein_df
    if cell_line in protein_df.columns:
        # Iterate through all the empty columns in the master row (ignoring 'ccl_name')
        for col in master_df.columns[1:]:

            if pd.isna(row[col]):  # Only update if the value is NaN (empty in Excel)
                # Check if the column name (e.g., protein/gene symbol) exists in protein_df
                if col in protein_df.index:
                    # Retrieve the scalar value from protein_df
                    value = protein_df.at[col, cell_line]

                    # Handle cases where value is a Series
                    if isinstance(value, pd.Series):
                        if value.size == 1:
                            value = value.item()  # Extract scalar if Series has only one element
                        else:
                            # Skip if the value is ambiguous (contains more than one item)
                            continue
                    
                    # Ensure we are assigning a scalar and that the value is not NaN
                    if pd.notna(value):
                        master_df.at[index, col] = value
                

master_df.to_csv('Updated_Master.csv', index=False)

print("Master file updated successfully.")

'''

df1 = pd.read_csv('')
df2 = pd.read_csv('')

df1.head()
df2.head()

X = df1.ilock[:,0:13] # the factors creating prediction
y = df2.ilock[:,13] # what i want to predict / outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 11, test_size = 0.2)

scaler = MinMaxScaler(feature_range = (0,1))

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

knn = KNeighborsClassifier(n_neigbors = 8)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(y_pred)

knn.score(X_test, y_test) # accuracy score

cm = confusion_matrix(y_test, y_pred)
print(cm) # top left: true positive, bottom right: true negative, top right: false positive, bottom left: false negative

cr = classification_report(y_test, y_pred)
print(cr)
'''