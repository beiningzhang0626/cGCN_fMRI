import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the CSV file
file_path = './HCP.csv'
data = pd.read_csv(file_path)

# Filter columns ending with '_AgeAdj'
age_adj_columns = [col for col in data.columns if col == "Subject" or col.endswith('_AgeAdj')]

# Create a new DataFrame with the filtered columns
filtered_data = data[age_adj_columns]

# Save the filtered data to a new CSV file
filtered_file_path = './HCP_filtered_AgeAdj.csv'
filtered_data.to_csv(filtered_file_path, index=False)

data_filtered = pd.read_csv('./HCP_filtered_AgeAdj.csv')

label_column = 'Subject'  
# Features and labels
X = data_filtered.drop(columns=[label_column]).select_dtypes(include=[np.number]).values  # Only numeric columns for features
y = np.arange(len(data_filtered))

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode labels if they are categorical
if y.dtype == object or y.dtype.name == 'category':
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

# Split the data into train, validation, and test sets (70%-20%-10%)
x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.33, random_state=42)

# Save to HDF5
h5_file = 'HCP.h5'
with h5py.File(h5_file, 'w') as f:
    f.create_dataset('x_train', data=x_train)
    f.create_dataset('x_val', data=x_val)
    f.create_dataset('x_test', data=x_test)
    f.create_dataset('y_train', data=y_train)
    f.create_dataset('y_val', data=y_val)
    f.create_dataset('y_test', data=y_test)

print(f"HDF5 file '{h5_file}' created successfully with the required structure.")
