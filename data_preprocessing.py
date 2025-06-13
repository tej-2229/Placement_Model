import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load Excel file
file_path = "Student_Placement_Data.xlsx"
data = pd.read_excel(file_path)

print(data.head())
print(data.info())
print(data.describe())

print(data[['Placement Status', 'Name of Company']].head(30))  

print("\nMissing Values:")
print(data.isnull().sum())

data['CTC'].replace(r'^\s*$', np.nan, regex=True, inplace=True)
data['CTC'].fillna(0, inplace=True)

missing_value_replacement = 'Not Placed'  

data['Name of Company'].fillna(missing_value_replacement, inplace=True)
data['Nature of Job'].fillna(missing_value_replacement, inplace=True)
data['Comapny Domain'].fillna(missing_value_replacement, inplace=True)
data['Industry Location'].fillna(missing_value_replacement, inplace=True)
data['Department Alocated'].fillna(missing_value_replacement, inplace=True)
data['Job Role'].fillna(missing_value_replacement, inplace=True)
data['Type of Offer'].fillna(missing_value_replacement, inplace=True)

print("\nMissing Values:")
print(data.isnull().sum())

print(data.columns)

data = data.drop(columns=['High School Marks'])
data = data.drop(columns=['Technical Score'])

print(data.columns)

data['Placement Status'] = LabelEncoder().fit_transform(data['Placement Status'])  
print(data['Placement Status'].head())

placement_counts = data['Placement Status'].value_counts()

print(placement_counts)

data.to_excel("processed_student_data.xlsx", index=False)