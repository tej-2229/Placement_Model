import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

file_path = "processed_student_data.xlsx"
data = pd.read_excel(file_path)

print(data.head())
print(data.info())
print(data.describe())

data['Placement Label'] = data['Placement Status'].map({0: 'Not Placed', 1: 'Placed'})

# Plotting
plt.figure(figsize=(12, 6))
sns.histplot(data['Placement Label'], kde=False)
plt.title("Placement Status Distribution")
plt.show()

numeric_data = data.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_data.corr()
print("Feature Correlation Matrix:", correlation_matrix)

if data['Placement Status'].dtype == 'object':
    data['Placement Status'] = LabelEncoder().fit_transform(data['Placement Status']) 

numeric_data = data.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_data.corr()['Placement Status'].drop('Placement Status') 
sorted_correlation = correlation_matrix.sort_values(ascending=False)
print("Sorted correlation with Placement Status:")
print(sorted_correlation)
sorted_correlation.to_pickle("placement_correlation.pkl")

skills = ['Java', 'Python', 'C++', 'ML', 'AI', 'SQL', 'Tableau', 'JavaScript', 'DSA', 'ReactJS', 'GenAI', 'MobileDev', 'WebDev', 'MongoDB']

# Calculate % of students having each skill in placed and not placed groups
skill_percentages = {}
for skill in skills:
    placed_pct = data[data['Placement Status'] == 1][skill].mean() * 100
    not_placed_pct = data[data['Placement Status'] == 0][skill].mean() * 100
    skill_percentages[skill] = {'Placed': placed_pct, 'Not Placed': not_placed_pct}

skill_df = pd.DataFrame(skill_percentages).T.reset_index().rename(columns={'index':'Skill'})


skill_df.plot(x='Skill', kind='bar', figsize=(12,6))
plt.ylabel('Percentage of Students with Skill (%)')
plt.title('Skill Prevalence by Placement Status')
plt.xticks(rotation=45)
plt.show()

data[['Verbal', 'Technical Score (out of 20)', 'Quants']].hist(bins=15, figsize=(12, 8))
plt.suptitle('Histograms of Numerical Features')
plt.show()


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='Placement Status', y='10th Marks', data=data)
plt.title('10th Marks by Placement Status')

plt.subplot(1, 2, 2)
sns.boxplot(x='Placement Status', y='12th Marks', data=data)
plt.title('12th Marks by Placement Status')

plt.tight_layout()
plt.show()


