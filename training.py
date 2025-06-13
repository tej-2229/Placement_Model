import pandas as pd # type: ignore
import joblib # type: ignore
from collections import Counter
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore
from imblearn.over_sampling import SMOTE # type: ignore
from sklearn.calibration import CalibratedClassifierCV # type: ignore

# Load the processed data
data = pd.read_excel("processed_student_data.xlsx")

selected_features = pd.read_pickle("selected_features.pkl").tolist()

X = data[selected_features]
y = data['Placement Status']

# Split data into training and testing sets (stratified for class balance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train RandomForest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Save the model
joblib.dump(model, "placement_model.pkl")

# Model Calibration
calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")

print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the calibrated model
joblib.dump(calibrated_model, "calibrated_placement_model.pkl")


# Function to generate suggestions
def generate_suggestions(student_data):
    suggestions = []


    if student_data['Graduation Marks'] < 70:
        suggestions.append("Work on improving your graduation scores.")
    if student_data['Technical Score (out of 20)'] < 16:
        suggestions.append("Enhance your technical skills by practicing coding challenges.")
    if student_data['Quants'] < 15:
        suggestions.append("Improve your quantitative aptitude.")
    if student_data['Verbal'] < 15:
        suggestions.append("Develop stronger verbal communication skills.")
    if student_data['Number of Projects'] < 2:
        suggestions.append("Work on more real-world projects to boost your profile.")
    if student_data['Number of Internships'] < 1:
        suggestions.append("Gain industry exposure through internships.")
    if student_data['Java'] == 0:
        suggestions.append("Learn Java to enhance your programming skillset.")
    if student_data['Python'] == 0:
        suggestions.append("Master Python for better job prospects.")
    if student_data['C++'] == 0:
        suggestions.append("Consider learning C++ for problem-solving skills.")
    if student_data['ML'] == 0:
        suggestions.append("Explore Machine Learning concepts.")
    if student_data['AI'] == 0:
        suggestions.append("Get familiar with Artificial Intelligence.")
    if student_data['SQL'] == 0:
        suggestions.append("Learn SQL for database management.")
    if student_data['Tableau'] == 0:
        suggestions.append("Improve your data visualization skills with Tableau.")
    if student_data['ReactJS'] == 0:
        suggestions.append("Develop expertise in ReactJS for front-end development.")
    if student_data['MongoDB'] == 0:
        suggestions.append("Learn MongoDB for NoSQL database knowledge.")
    if student_data['GenAI'] == 0:
        suggestions.append("Understand Generative AI trends.")
    if student_data['MobileDev'] == 0:
        suggestions.append("Expand into mobile app development.")
    if student_data['WebDev'] == 0:
        suggestions.append("Strengthen your web development skills.")
    
    return suggestions

test_student = pd.DataFrame({
    '10th Marks': [95],
    '12th Marks': [78],
    'Graduation Marks': [80.0],
    'Technical Score (out of 20)': [15],
    'Quants': [18],
    'Verbal': [16],
    'Number of Projects': [3],
    'Number of Internships': [2],
    'JavaScript': [0],
    'ML': [1],
    'AI': [1],
    'SQL': [1],
    'Python': [1],
    'Tableau': [0],
    'DSA': [1],
    'ReactJS': [0],
    'NodeJS': [0],            # moved here per order
    'Java': [1],
    'C++': [1],
    'CC': [0],
    'Logical Reasoning': [0],
    'GenAI': [0],
    'MobileDev': [0],
    'WebDev': [1],
    'MongoDB': [1]
})

test_student = test_student[selected_features]
# Load model and predict
model = joblib.load("placement_model.pkl")
prediction = model.predict(test_student)
probability = model.predict_proba(test_student)[:, 1][0] * 100

# Output result
#placement_status = "Placed" if prediction[0] == 1 else "Not Placed"
#print(f"Predicted Placement Status: {placement_status}")
print(f"You will be placed with a probability of {probability:.2f}%")

# Generate improvement suggestions
student_info = test_student.iloc[0].to_dict()
student_info['Placement Status'] = prediction[0]
suggestions = generate_suggestions(student_info)

print("Suggestions:")
for suggestion in suggestions:
    print(f"- {suggestion}")
