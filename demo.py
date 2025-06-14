import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold

# Load dataset
data = pd.read_excel("processed_student_data.xlsx")

# Load selected features
selected_features = pd.read_pickle("selected_features.pkl").tolist()

# Feature and target
X = data[selected_features]
y = data['Placement Status']

# Feature selection (optional: reduce correlated or low-variance features)
# Uncomment below to automatically remove low-variance features
# selector = VarianceThreshold(threshold=0.01)
# X = selector.fit_transform(X)

# Train-test split (stratified for class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define pipeline with SMOTE and Random Forest (limited depth)
pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42))
])

# Cross-validation to evaluate model performance
scores = cross_val_score(pipeline, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Fit the pipeline to training data
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Calibrate model for probability output
model_only = pipeline.named_steps['model']
smote_X, smote_y = SMOTE(random_state=42).fit_resample(X_train, y_train)
calibrated_model = CalibratedClassifierCV(model_only, method='sigmoid', cv='prefit')
calibrated_model.fit(smote_X, smote_y)

# Save models
joblib.dump(pipeline, "placement_model_pipeline.pkl")
joblib.dump(calibrated_model, "calibrated_placement_model.pkl")

# ---------- Test Prediction ----------
# Input test student (ensure same order as selected_features)
test_student = pd.DataFrame([{
    "Technical Score (out of 20)": 15,
    "Verbal": 16,
    "Quants": 18,
    "10th Marks": 95,
    "12th Marks": 78,
    "Graduation Marks": 80.0,
    "Number of Projects": 3,
    "Number of Internships": 2,
    "SQL": 1,
    "DSA": 1,
    "Java": 1,
    "Python": 1,
    "C++": 1,
    "JavaScript": 0,
    "ML": 1,
    "ReactJS": 0,
    "Tableau": 0,
    "AI": 1,
    "GenAI": 0,
    "MobileDev": 0,
    "WebDev": 1,
    "MongoDB": 1
}])

test_student = test_student[selected_features]  # ensure same order

# Predict placement probability
probability = calibrated_model.predict_proba(test_student)[:, 1][0] * 100
prediction = calibrated_model.predict(test_student)[0]
placement_status = "Placed" if prediction == 1 else "Not Placed"

print(f"\nPrediction: {placement_status}")
print(f"Probability of placement: {probability:.2f}%")

# ---------- Suggestions ----------
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

student_info = test_student.iloc[0].to_dict()
student_info['Placement Status'] = prediction
print("\nSuggestions:")
for s in generate_suggestions(student_info):
    print("-", s)
