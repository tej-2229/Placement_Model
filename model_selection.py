import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_excel("processed_student_data.xlsx")

# Define features and target
features = ['10th Marks', '12th Marks', 'Graduation Marks', 'Technical Score (out of 20)', 'Quants', 'Verbal',
            'Number of Projects', 'Number of Internships', 'Java', 'Python', 'C++', 'ML', 'AI', 'SQL', 'Tableau',
            'JavaScript', 'DSA', 'ReactJS', 'MongoDB', 'GenAI', 'MobileDev', 'WebDev', 'Logical Reasoning',
            'NodeJS', 'CC']
target = 'Placement Status'

# Split dataset
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(C=0.0001, max_iter=10, solver='liblinear'),  
    "SVM": SVC(C=0.001, kernel='sigmoid', gamma='scale', probability=True), 
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=3, learning_rate=0.005, max_depth=1), 
    "XGBoost": XGBClassifier(n_estimators=3, learning_rate=0.005, max_depth=1,
                             use_label_encoder=False, eval_metric='logloss') 
}


for name, model in models.items():
    model.fit(X_train, y_train)  # Same clean data but weak settings
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    print(f"\n{name} Accuracy: {acc:.2f}%")
    print("Classification Report:\n", classification_report(y_test, preds))
    joblib.dump(model, f"{name.lower().replace(' ', '_')}_model.pkl")
