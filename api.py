from flask import Flask, request, jsonify
import pandas as pd
import joblib
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load trained model and selected feature order
model = joblib.load("calibrated_placement_model.pkl")  
with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)  # ['Technical Score (out of 20)', 'Verbal', 'Quants', ...]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)

        tech_skills = data.get('technicalSkills', '').lower()  # Optional string input

        ml_input = {
            '10th Marks': data.get('10th Marks', 0),
            '12th Marks': data.get('12th Marks', 0),
            'Graduation Marks': data.get('Graduation Marks', 0),
            'Technical Score (out of 20)': data.get('Technical Score (out of 20)', 0),
            'Quants': data.get('Quants', 0),
            'Verbal': data.get('Verbal', 0),
            'Number of Projects': data.get('Number of Projects', 0),
            'Number of Internships': data.get('Number of Internships', 0),
            'Java': data.get('Java', 0),
            'Python': data.get('Python', 0),
            'C++': data.get('C++', 0),
            'ML': data.get('ML', 0),
            'AI': data.get('AI', 0),
            'SQL': data.get('SQL', 0),
            'Tableau': data.get('Tableau', 0),
            'JavaScript': data.get('JavaScript', 0),
            'DSA': data.get('DSA', 0),
            'ReactJS': data.get('ReactJS', 0),
            'MongoDB': data.get('MongoDB', 0),
            'GenAI': data.get('GenAI', 0),
            'MobileDev': data.get('MobileDev', 0),
            'WebDev': data.get('WebDev', 0),
        }

        # Convert to DataFrame and reorder columns as per training
        input_df = pd.DataFrame([ml_input])
        input_df = input_df[selected_features]  # This ensures order and feature match

        print("Input features:", list(input_df.columns))
        print("Input values:", input_df.values.tolist())

        # Check for NaN values
        if input_df.isnull().any().any():
            return jsonify({'error': 'Missing values in input'}), 400

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100

        suggestions = generate_suggestions(ml_input)

        return jsonify({
            'placement_status': int(prediction),
            'probability': round(float(probability), 2),
            'suggestions': suggestions
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': str(e)}), 400


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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
