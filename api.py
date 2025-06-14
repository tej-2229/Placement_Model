from flask import Flask, request, jsonify
import pandas as pd
import joblib
from collections import Counter
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("placement_model.pkl")
selected_features = joblib.load("selected_features.pkl") 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)

        tech_skills = data.get('technicalSkills', '').lower()

        ml_input = {
            '10th Marks': data.get('10thMarks', 0),
            '12th Marks': data.get('12thMarks', 0),
            'Graduation Marks': data.get('GraduationMarks', 0),
            'Technical Score (out of 20)': data.get('TechnicalScore', 15),
            'Quants': data.get('Quants', 15),
            'Verbal': data.get('Verbal', 15),
            'Number of Projects': data.get('projects', 0),
            'Number of Internships': data.get('internships', 0),
            'Java': 1 if 'java' in tech_skills else 0,
            'Python': 1 if 'python' in tech_skills else 0,
            'C++': 1 if 'c++' in tech_skills else 0,
            'ML': 1 if 'ml' in tech_skills or 'machine learning' in tech_skills else 0,
            'AI': 1 if 'ai' in tech_skills or 'artificial intelligence' in tech_skills else 0,
            'SQL': 1 if 'sql' in tech_skills else 0,
            'Tableau': 1 if 'tableau' in tech_skills else 0,
            'JavaScript': 1 if 'javascript' in tech_skills or 'js' in tech_skills else 0,
            'DSA': 1 if 'dsa' in tech_skills or 'data structure' in tech_skills else 0,
            'ReactJS': 1 if 'react' in tech_skills else 0,
            'MongoDB': 1 if 'mongodb' in tech_skills else 0,
            'GenAI': 1 if 'genai' in tech_skills or 'generative ai' in tech_skills else 0,
            'MobileDev': 1 if 'mobile' in tech_skills else 0,
            'WebDev': 1 if 'web' in tech_skills else 0,
        }

        for feature in selected_features:
            if feature not in ml_input:
                ml_input[feature] = 0

        input_data = pd.DataFrame([ml_input])[selected_features]
        
        input_data = pd.DataFrame([ml_input])
        print("DataFrame shape:", input_data.shape)
        print("DataFrame columns:", input_data.columns.tolist())

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100

        print(f"\nRaw prediction: {prediction}")      
        print(f"Raw probability: {probability}")     
        
        suggestions = generate_suggestions(ml_input)

        
        return jsonify({
            'placement_status': int(prediction),
            'probability': float(probability),
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


