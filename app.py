import sys, os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

# Load model and metadata
BASE = os.path.dirname(os.path.abspath(__file__))
clf = joblib.load(os.path.join(BASE, 'models', 'classifier.pkl'))
with open(os.path.join(BASE, 'models', 'symptoms.json')) as f:
    ALL_SYMPTOMS = json.load(f)
with open(os.path.join(BASE, 'models', 'disease_info.json')) as f:
    DISEASE_INFO = json.load(f)

SEVERITY_ORDER = {'Emergency': 5, 'High': 4, 'Chronic': 3, 'Moderate': 2, 'Mild': 1}

@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/app')
def dashboard():
    return render_template('app.html')

@app.route('/api/symptoms')
def get_symptoms():
    return jsonify({'symptoms': ALL_SYMPTOMS})

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    selected = data.get('symptoms', [])
    
    if not selected:
        return jsonify({'error': 'No symptoms selected'}), 400
    if len(selected) < 2:
        return jsonify({'error': 'Please select at least 2 symptoms for accurate prediction'}), 400

    # Build feature vector
    vec = [0] * len(ALL_SYMPTOMS)
    for s in selected:
        if s in ALL_SYMPTOMS:
            vec[ALL_SYMPTOMS.index(s)] = 1

    X = np.array([vec])
    proba = clf.predict_proba(X)[0]
    classes = clf.classes_

    # Top 3 predictions
    top_idx = np.argsort(proba)[::-1][:5]
    results = []
    for i in top_idx:
        disease = classes[i]
        prob = float(proba[i])
        if prob < 0.01:
            continue
        info = DISEASE_INFO.get(disease, {})
        matched = [s for s in selected if s in info.get('symptoms', [])]
        results.append({
            'disease': disease,
            'probability': round(prob * 100, 1),
            'description': info.get('description', ''),
            'severity': info.get('severity', 'Unknown'),
            'action': info.get('action', ''),
            'matched_symptoms': matched,
            'total_symptoms': len(info.get('symptoms', [])),
        })

    if not results:
        return jsonify({'error': 'Could not determine a diagnosis'}), 400

    return jsonify({
        'predictions': results[:3],
        'selected_count': len(selected),
        'disclaimer': 'This is an AI-based tool for informational purposes only. Always consult a qualified healthcare professional for medical advice.'
    })

@app.route('/api/diseases')
def get_diseases():
    diseases = []
    for name, info in DISEASE_INFO.items():
        diseases.append({
            'name': name,
            'severity': info.get('severity'),
            'symptom_count': len(info.get('symptoms', [])),
            'description': info.get('description', '')
        })
    diseases.sort(key=lambda x: SEVERITY_ORDER.get(x['severity'], 0), reverse=True)
    return jsonify({'diseases': diseases, 'total': len(diseases)})

if __name__ == '__main__':
    print("=" * 60)
    print("  Disease Classifier — Running at http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, port=5000)
