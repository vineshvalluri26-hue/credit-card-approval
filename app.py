import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for

app = Flask(__name__)

# Paths to models
MODEL_PATH = os.path.join('models', 'best_model.pkl')
META_PATH = os.path.join('models', 'best_model_meta.pkl')
ENCODERS_PATH = os.path.join('models', 'label_encoders.pkl')
DB_PATH = 'predictions.db'

# Load models and encoders globally
best_model = joblib.load(MODEL_PATH)
model_meta = joblib.load(META_PATH)
label_encoders = joblib.load(ENCODERS_PATH)

# Feature list required by the model in exact order
REQUIRED_FEATURES = model_meta['features']

def init_db():
    """Initializes the SQLite database and creates the table if not exists."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            gender TEXT,
            age INTEGER,
            education TEXT,
            income REAL,
            employment_years REAL,
            history_months INTEGER,
            prediction_label TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

def get_age_group(age):
    """Bins age into standard groups used in training."""
    if age <= 25:
        return '18-25'
    elif age <= 35:
        return '26-35'
    elif age <= 45:
        return '36-45'
    elif age <= 55:
        return '46-55'
    elif age <= 65:
        return '56-65'
    else:
        return '65+'

def get_income_group(income):
    """Bins income into standard groups used in training."""
    if income <= 100000:
        return 'Low'
    elif income <= 200000:
        return 'Medium'
    elif income <= 300000:
        return 'High'
    elif income <= 500000:
        return 'Very High'
    else:
        return 'Ultra High'

@app.route('/')
def home():
    """Renders Page 1: Landing Home Page."""
    return render_template('home.html')

@app.route('/apply')
def apply():
    """Renders Page 2: User Input Form."""
    # List of available options for dropdowns, dynamically retrieved from encoders
    dropdown_options = {
        'gender': list(label_encoders['CODE_GENDER'].classes_),
        'own_car': list(label_encoders['FLAG_OWN_CAR'].classes_),
        'own_realty': list(label_encoders['FLAG_OWN_REALTY'].classes_),
        'income_type': list(label_encoders['NAME_INCOME_TYPE'].classes_),
        'education_type': list(label_encoders['NAME_EDUCATION_TYPE'].classes_),
        'family_status': list(label_encoders['NAME_FAMILY_STATUS'].classes_),
        'housing_type': list(label_encoders['NAME_HOUSING_TYPE'].classes_),
        'occupation_type': list(label_encoders['OCCUPATION_TYPE'].classes_)
    }
    return render_template('apply.html', dropdowns=dropdown_options)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles real-time predictions, saves results to DB, and renders Page 3: Result."""
    try:
        # Determine source (JSON vs standard form input)
        if request.is_json:
            data = request.json
        else:
            data = request.form
        
        # 1. Parse raw input
        gender = data.get('CODE_GENDER')
        own_car = data.get('FLAG_OWN_CAR', 'N')
        own_realty = data.get('FLAG_OWN_REALTY', 'N')
        cnt_children = int(data.get('CNT_CHILDREN', 0))
        amt_income_total = float(data.get('AMT_INCOME_TOTAL', 0))
        name_income_type = data.get('NAME_INCOME_TYPE')
        name_education_type = data.get('NAME_EDUCATION_TYPE')
        name_family_status = data.get('NAME_FAMILY_STATUS')
        name_housing_type = data.get('NAME_HOUSING_TYPE')
        flag_work_phone = int(data.get('FLAG_WORK_PHONE', 0))
        flag_phone = int(data.get('FLAG_PHONE', 0))
        flag_email = int(data.get('FLAG_EMAIL', 0))
        occupation_type = data.get('OCCUPATION_TYPE')
        cnt_fam_members = float(data.get('CNT_FAM_MEMBERS', 1))
        
        # Credit history fields
        num_months = float(data.get('NUM_MONTHS', 24))
        months_history = num_months  # identical in training data
        
        age_years = float(data.get('AGE_YEARS', 30))
        employment_years = float(data.get('EMPLOYMENT_YEARS', 0))
        
        # 2. Derive/Engineer features
        income_per_member = amt_income_total / cnt_fam_members
        age_group = get_age_group(age_years)
        income_group = get_income_group(amt_income_total)
        has_children = 1 if cnt_children > 0 else 0
        
        # If employee duration is 0 and they say they are retired or income type is Pensioner
        flag_retired = 1 if (name_income_type == 'Pensioner' or data.get('IS_RETIRED') == 'on' or data.get('IS_RETIRED') == True) else 0
        if flag_retired == 1:
            employment_years = 0.0
            
        # 3. Create raw dict of all features
        raw_features = {
            'CODE_GENDER': gender,
            'FLAG_OWN_CAR': own_car,
            'FLAG_OWN_REALTY': own_realty,
            'CNT_CHILDREN': cnt_children,
            'AMT_INCOME_TOTAL': amt_income_total,
            'NAME_INCOME_TYPE': name_income_type,
            'NAME_EDUCATION_TYPE': name_education_type,
            'NAME_FAMILY_STATUS': name_family_status,
            'NAME_HOUSING_TYPE': name_housing_type,
            'FLAG_WORK_PHONE': flag_work_phone,
            'FLAG_PHONE': flag_phone,
            'FLAG_EMAIL': flag_email,
            'OCCUPATION_TYPE': occupation_type,
            'CNT_FAM_MEMBERS': cnt_fam_members,
            'NUM_MONTHS': num_months,
            'MONTHS_HISTORY': months_history,
            'FLAG_RETIRED': flag_retired,
            'AGE_YEARS': age_years,
            'EMPLOYMENT_YEARS': employment_years,
            'INCOME_PER_MEMBER': income_per_member,
            'AGE_GROUP': age_group,
            'INCOME_GROUP': income_group,
            'HAS_CHILDREN': has_children
        }
        
        # 4. Process Categorical Encodings
        encoded_features = {}
        for col, val in raw_features.items():
            if col in label_encoders:
                encoder = label_encoders[col]
                # Default to first class if not found (insurance against UI drift)
                if val not in encoder.classes_:
                    val = encoder.classes_[0]
                encoded_features[col] = encoder.transform([val])[0]
            else:
                encoded_features[col] = val
                
        # 5. Build input vector in correct column order
        input_data = pd.DataFrame([encoded_features])[REQUIRED_FEATURES]
        
        # 6. Predict using model
        # Target classes: Approved (0), Rejected (1)
        pred = int(best_model.predict(input_data)[0])
        proba = best_model.predict_proba(input_data)[0]
        
        approved_prob = proba[0]
        rejected_prob = proba[1]
        
        # If prediction is 0, approval confidence is approved_prob. If 1, rejection confidence is rejected_prob.
        confidence = approved_prob if pred == 0 else rejected_prob
        prediction_label = 'Approved' if pred == 0 else 'Rejected'
        
        # 7. Generate Explainability/Key Factors
        reasons = []
        
        # Post-processing / Business Rules Engine
        # Cap or override predictions for unrealistic profiles that exploit target leakage
        if pred == 0:  # Model predicted Approved
            # Rule 1: Insufficient Credit History
            if num_months < 6:
                pred = 1
                prediction_label = 'Rejected'
                approved_prob = 0.12
                rejected_prob = 0.88
                confidence = rejected_prob
                reasons = ["Insufficient credit history tracking length (less than 6 months).", 
                           "Lack of sufficient payment history timeline to establish creditworthiness."]
            # Rule 2: Young applicant with low job stability
            elif age_years < 22 and employment_years < 2.0 and flag_retired == 0:
                pred = 1
                prediction_label = 'Rejected'
                approved_prob = 0.15
                rejected_prob = 0.85
                confidence = rejected_prob
                reasons = ["Young applicant (under 22) with limited employment stability.",
                           "High risk due to combined entry-level demographic profile."]
            # Rule 3: Low income per family member
            elif income_per_member < 30000:
                pred = 1
                prediction_label = 'Rejected'
                approved_prob = 0.20
                rejected_prob = 0.80
                confidence = rejected_prob
                reasons = ["Low disposable income per family member.", 
                           "Insufficient financial safety margin relative to household size."]
            # Rule 4: Unemployed but not retired
            elif employment_years == 0 and flag_retired == 0:
                pred = 1
                prediction_label = 'Rejected'
                approved_prob = 0.10
                rejected_prob = 0.90
                confidence = rejected_prob
                reasons = ["Applicant reports zero years of employment and is not retired."]

        # Fallback to model-based reasons if no rules overrode the decision
        if len(reasons) == 0:
            if pred == 1: # Rejected
                if employment_years < 1.0 and flag_retired == 0:
                    reasons.append("Short employment duration (less than 1 year).")
                if income_per_member < 45000:
                    reasons.append("Low disposable income per family member.")
                if num_months < 12:
                    reasons.append("Limited active credit history.")
                if amt_income_total < 120000:
                    reasons.append("Total annual income is below target benchmarks.")
                if occupation_type == 'Unknown' and flag_retired == 0:
                    reasons.append("Unspecified or undocumented occupation class.")
                if len(reasons) == 0:
                    reasons.append("Overall risk profile from credit parameters exceeded safety limits.")
            else: # Approved
                if employment_years >= 3.0:
                    reasons.append("Stable employment history (3+ years).")
                if income_per_member >= 100000:
                    reasons.append("Strong income-to-family member ratio.")
                if num_months >= 24:
                    reasons.append("Established credit history (24+ months).")
                if own_realty == 'Y' or own_car == 'Y':
                    reasons.append("Asset ownership (property or vehicle) reduces credit risk.")
                if name_education_type in ['Higher education', 'Academic degree']:
                    reasons.append("Higher education level indicates low risk probability.")
                if len(reasons) == 0:
                    reasons.append("Satisfies all threshold credit and demographic risk scores.")

        # 8. Save prediction result to SQLite database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (
                timestamp, gender, age, education, income, employment_years, history_months, prediction_label, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Male' if gender == 'M' else 'Female',
            int(age_years),
            name_education_type,
            amt_income_total,
            employment_years,
            int(num_months),
            prediction_label,
            float(confidence)
        ))
        conn.commit()
        conn.close()

        # 9. Format response based on request source
        result_payload = {
            'prediction': pred,  # 0 or 1
            'prediction_label': prediction_label,
            'confidence': float(confidence),
            'approved_probability': float(approved_prob),
            'rejected_probability': float(rejected_prob),
            'reasons': reasons,
            'derived_metrics': {
                'income_per_member': round(income_per_member, 2),
                'age_group': age_group,
                'income_group': income_group,
                'flag_retired': flag_retired
            },
            'applicant_data': {
                'gender': 'Male' if gender == 'M' else 'Female',
                'age': int(age_years),
                'income': f"₹{amt_income_total:,.2f}",
                'employment': f"{employment_years} years" if flag_retired == 0 else "Retired/Pensioner",
                'education': name_education_type,
                'history': f"{int(num_months)} months",
                'fam_members': int(cnt_fam_members)
            }
        }

        if request.is_json:
            return jsonify({'success': True, **result_payload})
        else:
            return render_template('result.html', result=result_payload)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        if request.is_json:
            return jsonify({'success': False, 'error': str(e)}), 400
        else:
            return render_template('layout.html', error=str(e))

@app.route('/history')
def history():
    """Renders Page 4: Prediction History."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT timestamp, gender, age, education, income, employment_years, history_months, prediction_label, confidence FROM predictions ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()
    
    # Structure rows for template rendering
    predictions_history = []
    for r in rows:
        predictions_history.append({
            'timestamp': r[0],
            'gender': r[1],
            'age': r[2],
            'education': r[3],
            'income': f"₹{r[4]:,.2f}",
            'employment': f"{r[5]} years" if r[5] > 0 else "Retired",
            'history': f"{r[6]} months",
            'prediction_label': r[7],
            'confidence': f"{round(r[8] * 100, 1)}%"
        })
        
    return render_template('history.html', history=predictions_history)

@app.route('/history/clear', methods=['POST'])
def clear_history():
    """Purges the SQLite predictions table and redirects."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()
    return redirect(url_for('history'))

@app.route('/insights')
def insights():
    """Renders Page 5: Model Insights and plots."""
    # List model metrics to showcase architecture properties
    model_info = {
        'name': model_meta['model_name'],
        'accuracy': round(model_meta['accuracy'] * 100, 2),
        'f1_score': round(model_meta['f1_score'] * 100, 2),
        'roc_auc': round(model_meta['roc_auc'] * 100, 2)
    }
    return render_template('insights.html', model=model_info)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
