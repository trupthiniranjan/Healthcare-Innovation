from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import pandas as pd
import numpy as np
import os
import webbrowser          # <--- ADDED
from threading import Timer # <--- ADDED

app = Flask(__name__)

# Paths (assumes these are in the same working directory)
DATA_CSV_PATH = 'drug_data.csv'
MODEL_PATH = 'drug_recommendation_model.pkl'
FEATURE_COLS_PATH = 'feature_columns.pkl'
MLB_PATH = 'mlb_encoder.pkl'

# Globals
model = None
feature_columns = []
mlb = None
drug_df = None

def _strip_all_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace on all object/string columns in-place and return df."""
    obj_cols = df.select_dtypes(include=['object']).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).map(lambda x: x.strip())
    return df

def load_artifacts():
    """Load model artifacts (if present) and the CSV dataset into global drug_df."""
    global model, feature_columns, mlb, drug_df

    # Try to load ML artifacts if they exist (optional)
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_COLS_PATH) and os.path.exists(MLB_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            feature_columns = joblib.load(FEATURE_COLS_PATH)
            # ensure feature_columns is a lowercase list
            feature_columns = [str(c).strip().lower() for c in feature_columns]
            mlb = joblib.load(MLB_PATH)
            print("Loaded model artifacts.")
        except Exception as e:
            print("Warning: failed to load model artifacts:", e)
            model = None
            feature_columns = []
            mlb = None
    else:
        print("Model artifacts not found -- running in rule-based mode.")

    # Load CSV dataset
    if not os.path.exists(DATA_CSV_PATH):
        print(f"Error: {DATA_CSV_PATH} not found.")
        drug_df = None
        return

    try:
        df = pd.read_csv(DATA_CSV_PATH, dtype=str).fillna('')
        # normalize column names and string contents
        df.columns = [c.strip() for c in df.columns]
        df = _strip_all_strings(df)

        # Ensure a Drug_Name column exists (attempt alternatives)
        if 'Drug_Name' not in df.columns:
            for alt in ['drug_name', 'Name', 'name', 'Drug']:
                if alt in df.columns:
                    df['Drug_Name'] = df[alt]
                    break

        # Create normalized name column for robust lookups
        if 'Drug_Name' in df.columns:
            df['Drug_Name_norm'] = df['Drug_Name'].astype(str).str.lower().str.strip()
        else:
            df['Drug_Name_norm'] = ''

        # Build Symptoms_List: check symptom* columns first, else 'Symptoms' column, else empty list
        symptom_cols = [c for c in df.columns if c.lower().startswith('symptom')]
        if symptom_cols:
            def _row_to_symptoms(row):
                vals = []
                for v in row.values:
                    v = str(v).strip()
                    if v != '':
                        # If there are comma-separated symptoms in a cell, split them too
                        parts = [p.strip().lower() for p in v.split(',') if p.strip()]
                        vals.extend(parts)
                # dedupe while preserving order
                seen = set()
                out = []
                for s in vals:
                    if s not in seen:
                        seen.add(s)
                        out.append(s)
                return out
            df['Symptoms_List'] = df[symptom_cols].apply(_row_to_symptoms, axis=1)
        elif 'Symptoms' in df.columns:
            df['Symptoms_List'] = df['Symptoms'].apply(lambda x: [s.strip().lower() for s in str(x).split(',') if s.strip()])
        else:
            df['Symptoms_List'] = [[] for _ in range(len(df))]

        # Ensure consistent column names used by frontend/backends
        if 'Side_Effects' in df.columns and 'Side_Effect' not in df.columns:
            df['Side_Effect'] = df['Side_Effects']
        if 'Similar_Tablet' in df.columns and 'Similar_Tablets' not in df.columns:
            df['Similar_Tablets'] = df['Similar_Tablet']

        drug_df = df
        print(f"Loaded dataset with {len(df)} rows. Sample normalized names: {drug_df['Drug_Name_norm'].head(8).tolist()}")
    except Exception as e:
        print("Failed to load CSV:", e)
        drug_df = None

# Load on startup
load_artifacts()

# ---------- Routes ----------
@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception:
        return send_from_directory('.', 'index.html')

@app.route('/all_drugs', methods=['GET'])
def all_drugs():
    if drug_df is None:
        return jsonify([])
    out = []
    for _, r in drug_df.iterrows():
        out.append({
            "Drug_Name": r.get('Drug_Name',''),
            "Type": r.get('Type',''),
            "Price": r.get('Price',''),
            "Uses": r.get('Uses',''),
            "Side_Effect": r.get('Side_Effect',''),
            "Dosage": r.get('Dosage',''),
            "Similar_Tablets": r.get('Similar_Tablets',''),
            "Symptoms_List": r.get('Symptoms_List', [])
        })
    return jsonify(out)

@app.route('/drug_info', methods=['GET'])
def drug_info():
    if drug_df is None:
        return jsonify({"error":"Drug database not loaded."}), 500
    name = request.args.get('name', '')
    if not name:
        return jsonify({"error":"Please supply ?name=..."}), 400

    q = str(name).strip().lower()
    matches = drug_df[drug_df['Drug_Name_norm'] == q]

    if matches.empty:
        starts = drug_df[drug_df['Drug_Name_norm'].str.startswith(q)]
        if not starts.empty:
            matches = starts
    if matches.empty:
        try:
            contains = drug_df[drug_df['Drug_Name_norm'].str.contains(q, regex=False)]
        except Exception:
            contains = drug_df[drug_df['Drug_Name_norm'].str.contains(q)]
        if not contains.empty:
            matches = contains

    if matches.empty:
        return jsonify({"error":"Drug not found."}), 404

    row = matches.iloc[0]
    base_symptoms = set(row['Symptoms_List']) if isinstance(row.get('Symptoms_List'), list) else set()
    similar = []
    
    # Simple similarity based on overlapping symptoms
    for _, r in drug_df.iterrows():
        if str(r.get('Drug_Name','')).strip().lower() == str(row.get('Drug_Name','')).strip().lower():
            continue
        other_sym = r.get('Symptoms_List', [])
        if not isinstance(other_sym, list):
            other_sym = []
        overlap = base_symptoms.intersection(set(other_sym))
        if overlap:
            similar.append({
                "drug_name": r.get('Drug_Name',''),
                "type": r.get('Type',''),
                "price": r.get('Price',''),
                "uses": r.get('Uses',''),
                "side_effect": r.get('Side_Effect',''),
                "common_symptoms": list(overlap)
            })

    info = {
        "drug_name": row.get('Drug_Name',''),
        "type": row.get('Type',''),
        "price": row.get('Price',''),
        "uses": row.get('Uses',''),
        "side_effect": row.get('Side_Effect',''),
        "dosage": row.get('Dosage',''),
        "similar_tablets": row.get('Similar_Tablets',''),
        "common_symptoms": list(base_symptoms),
        "similar_drugs": similar[:8]
    }
    return jsonify(info)

@app.route('/recommend_drug', methods=['POST'])
def recommend_drug():
    if drug_df is None:
        return jsonify({"error":"Drug database not loaded."}), 500

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error":"Invalid JSON payload."}), 400

    user_symptoms = payload.get('symptoms', [])
    if not isinstance(user_symptoms, list) or len(user_symptoms) == 0:
        if isinstance(payload.get('symptoms'), str):
            user_symptoms = [s.strip() for s in payload.get('symptoms').split(',') if s.strip()]
        else:
            return jsonify({"error":"Please provide 'symptoms' as a non-empty list or string."}), 400

    q_symptoms = [str(s).strip().lower() for s in user_symptoms if s and str(s).strip()]
    if len(q_symptoms) == 0:
        return jsonify({"error":"No valid symptoms provided."}), 400

    results = []
    
    for _, r in drug_df.iterrows():
        drug_symptoms = r.get('Symptoms_List', [])
        if not isinstance(drug_symptoms, list):
            drug_symptoms = []
        
        matched_symptoms = set()
        for u_sym in q_symptoms:
            for d_sym in drug_symptoms:
                if u_sym in d_sym or d_sym in u_sym: 
                    matched_symptoms.add(d_sym)
        
        if len(matched_symptoms) > 0:
            score = len(matched_symptoms)
            results.append((score, r, list(matched_symptoms)))

    if results:
        results.sort(key=lambda x: (-x[0], str(x[1].get('Drug_Name',''))))
        recs = []
        for score, r, overlap in results[:15]:
            recs.append({
                "drug_name": r.get('Drug_Name',''),
                "confidence_score": round((score / max(1, len(q_symptoms))) * 100, 2),
                "type": r.get('Type',''),
                "price": r.get('Price',''),
                "uses": r.get('Uses',''),
                "side_effect": r.get('Side_Effect',''),
                "dosage": r.get('Dosage',''),
                "common_symptoms": overlap
            })
        return jsonify({"unseen_symptoms": [], "recommendations": recs})

    if model is not None and feature_columns:
        input_vector = pd.DataFrame(0, index=[0], columns=feature_columns)
        matched = []
        for s in q_symptoms:
            if s in input_vector.columns:
                input_vector.at[0, s] = 1
                matched.append(s)

        if len(matched) > 0:
            try:
                pred = model.predict(input_vector)[0]
                matches = drug_df[drug_df['Drug_Name_norm'] == str(pred).strip().lower()]
                if not matches.empty:
                    r = matches.iloc[0]
                    rec = {
                        "drug_name": r.get('Drug_Name',''),
                        "confidence_score": 90.0,
                        "type": r.get('Type',''),
                        "price": r.get('Price',''),
                        "uses": r.get('Uses',''),
                        "side_effect": r.get('Side_Effect',''),
                        "common_symptoms": matched
                    }
                    return jsonify({"unseen_symptoms": [], "recommendations": [rec]})
            except Exception:
                pass

    return jsonify({"error": "No drugs found matching those symptoms.", "recommendations": []})

@app.route('/_debug_names', methods=['GET'])
def _debug_names():
    if drug_df is None:
        return jsonify({"error": "no dataset"}), 500
    return jsonify({"count": len(drug_df), "sample_norm_names": drug_df['Drug_Name_norm'].head(50).tolist()})

# ---------- MAIN BLOCK (MODIFIED) ----------
def open_browser():
    # Use 127.0.0.1 (Localhost) instead of 0.0.0.0 for the browser
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    # This prevents the browser from opening twice if the auto-reloader is active
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        # Timer(1, ...) waits 1 second to give Flask time to start up
        Timer(1, open_browser).start()
    
    # Start the server
    app.run(host='0.0.0.0', port=5000, debug=True)