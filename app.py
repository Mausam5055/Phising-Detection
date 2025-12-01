from flask import Flask, render_template, request
import joblib
import numpy as np
import os
from src.feature_extractor import FeatureExtractor

app = Flask(__name__)

# Load Models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models_saved')

try:
    model = joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    imputer = joblib.load(os.path.join(MODEL_DIR, 'imputer.pkl'))
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    url = ""
    error = None
    
    if request.method == 'POST':
        url = request.form.get('url')
        if url:
            try:
                # 1. Extract Features
                extractor = FeatureExtractor(url)
                # Fetch content for better accuracy (optional, might be slow)
                # extractor.fetch_content() 
                features = extractor.extract_features()
                
                # 2. Preprocess
                # Impute (handle NaNs if any, though extractor returns 0s)
                features = imputer.transform(features)
                # Scale
                features = scaler.transform(features)
                
                # 3. Predict
                if model:
                    pred_prob = model.predict_proba(features)[0][1]
                    pred_class = int(pred_prob > 0.5)
                    
                    prediction = {
                        'class': 'Phishing' if pred_class == 1 else 'Legitimate',
                        'probability': round(pred_prob * 100, 2),
                        'is_phishing': pred_class == 1
                    }
                else:
                    error = "Model not loaded."
            except Exception as e:
                error = f"Error processing URL: {e}"
                print(e)
        else:
            error = "Please enter a URL."

    return render_template('index.html', prediction=prediction, url=url, error=error)

if __name__ == '__main__':
    app.run(debug=True)
