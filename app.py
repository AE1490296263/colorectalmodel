import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Load model
model_path = r"colorectal.joblib"
rf_model = joblib.load(model_path)
features = ['CEA', 'ALB', 'CIKP', 'Cyfra211', 'Ca', 'HGB']

# Configure matplotlib for non-interactive backend
plt.switch_backend('Agg')

def generate_shap_plot(input_data, prediction_type):
    # Convert input to DataFrame
    X_test = pd.DataFrame([input_data], columns=features)
    
    # Determine class index
    class_index = 1 if prediction_type == 'cancer' else 0
    class_name = "Colorectal Cancer" if prediction_type == 'cancer' else "Colorectal Polyp"
    
    # SHAP explanation
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    
    # Prepare visualization
    plt.figure()
    sample_features = X_test.iloc[0]
    shap_values_sample = shap_values[class_index].T[0]
    base_value = explainer.expected_value[class_index]
    
    features_with_values = np.array([
        f'{name}={value:.4f}' for name, value in zip(features, sample_features)
    ])
    
    shap.force_plot(
        base_value,
        shap_values_sample,
        features_with_values,
        matplotlib=True,
        show=False
    )
    
    plt.title(f"SHAP Force Plot - {class_name}", y=1.1)
    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'shap_plot.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    return plot_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input values
        input_data = [
            float(request.form['CEA']),
            float(request.form['ALB']),
            float(request.form['CIKP']),
            float(request.form['Cyfra211']),
            float(request.form['Ca']),
            float(request.form['HGB'])
        ]
        
        # Get prediction type
        prediction_type = request.form['prediction_type']
        
        # Make prediction
        X_test = pd.DataFrame([input_data], columns=features)
        probs = rf_model.predict_proba(X_test)[0]
        
        # Generate SHAP plot
        plot_path = generate_shap_plot(input_data, prediction_type)
        
        return render_template('index.html', 
                             cancer_prob=f"{probs[1]*100:.2f}%",
                             polyp_prob=f"{probs[0]*100:.2f}%",
                             show_results=True,
                             plot_path=plot_path)
    
    return render_template('index.html', show_results=False)

if __name__ == '__main__':
    os.makedirs(os.path.join('static', 'shap_plots'), exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)