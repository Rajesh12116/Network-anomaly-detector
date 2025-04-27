# Step 1: Import libraries
from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 2: Create Flask app
app = Flask(__name__)

# Step 3: Load trained model
model = joblib.load('network_anomaly_detector.pkl')

# Step 4: Home page
@app.route('/')
def home():
    return render_template('index.html')

# Step 5: Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return "No file uploaded!"
        
        data = pd.read_csv(file)

        # Label Encoding (text to numbers)
        le = LabelEncoder()
        for column in data.columns:
            if data[column].dtype == object:
                data[column] = le.fit_transform(data[column])

        # If missing columns, add dummy
        if data.shape[1] < 42:
            missing_cols = 42 - data.shape[1]
            for i in range(missing_cols):
                data[f'Dummy_{i}'] = 0

        # Make Prediction
        prediction = model.predict(data)
        
        result = 'Normal Traffic' if prediction[0] == 0 else 'Attack Detected'
        
        return render_template('result.html', prediction=result)

# Step 6: Run app
if __name__ == '__main__':
    app.run(debug=True)
