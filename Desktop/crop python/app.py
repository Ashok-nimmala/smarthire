import numpy as np
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get inputs individually (better control)
        n = float(request.form["n"])
        p = float(request.form["p"])
        k = float(request.form["k"])
        temp = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # ✅ Validation
        if not (0 <= n <= 150 and 0 <= p <= 150 and 0 <= k <= 150):
            return "Invalid NPK values"

        if not (0 <= temp <= 50):
            return "Temperature must be between 0–50°C"

        if not (0 <= humidity <= 100):
            return "Humidity must be between 0–100%"

        if not (0 <= ph <= 14):
            return "pH must be between 0–14"

        if not (0 <= rainfall <= 300):
            return "Rainfall must be between 0–300 mm"

        # Prepare features
        features = [[n, p, k, temp, humidity, ph, rainfall]]

        # Prediction
        prediction = model.predict(features)[0]

        # ✅ Fix: normalize for image matching
        result = prediction.strip().lower()

        return render_template("result.html", prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

# ✅ Single run block (Render compatible)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))