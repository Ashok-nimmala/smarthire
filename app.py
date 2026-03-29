import numpy as np
from flask import Flask, request, render_template
import pickle

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
        # Get input values
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]

        # Prediction
        prediction = model.predict(final_features)
        result = prediction[0].capitalize()

        return render_template("result.html", prediction=result)

    except Exception as e:
        return str(e)

# Run app
if __name__ == "__main__":
    app.run(debug=True, port=5002)

import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))