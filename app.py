# app.py
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# load saved artifacts that you created with joblib.dump(...)
vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("model.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "")
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]          # 1=FAKE, 0=REAL
    label = "FAKE" if pred == 1 else "REAL"
    return jsonify({"label": label})

if __name__ == "__main__":
    app.run(debug=True)
