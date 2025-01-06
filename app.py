from transformers import pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)
model = pipeline("sentiment-analysis")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    result = model(data["text"])
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
