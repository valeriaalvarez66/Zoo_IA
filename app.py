from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import json
from flask_cors import CORS  # 🔥 IMPORTANTE

app = Flask(__name__)
CORS(app)  # 🔥 PERMITE PETICIONES DESDE TU FRONTEND

modelo = joblib.load("models/modelo_zoo.pkl")

with open("json/columnas.json", "r") as f:
    columnas = json.load(f)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    entrada = [data.get(col, 0) for col in columnas]
    entrada = np.array(entrada).reshape(1, -1)
    prediccion = modelo.predict(entrada)[0]
    return jsonify({"prediccion": int(prediccion)})

if __name__ == "__main__":
    app.run(debug=True)
