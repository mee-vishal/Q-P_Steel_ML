from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

# ==========================================
# CREATE FLASK APP
# ==========================================
app = Flask(__name__)

# ==========================================
# ENABLE CORS
# ==========================================
CORS(app)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================
model = joblib.load("knn_model.pkl")

# ==========================================
# LABEL MAPPING
# ==========================================
label_mapping = {

    1: "M, B, RA",

    3: "M, F, B, RA",

    4: "M, F, RA",

    5: "M, F, RA, C",

    6: "M, RA",

    8: "M, RA, C"

}

# ==========================================
# FEATURE SCALING FUNCTION
# ==========================================
def feature_scale(C, Si, Mn, TMAE, Ac1, Ac3, MS, Qtemp, Ptemp):

    mean = {
        "C": 0.293711,
        "Si": 1.353189,
        "Mn": 2.130231,
        "TMAE": 0.017715,
        "Ac1": 742.445838,
        "Ac3": 829.178609,
        "MS": 357.581143,
        "Qtemp": 226.697143,
        "Ptemp": 367.342857
    }

    std = {
        "C": 0.175460,
        "Si": 0.538275,
        "Mn": 1.180975,
        "TMAE": 0.045991,
        "Ac1": 12.901875,
        "Ac3": 35.662434,
        "MS": 61.784294,
        "Qtemp": 77.922007,
        "Ptemp": 80.983794
    }

    scaled = [[
        (C - mean["C"]) / std["C"],
        (Si - mean["Si"]) / std["Si"],
        (Mn - mean["Mn"]) / std["Mn"],
        (TMAE - mean["TMAE"]) / std["TMAE"],
        (Ac1 - mean["Ac1"]) / std["Ac1"],
        (Ac3 - mean["Ac3"]) / std["Ac3"],
        (MS - mean["MS"]) / std["MS"],
        (Qtemp - mean["Qtemp"]) / std["Qtemp"],
        (Ptemp - mean["Ptemp"]) / std["Ptemp"]
    ]]

    return scaled

# ==========================================
# HOME ROUTE
# ==========================================
@app.route('/')
def home():

    return jsonify({
        "message": "Q&P Steel Prediction API Running"
    })

# ==========================================
# PREDICTION ROUTE
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():

    try:

        # ==========================================
        # GET INPUT DATA
        # ==========================================
        data = request.get_json()

        # ==========================================
        # EXTRACT FEATURES
        # ==========================================
        C = float(data['C'])
        Si = float(data['Si'])
        Mn = float(data['Mn'])
        TMAE = float(data['TMAE'])
        Ac1 = float(data['Ac1'])
        Ac3 = float(data['Ac3'])
        Ms = float(data['Ms'])
        QT = float(data['QT'])
        PT = float(data['PT'])

        # ==========================================
        # APPLY FEATURE SCALING
        # ==========================================
        scaled_input = feature_scale(
            C,
            Si,
            Mn,
            TMAE,
            Ac1,
            Ac3,
            Ms,
            QT,
            PT
        )

        # ==========================================
        # MODEL PREDICTION
        # ==========================================
        raw_prediction = model.predict(scaled_input)

        prediction = np.array(
            raw_prediction
        ).flatten()[0]

        prediction = int(prediction)

        # ==========================================
        # CONFIDENCE SCORE
        # ==========================================
        confidence = 90

        if hasattr(model, "predict_proba"):

            probs = model.predict_proba(
                scaled_input
            )

            confidence = round(
                float(np.max(probs)) * 100,
                2
            )

        # ==========================================
        # RETURN RESPONSE
        # ==========================================
        return jsonify({

            "success": True,

            "prediction_code": prediction,

            "prediction_label":
                label_mapping.get(
                    prediction,
                    f"Unknown Class {prediction}"
                ),

            "confidence": confidence

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500

# ==========================================
# RUN APP
# ==========================================
if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000
    )