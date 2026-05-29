
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
import logging

# ==========================================
# CONFIGURATION
# ==========================================

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    }
)

# ==========================================
# LOAD MODELS
# ==========================================

try:

    with open("knn_model.pkl", "rb") as f:
        knn_model = pickle.load(f)

    with open("random_forest_model.pkl", "rb") as f:
        rf_model = pickle.load(f)

    with open("svm_model.pkl", "rb") as f:
        svm_model = pickle.load(f)

    logging.info("All models loaded successfully.")

except Exception as e:

    logging.error(f"Model loading failed: {e}")
    raise

# ==========================================
# LABEL MAPPING
# ==========================================

LABEL_MAPPING = {
    1: "M, B, RA",
    3: "M, F, B, RA",
    4: "M, F, RA",
    5: "M, F, RA, C",
    6: "M, RA",
    8: "M, RA, C"
}

# ==========================================
# SCALING CONSTANTS
# ==========================================

MEAN = {
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

STD = {
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

# ==========================================
# FEATURE SCALING
# ==========================================

def feature_scale(
    C,
    Si,
    Mn,
    TMAE,
    Ac1,
    Ac3,
    MS,
    Qtemp,
    Ptemp
):

    return [[
        (C - MEAN["C"]) / STD["C"],
        (Si - MEAN["Si"]) / STD["Si"],
        (Mn - MEAN["Mn"]) / STD["Mn"],
        (TMAE - MEAN["TMAE"]) / STD["TMAE"],
        (Ac1 - MEAN["Ac1"]) / STD["Ac1"],
        (Ac3 - MEAN["Ac3"]) / STD["Ac3"],
        (MS - MEAN["MS"]) / STD["MS"],
        (Qtemp - MEAN["Qtemp"]) / STD["Qtemp"],
        (Ptemp - MEAN["Ptemp"]) / STD["Ptemp"]
    ]]

# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def home():

    return jsonify({
        "message": "Q&P Steel Multi-Model Prediction API",
        "models": [
            "KNN",
            "Random Forest",
            "SVM"
        ]
    })

@app.route("/health")
def health():

    return jsonify({
        "status": "healthy",
        "knn_loaded": knn_model is not None,
        "rf_loaded": rf_model is not None,
        "svm_loaded": svm_model is not None
    })

@app.route("/version")
def version():

    return jsonify({
        "version": "1.0.0",
        "models": [
            "KNN",
            "Random Forest",
            "SVM"
        ]
    })

# ==========================================
# PREDICTION ROUTE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        required_fields = [
            "C",
            "Si",
            "Mn",
            "TMAE",
            "Ac1",
            "Ac3",
            "Ms",
            "QT",
            "PT"
        ]

        for field in required_fields:

            if field not in data:

                return jsonify({
                    "success": False,
                    "error": f"Missing field: {field}"
                }), 400

        C = float(data["C"])
        Si = float(data["Si"])
        Mn = float(data["Mn"])
        TMAE = float(data["TMAE"])
        Ac1 = float(data["Ac1"])
        Ac3 = float(data["Ac3"])
        Ms = float(data["Ms"])
        QT = float(data["QT"])
        PT = float(data["PT"])

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

        # ==================================
        # KNN
        # ==================================

        knn_pred = int(
            knn_model.predict(
                scaled_input
            )[0]
        )

        knn_conf = round(
            float(
                np.max(
                    knn_model.predict_proba(
                        scaled_input
                    )
                )
            ) * 100,
            2
        )

        # ==================================
        # RANDOM FOREST
        # ==================================

        rf_pred = int(
            rf_model.predict(
                scaled_input
            )[0]
        )

        rf_conf = round(
            float(
                np.max(
                    rf_model.predict_proba(
                        scaled_input
                    )
                )
            ) * 100,
            2
        )

        # ==================================
        # SVM
        # ==================================

        svm_pred = int(
            svm_model.predict(
                scaled_input
            )[0]
        )

        svm_conf = round(
            float(
                np.max(
                    svm_model.predict_proba(
                        scaled_input
                    )
                )
            ) * 100,
            2
        )

        # ==================================
        # CONSENSUS
        # ==================================

        predictions = [
            knn_pred,
            rf_pred,
            svm_pred
        ]

        consensus = max(
            set(predictions),
            key=predictions.count
        )

        agreement = predictions.count(
            consensus
        )

        return jsonify({

            "success": True,

            "predictions": {

                "knn": {
                    "code": knn_pred,
                    "label": LABEL_MAPPING.get(
                        knn_pred,
                        "Unknown"
                    ),
                    "confidence": knn_conf
                },

                "random_forest": {
                    "code": rf_pred,
                    "label": LABEL_MAPPING.get(
                        rf_pred,
                        "Unknown"
                    ),
                    "confidence": rf_conf
                },

                "svm": {
                    "code": svm_pred,
                    "label": LABEL_MAPPING.get(
                        svm_pred,
                        "Unknown"
                    ),
                    "confidence": svm_conf
                }

            },

            "consensus": {

                "code": consensus,

                "label":
                    LABEL_MAPPING.get(
                        consensus,
                        "Unknown"
                    ),

                "agreement":
                    f"{agreement}/3"

            }

        })

    except Exception as e:

        logging.error(str(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        )
    )

