from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from groq import Groq
import os

# ==========================================
# CREATE FLASK APP
# ==========================================
app = Flask(__name__)

# ==========================================
# ENABLE CORS
# ==========================================
CORS(app)

# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load("knn_model.pkl")

# ==========================================
# GROQ CLIENT
# ==========================================
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

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
# SAFE PARAMETER LIMITS
# ==========================================
parameter_limits = {

    "C": (0.03, 1.08),
    "Si": (0.20, 2.60),
    "Mn": (0.24, 6.00),
    "TMAE": (0.00, 0.10),

    "Ac1": (690, 780),
    "Ac3": (700, 920),
    "Ms": (120, 460),

    "QT": (10, 550),
    "PT": (150, 650)
}

# ==========================================
# FEATURE SCALING
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
# LIMIT VALUES SAFELY
# ==========================================
def clamp(value, min_val, max_val):

    return max(min(value, max_val), min_val)

# ==========================================
# OPTIMIZATION ENGINE
# ==========================================
def optimize_parameters(goal, params):

    optimized = params.copy()

    if goal == "Higher Ductility":

        optimized["PT"] += 50
        optimized["QT"] -= 30
        optimized["Si"] += 0.3

    elif goal == "Higher Strength":

        optimized["C"] += 0.08
        optimized["QT"] += 40
        optimized["PT"] -= 30

    elif goal == "Better Toughness":

        optimized["PT"] += 20
        optimized["QT"] -= 10
        optimized["Mn"] += 0.2

    elif goal == "Increase Retained Austenite":

        optimized["PT"] += 60
        optimized["Si"] += 0.4
        optimized["QT"] -= 40

    elif goal == "Reduce Carbide Formation":

        optimized["Si"] += 0.5
        optimized["PT"] += 40

    elif goal == "Balanced Strength-Ductility":

        optimized["PT"] += 30
        optimized["QT"] -= 15
        optimized["Si"] += 0.2

    # ==========================================
    # APPLY SAFE LIMITS
    # ==========================================
    for key in optimized:

        min_val, max_val = parameter_limits[key]

        optimized[key] = round(
            clamp(
                optimized[key],
                min_val,
                max_val
            ),
            2
        )

    return optimized

# ==========================================
# GROQ AI ANALYSIS
# ==========================================
def generate_ai_analysis(

    prediction_label,
    confidence,
    current_params,
    optimized_params,
    goal

):

    prompt = f"""
You are an expert metallurgical engineer specializing
in Quenched and Partitioned (Q&P) steels.

Current Predicted Microstructure:
{prediction_label}

Prediction Confidence:
{confidence}%

Optimization Goal:
{goal}

CURRENT PARAMETERS:
{current_params}

OPTIMIZED PARAMETERS:
{optimized_params}

Analyze:

1. Why the current microstructure formed
2. How the optimized parameters improve the target property
3. Which parameters changed most significantly
4. Metallurgical reasoning behind the changes
5. Whether the same microstructure family is likely retained
6. Expected property improvements
7. Possible tradeoffs
8. Phase transformation reasoning

VERY IMPORTANT:
- Mention actual parameter changes
- Compare old vs new values
- Explain scientifically
- Keep response concise
- Use bullet points
- Sound like a metallurgy research expert
"""

    try:

        completion = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.5,

            max_tokens=700
        )

        return completion.choices[0].message.content

    except Exception as e:

        return f"AI analysis failed: {str(e)}"

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
        # INPUT DATA
        # ==========================================
        data = request.get_json()

        goal = data.get(
            "goal",
            "Balanced Strength-Ductility"
        )

        # ==========================================
        # EXTRACT PARAMETERS
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

        current_params = {

            "C": C,
            "Si": Si,
            "Mn": Mn,
            "TMAE": TMAE,

            "Ac1": Ac1,
            "Ac3": Ac3,
            "Ms": Ms,

            "QT": QT,
            "PT": PT
        }

        # ==========================================
        # SCALE FEATURES
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
        raw_prediction = model.predict(
            scaled_input
        )

        prediction = int(
            np.array(raw_prediction).flatten()[0]
        )

        prediction_label = label_mapping.get(
            prediction,
            f"Unknown Class {prediction}"
        )

        # ==========================================
        # CONFIDENCE
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
        # OPTIMIZATION
        # ==========================================
        optimized_params = optimize_parameters(

            goal,
            current_params
        )

        # ==========================================
        # AI ANALYSIS
        # ==========================================
        ai_analysis = generate_ai_analysis(

            prediction_label=prediction_label,

            confidence=confidence,

            current_params=current_params,

            optimized_params=optimized_params,

            goal=goal
        )

        # ==========================================
        # RESPONSE
        # ==========================================
        return jsonify({

            "success": True,

            "prediction_code": prediction,

            "prediction_label": prediction_label,

            "confidence": confidence,

            "goal": goal,

            "current_parameters": current_params,

            "optimized_parameters": optimized_params,

            "ai_analysis": ai_analysis
        })

    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500

# ==========================================
# RUN SERVER
# ==========================================
if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )

