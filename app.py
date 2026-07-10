from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("models/floods.save")
scaler = joblib.load("models/scaler.save")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form["MonsoonIntensity"]),
        float(request.form["TopographyDrainage"]),
        float(request.form["RiverManagement"]),
        float(request.form["Deforestation"]),
        float(request.form["Urbanization"]),
        float(request.form["ClimateChange"]),
        float(request.form["DamsQuality"]),
        float(request.form["Siltation"]),
        float(request.form["AgriculturalPractices"]),
        float(request.form["Encroachments"]),
        float(request.form["IneffectiveDisasterPreparedness"]),
        float(request.form["DrainageSystems"]),
        float(request.form["CoastalVulnerability"]),
        float(request.form["Landslides"]),
        float(request.form["Watersheds"]),
        float(request.form["DeterioratingInfrastructure"]),
        float(request.form["PopulationScore"]),
        float(request.form["WetlandLoss"]),
        float(request.form["InadequatePlanning"]),
        float(request.form["PoliticalFactors"])
    ]

    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "High Flood Risk"
    else:
        result = "Low Flood Risk"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)