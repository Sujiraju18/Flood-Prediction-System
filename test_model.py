import joblib
import numpy as np

model = joblib.load("models/floods.save")
scaler = joblib.load("models/scaler.save")

samples = {
    "All 0s": [0] * 20,
    "All 5s": [5] * 20,
    "All 10s": [10] * 20,
}

for name, values in samples.items():
    x = np.array(values).reshape(1, -1)
    x = scaler.transform(x)

    prediction = model.predict(x)
    probability = model.predict_proba(x)

    print(f"\n{name}")
    print("Prediction:", prediction)
    print("Probability:", probability)