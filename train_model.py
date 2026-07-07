import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("dataset/raw/train.csv")

df["Flood"] = (df["FloodProbability"] >= 0.5).astype(int)

df.drop(["id", "FloodProbability"], axis=1, inplace=True)

X = df.drop("Flood", axis=1)
y = df["Flood"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),

    "KNN": KNeighborsClassifier(),

    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )

}

best_model = None
best_accuracy = 0

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print("=" * 50)
    print(name)
    print("Accuracy:", accuracy)
    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

joblib.dump(best_model, "models/floods.save")

joblib.dump(scaler, "models/scaler.save")

print("\nBest Model Saved Successfully!")

print("Best Accuracy:", best_accuracy)