from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# ---- 1. Load dataset ----
df = pd.read_csv("Iris.csv")

# Bỏ cột Id, lấy features + target
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

# ---- 2. Train model ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline: scale + KNN
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
model.fit(X_train, y_train)
test_acc = model.score(X_test, y_test)

feature_names = X.columns.tolist()

# ---- 3. Flask routes ----
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    proba = None
    error = None
    inputs = {fn: "" for fn in feature_names}

    if request.method == "POST":
        try:
            vals = []
            for fn in feature_names:
                val = float(request.form.get(fn, "").strip())
                inputs[fn] = request.form.get(fn, "").strip()
                vals.append(val)

            pred = model.predict([vals])[0]
            probs = model.predict_proba([vals])[0]
            proba = max(probs)

            prediction = str(pred)
        except Exception as e:
            error = f"Lỗi nhập liệu: {e}"

    return render_template("index.html",
                           feature_names=feature_names,
                           inputs=inputs,
                           prediction=prediction,
                           proba=proba,
                           test_acc=test_acc,
                           error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
