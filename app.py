import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "demand_model.pkl"), "rb") as f:
    rf = pickle.load(f)
with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

inventory_df = pd.read_csv(os.path.join(BASE_DIR, "inventory.csv"))


# 📌 Prediction Function (Correct Logic)
def predict_demand(category, year, month):
    if category not in le.classes_:
        return None, None, f"Category '{category}' not found in model data"

    # Encode category
    cat_encoded = le.transform([category])[0]
    X_input = np.array([[cat_encoded, year, month]])

    # Predict demand (integer)
    predicted_qty = int(round(rf.predict(X_input)[0]))

    # Get safety stock (integer)
    safety_value = int(
        inventory_df.loc[inventory_df["Category"] == category, "Safety_Stock"].values[0]
    )

    # Compare with correct logic
    difference = predicted_qty - safety_value
    if predicted_qty > safety_value:
        status = f"Under Stock: Need {difference} more units"
    elif predicted_qty < safety_value:
        status = f"Over Stock: {abs(difference)} extra units available"
    else:
        status = "Enough Stock"

    return predicted_qty, safety_value, status


# 📌 Flask Route
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    categories = inventory_df["Category"].unique().tolist()

    if request.method == "POST":
        category = request.form["category"]
        year = int(request.form["year"])
        month = int(request.form["month"])

        pred_qty, inv, status = predict_demand(category, year, month)
        result = {
            "category": category,
            "year": year,
            "month": month,
            "pred_qty": pred_qty,
            "inventory": inv,
            "status": status
        }

    return render_template("index.html", categories=categories, result=result)


if __name__ == "__main__":
    app.run(host-'0.0.0.0', debug-True)

