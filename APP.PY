from flask import Flask, render_template, request
import pandas as pd
import joblib

# 1️⃣ Initialize Flask app
app = Flask(__name__)

# 2️⃣ Load trained models
rf_model = joblib.load("models/rf_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")

# 3️⃣ Home route - shows input form
@app.route("/")
def home():
    return render_template("index.html")

# 4️⃣ Prediction route - handles form submission
@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    rating = float(request.form["rating"])
    learners = int(request.form["learners"])
    duration = float(request.form["duration"])
    is_paid = int(request.form["is_paid"])

    # Prepare input for models
    input_df = pd.DataFrame([[rating, learners, duration, is_paid]],
                            columns=["Rating", "Learners", "Duration_Months", "Is_Paid"])

    # Make predictions
    rf_pred = rf_model.predict(input_df)[0]
    xgb_pred = xgb_model.predict(input_df)[0]

    # Optional: get probabilities
    rf_prob = rf_model.predict_proba(input_df)[0][1]  # probability of success
    xgb_prob = xgb_model.predict_proba(input_df)[0][1]

    # Render result template
    return render_template("result.html", 
                           rf_pred=rf_pred, xgb_pred=xgb_pred, 
                           rf_prob=round(rf_prob*100, 2), 
                           xgb_prob=round(xgb_prob*100, 2))

# 5️⃣ Run the app
if __name__ == "__main__":
    app.run(debug=True)
