from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("loan_model.pkl")
feature_names = joblib.load("feature_names.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["age"])
    income = float(request.form["income"])
    loan = float(request.form["loan"])
    credit = int(request.form["credit"])

    input_data = pd.DataFrame(
        [[age, income, loan, credit]],
        columns=feature_names
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        result = f"✅ Loan Approved\nApproval Probability: {probability:.2%}"
    else:
        result = f"❌ Loan Rejected\nApproval Probability: {probability:.2%}"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)