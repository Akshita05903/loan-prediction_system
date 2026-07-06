import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("dataset/loan.csv")

# Features (Input)
X = df[["Age", "Income", "LoanAmount", "CreditScore"]]

# Target (Output)
y = df["Approved"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "loan_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

# Accuracy
accuracy = model.score(X_test, y_test)

print("Model Trained Successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")