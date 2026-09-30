
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create sample customer data
np.random.seed(42)
n = 500

age = np.random.randint(18, 70, n)
tenure = np.random.randint(1, 11, n)
monthly_charges = np.random.randint(20, 150, n)
support_calls = np.random.randint(0, 10, n)

# Create churn labels
score = (0.01 * (monthly_charges - 80)
         - 0.15 * (tenure - 5)
         + 0.35 * support_calls
         + 0.02 * (age - 40))

probability = 1 / (1 + np.exp(-score))
churn = (probability > 0.5).astype(int)

# Create dataset
data = pd.DataFrame({
    "age": age,
    "tenure": tenure,
    "monthly_charges": monthly_charges,
    "support_calls": support_calls,
    "churn": churn
})

print(data.head())

X = data[["age", "tenure",
          "monthly_charges", "support_calls"]]
y = data["churn"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Save model and scaler
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Training completed successfully!")