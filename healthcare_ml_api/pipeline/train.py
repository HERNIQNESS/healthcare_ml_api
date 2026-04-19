import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# Load cleaned data
from database.db_connection import engine

df = pd.read_sql("SELECT * FROM cleaned_data", engine)

print("Columns:", df.columns)

# ✅ FIXED ENCODING
encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Define target
target = "Test Results"

X = df.drop(target, axis=1)
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Models
xgb = XGBClassifier()
lr = LogisticRegression(max_iter=2000)

# Train
xgb.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Evaluate
print("\n=== XGBoost ===")
print(classification_report(y_test, xgb.predict(X_test)))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb.predict(X_test)))

print("\n=== Logistic Regression ===")
print(classification_report(y_test, lr.predict(X_test)))

# Save model + encoders
joblib.dump(xgb, "models/model.joblib")
joblib.dump(encoders, "models/encoders.joblib")

print("\n Model and encoders saved.")

print("Feature Order:", X.columns.tolist())