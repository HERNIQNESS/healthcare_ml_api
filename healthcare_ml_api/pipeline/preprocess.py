import pandas as pd

df = pd.read_csv("data/raw/healthcare_dataset.csv")

# Drop useless columns
df = df.drop(columns=["Name", "Doctor", "Hospital", "Room Number"], errors="ignore")

# Convert dates
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])

# Feature engineering
df["Length of Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days

# Drop original date columns
df = df.drop(columns=["Date of Admission", "Discharge Date"])

# Clean categorical text
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip().str.lower()

# Save cleaned data
df.to_csv("data/cleaned/cleaned.csv", index=False)

print("Cleaned data saved")
print(df.head())