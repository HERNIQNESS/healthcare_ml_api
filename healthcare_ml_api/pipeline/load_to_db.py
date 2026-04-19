import pandas as pd
from database.db_connection import engine

# Load cleaned data
df = pd.read_csv("data/cleaned/cleaned.csv")

# Save to PostgreSQL
df.to_sql("cleaned_data", engine, if_exists="replace", index=False)

print("Data loaded into PostgreSQL")