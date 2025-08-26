# migrate_to_sqlite.py
import pandas as pd
import sqlite3

# Input CSV
csv_file = "tweet_data.csv"

# Output SQLite DB
db_file = "tweets.db"

# Load CSV
print("📂 Loading CSV:", csv_file)
df = pd.read_csv(csv_file)

# Add timestamp if not exists (so we can sort by latest later)
if "timestamp" not in df.columns:
    df["timestamp"] = pd.Timestamp.now().isoformat()

# Create DB connection
conn = sqlite3.connect(db_file)

# Write CSV into table "tweets"
df.to_sql("tweets", conn, if_exists="replace", index=False)

conn.commit()
conn.close()

print(f"✅ Migration complete! All {len(df)} rows moved into {db_file}")
