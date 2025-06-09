import pandas as pd
from sklearn.model_selection import train_test_split

# Load full dataset
df = pd.read_csv("custom_emotion_data.csv")

# Split features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split into train/test (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Combine X and y back into DataFrames
train_df = pd.concat([X_train, y_train], axis=1)
test_df  = pd.concat([X_test, y_test], axis=1)

# Save to new CSVs
train_df.to_csv("emotion_train2.csv", index=False)
test_df.to_csv("emotion_test2.csv", index=False)

print("✅ Split completed and saved to emotion_train.csv and emotion_test.csv")