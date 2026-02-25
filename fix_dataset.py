import pandas as pd

# Load dataset
df = pd.read_csv("data/signs.csv")

# Function to limit samples per label
def limit_label(df, label, max_count):
    label_df = df[df["label"] == label]
    other_df = df[df["label"] != label]
    
    label_df = label_df.iloc[:max_count]  # keep first N samples
    
    return pd.concat([other_df, label_df])

# Limit letters to 50 each
df = limit_label(df, "a", 50)
df = limit_label(df, "b", 50)
df = limit_label(df, "c", 50)
df = limit_label(df, "d", 50)
df = limit_label(df, "e", 50)

# Save cleaned dataset
df.to_csv("data/signs.csv", index=False)

print("Dataset balanced successfully.")
print(df["label"].value_counts())