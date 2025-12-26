import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# ==============================
# 1. LOAD DATASET
# ==============================
df = pd.read_csv("customer_churn_dataset-training-master.csv")
print("Initial shape:", df.shape)

# ==============================
# 2. TARGET COLUMN (ALREADY NUMERIC)
# ==============================
df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')
df = df.dropna(subset=['Churn'])

print("After cleaning shape:", df.shape)
print(df['Churn'].value_counts())

# ==============================
# 3. FEATURES & TARGET
# ==============================
X = df.drop(['Churn', 'CustomerID'], axis=1)
y = df['Churn']

print("X shape:", X.shape)
print("y shape:", y.shape)
# Save feature order
feature_names = X.columns.tolist()
pickle.dump(feature_names, open("feature_names.pkl", "wb"))

# ==============================
# 4. ENCODE CATEGORICAL FEATURES
# ==============================
le = LabelEncoder()
for col in X.select_dtypes(include='object').columns:
    X[col] = le.fit_transform(X[col])

# ==============================
# 5. TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# 6. FEATURE SCALING
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# ==============================
# 7. TRAIN MODEL
# ==============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==============================
# 8. SAVE MODEL & SCALER
# ==============================
pickle.dump(model, open("churn_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ churn_model.pkl and scaler.pkl created successfully")
