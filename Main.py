import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from preprocessing import build_preprocessor
from train import train_random_forest
from evaluate import evaluate_model


# Load dataset
df = pd.read_csv("../data/german_credit.csv")

# Clean target
df["credit_risk"] = df["credit_risk"].str.strip().str.lower().map({
    "good": 0,
    "bad": 1
})

X = df.drop("credit_risk", axis=1)
y = df["credit_risk"]

num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocess
preprocessor = build_preprocessor(num_features, cat_features)
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Balance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)

# Train
model = train_random_forest(X_train_res, y_train_res)

# Evaluate
metrics = evaluate_model(model, X_test_proc, y_test)
print(metrics)

# Save model
joblib.dump(model, "../models/random_forest.pkl")
joblib.dump(preprocessor, "../models/preprocessor.pkl")
