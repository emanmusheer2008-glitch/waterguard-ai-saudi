import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.impute import SimpleImputer
import joblib

print("=" * 70)
print("WATERGUARD AI SAUDI — MODEL V1")
print("=" * 70)

DATA = Path("data")
OUT = Path("outputs")
MODELS = Path("models")
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD SCADA
# ---------------------------------------------------------
print("\n[1/7] Loading SCADA data...")

pressures = pd.read_excel(
    DATA / "2018_SCADA.xlsx",
    sheet_name="Pressures (m)"
)

demands = pd.read_excel(
    DATA / "2018_SCADA.xlsx",
    sheet_name="Demands (L_h)"
)

flows = pd.read_excel(
    DATA / "2018_SCADA.xlsx",
    sheet_name="Flows (m3_h)"
)

levels = pd.read_excel(
    DATA / "2018_SCADA.xlsx",
    sheet_name="Levels (m)"
)

for df in [pressures, demands, flows, levels]:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

print("Pressure shape:", pressures.shape)
print("Demand shape:", demands.shape)
print("Flow shape:", flows.shape)
print("Level shape:", levels.shape)

# ---------------------------------------------------------
# 2. ENGINEER NETWORK FEATURES
# ---------------------------------------------------------
print("\n[2/7] Engineering SCADA features...")

features = pd.DataFrame()
features["Timestamp"] = pressures["Timestamp"]

p = pressures.drop(columns="Timestamp")
d = demands.drop(columns="Timestamp")
f = flows.drop(columns="Timestamp")
l = levels.drop(columns="Timestamp")

# Network pressure behaviour
features["pressure_mean"] = p.mean(axis=1)
features["pressure_min"] = p.min(axis=1)
features["pressure_max"] = p.max(axis=1)
features["pressure_std"] = p.std(axis=1)
features["pressure_range"] = p.max(axis=1) - p.min(axis=1)

# Demand behaviour
features["demand_total"] = d.sum(axis=1)
features["demand_mean"] = d.mean(axis=1)
features["demand_std"] = d.std(axis=1)

# Flow behaviour
features["flow_total"] = f.sum(axis=1)
features["flow_mean"] = f.mean(axis=1)
features["flow_std"] = f.std(axis=1)

# Tank
features["tank_level"] = l.iloc[:, 0]

# Time features
features["hour"] = features["Timestamp"].dt.hour
features["dayofweek"] = features["Timestamp"].dt.dayofweek
features["month"] = features["Timestamp"].dt.month

# Short-term changes
for col in [
    "pressure_mean",
    "pressure_min",
    "demand_total",
    "flow_total",
    "tank_level"
]:
    features[f"{col}_change_5m"] = features[col].diff()

print("Engineered features:", features.shape[1] - 1)

# ---------------------------------------------------------
# 3. LOAD GROUND TRUTH
# ---------------------------------------------------------
print("\n[3/7] Loading leakage ground truth...")

leaks = pd.read_csv(
    DATA / "2018_Leakages.csv",
    sep=";",
    decimal=",",
    parse_dates=["Timestamp"]
)

leak_cols = [c for c in leaks.columns if c != "Timestamp"]
leaks["total_leak"] = leaks[leak_cols].sum(axis=1)

# V1 operational severity definition
SEVERE_THRESHOLD = 40.0
leaks["severe_leak"] = (
    leaks["total_leak"] >= SEVERE_THRESHOLD
).astype(int)

print(
    f"Severe target (total leakage >= {SEVERE_THRESHOLD}):",
    f"{leaks['severe_leak'].mean()*100:.2f}% of timestamps"
)

# ---------------------------------------------------------
# 4. MERGE
# ---------------------------------------------------------
print("\n[4/7] Merging sensors with labels...")

df = features.merge(
    leaks[["Timestamp", "total_leak", "severe_leak"]],
    on="Timestamp",
    how="inner"
)

df = df.sort_values("Timestamp").reset_index(drop=True)

feature_cols = [
    c for c in features.columns
    if c != "Timestamp"
]

print("Final rows:", len(df))
print("Features:", len(feature_cols))

# ---------------------------------------------------------
# 5. TIME-BASED TRAIN / TEST SPLIT
# ---------------------------------------------------------
print("\n[5/7] Creating chronological train/test split...")

# Important: NO random split.
# First 70% of the year trains the model.
# Final 30% tests future generalisation.
split = int(len(df) * 0.70)

train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

X_train = train[feature_cols]
y_train = train["severe_leak"]

X_test = test[feature_cols]
y_test = test["severe_leak"]

print(
    "Train:",
    train["Timestamp"].min(),
    "→",
    train["Timestamp"].max()
)

print(
    "Test:",
    test["Timestamp"].min(),
    "→",
    test["Timestamp"].max()
)

print("Train severe rate:", round(y_train.mean()*100, 2), "%")
print("Test severe rate:", round(y_test.mean()*100, 2), "%")

# ---------------------------------------------------------
# 6. MODEL
# ---------------------------------------------------------
print("\n[6/7] Training Random Forest...")

imputer = SimpleImputer(strategy="median")

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=14,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_imp, y_train)

pred = model.predict(X_test_imp)
prob = model.predict_proba(X_test_imp)[:, 1]

# ---------------------------------------------------------
# 7. EVALUATION
# ---------------------------------------------------------
print("\n[7/7] Evaluating...")
print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, pred, digits=4))

precision = precision_score(y_test, pred, zero_division=0)
recall = recall_score(y_test, pred, zero_division=0)
f1 = f1_score(y_test, pred, zero_division=0)

try:
    auc = roc_auc_score(y_test, prob)
except ValueError:
    auc = np.nan

print("WATERGUARD V1 RESULTS")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")

# Feature importance
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTOP 10 FEATURES")
print(importance.head(10).to_string(index=False))

importance.to_csv(
    OUT / "feature_importance_v1.csv",
    index=False
)

results = test[
    ["Timestamp", "total_leak", "severe_leak"]
].copy()

results["predicted_severe"] = pred
results["risk_probability"] = prob

results.to_csv(
    OUT / "predictions_v1.csv",
    index=False
)

joblib.dump(model, MODELS / "waterguard_rf_v1.joblib")
joblib.dump(imputer, MODELS / "imputer_v1.joblib")
joblib.dump(feature_cols, MODELS / "features_v1.joblib")

print("\nSaved:")
print(" models/waterguard_rf_v1.joblib")
print(" outputs/predictions_v1.csv")
print(" outputs/feature_importance_v1.csv")

print("\nMODEL V1 COMPLETE")
