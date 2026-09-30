import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.impute import SimpleImputer
import joblib

print("=" * 70)
print("WATERGUARD AI SAUDI — MODEL V2")
print("=" * 70)

DATA = Path("data")
OUT = Path("outputs")
MODELS = Path("models")

print("\n[1/6] Loading SCADA...")

p = pd.read_excel(DATA/"2018_SCADA.xlsx", sheet_name="Pressures (m)")
d = pd.read_excel(DATA/"2018_SCADA.xlsx", sheet_name="Demands (L_h)")
f = pd.read_excel(DATA/"2018_SCADA.xlsx", sheet_name="Flows (m3_h)")
l = pd.read_excel(DATA/"2018_SCADA.xlsx", sheet_name="Levels (m)")

for x in [p,d,f,l]:
    x["Timestamp"] = pd.to_datetime(x["Timestamp"])

print("[2/6] Engineering features...")

X = pd.DataFrame({"Timestamp":p["Timestamp"]})

P=p.drop(columns="Timestamp")
D=d.drop(columns="Timestamp")
F=f.drop(columns="Timestamp")
L=l.drop(columns="Timestamp")

# Keep individual pressure sensors.
# These can contain local hydraulic information lost by averages.
for col in P.columns:
    X[f"P_{col}"] = P[col]

# Network summaries
X["pressure_mean"]=P.mean(axis=1)
X["pressure_min"]=P.min(axis=1)
X["pressure_max"]=P.max(axis=1)
X["pressure_std"]=P.std(axis=1)
X["pressure_range"]=P.max(axis=1)-P.min(axis=1)

X["demand_total"]=D.sum(axis=1)
X["demand_mean"]=D.mean(axis=1)
X["demand_std"]=D.std(axis=1)

for col in F.columns:
    X[f"F_{col}"]=F[col]

X["flow_total"]=F.sum(axis=1)
X["flow_mean"]=F.mean(axis=1)
X["flow_std"]=F.std(axis=1)
X["tank_level"]=L.iloc[:,0]

# Cyclical time-of-day representation.
# Intentionally NO month feature.
hour = (
    X["Timestamp"].dt.hour +
    X["Timestamp"].dt.minute/60
)

X["hour_sin"]=np.sin(2*np.pi*hour/24)
X["hour_cos"]=np.cos(2*np.pi*hour/24)

# Short-term hydraulic changes
for col in ["pressure_mean","pressure_min",
            "pressure_std","flow_total","tank_level"]:
    X[f"{col}_diff_5m"]=X[col].diff()
    X[f"{col}_diff_30m"]=X[col].diff(6)

print("Features:", len(X.columns)-1)

print("[3/6] Creating ground truth...")

leaks=pd.read_csv(
    DATA/"2018_Leakages.csv",
    sep=";", decimal=",",
    parse_dates=["Timestamp"]
)

leakcols=[c for c in leaks.columns if c!="Timestamp"]
leaks["total_leak"]=leaks[leakcols].sum(axis=1)
leaks["target"]=(leaks["total_leak"]>=40).astype(int)

df=X.merge(
    leaks[["Timestamp","total_leak","target"]],
    on="Timestamp"
).sort_values("Timestamp").reset_index(drop=True)

features=[c for c in X.columns if c!="Timestamp"]

# -------------------------------------------------
# Chronological 60 / 10 / 30 split
# -------------------------------------------------
n=len(df)
train_end=int(n*.60)
val_end=int(n*.70)

train=df.iloc[:train_end]
val=df.iloc[train_end:val_end]
test=df.iloc[val_end:]

print("\nTRAIN:",train.Timestamp.min(),"→",train.Timestamp.max())
print("VALID:",val.Timestamp.min(),"→",val.Timestamp.max())
print("TEST :",test.Timestamp.min(),"→",test.Timestamp.max())

imputer=SimpleImputer(strategy="median")

Xtr=imputer.fit_transform(train[features])
Xv=imputer.transform(val[features])
Xt=imputer.transform(test[features])

ytr=train["target"].values
yv=val["target"].values
yt=test["target"].values

print("[4/6] Training Random Forest...")

model=RandomForestClassifier(
    n_estimators=300,
    max_depth=16,
    min_samples_leaf=4,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

model.fit(Xtr,ytr)

print("[5/6] Selecting threshold USING VALIDATION ONLY...")

val_prob=model.predict_proba(Xv)[:,1]

thresholds=np.arange(.05,.81,.01)
rows=[]

for t in thresholds:
    pred=(val_prob>=t).astype(int)
    rows.append({
        "threshold":t,
        "precision":precision_score(yv,pred,zero_division=0),
        "recall":recall_score(yv,pred,zero_division=0),
        "f1":f1_score(yv,pred,zero_division=0)
    })

threshold_df=pd.DataFrame(rows)
best=threshold_df.loc[threshold_df["f1"].idxmax()]
threshold=float(best["threshold"])

print("\nBest validation threshold:",round(threshold,2))
print(best.to_string())

print("\n[6/6] FINAL TEST — untouched future data")

test_prob=model.predict_proba(Xt)[:,1]
pred=(test_prob>=threshold).astype(int)

precision=precision_score(yt,pred,zero_division=0)
recall=recall_score(yt,pred,zero_division=0)
f1=f1_score(yt,pred,zero_division=0)
auc=roc_auc_score(yt,test_prob)

print("\nCONFUSION MATRIX")
print(confusion_matrix(yt,pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(yt,pred,digits=4))

print("\nWATERGUARD V2 FINAL RESULTS")
print(f"Threshold : {threshold:.2f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")

importance=pd.DataFrame({
    "feature":features,
    "importance":model.feature_importances_
}).sort_values("importance",ascending=False)

print("\nTOP 15 FEATURES")
print(importance.head(15).to_string(index=False))

results=test[
    ["Timestamp","total_leak","target"]
].copy()

results["prediction"]=pred
results["risk_probability"]=test_prob

results.to_csv(OUT/"predictions_v2.csv",index=False)
importance.to_csv(OUT/"feature_importance_v2.csv",index=False)
threshold_df.to_csv(OUT/"threshold_validation_v2.csv",index=False)

joblib.dump(model,MODELS/"waterguard_rf_v2.joblib")
joblib.dump(imputer,MODELS/"imputer_v2.joblib")
joblib.dump(features,MODELS/"features_v2.joblib")
joblib.dump(threshold,MODELS/"threshold_v2.joblib")

print("\nV2 COMPLETE")
