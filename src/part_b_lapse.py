"""Part B tasks 4-8: lapse problem, preprocessing, tuning, balancing, persistence, retention list."""
import numpy as np, pandas as pd, joblib
from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

CUTOFF = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")

# ---------- load ----------
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
cust = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
products = pd.read_csv("data/clean/products.csv")
orders = orders.drop(columns=[c for c in ["category"] if c in orders.columns]) \
               .merge(products[["product_id", "category"]], on="product_id", how="left")

# ---------- task 4: cohort, label, features (pre-cutoff orders only) ----------
pre = orders[orders["order_ts"] < CUTOFF]
comp = pre[pre["status"] == "completed"]

g = comp.groupby("customer_id")
feat = pd.DataFrame({
    "last_ts": g["order_ts"].max(),
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount": g["discount_pct"].mean(),
    "avg_delivery": g["delivery_minutes"].mean(),
    "fav_category": g["category"].agg(
        lambda s: sorted(s.value_counts().items(), key=lambda kv: (-kv[1], kv[0]))[0][0]),
})
feat["recency"] = (CUTOFF - feat["last_ts"]).dt.days
feat["cancel_rate"] = (pre["status"] == "cancelled").groupby(pre["customer_id"]).mean() \
                          .reindex(feat.index).fillna(0.0)
feat = feat.drop(columns="last_ts")
feat = feat.join(cust.set_index("customer_id")[["city", "age", "gender", "signup_date"]])
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
feat = feat.drop(columns="signup_date")

sept = orders[(orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < END) & (orders["status"] == "completed")]
active = set(sept["customer_id"])
y = pd.Series([0 if c in active else 1 for c in feat.index], index=feat.index, name="lapsed")

num_cols = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
cat_cols = ["city", "gender", "fav_category"]
X = feat[num_cols + cat_cols]

assert not X.isna().any().any(), "missing values in features"
print("Cohort:", len(X), "(ref 1156) | lapsed:", int(y.sum()), f"({y.mean()*100:.2f}%) (ref 443, 38.32%)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Train/test:", len(X_train), len(X_test), "(ref 924/232)")

def metrics(y_true, y_pred):
    return dict(precision=precision_score(y_true, y_pred, zero_division=0),
                recall=recall_score(y_true, y_pred, zero_division=0),
                f1=f1_score(y_true, y_pred, zero_division=0))

# ---------- baseline ----------
base_pred = np.zeros(len(y_test), dtype=int)
print("Baseline accuracy:", round(accuracy_score(y_test, base_pred) * 100, 1), "% (ref 61.6) | lapsed F1:",
      f1_score(y_test, base_pred, zero_division=0), "(ref 0)")

# ---------- task 5: preprocessing (fit on train only) ----------
def make_prep():
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ])

prep_check = make_prep().fit(X_train)
learned = prep_check.named_transformers_["num"].mean_[num_cols.index("frequency")]
print(f"Scaler-learned frequency mean (train only): {learned:.4f} | full-cohort frequency mean: {X['frequency'].mean():.4f}")

# ---------- task 6: tuning ----------
lr_pipe = Pipeline([("prep", make_prep()), ("clf", LogisticRegression(max_iter=1000))])
lr_gs = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
lr_gs.fit(X_train, y_train)

rf_pipe = Pipeline([("prep", make_prep()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_gs = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1")
rf_gs.fit(X_train, y_train)

print("\nLR best params:", lr_gs.best_params_, "| best CV F1:", round(lr_gs.best_score_, 4), "(ref C=1)")
print("RF best params:", rf_gs.best_params_, "| best CV F1:", round(rf_gs.best_score_, 4), "(ref depth 8, leaf 10)")

results = {}
for name, gs in [("LogReg", lr_gs), ("RandomForest", rf_gs)]:
    pred = gs.best_estimator_.predict(X_test)
    results[name] = (confusion_matrix(y_test, pred, labels=[0, 1]), metrics(y_test, pred), pred)
    print(f"\n{name} test confusion matrix [[TN FP],[FN TP]]:\n{results[name][0]}")
    print(f"{name} test lapsed-class:", {k: round(v, 4) for k, v in results[name][1].items()},
          "(ref F1: LR ~0.62, RF ~0.65)")

# model choice by CROSS-VALIDATED F1 (test set not used for selection)
sel_name = "RandomForest" if rf_gs.best_score_ >= lr_gs.best_score_ else "LogReg"
sel_gs = rf_gs if sel_name == "RandomForest" else lr_gs
sel_model = sel_gs.best_estimator_
print("\nSelected model (higher CV F1):", sel_name, "| test F1:", round(results[sel_name][1]["f1"], 4), "(must be >= 0.55)")

# ---------- task 7: oversample lapsed rows in TRAIN only ----------
pos_idx = np.where(y_train.values == 1)[0]
n_add = int((y_train == 0).sum() - (y_train == 1).sum())
rng = np.random.RandomState(42)
extra = rng.choice(pos_idx, size=n_add, replace=True)
X_bal = pd.concat([X_train, X_train.iloc[extra]])
y_bal = pd.concat([y_train, y_train.iloc[extra]])
print("\nBalanced train counts:", y_bal.value_counts().to_dict(), "(ref 570 each)")

lr_bal = clone(lr_gs.best_estimator_).fit(X_bal, y_bal)
bal_pred = lr_bal.predict(X_test)
m0, m1 = results["LogReg"][1], metrics(y_test, bal_pred)
print("LR unbalanced:", {k: round(v, 4) for k, v in m0.items()})
print("LR balanced  :", {k: round(v, 4) for k, v in m1.items()}, "(ref recall ~0.64 -> ~0.85, precision ~0.6)")

# ---------- task 8: interpretation ----------
names = rf_gs.best_estimator_.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_gs.best_estimator_.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop 5 RF importances:\n", imp.head(5).round(4))

lr_names = lr_gs.best_estimator_.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(lr_gs.best_estimator_.named_steps["clf"].coef_[0], index=lr_names)
print("\nLR coefficients (numeric features):\n", coefs[[f"num__{c}" for c in num_cols]].round(4))

# persist and reload
joblib.dump(sel_model, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
assert np.array_equal(sel_model.predict(X_test), reloaded.predict(X_test))
print("\nReloaded model reproduces test predictions exactly: True")

# retention list (illustrative, Premium flag is whole-window)
seg = pd.read_csv("outputs/customer_segments.csv")
premium_ids = set(seg.loc[seg["is_premium"].astype(str) == "True", "customer_id"])
proba = pd.Series(reloaded.predict_proba(X_test)[:, 1], index=X_test.index)
rl = pd.DataFrame({"customer_id": X_test.index, "lapse_probability": proba.values,
                   "actually_lapsed": y_test.values})
rl = rl[rl["customer_id"].isin(premium_ids)].sort_values("lapse_probability", ascending=False).head(10)
rl.to_csv("outputs/retention_list.csv", index=False)
print("\nRetention list rows:", len(rl), "| truly lapsed:", int(rl["actually_lapsed"].sum()))
print(rl.to_string(index=False))