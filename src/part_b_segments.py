import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

Path("outputs").mkdir(exist_ok=True)
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
comp = o[o.status == "completed"]

# ---- Task 1: RFM ----
snap = pd.Timestamp("2025-10-01 00:00:00")
g = comp.groupby("customer_id")
rfm = pd.DataFrame({
    "recency": (snap - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
}).reset_index()
print("customers:", len(rfm), "(ref 1175)")
print("recency median/min/max:", rfm.recency.median(), rfm.recency.min(), rfm.recency.max(), "(ref 19/0/180)")
print("frequency mean/max:", round(rfm.frequency.mean(), 2), rfm.frequency.max(), "(ref 9.46/33)")
print("monetary mean/max:", round(rfm.monetary.mean(), 2), round(rfm.monetary.max(), 2), "(ref 2552.06/14766.55)")

# ---- Task 2: K-means ----
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
print("\nK  inertia  silhouette")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    print(k, round(km.inertia_, 1), round(silhouette_score(X, km.labels_), 3))

# ---- Task 3: final K = 4 ----
K = 4
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"),
                                     recency=("recency", "mean"),
                                     frequency=("frequency", "mean"),
                                     monetary=("monetary", "mean")).round(2)
print("\nSegment profile:")
print(prof.to_string())

pca = PCA(n_components=2).fit(X)
ev = pca.explained_variance_ratio_
print("\nPCA explained variance:", ev.round(3), "total", round(ev.sum(), 3))

prof.to_csv("outputs/_segment_profile.csv")
rfm.to_pickle("outputs/_rfm_tmp.pkl")