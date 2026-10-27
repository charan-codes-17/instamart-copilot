import pandas as pd

rfm = pd.read_pickle("outputs/_rfm_tmp.pkl")
prof = rfm.groupby("segment_id")["monetary"].mean()
premium_id = prof.idxmax()

names = {0: "Casual Regulars", 1: "Premium Big Spenders",
         2: "Loyal Frequent Shoppers", 3: "Lapsing Dormant"}
# safety: make the Premium name follow the data, not my guess
if names[premium_id] != "Premium Big Spenders":
    raise SystemExit(f"Premium segment is {premium_id}; fix the names dict")

rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
rfm[["customer_id", "recency", "frequency", "monetary",
     "segment_id", "segment_name", "is_premium"]].to_csv("outputs/customer_segments.csv", index=False)
print(rfm.groupby(["segment_id", "segment_name", "is_premium"]).size())