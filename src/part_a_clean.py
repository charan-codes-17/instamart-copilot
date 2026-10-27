import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

RAW, CLEAN = Path("data/raw"), Path("data/clean")
CLEAN.mkdir(parents=True, exist_ok=True)

cust = pd.read_csv(RAW / "customers.csv")
prod = pd.read_csv(RAW / "products.csv")
orders = pd.read_csv(RAW / "orders.csv")

# ---------------- Task 2: audit ----------------
print("=== SHAPES ===")
for n, d in [("customers", cust), ("products", prod), ("orders", orders)]:
    print(n, d.shape)
    print(d.dtypes.to_string())
    miss = d.isna().sum()
    print(pd.DataFrame({"missing": miss, "pct": (miss / len(d) * 100).round(2)}).to_string())
    print()

amt_num = pd.to_numeric(orders["order_amount"], errors="coerce")
age_num = pd.to_numeric(cust["age"], errors="coerce")
audit = {
    "orders raw rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "pending rows": int((orders["status"] == "pending").sum()),
    "distinct order_id": orders["order_id"].nunique(),
    "non-numeric order_amount": int(amt_num.isna().sum()),
    "negative delivery_minutes": int((orders["delivery_minutes"] < 0).sum()),
    "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
    "blank age": int(age_num.isna().sum()),
    "age outside 10-100": int(((age_num < 10) | (age_num > 100)).sum()),
    "distinct raw city spellings": cust["city"].nunique(),
    "distinct non-blank gender spellings": cust["gender"].dropna().nunique(),
    "blank gender": int(cust["gender"].isna().sum()),
}
print("=== AUDIT TABLE ===")
for k, v in audit.items():
    print(f"{k:40s} {v}")

# ---------------- Task 3: clean customers ----------------
c = cust.copy()
c["city"] = c["city"].str.strip().str.title()
gmap = {"male": "male", "m": "male", "female": "female", "f": "female"}
c["gender"] = c["gender"].fillna("").str.strip().str.lower().map(gmap).fillna("unknown")
c["age"] = pd.to_numeric(c["age"], errors="coerce")
n_bad_age = int(((c["age"] < 10) | (c["age"] > 100)).sum())
c.loc[(c["age"] < 10) | (c["age"] > 100), "age"] = np.nan
n_missing_age = int(c["age"].isna().sum())
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
c["age"] = c["age"].round().astype(int)
c["signup_date"] = pd.to_datetime(c["signup_date"])

print("\n=== CUSTOMERS CLEANED ===")
print(c["city"].value_counts().to_string())
print(c["gender"].value_counts().to_string())
print("missing ages:", c["age"].isna().sum(), "| min/max age:", c["age"].min(), c["age"].max())
print("ages set to missing (out of range):", n_bad_age, "| total ages filled:", n_missing_age)

# ---------------- Task 4: clean orders ----------------
o = orders.copy()
log = {"raw rows": len(o)}
o["order_ts"] = pd.to_datetime(o["order_ts"])
before = len(o); o = o.drop_duplicates()
log["exact duplicates removed"] = before - len(o)
before = len(o)
o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
log["older records removed (keep latest per order_id)"] = before - len(o)

o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
o = o.merge(prod[["product_id", "category", "price_inr"]], on="product_id", how="left")
miss = o["order_amount"].isna()
log["order_amount filled from product price"] = int(miss.sum())
o.loc[miss, "order_amount"] = (o.loc[miss, "price_inr"] * o.loc[miss, "quantity"]
                               * (1 - o.loc[miss, "discount_pct"] / 100)).round(2)

neg = o["delivery_minutes"] < 0
log["negative delivery_minutes set to missing"] = int(neg.sum())
o.loc[neg, "delivery_minutes"] = np.nan
comp = o["status"] == "completed"
med = o[comp].groupby("category")["delivery_minutes"].transform("median")
need = comp & o["delivery_minutes"].isna()
log["completed delivery_minutes imputed"] = int(need.sum())
o.loc[need, "delivery_minutes"] = med[need]
# cancelled stay missing on purpose

o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)
print("\n=== ORDERS CLEANING LOG ===")
for k, v in log.items():
    print(f"{k:55s} {v}")
print("final rows:", len(o))
print(o["status"].value_counts().to_string())
print("missing amount:", o["order_amount"].isna().sum())
print("missing delivery (completed):", o.loc[o.status == "completed", "delivery_minutes"].isna().sum())
print("missing delivery (cancelled):", o.loc[o.status == "cancelled", "delivery_minutes"].isna().sum())
print("completed revenue:", round(o.loc[o.status == "completed", "order_amount"].sum(), 2))

# ---------------- Task 5: store ----------------
o_out = o.copy()
o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c_out = c.copy()
c_out["signup_date"] = c_out["signup_date"].dt.strftime("%Y-%m-%d")

c_out.to_csv(CLEAN / "customers.csv", index=False)
prod.to_csv(CLEAN / "products.csv", index=False)
o_out.to_csv(CLEAN / "orders.csv", index=False)

with sqlite3.connect("data/instamart.db") as con:
    c_out.to_sql("customers", con, if_exists="replace", index=False)
    prod.to_sql("products", con, if_exists="replace", index=False)
    o_out.to_sql("orders", con, if_exists="replace", index=False)
print("\nWrote data/clean/*.csv and data/instamart.db")