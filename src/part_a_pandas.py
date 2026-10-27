import sqlite3
import pandas as pd
import numpy as np

o = pd.read_csv("data/clean/orders.csv")
c = pd.read_csv("data/clean/customers.csv")
p = pd.read_csv("data/clean/products.csv")
comp = o[o.status == "completed"].merge(c[["customer_id", "city"]], on="customer_id") \
                                 .merge(p[["product_id", "category"]], on="product_id")

# (a) pivot cross-check
pv = pd.pivot_table(comp, values="order_amount", index="city", columns="category",
                    aggfunc="sum", margins=True, margins_name="Total")
print(pv.round(2).to_string())
print("\nPivot grand total:", round(pv.loc["Total", "Total"], 2), "| SQL total: 2998668.80")

g = comp.groupby("city").agg(total=("order_amount", "sum"),
                             mean=("order_amount", "mean"),
                             count=("order_amount", "count")).round(2)
print("\nPer-city sum / mean / count:")
print(g.to_string())

# (b) distribution
x = comp["order_amount"]
q1, q3 = x.quantile(0.25), x.quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("\nmean", round(x.mean(), 2), "| median", x.median(), "| skew", round(x.skew(), 2))
print("Q1", q1, "| Q3", q3, "| IQR", iqr)
print("lower fence", lo, "| upper fence", hi)
print("outliers below:", int((x < lo).sum()), "| above:", int((x > hi).sum()))

corr = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print("\nCorrelation matrix:")
print(corr.round(3).to_string())
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
top = pairs.abs().idxmax()
print("Largest |corr| pair:", top, round(pairs[top], 3))