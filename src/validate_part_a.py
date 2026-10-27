import pandas as pd

c = pd.read_csv("data/clean/customers.csv")
o = pd.read_csv("data/clean/orders.csv")
comp = o[o.status == "completed"]

checks = {
    "orders rows == 12309": len(o) == 12309,
    "completed == 11110": (o.status == "completed").sum() == 11110,
    "cancelled == 1199": (o.status == "cancelled").sum() == 1199,
    "no pending": (o.status == "pending").sum() == 0,
    "no missing amount": o.order_amount.isna().sum() == 0,
    "completed delivery missing == 0": comp.delivery_minutes.isna().sum() == 0,
    "cancelled delivery missing == 1199": o[o.status == "cancelled"].delivery_minutes.isna().sum() == 1199,
    "revenue == 2998668.80": round(comp.order_amount.sum(), 2) == 2998668.80,
    "six cities": c.city.nunique() == 6,
    "Pune == 227": (c.city == "Pune").sum() == 227,
    "female == 610": (c.gender == "female").sum() == 610,
    "male == 560": (c.gender == "male").sum() == 560,
    "unknown == 30": (c.gender == "unknown").sum() == 30,
    "no missing age": c.age.isna().sum() == 0,
    "age within 18-60": c.age.between(18, 60).all(),
}
for k, v in checks.items():
    print("PASS" if v else "FAIL", k)
if all(checks.values()):
    print("ALL CHECKPOINTS PASSED")