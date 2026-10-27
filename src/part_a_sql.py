import re
import sqlite3
import pandas as pd

text = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"^-- (Q\d)[^\n]*\n", text, flags=re.M)
queries = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}

con = sqlite3.connect("data/instamart.db")
res = {q: pd.read_sql_query(sql, con) for q, sql in queries.items()}

print("Q1 customers:", len(res["Q1"]), "(ref 30)")

q2 = res["Q2"]
print("Q2 rows:", len(q2), "(ref 30) | sum:", round(q2.revenue.sum(), 2), "(ref 2998668.80)")
print("Q2 largest:", q2.iloc[0].to_dict(), "(ref Bengaluru/Electronics 233909.45)")

print("Q3 products:", len(res["Q3"]), "(ref 11)")
print("Q4 customers:", len(res["Q4"]), "(ref 25)")

q5 = res["Q5"]
print("Q5 customers:", len(q5), "(ref 550) | avg:", q5.avg_total.iloc[0], "(ref 2552.06)")

q6 = res["Q6"]
print("Q6 top per city:")
print(q6[q6.row_num == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows where RANK != ROW_NUMBER:", int((q6.rnk != q6.row_num).sum()), "(ref 8)")

q7 = res["Q7"]
print("Q7 daily rows:", len(q7), "(ref 183) | last cumulative:", q7.cumulative_revenue.iloc[-1], "(ref 2998668.80)")
con.close()