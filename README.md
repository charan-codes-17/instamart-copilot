# Instamart Customer Intelligence & Support Copilot

## Overview

This repository contains the Instamart Customer Intelligence & Support Copilot, built in four connected Parts on synthetic, fictional data (all amounts in INR).

| Part | What it does | Main files |
|---|---|---|
| A | Cleans the raw exports, stores them in SQLite, answers business questions in SQL and pandas | `src/part_a_*.py`, `sql/queries.sql`, `data/instamart.db` |
| B | RFM segments (K-means, K=4) and a lapse-prediction model | `src/part_b_*.py`, `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv` |
| C | Support-ticket triage with privacy, retry, validation and budget guardrails | `src/part_c_triage.py`, `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv` |
| D | Policy RAG assistant and a tool-using refund agent with safe stopping and resume | `src/part_d_rag.py`, `src/part_d_agent.py`, `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/` |

`generate_data.py` and `mock_llm.py` are the supplied files and are committed unmodified. No API key, account or network call is used anywhere: every model call goes through the offline `mock_llm.py`. Libraries: Python standard library, pandas, numpy, scikit-learn, joblib. Parts chain: Part B reads Part A's clean CSVs, Part C reads Part B's segment file, and Part D's agent reads Part A's database.

## Reproduce

Run from the repository root, in this order (Python 3.11 or newer). On Linux/macOS activate with `source .venv/bin/activate`; on Windows PowerShell use `.\.venv\Scripts\Activate.ps1`.

1. `python -m venv .venv`
2. Activate the environment (see above)
3. `pip install -r requirements.txt`
4. `python generate_data.py` (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
5. `python src/part_a_clean.py`
6. `python src/validate_part_a.py` (prints `ALL CHECKPOINTS PASSED`)
7. `python src/part_a_sql.py`
8. `python src/part_a_pandas.py`
9. `python src/part_b_segments.py`
10. `python src/part_b_finalize.py` (reads the temporary file written by step 9)
11. `python src/part_b_lapse.py`
12. `python src/part_c_triage.py`
13. `python src/part_d_rag.py`
14. `python src/part_d_agent.py` (clears and rewrites the ledger, run log and `state/` first, so it can be re-run safely)

Every script uses relative paths and fixed seeds (`random_state=42`, `random.seed(42)`). The two temporary files `outputs/_rfm_tmp.pkl` and `outputs/_segment_profile.csv` are created by step 9 and are not committed.

## Part A

### A1. Setup

Repository on branch `main`, isolated virtual environment (`python -m venv .venv`), libraries installed with `pip`, `requirements.txt` written with `pip freeze`, `.gitignore` listing `.env`, `.venv/` and `__pycache__/`. The two supplied files were copied unmodified and `python generate_data.py` printed `generated: customers=1200 products=40 orders_raw=13084 tickets=164`.

### A2. Audit of the raw data

| Table | Shape | Columns and data types |
|---|---|---|
| customers | 1200 x 5 | customer_id str, city str, signup_date str, age float64, gender str |
| products | 40 x 4 | product_id str, product_name str, category str, price_inr int64 |
| orders | 13084 x 9 | order_id str, customer_id str, product_id str, quantity int64, order_ts str, status str, discount_pct int64, order_amount str, delivery_minutes float64 |

Missing values (all other columns have none): customers `age` 61 (5.08%), `gender` 30 (2.50%); orders `order_amount` 53 (0.41%), `delivery_minutes` 2,541 (19.42%). The 53 missing amounts are the `N/A` entries, which pandas reads as missing; the remaining unparseable entries are the text `error`, so together there are 120 non-numeric amounts.

| Audit figure | Value |
|---|---|
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Rows with status `pending` | 656 |
| Distinct `order_id` | 12,309 |
| Non-numeric `order_amount` | 120 |
| Negative `delivery_minutes` | 106 |
| Blank `delivery_minutes` | 2,541 |
| Blank `age` | 61 |
| `age` outside 10-100 | 6 |
| Distinct raw `city` spellings | 24 |
| Distinct non-blank raw `gender` spellings | 10 |
| Blank `gender` | 30 |

### A3. Cleaning `customers`

Rules: (a) strip spaces and apply title case to `city`; (b) map every gender spelling to `male`, `female` or `unknown` (blank becomes `unknown`); (c) set ages outside 10-100 to missing, then fill all missing ages with the median age of the customer's city (`groupby` + `transform`); (d) convert `signup_date` to datetime.

Result: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; female 610, male 560, unknown 30. Six out-of-range ages were set to missing and 67 ages were filled in total (61 blank + 6). There are 0 missing ages and every age lies between 18 and 60.

### A4. Cleaning `orders`

| Step | Rule | Rows or values affected |
|---|---|---|
| 1 | Convert `order_ts` to datetime | all rows |
| 2 | Remove exact duplicate rows | 119 removed |
| 3 | Keep only the record with the latest `order_ts` per `order_id` | 656 older (pending) records removed |
| 4 | Convert `order_amount` to numeric; fill missing from `round(price_inr * quantity * (1 - discount_pct/100), 2)` | 112 amounts filled |
| 5a | Set negative `delivery_minutes` to missing | 105 values set to missing |
| 5b | Fill missing delivery of `completed` orders with the median of that product category among completed orders (`groupby` + `transform`) | 774 values imputed |
| 5c | Leave `delivery_minutes` missing for `cancelled` orders (never delivered, so imputing would invent data) | 1,199 left missing |

The audit counts 106 negative delivery times but the cleaning step changes 105, because one negative value sat on a row that was removed in step 2 or 3.

Result: 12,309 rows (one per `order_id`); 11,110 completed and 1,199 cancelled; no pending rows; 0 missing `order_amount`; 0 missing delivery among completed orders; completed revenue INR 2,998,668.80.

### A5. Storage and validation

The clean tables are written to `data/clean/customers.csv`, `products.csv`, `orders.csv` and to `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `src/validate_part_a.py` asserts 15 reference checkpoints and prints `ALL CHECKPOINTS PASSED`.

### A6. SQL analysis (`sql/queries.sql`, run with `pd.read_sql_query`)

| Query | Technique | Result |
|---|---|---|
| Q1 | WHERE (`order_amount > 500`) then HAVING (total > 8000) | 30 customers |
| Q2 | Three-table join, revenue by city and category | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 | Scalar subquery, price above catalogue average | 11 products |
| Q4 | LEFT JOIN, customers with no completed order | 25 customers |
| Q5 | Two CTEs, customers above the average total spend | 550 customers; average INR 2,552.06 |
| Q6 | RANK, DENSE_RANK, ROW_NUMBER partitioned by city | see below |
| Q7 | Daily revenue with running total | 183 rows; final cumulative INR 2,998,668.80 |

Q6 top customer per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10). Exactly 8 rows have `RANK()` different from `ROW_NUMBER()`: when two customers in a city have the same total spend, `RANK()` gives them the same rank while `ROW_NUMBER()` still gives them different consecutive numbers, so the two differ for the tied rows and for the rows after them. Spend is rounded to 2 decimals inside the CTE so that equal totals tie exactly.

### A7. pandas cross-check and exploratory analysis

The `pd.pivot_table(..., margins=True)` grand total equals the SQL total, INR 2,998,668.80.

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed `order_amount`: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, IQR 232.20, fences -283.50 and 645.30, with 0 outliers below the lower fence and 897 above the upper fence. The correlation with the largest absolute value is `order_amount` with `discount_pct` at about -0.095; `order_amount` with `quantity` is +0.087.

- **What is this data about?** Six months of quick-commerce orders (April to September 2025) across six cities, with products, discounts and delivery times.
- **Distributions and statistics?** Order amounts are strongly right-skewed: the mean (269.91) is about double the median (135.00), and most orders are small grocery and dairy baskets.
- **Outliers or anomalies?** 897 orders exceed the upper fence of 645.30. They are genuine electronics purchases, so they are kept, not treated as errors. The real anomalies were the planted duplicates, impossible ages and negative delivery times, which were cleaned earlier.
- **Relationships?** All correlations are weak; the largest, `order_amount` vs `discount_pct` at -0.095, is small, and a correlation of this size does not establish causation.

## Part B


### B1. RFM table
Snapshot 2025-10-01 00:00:00, completed orders only. 1,175 customers. Recency median 19 days (min 0, max 180); frequency mean 9.46 (max 33); monetary mean INR 2,552.06 (max INR 14,766.55).

### B2. K-means
RFM columns standardised with StandardScaler; `KMeans(n_clusters=K, n_init=10, random_state=42)`.

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

The elbow is at K=4: inertia falls by 41% from K=3 to K=4 (1268.2 to 743.1) but only 17% from K=4 to K=5. K=4 also has the highest silhouette (0.526) among {3, 4, 5} and gives four segments the business can act on, so **final K = 4**.
Inertia always falls as K grows because more centroids can always sit closer to the points (at K = number of customers it would be 0), so K=8 beats K=2 automatically. That fact alone cannot pick K; we need the elbow, the silhouette and interpretability.

### B3. Segments
Cluster numbers are only labels and carry no ordering.

| Segment | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Premium | Recommended action |
|---|---|---|---|---|---|---|---|
| 0 | Casual Regulars | 436 | 28.86 | 3.77 | 1,057.62 | No | Category-based offers to lift order frequency |
| 1 | Premium Big Spenders | 138 | 24.12 | 6.77 | 7,129.70 | Yes | Priority support and early access to electronics offers |
| 2 | Loyal Frequent Shoppers | 419 | 8.77 | 19.40 | 3,410.02 | No | Loyalty rewards; keep delivery fast |
| 3 | Lapsing Dormant | 182 | 109.04 | 2.20 | 685.99 | No | Win-back coupon campaign |

Premium = the segment with the highest mean monetary value. PCA (2 components) on the scaled RFM matrix: explained variance 0.634 and 0.218, total 0.852. Output: `outputs/customer_segments.csv`.

### B4. Lapse problem
Cutoff 2025-09-01. Cohort: customers with at least one completed order before the cutoff = 1,156. Label `lapsed = 1` if no completed order dated 2025-09-01 to 2025-09-30: 443 lapsed (38.32%). Split `train_test_split(test_size=0.2, stratify=y, random_state=42)`: 924 train, 232 test. Baseline "always not lapsed" on test: accuracy 61.6%, lapsed-class F1 0.
Features (orders before the cutoff only): recency, frequency, monetary, average discount_pct, cancellation rate, average delivery_minutes, favourite category, city, age, gender, tenure in days. No order dated on or after 2025-09-01 entered any feature, and the task 3 segment labels were not used as features.

### B5. Preprocessing without leakage
A ColumnTransformer inside a scikit-learn Pipeline: StandardScaler on numeric columns and OneHotEncoder(drop="first") on city, gender and favourite category. No encoder, scaler or resampler saw test rows: they are fitted on training rows only (and, inside GridSearchCV, on each training fold only). An assertion confirms no feature column has a missing value.
Scaler-learned mean of frequency (train only) = 8.1147 versus the full-cohort mean 8.0285; they differ because the scaler sees only the 924 training rows, not the test rows.

### B6. Tuning and evaluation
| Model | Grid | Best params | Best CV F1 |
|---|---|---|---|
| Logistic regression | C in {0.01, 0.1, 1, 10, 100} | C=100 | 0.6506 |
| Random forest | max_depth {3,5,8} x min_samples_leaf {5,10,20} | depth 8, leaf 10 | 0.6811 |

Test set (lapsed class as positive; confusion matrix rows = actual, columns = predicted [not lapsed, lapsed]):

| Model | Confusion matrix | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (never lapsed) | [[143, 0], [89, 0]] | 0 | 0 | 0 |
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[113, 30], [31, 58]] | 0.659 | 0.652 | 0.655 |

**Selected model: random forest** (chosen on cross-validated F1, test F1 0.655 is at least 0.55). Missing a customer who is about to lapse costs all of that customer's future revenue, while wrongly targeting a loyal customer costs only a small coupon or message, so recall matters more than precision here. Both models reach the same recall (0.652), but the forest has the higher cross-validated F1 (0.681 versus 0.651) and slightly higher test precision (0.659 versus 0.644), so it wastes fewer offers. If the team wants to catch more lapsers, the decision threshold can be lowered on top of this model.

### B7. Class-balance experiment
Lapsed training rows were randomly duplicated (with replacement, random_state=42) until both classes had 570 rows; the best logistic regression was refitted; the test set is unchanged.

| Logistic regression | Precision | Recall | F1 |
|---|---|---|---|
| Original training set | 0.644 | 0.652 | 0.648 |
| Balanced training set | 0.589 | 0.854 | 0.697 |

Recall moved most (+0.20); precision did not rise (it fell by 0.055). For the retention team this means catching about 85% of lapsing customers instead of 65%, at the price of about 41% of the flagged customers being loyal ones, which is acceptable only if the offer is cheap.

### B8. Interpretation and persistence
Top 5 random-forest importances: frequency 0.280, monetary 0.271, recency 0.115, avg_delivery 0.061, cancel_rate 0.053. How often and how much a customer buys are the strongest signals; time since last order and slow deliveries come next.
Logistic regression coefficients (standardised features, C=100): frequency -1.70 (more orders, much less likely to lapse); recency +0.04 (a longer gap makes lapse slightly more likely, a very weak effect); monetary +0.28 (positive once frequency is held fixed, which is counter-intuitive and is most likely an artefact of monetary and frequency being strongly correlated, so it should not be read as "big spenders lapse").
The selected model was saved with `joblib.dump` to `outputs/lapse_model.joblib`; the reloaded model reproduces the original test predictions exactly.
Retention list (`outputs/retention_list.csv`): top 10 Premium customers of the test set by predicted lapse probability (highest 0.487); 1 of the 10 truly lapsed. This list is illustrative and not a leakage-free evaluation, because the Premium flag comes from the whole-window RFM of task 3, which includes the September window that defines the lapse label.


## Part C


### C1. Input guardrails
Of 164 tickets, 4 are empty (status `skipped_empty`) and 4 exceed `MAX_INPUT_TOKENS = 400` (status `flagged_long`, each estimated at 734 tokens). None of these 8 was sent to the model: model calls were made for exactly the 156 eligible tickets (asserted in code).

### C2. Redaction
Patterns applied in this order: email, phone, order id.

| Type | Substitutions |
|---|---|
| Email | 55 |
| Phone | 91 |
| Order id | 132 |
| Total | 278 |

Re-applying the three patterns to the redacted texts finds 0 matches. The user message sent to the model is exactly the redacted text, and an assertion confirms no email, phone or order id appears in any prompt.

### C3. Prompts
Zero-shot prompt: about 33 estimated tokens, contains `CLASSIFY`, the four categories and the JSON-only format, no `EXAMPLES:`. Few-shot prompt: about 120 estimated tokens, the same text plus an `EXAMPLES:` line with four labelled examples (billing, technical, fraud, delivery).

### C4. Retries
Maximum 3 retries per ticket; wait before retry n = `0.05 x 2**n x random.uniform(0.9, 1.15)`; one log line per retry; `mock_llm.reset()` before every full run. A full run has 19 `TransientLLMError`s, all recovered by one retry (every retry is retry number 0), 0 permanent errors. Observed waits: 0.0451 to 0.0574 s, inside the 0.045 to 0.0575 s range.

### C5. Output validation
Replies are parsed as JSON (`classified_json` if the category is allowed). Otherwise the lowercase words from `re.findall(r"[a-z]+", text.lower())` are checked and the reply is accepted (`classified_keyword`) only if exactly one allowed category word is present. Everything else is `flagged_invalid_output` and goes to human review; nothing is guessed.

| Status | Count (each full run) |
|---|---|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |

### C6. Zero-shot versus few-shot

| Run | Classified | Correct | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|---|
| Zero-shot | 147 | 126 | 85.7% | 9,556 | 65.01 |
| Few-shot | 147 | 147 | 100.0% | 23,128 | 157.33 |

Zero-shot confusion table (rows = true category, columns = predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

The few-shot prompt costs about 92 more tokens per classified ticket (157.33 versus 65.01), 13,572 more tokens in total, and removes all 21 zero-shot errors, so each corrected ticket costs about 646 tokens. It is worth paying when errors are expensive: 11 of the 34 fraud tickets (32%) were mislabelled as billing and would miss the security team's 2-hour handover. It is not worth paying for low-stakes bulk sorting where an error only means a slightly slower queue.

### C7. Budget and routing
Budget demo: zero-shot with `TOKEN_BUDGET = 6000` (prompt about 33 tokens) stopped after 6,017 tokens (the check happens before each call, so the last call may cross the budget); 58 tickets ended `budget_exhausted` and went to human review. Saved to `outputs/triage_budget_demo.csv`.

Final run: few-shot, `TOKEN_BUDGET = 100000`, saved to `outputs/triage_results.csv` (164 rows). Routing: skipped, flagged and exhausted tickets go to `human_review` (NORMAL); otherwise fraud goes to `security_team` (HIGH); otherwise premium customers (from `outputs/customer_segments.csv`) go to `premium_desk` (HIGH); everything else goes to `support_queue` (NORMAL). A customer missing from the segment file is treated as not premium.

| Route | Priority | Tickets |
|---|---|---|
| human_review | NORMAL | 17 (4 empty, 4 long, 9 invalid output) |
| security_team | HIGH | 34 |
| premium_desk | HIGH | 11 |
| support_queue | NORMAL | 102 |

Rule-validated by code (JSON or single-keyword check): 147 tickets. Sent to humans: 17 tickets.


## Part D

Script: `src/part_d_rag.py` (tasks 1-4; the agent in tasks 5-9 imports `retrieve` from it).

### Tasks 1-4: Policy RAG assistant

**Chunking (task 1).** Each of the five policy files is split into 40-word chunks with a 10-word overlap (a new chunk starts 30 words after the previous one, and splitting stops after the first chunk whose end reaches or passes the last word). Every chunk keeps its source file name.

| File | Chunks |
|---|---|
| refund_policy.txt | 4 |
| delivery_policy.txt | 3 |
| escalation_policy.txt | 3 |
| payment_policy.txt | 3 |
| privacy_policy.txt | 2 |
| **Total** | **15** |

**Retrieval (task 2).** `TfidfVectorizer()` is fitted on the 15 chunk texts, a query is transformed with the same fitted vectorizer, and chunks are ranked by cosine similarity. `retrieve(query, k=3)` returns dictionaries with `source`, `score` and `text`. For the query `refund policy late delivery` the top three chunks all come from `refund_policy.txt` (scores 0.2719, 0.2514, 0.2454).

TF (term frequency) measures how often a word appears inside a chunk, and IDF (inverse document frequency) lowers the weight of words that appear in many chunks, so rare, distinctive words count more. A limitation is that TF-IDF only matches exact words: a question phrased as "money back" would not match a chunk that says "refund", because the two phrasings share no words even though they mean the same thing.

**Augment and answer (task 3).** `answer(question)` retrieves three chunks and sends two messages to `mock_llm.chat`: a system message containing `POLICY ASSISTANT` (answer only from the context, otherwise say the information could not be found) and a user message of the form `CONTEXT:`, the three chunks each prefixed with `[<source file name>]`, then `QUESTION:` and the question. It returns the model text and the sources used.

**Evaluation (task 4).** Questions come from `data/raw/policy_eval.csv` (12 answerable, 2 not covered by any document).

| Metric | Overlap 10 | Overlap 0 |
|---|---|---|
| Chunks | 15 | 13 |
| Answerable: `expected_doc` in top 3 | 12/12 | 12/12 |
| Answerable: answer contains `expected_fact` | 12/12 | 10/12 |
| Unanswerable: fallback `I could not find...` | 2/2 | 2/2 |

With zero overlap two questions break. "What delivery time makes a completed order eligible for a full refund?" returns only "receives a full refund of the order amount.", because the sentence containing "more than 45 minutes" is cut across a chunk boundary and the condition ends up in the previous chunk. "How soon is money returned when a payment fails but is deducted?" returns the sentence up to "returned automatically", and the words "within 48 hours" fall into the next chunk. In both cases the right document was retrieved, but the answer sentence was split in two. The 10-word overlap repeats the boundary words in both neighbouring chunks, so these short facts stay together in at least one chunk.


### Tasks 5-8: Refund agent

Script: `src/part_d_agent.py` (imports `retrieve` from `src/part_d_rag.py`). Running it clears and rewrites `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `outputs/agent_demos.json` and `state/`, so every run reproduces the same files.

**Tools (task 5).** `TOOLS_SPEC` lists five tools and `TOOL_MAP` maps each name to its Python function: `get_order` (reads `data/instamart.db`), `search_policy` (returns `retrieve` output under `results`), `calculate_refund` (100% of the order amount if delivery took more than 45 minutes, otherwise 0), `issue_refund` and `escalate_to_human` (each append one JSON line with a timestamp and the tool name to `outputs/refund_ledger.jsonl`). The agent looks the requested name up in `TOOL_MAP` and calls the function with the model's `arguments`.

**ReAct loop (task 6).** `run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)` calls the model, executes any requested tool, appends the assistant and tool messages, and repeats until the model gives a final answer. The status is the word before the colon in that answer. All seven scenarios give the expected outcome, and every refund amount equals `expected_refund_inr` within 0.01.

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,067 |
| S2 | REFUND | 135.00 | 5 | 1,067 |
| S3 | ESCALATE | 1,999.00 (escalated, not issued) | 5 | 1,054 |
| S4 | DENIED | 0.00 | 4 | 741 |
| S5 | DENIED | 0.00 | 3 | 439 |
| S6 | NOT_FOUND | 0.00 | 2 | 140 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,204 |

S7 contains `[LOOP]`, so the mock keeps requesting `search_policy`. It uses all six model calls and stops with `FALLBACK_MAX_ITERATIONS`, below the default token limit of 6000, so the token limit does not stop it early.

**Stop safely (task 7).** Three limits are enforced: `max_iterations` model calls, `token_limit` (checked after each call, before the requested tool is executed) and `time_limit_seconds` (checked at the top of each iteration with `time.monotonic()`). Every fallback returns the steps completed so far and a short human-handoff summary in `final_answer`.

| Demonstration | Status | Model calls | Tokens used |
|---|---|---|---|
| S1 with `token_limit=200` | FALLBACK_TOKEN_LIMIT | 3 | 456 |
| S1 with `time_limit_seconds=0` | FALLBACK_TIMEOUT | 0 | 0 |

In the token-limit demo, the tool requested by the third call was not executed (only `get_order` and `search_policy` had run). In the timeout demo the loop stops before the first model call.

**Checkpoint and resume (task 8).** After every tool observation the agent writes `state/<run_id>.json` containing the run id, scenario id, full message list, `tokens_used`, step number and the tools executed so far. `resume_agent(run_id)` reloads that file and continues. For S1 a simulated crash was raised right after the second tool execution. The checkpoint then listed `get_order, search_policy`, and after `resume_agent` the tools executed across both parts were, in order:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`

That is four executions, none repeated. The final status is `REFUND` with INR 128.25 (5 model calls, 1,067 tokens). The ledger therefore has four lines: S1, S2 and S3 from the scenario runs plus one more S1 line (run id `S1_crash`) written by the crash-and-resume demonstration. That extra line is a demo of resume and not a duplicate refund in the scenario runs.

### Task 9: ReAct versus plan-and-execute

ReAct makes one model call per step. Measured: S1 and S2 took 5 calls (1,067 tokens), S3 5 (1,054), S4 4 (741), S5 3 (439), S6 2 (140). Every call re-sends the growing message history, so tokens grow faster than calls. Plan-and-execute asks the model once for the full step list, runs the tools without the model, and calls it again only to write the answer or to replan. A normal refund would need about 2 calls instead of 5 and fewer tokens, because the history is not re-sent at each step.

The cost is rigidity. The plan is written before any observation, so it cannot know that an order is missing (S6), cancelled (S5), on time (S4) or above INR 1000 (S3, which needed `escalate_to_human` instead of `issue_refund`). It must either contain conditional branches or replan after an unexpected observation, and each replan adds calls and tokens back. In ReAct each observation decides the next step: S6 stopped after 2 calls on NOT_FOUND, and S5 stopped after the policy search without calling `calculate_refund`.

For this refund workflow I would choose ReAct. Money is involved, the correct path depends on real order data, and the extra two or three calls are cheap. The hard limits (6 calls, token cap, timeout) already bound its cost, as S7 shows: it was stopped at 6 calls and 3,204 tokens.



## Limitations and Next Steps

- **The Premium flag is a post-hoc whole-window label.** It comes from RFM over the whole order window, which includes September, the month that defines the lapse label, so the retention list is illustrative and not a leakage-free evaluation (1 of its 10 customers truly lapsed). The model's own features use only orders before 2025-09-01.
- **Synthetic data and a mock model.** All results come from generated data and the rule-based `mock_llm.py`. The 85.7% zero-shot and 100% few-shot accuracies reflect the mock's keyword behaviour and would not carry over to a real model.
- **Single split.** The lapse model was evaluated on one 80/20 split (232 test customers), so test F1 (0.655 for the random forest) carries noticeable uncertainty. Repeated cross-validation would give a steadier estimate.
- **Decision threshold.** The lapse model uses the default 0.5 threshold. Lowering it would catch more lapsing customers at the cost of more wrongly targeted loyal ones, which suits a retention campaign where a missed lapser costs more than an unnecessary offer.
- **TF-IDF retrieval** matches exact words only, so a differently worded question can miss the right chunk. Embeddings or a synonym list would help. Chunk size and overlap were fixed by the brief and not tuned.
- **Refund agent.** The refund rules are hard-coded (completed, more than 45 minutes, escalation above INR 1000) and the 7-day window is assumed met. A real deployment would check the date window, fraud signals and duplicate requests.
- **Imputation.** Missing ages and delivery times are filled with medians, which keeps the data complete but understates real variation.
- **Next steps:** a leakage-free Premium definition computed before the cutoff, threshold tuning for the lapse model, a plan-and-execute variant of the agent with replanning, and a single script that runs the whole Reproduce sequence.