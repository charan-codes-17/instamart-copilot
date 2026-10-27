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