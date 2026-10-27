"""Part C: support-ticket triage with guardrails, retries, output validation, token budget and routing."""
import json, os, random, re, sys, time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for mock_llm
import mock_llm
from mock_llm import chat, est_tokens, TransientLLMError

random.seed(42)
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
CATEGORIES = ("billing", "delivery", "technical", "fraud")
HUMAN_STATUSES = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "llm_error"}
OUT_COLS = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]

# ---------------- task 2: redaction (order matters) ----------------
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

def redact(text):
    counts = {}
    for name, pat, repl in PATTERNS:
        text, n = pat.subn(repl, text)
        counts[name] = n
    return text, counts

def assert_no_pii(messages):
    for m in messages:
        for name, pat, _ in PATTERNS:
            assert not pat.search(m["content"]), f"PII ({name}) found in a prompt"

# ---------------- task 3: prompts ----------------
ZERO_SHOT = ('You CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud. '
             'Reply with JSON only, no other text, in the form {"category": "...", "reason": "..."}.')
FEW_SHOT = ZERO_SHOT + "\nEXAMPLES:\n" + "\n".join([
    'Ticket: I was charged twice for order [ORDER_ID]. -> {"category": "billing", "reason": "duplicate charge"}',
    'Ticket: The app crashes every time I open the cart. -> {"category": "technical", "reason": "app crash"}',
    'Ticket: There is an unauthorized transaction on my card. -> {"category": "fraud", "reason": "unauthorized transaction"}',
    'Ticket: My order has not arrived and the rider is not picking up calls. -> {"category": "delivery", "reason": "order not arrived"}'])
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
assert "CLASSIFY" in FEW_SHOT and "EXAMPLES:" in FEW_SHOT

# ---------------- task 4: resilient call ----------------
def call_with_retry(messages, tid, stats, show_log):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return chat(messages)
        except TransientLLMError:
            stats["errors"] += 1
            if attempt == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** attempt * random.uniform(0.9, 1.15)   # attempt == retry number n
            stats["waits"].append(wait)
            if show_log:
                print(f"  [retry] ticket={tid} retry_number={attempt} wait={wait:.4f}s")
            time.sleep(wait)

# ---------------- task 5: validation ----------------
def validate(content):
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (content or "").lower()))
        found = [c for c in CATEGORIES if c in words]
        return ("classified_keyword", found[0]) if len(found) == 1 else ("flagged_invalid_output", "")
    if isinstance(obj, dict) and obj.get("category") in CATEGORIES:
        return "classified_json", obj["category"]
    return "flagged_invalid_output", ""

# ---------------- pipeline ----------------
def run_pipeline(tickets, system_prompt, token_budget=None, show_log=False):
    mock_llm.reset()                                   # immediately before each full run
    stats = {"errors": 0, "waits": [], "sent": []}
    running, rows = 0, []
    for t in tickets:
        text = t["text"]
        row = {"ticket_id": t["ticket_id"], "customer_id": t["customer_id"], "status": "", "category": "",
               "tokens_used": 0, "true_category": t["true_category"]}
        if not text.strip():
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif token_budget is not None and running >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            clean, _ = redact(text)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": clean}]
            assert_no_pii(messages)
            stats["sent"].append(t["ticket_id"])
            try:
                resp = call_with_retry(messages, t["ticket_id"], stats, show_log)
            except TransientLLMError:
                row["status"] = "llm_error"
            else:
                used = resp.usage.total_tokens
                running += used
                row["tokens_used"] = used
                row["status"], row["category"] = validate(resp.choices[0].message.content)
        rows.append(row)
    return pd.DataFrame(rows), stats

def summarise(df, stats, label):
    counts = df["status"].value_counts().to_dict()
    cl = df[df["status"].isin(["classified_json", "classified_keyword"])]
    correct = int((cl["category"] == cl["true_category"]).sum())
    total = int(df["tokens_used"].sum())
    w = stats["waits"]
    print(f"\n=== {label} ===")
    print("status counts:", counts)
    print(f"classified: {len(cl)} | correct: {correct} | accuracy: {correct / len(cl) * 100:.1f}%")
    print(f"total tokens: {total} | tokens per classified ticket: {total / len(cl):.2f}")
    print(f"TransientLLMErrors: {stats['errors']} | retries: {len(w)} | permanent errors: {counts.get('llm_error', 0)}",
          f"| wait range: {min(w):.4f}-{max(w):.4f}s" if w else "")
    return cl

def route_row(status, category, is_premium):
    if status in HUMAN_STATUSES:
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if is_premium:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"

def add_routing(df, seg):
    names = seg.set_index("customer_id")["segment_name"].to_dict()
    prem = seg.set_index("customer_id")["is_premium"].astype(str).eq("True").to_dict()
    out = df.copy()
    out["segment_name"] = out["customer_id"].map(names).fillna("")
    routed = [route_row(s, c, prem.get(cid, False)) for s, c, cid in zip(out["status"], out["category"], out["customer_id"])]
    out["route"] = [r[0] for r in routed]
    out["priority"] = [r[1] for r in routed]
    return out[OUT_COLS]

# ---------------- main ----------------
tk = pd.read_csv("data/raw/tickets.csv", keep_default_na=False, dtype=str)
tickets = tk.to_dict("records")                         # file order
seg = pd.read_csv("outputs/customer_segments.csv")

# task 1: guard counts
empty = sum(1 for t in tickets if not t["text"].strip())
long_ = [est_tokens(t["text"]) for t in tickets if t["text"].strip() and est_tokens(t["text"]) > MAX_INPUT_TOKENS]
print(f"tickets: {len(tickets)} | empty: {empty} (ref 4) | long: {len(long_)} (ref 4) | long est_tokens: {sorted(set(long_))} (ref 734)")

# task 2: redaction counts over all 164 tickets + re-apply check
tot = {"email": 0, "phone": 0, "order_id": 0}
left = 0
for t in tickets:
    clean, c = redact(t["text"])
    for k in tot:
        tot[k] += c[k]
    left += sum(len(p.findall(clean)) for _, p, _ in PATTERNS)
print("redactions:", tot, "total:", sum(tot.values()), "(ref 55 / 91 / 132 = 278) | left after redaction:", left, "(ref 0)")
print("ZERO_SHOT est_tokens:", est_tokens(ZERO_SHOT), "| FEW_SHOT est_tokens:", est_tokens(FEW_SHOT))

# task 4-6: full runs
print("\nRetry log (zero-shot full run):")
zs_df, zs_stats = run_pipeline(tickets, ZERO_SHOT, show_log=True)
zs_cl = summarise(zs_df, zs_stats, "ZERO-SHOT (no budget)")
fs_df, fs_stats = run_pipeline(tickets, FEW_SHOT)
fs_cl = summarise(fs_df, fs_stats, "FEW-SHOT (no budget)")
eligible = len(tickets) - empty - len(long_)
assert len(zs_stats["sent"]) == len(fs_stats["sent"]) == eligible, "an empty/long ticket reached the model"
print(f"\nmodel calls were made only for the {eligible} eligible tickets (assertion passed); no PII in any prompt (assertion passed)")

print("\nZero-shot confusion table (rows = true, columns = predicted):")
print(pd.crosstab(zs_cl["true_category"], zs_cl["category"]))

# task 7a: budget demo (zero-shot)
BUDGET_DEMO = 6000
bd_df, bd_stats = run_pipeline(tickets, ZERO_SHOT, token_budget=BUDGET_DEMO)
summarise(bd_df, bd_stats, f"ZERO-SHOT with TOKEN_BUDGET={BUDGET_DEMO}")
n_ex = int((bd_df["status"] == "budget_exhausted").sum())
print(f"budget_exhausted tickets: {n_ex} (need at least 1; if 0, lower BUDGET_DEMO)")
os.makedirs("outputs", exist_ok=True)
add_routing(bd_df, seg).to_csv("outputs/triage_budget_demo.csv", index=False)

# task 7b: final results (few-shot, budget 100000)
fin_df, fin_stats = run_pipeline(tickets, FEW_SHOT, token_budget=100000)
summarise(fin_df, fin_stats, "FINAL: FEW-SHOT with TOKEN_BUDGET=100000")
final = add_routing(fin_df, seg)
final.to_csv("outputs/triage_results.csv", index=False)
assert len(final) == 164
print("\ntriage_results rows:", len(final))
print(final.groupby(["route", "priority"]).size())
print("status x route:\n", final.groupby(["status", "route"]).size())