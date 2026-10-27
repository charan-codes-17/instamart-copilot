"""Part D, tasks 1-4: chunking, TF-IDF retrieval, grounded answering, evaluation.
Run from the repo root:  python src\\part_d_rag.py
The agent script imports retrieve() and answer() from this module."""
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mock_llm  # noqa: E402

POLICY_DIR = os.path.join(ROOT, "data", "policies")
EVAL_CSV = os.path.join(ROOT, "data", "raw", "policy_eval.csv")
CHUNK_WORDS = 40
DEFAULT_OVERLAP = 10

SYSTEM_PROMPT = (
    "You are a POLICY ASSISTANT for Instamart support. Answer the question using only "
    "the supplied CONTEXT. If the context does not contain the answer, reply exactly: "
    "I could not find that information in the policy documents."
)


# ---------- Task 1: chunking ----------
def chunk_text(text, size=CHUNK_WORDS, overlap=DEFAULT_OVERLAP):
    """Word chunks of `size` words, each starting (size - overlap) words after the previous.
    Stops after the first chunk whose end reaches or passes the last word."""
    words = text.split()
    step = size - overlap
    chunks, start = [], 0
    while True:
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


def load_chunks(overlap=DEFAULT_OVERLAP):
    out = []
    for fn in sorted(os.listdir(POLICY_DIR)):
        if not fn.endswith(".txt"):
            continue
        with open(os.path.join(POLICY_DIR, fn), encoding="utf-8") as f:
            text = f.read()
        for c in chunk_text(text, CHUNK_WORDS, overlap):
            out.append({"source": fn, "text": c})
    return out


# ---------- Task 2: TF-IDF retrieval ----------
class Index:
    def __init__(self, overlap=DEFAULT_OVERLAP):
        self.overlap = overlap
        self.chunks = load_chunks(overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        q = self.vectorizer.transform([query])          # same fitted vectorizer
        sims = cosine_similarity(q, self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]),
                 "text": self.chunks[i]["text"]} for i in order]


_INDEX = Index(DEFAULT_OVERLAP)


def retrieve(query, k=3):
    return _INDEX.retrieve(query, k)


# ---------- Task 3: augment and answer ----------
def answer(question, index=None):
    idx = index or _INDEX
    hits = idx.retrieve(question, k=3)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}])
    return resp.choices[0].message.content, [h["source"] for h in hits]


# ---------- Task 4: evaluation ----------
def evaluate(index, label):
    df = pd.read_csv(EVAL_CSV, keep_default_na=False)
    ans_q = df[df["expected_doc"] != "NONE"]
    none_q = df[df["expected_doc"] == "NONE"]
    ret_ok = fact_ok = fb_ok = 0
    failures = []
    for _, r in ans_q.iterrows():
        text, sources = answer(r["question"], index)
        got_doc = r["expected_doc"] in sources
        got_fact = r["expected_fact"] in text
        ret_ok += got_doc
        fact_ok += got_fact
        if not (got_doc and got_fact):
            failures.append((r["question"], r["expected_doc"], r["expected_fact"], got_doc, got_fact, text))
    for _, r in none_q.iterrows():
        text, _ = answer(r["question"], index)
        fb_ok += text.startswith("I could not find")
    print(f"\n=== {label} (overlap={index.overlap}, chunks={len(index.chunks)}) ===")
    print(f"expected_doc in top 3 : {ret_ok}/{len(ans_q)}   (ref 12/12 at overlap 10)")
    print(f"expected_fact in answer: {fact_ok}/{len(ans_q)}   (ref 12/12 at overlap 10)")
    print(f"fallback on unanswerable: {fb_ok}/{len(none_q)}   (ref 2/2)")
    for q, d, f, gd, gf, t in failures:
        print(f"  FAIL: {q}\n    expected_doc={d} found={gd} | expected_fact='{f}' found={gf}\n    answer: {t}")
    return ret_ok, fact_ok, fb_ok, failures


def main():
    # Task 1
    chunks = _INDEX.chunks
    counts = pd.Series([c["source"] for c in chunks]).value_counts()
    print("chunks total:", len(chunks), "(ref 15)")
    print(counts.to_string(), "\n(ref: refund 4, delivery 3, escalation 3, payment 3, privacy 2)")
    first = chunks[0]["text"].split()
    print("first chunk words:", len(first), "| first 6 words:", " ".join(first[:6]))

    # Task 2
    hits = retrieve("refund policy late delivery", k=3)
    print("\nquery 'refund policy late delivery':")
    for h in hits:
        print(f"  {h['source']:<22} score={h['score']:.4f}")
    assert all(h["source"] == "refund_policy.txt" for h in hits), "top 3 must be refund_policy.txt"
    print("top 3 all refund_policy.txt: OK")

    # Task 3 sample
    t, s = answer("Who must approve a refund above INR 1000?")
    print("\nsample answer:", t, "| sources:", s)

    # Task 4
    evaluate(_INDEX, "overlap 10")
    evaluate(Index(0), "overlap 0")


if __name__ == "__main__":
    main()