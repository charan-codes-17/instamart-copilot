"""Part D, tasks 5-8: tools, ReAct loop, safe stopping, checkpoint/resume, 7 scenarios + demos.
Run from the repo root:  python src\\part_d_agent.py
The script clears outputs\\agent_runs.jsonl, outputs\\refund_ledger.jsonl and state\\*.json at the
start, so every run reproduces the same committed files."""
import glob
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import mock_llm  # noqa: E402
from part_d_rag import retrieve  # noqa: E402

DB_PATH = os.path.join(ROOT, "data", "instamart.db")
OUT_DIR = os.path.join(ROOT, "outputs")
STATE_DIR = os.path.join(ROOT, "state")
LEDGER = os.path.join(OUT_DIR, "refund_ledger.jsonl")
RUNS = os.path.join(OUT_DIR, "agent_runs.jsonl")
DEMOS = os.path.join(OUT_DIR, "agent_demos.json")
SCENARIOS = os.path.join(ROOT, "data", "raw", "agent_scenarios.csv")

CURRENT_RUN_ID = None


class SimulatedCrash(Exception):
    """Raised on purpose after a checkpoint to demonstrate resume."""


# ---------------- Task 5: tools ----------------
def get_order(order_id):
    con = sqlite3.connect(DB_PATH)
    try:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes "
                          "FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    finally:
        con.close()
    if row is None:
        return {"error": "order not found"}
    dm = row[3]
    if dm is not None and float(dm).is_integer():
        dm = int(dm)
    return {"order_id": row[0], "status": row[1], "order_amount": round(float(row[2]), 2),
            "delivery_minutes": dm}


def search_policy(query):
    results = retrieve(query, k=3)
    for r in results:
        r["score"] = round(r["score"], 4)
    return {"results": results}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and float(delivery_minutes) > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0.0}


def _ledger(tool, order_id, refund_inr):
    os.makedirs(OUT_DIR, exist_ok=True)
    line = {"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool,
            "order_id": order_id, "refund_inr": refund_inr, "run_id": CURRENT_RUN_ID}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def issue_refund(order_id, refund_inr):
    _ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}


def escalate_to_human(order_id, refund_inr):
    _ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}


def _spec(name, description, props, required):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}


TOOLS_SPEC = [
    _spec("get_order", "Look up an order by id and return its status, amount and delivery minutes.",
          {"order_id": {"type": "string"}}, ["order_id"]),
    _spec("search_policy", "Search the policy documents and return the three best matching chunks.",
          {"query": {"type": "string"}}, ["query"]),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}},
          ["order_amount", "delivery_minutes"]),
    _spec("issue_refund", "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead and record it.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy,
            "calculate_refund": calculate_refund, "issue_refund": issue_refund,
            "escalate_to_human": escalate_to_human}

SYSTEM_PROMPT = ("You are a refund AGENT for Instamart support. Use the tools get_order, search_policy, "
                 "calculate_refund, issue_refund and escalate_to_human. Finish with one line that starts "
                 "with REFUND, ESCALATE, DENIED or NOT_FOUND, then a colon and a short reason.")


def execute_tool(name, args):
    fn = TOOL_MAP.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(**args)
    except Exception as exc:  # a tool failure becomes an observation, not a crash
        return {"error": str(exc)}


# ---------------- Task 8: checkpoints ----------------
def save_checkpoint(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    ck = {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "messages": state["messages"],
          "tokens_used": state["tokens_used"], "llm_calls": state["llm_calls"], "step": len(state["steps"]),
          "tools_executed": [s["tool"] for s in state["steps"]], "steps": state["steps"],
          "params": state["params"]}
    with open(os.path.join(STATE_DIR, f"{state['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(ck, f, indent=1)


def _result(state, status, final_answer):
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status,
            "final_answer": final_answer, "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"]}


def _fallback(state, status):
    tools = ", ".join(s["tool"] for s in state["steps"]) or "none"
    text = (f"{status}: stopped after {state['llm_calls']} model calls and {len(state['steps'])} tool "
            f"steps ({tools}). Please handle this refund request manually.")
    return _result(state, status, text)


# ---------------- Tasks 6-7: ReAct loop with limits ----------------
def _loop(state, crash_after_tool=None):
    global CURRENT_RUN_ID
    CURRENT_RUN_ID = state["run_id"]
    p = state["params"]
    t0 = time.monotonic()
    while state["llm_calls"] < p["max_iterations"]:
        if time.monotonic() - t0 >= p["time_limit_seconds"]:
            return _fallback(state, "FALLBACK_TIMEOUT")
        resp = mock_llm.chat(state["messages"], TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > p["token_limit"]:
            return _fallback(state, "FALLBACK_TOKEN_LIMIT")      # tool not executed
        msg = resp.choices[0].message
        if not msg.tool_calls:
            text = msg.content or ""
            status = text.split(":")[0].strip() if ":" in text else "UNKNOWN"
            return _result(state, status, text)
        calls = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in msg.tool_calls]
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": calls})
        for c in calls:
            obs = execute_tool(c["name"], c["arguments"])
            state["messages"].append({"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                                      "content": json.dumps(obs)})
            state["steps"].append({"tool": c["name"], "arguments": c["arguments"], "observation": obs})
            save_checkpoint(state)
            if crash_after_tool is not None and len(state["steps"]) == crash_after_tool:
                raise SimulatedCrash(f"simulated crash after tool execution {crash_after_tool}")
    return _fallback(state, "FALLBACK_MAX_ITERATIONS")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tool=None):
    state = {"run_id": run_id or f"run_{scenario_id}", "scenario_id": scenario_id,
             "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "steps": [],
             "params": {"max_iterations": max_iterations, "token_limit": token_limit,
                        "time_limit_seconds": time_limit_seconds}}
    return _loop(state, crash_after_tool)


def resume_agent(run_id):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        ck = json.load(f)
    state = {"run_id": ck["run_id"], "scenario_id": ck["scenario_id"], "messages": ck["messages"],
             "tokens_used": ck["tokens_used"], "llm_calls": ck["llm_calls"], "steps": ck["steps"],
             "params": ck["params"]}
    return _loop(state)


def refund_amount(res):
    for s in reversed(res["steps"]):
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return float(s["observation"]["refund_inr"])
    return 0.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    for path in [RUNS, LEDGER] + glob.glob(os.path.join(STATE_DIR, "*.json")):
        if os.path.exists(path):
            os.remove(path)

    sc = pd.read_csv(SCENARIOS, keep_default_na=False)
    print("| id | expected | status | refund_inr | expected_inr | calls | tokens | OK |")
    for _, r in sc.iterrows():
        res = run_agent(r["message"], r["scenario_id"])
        amt = refund_amount(res)
        exp = float(r["expected_refund_inr"])
        ok = res["status"].startswith(r["expected_outcome"]) and abs(amt - exp) <= 0.01
        print(f"| {r['scenario_id']} | {r['expected_outcome']} | {res['status']} | {amt:.2f} | {exp:.2f} | "
              f"{res['llm_calls']} | {res['tokens_used']} | {ok} |")
        line = {"scenario_id": res["scenario_id"], "status": res["status"], "final_answer": res["final_answer"],
                "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"], "refund_inr": amt,
                "steps": res["steps"]}
        with open(RUNS, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")

    s1 = sc[sc["scenario_id"] == "S1"].iloc[0]["message"]
    demos = {}

    r = run_agent(s1, "S1", token_limit=200, run_id="S1_tokenlimit")
    demos["token_limit_200"] = {"status": r["status"], "llm_calls": r["llm_calls"],
                                "tokens_used": r["tokens_used"], "final_answer": r["final_answer"]}
    print("\nDEMO token_limit=200      ->", r["status"], "| calls", r["llm_calls"], "| tokens", r["tokens_used"])

    r = run_agent(s1, "S1", time_limit_seconds=0, run_id="S1_timeout")
    demos["time_limit_0"] = {"status": r["status"], "llm_calls": r["llm_calls"],
                             "tokens_used": r["tokens_used"], "final_answer": r["final_answer"]}
    print("DEMO time_limit_seconds=0 ->", r["status"], "| calls", r["llm_calls"], "| tokens", r["tokens_used"])

    try:
        run_agent(s1, "S1", run_id="S1_crash", crash_after_tool=2)
        raise AssertionError("the simulated crash did not happen")
    except SimulatedCrash as exc:
        print("\nDEMO crash raised:", exc)
    with open(os.path.join(STATE_DIR, "S1_crash.json"), encoding="utf-8") as f:
        ck = json.load(f)
    print("checkpoint tools_executed before resume:", ck["tools_executed"])
    r = resume_agent("S1_crash")
    tools = [s["tool"] for s in r["steps"]]
    print("tools executed across crash + resume:", tools)
    print("final:", r["status"], f"{refund_amount(r):.2f}", "| calls", r["llm_calls"], "| tokens", r["tokens_used"])
    assert tools == ["get_order", "search_policy", "calculate_refund", "issue_refund"], "tool list wrong"
    assert r["status"] == "REFUND" and abs(refund_amount(r) - 128.25) <= 0.01
    demos["crash_resume"] = {"tools_executed": tools, "status": r["status"],
                             "refund_inr": refund_amount(r), "llm_calls": r["llm_calls"]}
    with open(DEMOS, "w", encoding="utf-8") as f:
        json.dump(demos, f, indent=1)
    print("\ncrash/resume assertions passed")


if __name__ == "__main__":
    main()