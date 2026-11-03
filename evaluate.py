"""Evaluate on the 40 test questions.  Examples:
  python evaluate.py --retrieval-only                       # no model needed
  python evaluate.py --backend ollama --model qwen2.5:7b-instruct
  python evaluate.py --backend transformers --model Qwen/Qwen2.5-3B-Instruct
  python evaluate.py --backend mock-oracle                  # plumbing test only, NOT a result
Ground truth (expected.csv) was annotated by hand from the FAQ: expected_faq_id 0 = not covered."""
import argparse, csv, json, re, statistics as st, sys, time
from assistant import (Assistant, OllamaBackend, TransformersBackend, MockBackend, Retriever,
                       detect_language, load_faq, _bn_ratio)

def rd(p): return list(csv.DictReader(open(p, newline="", encoding="utf-8-sig")))

def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faq", default="bank-faq.csv"); ap.add_argument("--questions", default="test-questions.csv")
    ap.add_argument("--expected", default="expected.csv"); ap.add_argument("--out", default="results.csv")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "transformers", "mock-oracle"])
    ap.add_argument("--model", default=None); ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--floor", type=float, default=0.20); ap.add_argument("--4bit", dest="q4", action="store_true")
    a = ap.parse_args()

    qs = {int(r["id"]): r["question"] for r in rd(a.questions)}
    ex = {int(r["id"]): (int(r["expected_faq_id"]), r["expected_lang"]) for r in rd(a.expected)}
    faq_by_q = {f.question: f.id for f in load_faq(a.faq)}

    if a.retrieval_only:
        R = Retriever(load_faq(a.faq)); n1 = n3 = nans = lang = 0; below = 0
        for i, q in qs.items():
            lg = detect_language(q); lang += lg == ex[i][1]; c = R.search(q, lg, 3); e = ex[i][0]
            if e: nans += 1; n1 += c[0][0].id == e; n3 += e in [f.id for f, _ in c]
            elif c[0][1] < a.floor: below += 1
        print(f"language detection {lang}/{len(qs)}; retrieval recall@1 {n1}/{nans}, recall@3 {n3}/{nans}; "
              f"{below}/{len(qs)-nans} uncovered questions stopped by the score floor (rest need the model judge)")
        return

    if a.backend == "mock-oracle":
        def script(msgs):
            u = msgs[-1]["content"]
            if "Answer (one digit only)" in u:
                qq = re.search(r"Question: (.*)\nAnswer \(one digit only\)", u).group(1)
                want = next(ex[i][0] for i, q in qs.items() if q == qq)
                for m in re.finditer(r"\[(\d)\] Q: (.*?) A: ", u.split("Now the real case.")[1]):
                    if faq_by_q.get(m.group(2)) == want: return m.group(1)
                return "0"
            return re.search(r"FAQ answer: (.*)\nCustomer question", u).group(1)  # English text -> exercises fallback
        backend = MockBackend(script)
        print("!! mock-oracle: judge is told the answer. This only tests plumbing; numbers are NOT results.")
    elif a.backend == "ollama":
        backend = OllamaBackend(a.model or "qwen2.5:7b-instruct")
    else:
        backend = TransformersBackend(a.model or "Qwen/Qwen2.5-3B-Instruct", load_in_4bit=a.q4)

    bot = Assistant(a.faq, backend, floor=a.floor)
    bot.answer("warm up")  # exclude model load time from latency numbers
    rows = []
    for i, q in qs.items():
        r = bot.answer(q); e, el = ex[i]
        rows.append(dict(id=i, question=q, exp_faq=e, exp_lang=el, lang=r.language, faq=r.faq_id, handoff=int(r.handoff),
                         reason=r.reason, reply=r.reply, t_retrieve=round(r.t_retrieve, 4), t_judge=round(r.t_judge, 3),
                         t_write=round(r.t_write, 3), t_total=round(r.t_total, 3)))
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    ans = [r for r in rows if r["exp_faq"]]; oos = [r for r in rows if not r["exp_faq"]]
    good = [r for r in ans if r["faq"] == r["exp_faq"] and not r["handoff"]]
    falseidk = [r for r in ans if r["handoff"]]; wrong = [r for r in ans if not r["handoff"] and r["faq"] != r["exp_faq"]]
    idk_ok = [r for r in oos if r["handoff"]]
    clean = [r for r in good if r["reason"] in ("answered_llm", "answered_verbatim")]
    fb = [r for r in good if r["reason"].startswith("fallback")]
    lat = [r["t_total"] for r in rows]
    def lg(r): return r["lang"] == r["exp_lang"]
    print(f"\nModel: {a.backend} {a.model or ''}   (floor={a.floor})")
    print(f"Language detected correctly:          {sum(map(lg, rows))}/{len(rows)}")
    print(f"Answerable, routed to correct FAQ:    {len(good)}/{len(ans)}   (false 'I don't know': {len(falseidk)}, wrong FAQ: {len(wrong)})")
    print(f"  ...of which clean reply in own language: {len(clean)}, English fallback after guard failure: {len(fb)}")
    print(f"Not covered, correctly said IDK:      {len(idk_ok)}/{len(oos)}   (wrongly answered: {len(oos)-len(idk_ok)})")
    print(f"Overall correct decision:             {len(good)+len(idk_ok)}/{len(rows)}")
    print(f"Latency per question (s): mean {st.mean(lat):.2f}, p50 {pct(lat,.5):.2f}, p95 {pct(lat,.95):.2f}, max {max(lat):.2f}")
    print(f"  stage means: retrieve {st.mean(r['t_retrieve'] for r in rows):.4f}, judge {st.mean(r['t_judge'] for r in rows):.2f}, "
          f"write {st.mean(r['t_write'] for r in rows):.2f}")
    for r in falseidk + wrong + [x for x in oos if not x["handoff"]]:
        print(f"  CHECK id={r['id']} exp={r['exp_faq']} got={r['faq']} idk={r['handoff']} :: {r['question'][:60]}")
    print(f"Per-question detail (with replies for human review of Bangla/Banglish quality): {a.out}")

if __name__ == "__main__": main()
