"""Customer chat.  CLI:  python chat.py --model qwen2.5:7b-instruct
Web UI (local only): python chat.py --web   -> http://127.0.0.1:8000
Nothing is logged and nothing leaves this machine."""
import argparse, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from assistant import Assistant, OllamaBackend, TransformersBackend

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Nirbhor Bank Assistant</title><style>
body{font:16px system-ui,sans-serif;max-width:640px;margin:0 auto;padding:12px;display:flex;flex-direction:column;height:100vh;box-sizing:border-box}
h1{font-size:18px;margin:4px 0}small{color:#666}#log{flex:1;overflow:auto;border:1px solid #ccc;border-radius:8px;padding:8px;margin:8px 0}
.m{margin:6px 0;padding:8px 10px;border-radius:10px;max-width:85%;white-space:pre-wrap}.u{background:#dbeafe;margin-left:auto}.b{background:#f1f5f9}
.h{background:#fef3c7}form{display:flex;gap:6px}input{flex:1;padding:10px;font-size:16px}button{padding:10px 16px}
@media(prefers-color-scheme:dark){body{background:#111;color:#eee}#log{border-color:#444}.u{background:#1e3a8a}.b{background:#222}.h{background:#4a3b0a}input{background:#222;color:#eee}}
</style><h1>Nirbhor Bank Assistant</h1><small>English · বাংলা · Banglish. Runs on the bank's own computer; your messages are not sent outside.</small>
<div id=log></div><form id=f><input id=q autocomplete=off placeholder="Ask a question / প্রশ্ন লিখুন" autofocus><button>Send</button></form>
<script>
const log=document.getElementById('log');
function add(t,c){const d=document.createElement('div');d.className='m '+c;d.textContent=t;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
f.onsubmit=async e=>{e.preventDefault();const m=q.value.trim();if(!m)return;q.value='';add(m,'u');const w=add('…','b');
try{const r=await(await fetch('/ask',{method:'POST',body:JSON.stringify({message:m})})).json();
w.textContent=r.reply+(r.handoff?'\\n\\n[Connecting you to a bank representative…]':'');if(r.handoff)w.className='m h'}
catch(x){w.textContent='Error: the assistant is not reachable.'}}
</script>"""

def build(args):
    be = (TransformersBackend(args.model, load_in_4bit=args.q4) if args.backend == "transformers"
          else OllamaBackend(args.model))
    return Assistant(args.faq, be, floor=args.floor)

def serve(bot, port):
    lock = threading.Lock()
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass  # no request logging (customer data)
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
            self.wfile.write(PAGE.encode())
        def do_POST(self):
            msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["message"][:500]
            with lock: r = bot.answer(msg)
            out = json.dumps({"reply": r.reply, "handoff": r.handoff, "language": r.language}, ensure_ascii=False).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8"); self.end_headers()
            self.wfile.write(out)
    print(f"Open http://127.0.0.1:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faq", default="bank-faq.csv"); ap.add_argument("--backend", default="ollama", choices=["ollama", "transformers"])
    ap.add_argument("--model", default="qwen2.5:7b-instruct"); ap.add_argument("--floor", type=float, default=0.20)
    ap.add_argument("--4bit", dest="q4", action="store_true"); ap.add_argument("--web", action="store_true"); ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(); bot = build(args)
    if args.web: return serve(bot, args.port)
    print("Nirbhor Bank assistant. Type a question (English / বাংলা / Banglish). /quit to exit.")
    while True:
        try: q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt): break
        if q in ("/quit", "/exit"): break
        if not q: continue
        r = bot.answer(q)
        print(f"Bot: {r.reply}")
        if r.handoff: print("     [handoff: transferring to a human agent]")
        print(f"     ({r.language}, FAQ #{r.faq_id or '-'}, {r.t_total:.1f}s)")

if __name__ == "__main__": main()
