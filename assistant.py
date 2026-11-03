"""
Nirbhor Bank PLC - offline bilingual FAQ assistant (English / Bangla / Banglish).

Pipeline (everything runs on the local machine; the only network call is to a
local model server on localhost, or none at all with the transformers backend):

  1. detect_language()   rule based: en | bn (Bangla script) | bl (Banglish)
  2. Retriever           cross-lingual concept index over the FAQ -> top-k candidates
  3. LLM "judge"         small model picks which candidate directly answers, or 0
  4. LLM "writer"        small model rewrites the FAQ answer in the customer's language
  5. Guards              numbers must come from the FAQ, script must match language;
                         otherwise fall back to the English FAQ text (never invent)
  6. IDK                 fixed, pre-written "I don't know, let me connect you" message

Only stdlib is required. sentence-transformers is optional (dense retrieval).
"""
from __future__ import annotations

import csv
import json
import math
import re
import time
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# 1. Language detection
# --------------------------------------------------------------------------
_BN_CHAR = re.compile(r"[\u0980-\u09FF]")
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return text.replace("\u200c", "").replace("\u200d", "").lower()


def bl_norm(t: str) -> str:
    """Collapse the common spelling variation of Bangla written in Latin letters."""
    t = t.lower()
    for a, b in (("bh", "b"), ("dh", "d"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("ph", "f"),
                 ("sh", "s"), ("ch", "c"), ("ee", "i"), ("oo", "u"), ("v", "b"), ("z", "j"),
                 ("w", "o"), ("y", "i")):
        t = t.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", t)


# Function words that are (almost) never English; one hit => Banglish.
_BL_MARKERS_RAW = """ami amar amake amader apni apnar apnader apnake ki kibhabe kivabe kibabe kemon kemone
koto kato kothay kobe kokhon kon korbo korte korbe korle kori korun korini hobe hole hoy hoye hote jay jabe
jonno achhe ache nai bolun chai chaile er ta ti gulo theke lage lagbe pabo paben bhule gechi geche gesi
kora kore nite dite janabo ekta ekti shomoy somoy khola thake bochor boyosh tahole kine thakbe hobe
diyechi dilam bodlate bodlano korchi korsi jodi kintu ebong"""
_BL_MARKERS = {bl_norm(w) for w in _BL_MARKERS_RAW.split()}


def detect_language(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    bn = sum(1 for c in letters if _BN_CHAR.match(c))
    if bn / len(letters) > 0.3:
        return "bn"
    toks = re.findall(r"[a-z]+", text.lower())
    return "bl" if any(bl_norm(t) in _BL_MARKERS for t in toks) else "en"


# --------------------------------------------------------------------------
# 2. Cross-lingual concept glossary (banking vocabulary, NOT question specific)
#    concept: "english forms | bangla forms | banglish forms"
# --------------------------------------------------------------------------
GLOSSARY = {
    "account": "account,accounts | অ্যাকাউন্ট,একাউন্ট,এ্যাকাউন্ট,হিসাব | account,akaunt,ekaunt,hisab,acc",
    "savings": "savings,saving | সেভিংস,সঞ্চয়ী,সঞ্চয় | savings,saving,sanchoy,sonchoy",
    "open": "open,opening,opened | খোলা,খুলতে,খুলব,খুলবো,খুলে | khola,kholte,khulte,kholar,khulbo,kholbo,open",
    "document": "document,documents,paper,papers | কাগজ,কাগজপত্র,ডকুমেন্ট,ডকুমেন্টস | kagoj,kagojpotro,dokument,documents",
    "need": "need,require,required,requirement,requirements | লাগে,লাগবে,প্রয়োজন,দরকার | lage,lagbe,lagche,dorkar,proyojon",
    "balance": "balance | ব্যালেন্স | balance,balans",
    "minimum": "minimum,min,least,lowest | সর্বনিম্ন,ন্যূনতম,কমপক্ষে | minimum,sorbonimno,shorbonimno,nunotom,kompokkhe,min",
    "maximum": "maximum,max,highest | সর্বোচ্চ,সর্বাধিক | maximum,sorboccho,shorboccho,sorbocco,max",
    "limit": "limit,limits,cap | লিমিট,সীমা | limit,sima",
    "fee": "fee,fees,charge,charges,charged,cost,costs | ফি,ফী,চার্জ,খরচ | fee,charge,khoroch,chorgo,cost",
    "year": "year,years,annual,yearly,annually | বার্ষিক,বছর,বছরে | bochor,bosor,barshik,bochore,year",
    "card": "card,cards | কার্ড | card,kard",
    "debit": "debit | ডেবিট | debit",
    "credit": "credit | ক্রেডিট | credit",
    "lost": "lost,lose,stolen,theft,missing | হারিয়ে,হারিয়েছি,হারানো,চুরি | hariye,haraye,haray,harano,chori,lost",
    "block": "block,blocked,freeze | ব্লক | block",
    "pin": "pin | পিন | pin",
    "reset": "reset,forgot,forget,forgotten,recover | ভুলে,রিসেট | bhule,reset,bhulegechi",
    "app": "app,application | অ্যাপ,এ্যাপ | app,eyap",
    "mobile": "mobile,phone,cell,cellphone,sim | মোবাইল,ফোন | mobile,phone,fon,mobail",
    "number": "number,numbers | নম্বর,নাম্বার | nombor,number,nambar,nomber",
    "register": "register,registered,registration,signup,enroll | রেজিস্টার,রেজিস্ট্রেশন,নিবন্ধন | register,registration,nibondhon",
    "transfer": "transfer,transfers,send | ট্রান্সফার,পাঠানো,পাঠাতে,পাঠাব | transfer,pathano,pathate,pathabo",
    "day": ("day,days,daily,sunday,monday,tuesday,wednesday,thursday,friday,saturday | "
            "দিন,দিনে,দৈনিক,প্রতিদিন | din,dine,prodin,doinik,daily"),
    "beftn": "beftn | বিইএফটিএন | beftn",
    "rtgs": "rtgs | আরটিজিএস | rtgs",
    "atm": "atm | এটিএম | atm",
    "withdraw": "withdraw,withdrawal,withdrawn,cash | তোলা,তুলতে,উত্তোলন,তুলব | tola,tolte,uthano,uttolon,withdraw,tulbo,tolbo",
    "branch": "branch,branches,office | শাখা,ব্রাঞ্চ,শাখাগুলো | branch,shakha,brance",
    "hours": "hours,hour,time,timing,schedule | সময়,সময় | shomoy,somoy,time",
    "statement": "statement,statements | স্টেটমেন্ট,বিবরণী | statement",
    "change": "change,changing,modify,alter | বদলাতে,বদল,পরিবর্তন | bodlate,bodlano,bodol,poriborton,porivorton,change,bodlai",
    "update": "update,updated,updating | আপডেট | update",
    "dormant": "dormant,inactive | ডরম্যান্ট,ডরমেন্ট,নিষ্ক্রিয় | dormant,inactive,niskriyo",
    "reactivate": "reactivate,reactivation,activate | সক্রিয় | reactivate,sokriyo",
    "joint": "joint,jointly | যৌথ,জয়েন্ট | joint",
    "nominee": "nominee,nomination | নমিনি,মনোনীত | nominee,nomini",
    "fdr": "fdr,fixed | এফডিআর,ফিক্সড | fdr",
    "deposit": "deposit,deposits,deposited | ডিপোজিট,জমা | deposit,joma",
    "early": "early,premature,prematurely,break,broke,encash,encashment,encashing,before | ভাঙ্গলে,ভাঙলে,ভাঙ্গা,ভাঙতে,আগে | bhangle,bhanga,bhangte,age,bhange",
    "interest": "interest | সুদ,সুদের,মুনাফা | sud,sudh,interest,munafa",
    "rate": "rate,rates,percent,percentage | হার,রেট | rate,har",
    "loan": "loan,loans,borrow,borrowing,lend | লোন,ঋণ,ঋন | loan,lon,rin",
    "personal": "personal | পার্সোনাল,ব্যক্তিগত | personal",
    "home": "home,house,property,housing,flat,apartment | হোম,বাড়ি,বাসা,সম্পত্তি,ফ্ল্যাট | home,bari,basha,flat,property,shompotti",
    "eligible": "eligible,eligibility,qualify,qualification,who | যোগ্যতা,যোগ্য | jogyota,jogyo,eligible,eligibility",
    "income": "income,salary,salaried,earn,earning | আয়,বেতন,ইনকাম | income,ay,beton,salary",
    "age": "age,old,aged,minor | বয়স,বয়সী,বয়েস | boyosh,boyos,boyes",
    "pay": "pay,payment,payments,paid,repay,repayment | দিতে,দিবো,পরিশোধ,পেমেন্ট | dite,dibo,pay,payment,porishodh",
    "due": "due,deadline | বকেয়া | due,deadline",
    "bill": "bill,bills | বিল | bill,bil",
    "late": "late,delay,delayed,miss,missed,overdue | দেরি,দেরিতে,বিলম্ব | deri,late,delay",
    "dispute": ("dispute,disputed,unauthorized,unauthorised,fraud,fraudulent,complaint,complain,investigate | "
                "অভিযোগ,জালিয়াতি,প্রতারণা,করিনি | obhijog,ovijog,complain,fraud,jalijati,korini,koreni"),
    "report": "report,reported | রিপোর্ট,জানাব,জানাবো,জানাতে | report,janabo,janate,janai",
    "transaction": "transaction,transactions | ট্রানজেকশন,ট্রানজ্যাকশন,লেনদেন | transaction,transection,lenden,trxn",
    "sms": "sms | এসএমএস | sms",
    "alert": "alert,alerts,notification,notifications | অ্যালার্ট,এলার্ট,নোটিফিকেশন | alert,alart,notification",
    "remittance": ("remittance,remit,overseas,abroad,foreign,international | "
                   "রেমিট্যান্স,রেমিটেন্স,বিদেশ,প্রবাস,প্রবাসী | remittance,remitance,bidesh,probash,abroad"),
    "receive": ("receive,receiving,received,incoming,get,gets,obtain,collect,collected | "
                "পাওয়া,পেতে,গ্রহণ,নিতে | pete,pawa,receive,nite,nibo,newa,nea"),
    "cheque": "cheque,cheques,check,checks,checkbook,chequebook | চেক,চেকবই | cheque,check,chek,chq",
    "book": "book,books,leaf,leaves,booklet | বই,বুক,পাতা | boi,book,leaf",
    "stop": "stop,stopped,stopping,halt,cancel | বন্ধ,স্টপ,থামাতে | stop,bondho,bandho,bondo",
    "close": "close,closed,closing,closure,terminate,shut | বন্ধ,ক্লোজ | close,bondho,bandho,closing,bondo",
    "kyc": "kyc | কেওয়াইসি | kyc",
    "hotline": "hotline,call | হটলাইন,কল | hotline,call",
    "nid": "nid,passport,id | এনআইডি,পাসপোর্ট,আইডি | nid,passport",
    "photo": "photo,photos,picture | ছবি,ফটো | photo,chobi",
    "tin": "tin | টিন | tin",
    "vat": "vat | ভ্যাট | vat",
    "month": "month,months,monthly | মাস,মাসিক | mash,mas,maash,mashik",
    "free": "free | বিনামূল্যে,ফ্রি | free",
}

_EN_STOP = set("""a an the is are was were be been do does did can could will would should i me my we you your our
it its of to in on at for from with by and or if as that this what when where how which whom why about any there
their they them than then so not no yes also have has had please tell much many other bank nirbhor am want like
ever up out into over per""".split())
_BN_STOP = {unicodedata.normalize("NFC", w) for w in """কী কি কত কবে কখন কোথায় কেন কীভাবে কিভাবে কোন কোনো কোনও আমার আমি আমাকে
আমাদের আপনাদের আপনার আপনি আছে আছি নেই না হয় হবে হতে হলে হচ্ছে করব করবো করতে করা করে করি করুন এ এই এটি এটা ও বা এবং
থেকে জন্য বলুন তো যায় যাবে পারি পারব পারবো চাই গেছে গেলে তাহলে আজকের এখন সব কিছু টাকা টাকার টাকায় দিয়ে সাথে সঙ্গে মধ্যে
পর পরে যদি তবে তার তাদের আর একটি একটা এক""".split()}
_BL_STOP_RAW = """ami amar amake amader apni apnar apnader apnake ki kibhabe kivabe kibabe kemon kemone koto kato kothay
kobe kokhon kon korbo korte korbe korle kori kora korun hobe hole hoy hoye hote jay jabe jonno achhe ache nai bolun
chai chaile er ta ti gulo theke te ke e o ba ar ebong na to tai gele gechi geche gesi pari parbo parben dilam ekta
ekti ek ajker ajke ekhon sob shob kichu diye sathe majhe por pore jodi tahole tar taka tk takar thake thakbe"""
_BL_STOP = {bl_norm(w) for w in _BL_STOP_RAW.split()}
_BN_SUFFIXES = sorted(["গুলোর", "গুলো", "গুলি", "দের", "ের", "তে", "কে", "টির", "টা", "টি", "র", "ে", "য়", "ও", "ই"],
                      key=len, reverse=True)


def stem(w: str) -> str:
    w = w.lower()
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 4 and w.endswith(("ches", "sses", "xes")):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def _lev(a: str, b: str, cap: int) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class Lexicon:
    def __init__(self):
        self.en = defaultdict(set)       # stem(en form)            -> concepts
        self.bn = defaultdict(set)       # NFC bangla form          -> concepts
        self.bl = defaultdict(set)       # bl_norm(banglish / en)   -> concepts
        for concept, spec in GLOSSARY.items():
            en, bn, bl = [[x.strip() for x in part.split(",") if x.strip()] for part in spec.split("|")]
            for w in en:
                self.en[stem(w)].add(concept)
                self.bl[bl_norm(w)].add(concept)
            for w in bn:
                self.bn[unicodedata.normalize("NFC", w)].add(concept)
            for w in bl:
                self.bl[bl_norm(w)].add(concept)
        self._bl_keys = list(self.bl)
        self._en_keys = list(self.en)

    def concepts(self, text: str, lang: str):
        """Return (concepts, n_unmatched). Unknown English words index as their own stem."""
        out, unmatched = [], 0
        text = _clean(text).translate(_BN_DIGITS)
        for tok in re.findall(r"[\u0980-\u09FF]+|[a-z]+|\d+", text):
            if tok.isdigit():
                continue
            if _BN_CHAR.match(tok):
                if tok in _BN_STOP:
                    continue
                hit = self.bn.get(tok)
                if not hit:
                    for suf in _BN_SUFFIXES:
                        if tok.endswith(suf) and len(tok) > len(suf) + 1:
                            hit = self.bn.get(tok[: -len(suf)])
                            if hit:
                                break
                            base = tok[: -len(suf)]
                            if base in _BN_STOP:
                                hit = set()
                                break
                if hit:
                    out.extend(sorted(hit))
                elif hit is None:
                    unmatched += 1
                continue
            # Latin script
            n = bl_norm(tok)
            if lang == "bl":
                if n in _BL_STOP:
                    continue
            if tok in _EN_STOP:
                continue
            hit = None
            if lang == "bl":
                hit = self.bl.get(n) or self.en.get(stem(tok))
            else:
                hit = self.en.get(stem(tok))
            if not hit and len(tok) >= 5:
                cap = 1 if len(tok) < 9 else 2
                keys, table, key = (self._bl_keys, self.bl, n) if lang == "bl" else (self._en_keys, self.en, stem(tok))
                best = min(((_lev(key, k, cap), k) for k in keys), default=(cap + 1, None))
                if best[0] <= cap and (lang == "bl" or best[0] <= 1):
                    hit = table[best[1]]
            if hit:
                out.extend(sorted(hit))
            else:
                out.append("w:" + stem(tok))
                unmatched += 1
        return out, unmatched


# --------------------------------------------------------------------------
# 2b. Retriever (concept TF-IDF; optional dense fusion)
# --------------------------------------------------------------------------
@dataclass
class FAQ:
    id: int
    topic: str
    question: str
    answer: str


def load_faq(path: str) -> list[FAQ]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [FAQ(int(r["id"]), r["topic"], r["question"], r["answer"]) for r in csv.DictReader(f)]


class Retriever:
    def __init__(self, faqs: list[FAQ], dense_model: str | None = None):
        self.faqs, self.lex = faqs, Lexicon()
        docs = []
        for f in faqs:
            c = Counter()
            for text, w in ((f.topic, 2), (f.question, 2), (f.answer, 1)):
                for t in self.lex.concepts(text, "en")[0]:
                    c[t] += w
            docs.append(c)
        n = len(docs)
        df = Counter(t for d in docs for t in d)
        self.idf = {t: math.log((n + 1) / (df[t] + 1)) + 1 for t in df}
        self.idf_max = math.log(n + 1) + 1
        self.vecs = []
        for d in docs:
            v = {t: (1 + math.log(c)) * self.idf[t] for t, c in d.items()}
            norm = math.sqrt(sum(x * x for x in v.values()))
            self.vecs.append({t: x / norm for t, x in v.items()})
        self.dense = None
        if dense_model:  # optional; needs `pip install sentence-transformers` and a local copy of the model
            from sentence_transformers import SentenceTransformer
            self.dense = SentenceTransformer(dense_model)
            self.dense_vecs = self.dense.encode([f"{f.topic}. {f.question}" for f in faqs], normalize_embeddings=True)

    def search(self, query: str, lang: str, k: int = 3):
        concepts, unmatched = self.lex.concepts(query, lang)
        q = {t: (1 + math.log(c)) * self.idf.get(t, self.idf_max) for t, c in Counter(concepts).items()}
        qn = math.sqrt(sum(x * x for x in q.values())) or 1.0
        scores = [sum(w * v.get(t, 0.0) for t, w in q.items()) / qn for v in self.vecs]
        if self.dense is not None:
            e = self.dense.encode([query], normalize_embeddings=True)[0]
            ds = (self.dense_vecs @ e).tolist()
            scores = [0.6 * s + 0.4 * max(d, 0.0) for s, d in zip(scores, ds)]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.faqs[i], scores[i]) for i in order]


# --------------------------------------------------------------------------
# 3/4. LLM backends and prompts
# --------------------------------------------------------------------------
class OllamaBackend:
    """Talks to a local Ollama server (http://localhost:11434). Model stays on this machine."""

    def __init__(self, model="qwen2.5:7b-instruct", host="http://localhost:11434", num_ctx=2048):
        self.model, self.host, self.num_ctx = model, host.rstrip("/"), num_ctx

    def chat(self, messages, max_tokens=200, temperature=0.0) -> str:
        body = {"model": self.model, "messages": messages, "stream": False, "keep_alive": "30m",
                "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": self.num_ctx}}
        req = urllib.request.Request(self.host + "/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["message"]["content"].strip()


class TransformersBackend:
    """In-process Hugging Face model (use on Colab/Kaggle GPU or a local GPU)."""

    def __init__(self, model="Qwen/Qwen2.5-3B-Instruct", load_in_4bit=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        kw = {"device_map": "auto", "torch_dtype": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw)
        self.torch = torch

    def chat(self, messages, max_tokens=200, temperature=0.0) -> str:
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)  # works on all versions
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        gen = dict(max_new_tokens=max_tokens, do_sample=temperature > 0, pad_token_id=self.tok.eos_token_id)
        if temperature > 0:
            gen["temperature"] = temperature
        with self.torch.no_grad():
            out = self.model.generate(**enc, **gen)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class MockBackend:
    """For plumbing tests only. `script` is a function(messages)->str."""

    def __init__(self, script):
        self.script = script

    def chat(self, messages, max_tokens=200, temperature=0.0) -> str:
        return self.script(messages)


JUDGE_FEWSHOT = """You are a strict checker for a bank's FAQ assistant. You get numbered FAQ entries and a customer question
(English, Bangla or Banglish = Bangla in English letters). Reply with ONLY the number of the entry that DIRECTLY answers
the question. Reply 0 if no entry directly answers it. Reply 0 when the question is only on the same topic but asks for
something the entries do not state: other products, live rates or prices, refunds, waivers or exceptions, locations or
contact details that are not listed, another person's account data, or questions about other countries or services.

Example 1
[1] Q: What is the annual fee for the debit card? A: The first year is free. From the second year the fee is BDT 400.
[2] Q: What is the ATM withdrawal limit? A: Up to BDT 40,000 per day.
Question: Debit card er fee koto?
Answer: 1

Example 2
[1] Q: What is the annual fee for the debit card? A: The first year is free. From the second year the fee is BDT 400.
Question: Can you refund my debit card annual fee?
Answer: 0

Example 3
[1] Q: What are the branch opening hours? A: Open Sunday to Thursday, 10 AM to 4 PM.
Question: Dhaka te kothay kothay apnader branch ache?
Answer: 0

Example 4
[1] Q: How much can I borrow for a home loan? A: Up to 60% of the property value.
[2] Q: Who is eligible for a personal loan? A: Age 25 to 60, salaried.
Question: গাড়ি কেনার জন্য লোন পাওয়া যাবে?
Answer: 0

Example 5
[1] Q: How much do SMS alerts cost? A: A flat BDT 100 per year.
Question: এসএমএস অ্যালার্ট কীভাবে বন্ধ করব?
Answer: 0
"""

LANG_INSTR = {
    "bn": ("Bangla, written in Bangla script (বাংলা). Write all numbers with English digits (0-9). "
           "Keep names and terms like BDT, NB Go, BEFTN, RTGS, FDR, OTP, NID, KYC, ATM, TIN, VAT in English letters."),
    "bl": ("Banglish: the Bangla language written in English (Latin) letters, in a polite everyday style, "
           "for example 'Apnar card ta block korte 16999 e call korun.' Do NOT use Bangla script."),
    "en": "English.",
}
WRITER_SYS = (
    "You are the customer assistant of Nirbhor Bank PLC. Use ONLY the FAQ answer given to you. Never add facts, "
    "numbers, fees, conditions or advice that are not in it. Keep every number, amount, time, phone number, and "
    "product or app name exactly as written. Reply in 1-3 short sentences. Do not mention the FAQ."
)
WRITER_EXAMPLE = {
    "bl": ("Example\nFAQ answer: The annual fee is BDT 400 from the second year. The first year is free.\n"
           "Reply: Prothom bochor free. Ditiyo bochor theke annual fee BDT 400."),
    "bn": ("Example\nFAQ answer: The annual fee is BDT 400 from the second year. The first year is free.\n"
           "Reply: প্রথম বছর ফ্রি। দ্বিতীয় বছর থেকে বার্ষিক ফি BDT 400।"),
    "en": "",
}

IDK = {
    "en": "I'm sorry, I don't know the answer to that from our FAQ. Let me connect you to a person who can help.",
    "bn": "দুঃখিত, এই প্রশ্নের উত্তর আমার জানা নেই। আমি আপনাকে আমাদের একজন প্রতিনিধির সাথে সংযুক্ত করে দিচ্ছি।",
    "bl": "Dukkhito, eta amar jana nei. Ami apnake amader ekjon manusher (representative) sathe connect kore dichhi.",
}
FALLBACK_LEAD = {
    "bn": "তথ্যটি ইংরেজিতে দিচ্ছি: ",
    "bl": "Tothyo ti English e dichhi: ",
    "en": "",
}


def _numbers(text: str) -> set[str]:
    text = text.translate(_BN_DIGITS)
    return {re.sub(r"[,.]+$", "", n).replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def _bn_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if _BN_CHAR.match(c)) / len(letters) if letters else 0.0


def guard_reply(reply: str, faq_answer: str, question: str, lang: str, min_num_recall: float = 0.6):
    """Return (ok, why). Cheap, deterministic checks on what the small model wrote."""
    if not reply or len(reply) > 700:
        return False, "empty_or_long"
    r = _bn_ratio(reply)
    if lang == "bn" and r < 0.4:
        return False, "not_bangla_script"
    if lang == "bl" and r > 0.05:
        return False, "bangla_script_in_banglish"
    if lang in ("en", "bl") and not re.search(r"[A-Za-z]", reply):
        return False, "no_latin"
    if lang == "bl" and detect_language(reply) != "bl":
        return False, "banglish_expected_but_looks_english"
    allowed = _numbers(faq_answer) | _numbers(question)
    got = _numbers(reply)
    if got - allowed:
        return False, f"invented_numbers:{sorted(got - allowed)}"
    need = _numbers(faq_answer)
    if need and len(need & got) / len(need) < min_num_recall:
        return False, "dropped_numbers"
    return True, "ok"


# --------------------------------------------------------------------------
# 5. The assistant
# --------------------------------------------------------------------------
@dataclass
class Result:
    question: str
    language: str
    reply: str
    faq_id: int = 0
    handoff: bool = False
    reason: str = ""
    candidates: list = field(default_factory=list)
    t_retrieve: float = 0.0
    t_judge: float = 0.0
    t_write: float = 0.0

    @property
    def t_total(self):
        return self.t_retrieve + self.t_judge + self.t_write


class Assistant:
    def __init__(self, faq_path: str, backend, *, top_k=3, floor=0.20, llm_for_english=False, dense_model=None):
        self.faqs = {f.id: f for f in load_faq(faq_path)}
        self.retriever = Retriever(list(self.faqs.values()), dense_model)
        self.backend, self.top_k, self.floor, self.llm_en = backend, top_k, floor, llm_for_english

    # -- judge ---------------------------------------------------------
    def _judge(self, question, cands) -> int:
        entries = "\n".join(f"[{i}] Q: {f.question} A: {f.answer}" for i, (f, _) in enumerate(cands, 1))
        prompt = f"{JUDGE_FEWSHOT}\nNow the real case.\n{entries}\nQuestion: {question}\nAnswer (one digit only):"
        out = self.backend.chat([{"role": "user", "content": prompt}], max_tokens=3, temperature=0.0)
        m = re.search(r"\d", out.translate(_BN_DIGITS))
        n = int(m.group()) if m else 0
        return n if 1 <= n <= len(cands) else 0

    # -- writer --------------------------------------------------------
    def _write(self, question, faq: FAQ, lang) -> tuple[str, str]:
        ex = WRITER_EXAMPLE[lang]
        user = (f"{ex}\n\nNow the real case.\nFAQ answer: {faq.answer}\nCustomer question: {question}\n"
                f"Write the reply in {LANG_INSTR[lang]}\nReply:").strip()
        msgs = [{"role": "system", "content": WRITER_SYS}, {"role": "user", "content": user}]
        why = "no_attempt"
        for temp in (0.0, 0.3):  # one retry with a little randomness before falling back
            reply = self.backend.chat(msgs, max_tokens=220, temperature=temp)
            reply = re.sub(r"^(Reply|উত্তর)\s*[:：]\s*", "", reply.strip())
            ok, why = guard_reply(reply, faq.answer, question, lang)
            if ok:
                return reply, "answered_llm"
        return FALLBACK_LEAD[lang] + faq.answer, f"fallback_english({why})"

    # -- main ----------------------------------------------------------
    def answer(self, question: str) -> Result:
        lang = detect_language(question)
        t0 = time.perf_counter()
        cands = self.retriever.search(question, lang, self.top_k)
        t1 = time.perf_counter()
        res = Result(question, lang, "", candidates=[(f.id, round(s, 3)) for f, s in cands], t_retrieve=t1 - t0)

        def idk(reason):
            res.reply, res.handoff, res.reason = IDK[lang], True, reason
            return res

        if not cands or cands[0][1] < self.floor:
            return idk("below_floor")
        pick = self._judge(question, cands)
        res.t_judge = time.perf_counter() - t1
        if pick == 0:
            return idk("judge_none")
        faq = cands[pick - 1][0]
        res.faq_id = faq.id
        t2 = time.perf_counter()
        if lang == "en" and not self.llm_en:
            res.reply, res.reason = faq.answer, "answered_verbatim"
        else:
            res.reply, res.reason = self._write(question, faq, lang)
        res.t_write = time.perf_counter() - t2
        return res
