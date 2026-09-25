"""
Text -> phonemes (ARPAbet), used to drive the mouth.

Lip-sync only needs to know *which mouth shape* each sound needs, so this does
not have to be a perfect pronunciation engine. Order of preference per word:

  1. the CMU Pronouncing Dictionary, if the optional ``cmudict`` package is
     installed (``pip install cmudict``) - exact for ~130k English words;
  2. a small table of irregular everyday words;
  3. letter-to-sound rules, which handle the rest (product names such as
     "Nvidia", "Anthropic", "Mistral" ...).

Numbers, currency, percentages and acronyms are expanded to the words a
text-to-speech voice would actually say ("$5 billion" -> "five billion
dollars", "GPT" -> "G P T"), so the phoneme sequence matches the narration.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
_V = "aeiou"

# ----------------------------------------------------------------------------
# number / symbol expansion
# ----------------------------------------------------------------------------
_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
         "sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = [(10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (10 ** 3, "thousand")]
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
        "nine": "ninth", "twelve": "twelfth"}


def _below_1000(n: int) -> str:
    out = []
    if n >= 100:
        out.append(_ONES[n // 100] + " hundred")
        n %= 100
    if n >= 20:
        out.append(_TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else ""))
    elif n > 0 or not out:
        out.append(_ONES[n])
    return " ".join(out)


def num_to_words(n: int) -> str:
    if n < 0:
        return "minus " + num_to_words(-n)
    if n < 1000:
        return _below_1000(n)
    parts = []
    for val, name in _SCALES:
        if n >= val:
            parts.append(_below_1000(n // val) + " " + name)
            n %= val
    if n:
        parts.append(_below_1000(n))
    return " ".join(parts)


def _year_to_words(n: int) -> str:
    a, b = divmod(n, 100)
    if n == 2000:
        return "two thousand"
    if 2000 < n < 2010:
        return "two thousand " + _ONES[b]
    if b == 0:
        return _below_1000(a) + " hundred"
    return _below_1000(a) + " " + ("oh " + _ONES[b] if b < 10 else _below_1000(b))


def _ordinal(word: str) -> str:
    last = word.split()[-1].split("-")[-1]
    if last in _ORD:
        new = _ORD[last]
    elif last.endswith("y"):
        new = last[:-1] + "ieth"
    else:
        new = last + "th"
    return word[: len(word) - len(last)] + new


def _digits_out(s: str) -> str:
    return " ".join(_ONES[int(c)] for c in s)


def _expand_numbers(text: str) -> str:
    t = text
    # currency with optional scale word: $5 billion -> five billion dollars
    def money(m):
        num, scale = m.group(1).replace(",", ""), (m.group(2) or "").strip().lower()
        scale = {"k": "thousand", "m": "million", "b": "billion", "t": "trillion"}.get(scale, scale)
        if "." in num:
            i, f = num.split(".", 1)
            spoken = num_to_words(int(i or 0)) + " point " + _digits_out(f)
        else:
            spoken = num_to_words(int(num))
        return f" {spoken} {scale} dollars ".replace("  ", " ")
    t = re.sub(r"\$\s?(\d[\d,]*(?:\.\d+)?)(\s*(?:thousand|million|billion|trillion|[kKmMbBtT])\b)?", money, t)
    t = re.sub(r"(\d)\s?%", r"\1 percent", t)
    t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: " " + _ordinal(num_to_words(int(m.group(1)))) + " ", t)

    def dec(m):                                        # 4.5 -> four point five
        return f" {num_to_words(int(m.group(1)))} point {_digits_out(m.group(2))} "
    t = re.sub(r"\b(\d+)\.(\d+)\b", dec, t)

    def big(m):
        raw = m.group(0).replace(",", "")
        n = int(raw)
        if len(raw) == 4 and 1900 <= n <= 2099 and "," not in m.group(0):
            return " " + _year_to_words(n) + " "
        if len(raw) > 15:
            return " " + _digits_out(raw) + " "
        return " " + num_to_words(n) + " "
    t = re.sub(r"\d[\d,]*\d|\d", big, t)
    t = t.replace("&", " and ").replace("+", " plus ").replace("@", " at ")
    return t


# ----------------------------------------------------------------------------
# acronyms / spelling
# ----------------------------------------------------------------------------
_LETTER = {
    "a": "EY", "b": "B IY", "c": "S IY", "d": "D IY", "e": "IY", "f": "EH F", "g": "JH IY", "h": "EY CH",
    "i": "AY", "j": "JH EY", "k": "K EY", "l": "EH L", "m": "EH M", "n": "EH N", "o": "OW", "p": "P IY",
    "q": "K Y UW", "r": "AA R", "s": "EH S", "t": "T IY", "u": "Y UW", "v": "V IY", "w": "D AH B AH L Y UW",
    "x": "EH K S", "y": "W AY", "z": "Z IY",
}
_SAY_AS_WORD = {"NASA", "NVIDIA", "OPEC", "NATO", "LAMA", "SORA", "GROK", "BERT", "META", "TESLA", "APPLE",
                "GOOGLE", "AMAZON", "INTEL", "ARM", "SAP", "ASUS", "WWDC", "COVID", "SIRI", "ALEXA", "LORA"}
_SPELL = {"AI", "GPT", "API", "CEO", "CTO", "CFO", "GPU", "CPU", "TPU", "LLM", "LLMS", "USA", "US", "UK", "EU", "AMD",
          "IBM", "AWS", "SDK", "UI", "UX", "PC", "TV", "VR", "AR", "XR", "IPO", "ETF", "FBI", "CIA", "NSA", "SEC",
          "FDA", "FTC", "DOJ", "AGI", "ML", "NLP", "OCR", "SQL", "HTML", "CSS", "MA", "UN", "AGI", "SaaS", "RAG",
          "MCP", "IDE", "OS", "iOS", "HBM", "TSMC", "ASML", "R&D", "PDF", "URL", "USB", "SSD", "RAM", "ROI", "KPI"}


def _split_camel(tok: str) -> list[str]:
    # OpenAI -> Open AI, DeepMind -> Deep Mind, ChatGPT -> Chat GPT
    parts = re.findall(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+|[a-z]+|[A-Z]", tok)
    return parts if parts and "".join(parts) == tok else [tok]


# ----------------------------------------------------------------------------
# irregular words (ARPAbet, no stress digits; primary stress is on the first vowel unless noted with ')
# ----------------------------------------------------------------------------
_EXC = {
    "the": "DH AH", "a": "AH", "an": "AH N", "of": "AH V", "to": "T UW", "and": "AH N D", "is": "IH Z", "was": "W AH Z",
    "are": "AA R", "were": "W ER", "be": "B IY", "been": "B IH N", "for": "F AO R", "you": "Y UW", "your": "Y AO R",
    "one": "W AH N", "once": "W AH N S", "two": "T UW", "said": "S EH D", "says": "S EH Z", "have": "HH AE V",
    "has": "HH AE Z", "do": "D UW", "does": "D AH Z", "done": "D AH N", "from": "F R AH M", "they": "DH EY",
    "their": "DH EH R", "there": "DH EH R", "where": "W EH R", "what": "W AH T", "who": "HH UW", "whom": "HH UW M",
    "why": "W AY", "when": "W EH N", "would": "W UH D", "could": "K UH D", "should": "SH UH D", "people": "P IY P AH L",
    "because": "B IH K AH Z", "through": "TH R UW", "though": "DH OW", "thought": "TH AO T", "enough": "IH N AH F",
    "know": "N OW", "known": "N OW N", "new": "N UW", "news": "N UW Z", "this": "DH IH S", "that": "DH AE T",
    "these": "DH IY Z", "those": "DH OW Z", "them": "DH EH M", "then": "DH EH N", "than": "DH AE N", "with": "W IH DH",
    "without": "W IH DH AW T", "other": "AH DH ER", "another": "AH N AH DH ER", "mother": "M AH DH ER",
    "father": "F AA DH ER", "brother": "B R AH DH ER", "together": "T AH G EH DH ER", "weather": "W EH DH ER",
    "whether": "W EH DH ER", "rather": "R AE DH ER", "either": "IY DH ER", "neither": "N IY DH ER",
    "he": "HH IY", "she": "SH IY", "we": "W IY", "me": "M IY", "my": "M AY", "by": "B AY", "so": "S OW", "no": "N OW",
    "go": "G OW", "up": "AH P", "out": "AW T", "about": "AH B AW T", "into": "IH N T UW", "onto": "AA N T UW",
    "over": "OW V ER", "under": "AH N D ER", "after": "AE F T ER", "before": "B IH F AO R", "between": "B IH T W IY N",
    "great": "G R EY T", "break": "B R EY K", "steak": "S T EY K", "eight": "EY T", "weight": "W EY T",
    "height": "HH AY T", "night": "N AY T", "light": "L AY T", "right": "R AY T", "might": "M AY T",
    "sight": "S AY T", "tight": "T AY T", "high": "HH AY", "sign": "S AY N", "design": "D IH Z AY N",
    "laugh": "L AE F", "cough": "K AO F", "tough": "T AH F", "rough": "R AH F", "though": "DH OW",
    "world": "W ER L D", "word": "W ER D", "work": "W ER K", "worth": "W ER TH", "were": "W ER",
    "come": "K AH M", "some": "S AH M", "love": "L AH V", "above": "AH B AH V", "move": "M UW V", "prove": "P R UW V",
    "give": "G IH V", "live": "L IH V", "have": "HH AE V", "gave": "G EY V", "get": "G EH T", "gets": "G EH T S",
    "target": "T AA R G IH T", "begin": "B IH G IH N", "girl": "G ER L", "gift": "G IH F T", "forget": "F ER G EH T",
    "any": "EH N IY", "many": "M EH N IY", "very": "V EH R IY", "every": "EH V R IY", "family": "F AE M AH L IY",
    "money": "M AH N IY", "company": "K AH M P AH N IY", "country": "K AH N T R IY", "young": "Y AH NG",
    "touch": "T AH CH", "double": "D AH B AH L", "trouble": "T R AH B AH L", "couple": "K AH P AH L",
    "group": "G R UW P", "soup": "S UW P", "youth": "Y UW TH", "four": "F AO R", "court": "K AO R T",
    "source": "S AO R S", "course": "K AO R S", "our": "AW ER", "hour": "AW ER", "honest": "AA N AH S T",
    "heir": "EH R", "half": "HH AE F", "calm": "K AA M", "palm": "P AA M", "walk": "W AO K", "talk": "T AO K",
    "chalk": "CH AO K", "want": "W AA N T", "watch": "W AA CH", "water": "W AO T ER", "war": "W AO R",
    "warm": "W AO R M", "was": "W AH Z", "wash": "W AA SH", "swan": "S W AA N", "quality": "K W AA L AH T IY",
    "head": "HH EH D", "dead": "D EH D", "bread": "B R EH D", "ready": "R EH D IY", "health": "HH EH L TH",
    "breath": "B R EH TH", "death": "D EH TH", "threat": "TH R EH T", "heavy": "HH EH V IY", "instead": "IH N S T EH D",
    "ahead": "AH HH EH D", "spread": "S P R EH D", "measure": "M EH ZH ER", "pleasure": "P L EH ZH ER",
    "treasure": "T R EH ZH ER", "vision": "V IH ZH AH N", "decision": "D IH S IH ZH AH N",
    "good": "G UH D", "book": "B UH K", "look": "L UH K", "took": "T UH K", "foot": "F UH T", "stood": "S T UH D",
    "wood": "W UH D", "put": "P UH T", "full": "F UH L", "pull": "P UH L", "push": "P UH SH", "bull": "B UH L",
    "how": "HH AW", "now": "N AW", "cow": "K AW", "wow": "W AW", "allow": "AH L AW", "down": "D AW N",
    "town": "T AW N", "brown": "B R AW N", "power": "P AW ER", "flower": "F L AW ER", "tower": "T AW ER",
    "show": "SH OW", "low": "L OW", "slow": "S L OW", "grow": "G R OW", "flow": "F L OW", "below": "B IH L OW",
    "video": "V IH D IY OW", "hello": "HH AH L OW", "radio": "R EY D IY OW", "studio": "S T UW D IY OW",
    "data": "D EY T AH", "area": "EH R IY AH", "idea": "AY D IY AH", "real": "R IY L", "really": "R IY L IY",
    "media": "M IY D IY AH", "social": "S OW SH AH L", "special": "S P EH SH AH L", "official": "AH F IH SH AH L",
    "business": "B IH Z N AH S", "usually": "Y UW ZH AH W AH L IY", "actually": "AE K CH AH W AH L IY",
    "million": "M IH L Y AH N", "billion": "B IH L Y AH N", "trillion": "T R IH L Y AH N",
    "percent": "P ER S EH N T", "dollars": "D AA L ER Z", "dollar": "D AA L ER", "february": "F EH B Y UW EH R IY",
    "tuesday": "T UW Z D EY", "wednesday": "W EH N Z D EY", "thursday": "TH ER Z D EY", "monday": "M AH N D EY",
    "friday": "F R AY D EY", "saturday": "S AE T ER D EY", "sunday": "S AH N D EY",
    "ai": "EY AY", "openai": "OW P AH N EY AY", "nvidia": "EH N V IH D IY AH", "anthropic": "AE N TH R AA P IH K",
    "gemini": "JH EH M AH N AY", "llama": "L AA M AH", "mistral": "M IH S T R AH L", "deepseek": "D IY P S IY K",
    "microsoft": "M AY K R AH S AA F T", "google": "G UW G AH L", "amazon": "AE M AH Z AA N", "tesla": "T EH S L AH",
    "china": "CH AY N AH", "chinese": "CH AY N IY Z", "europe": "Y UH R AH P", "european": "Y UH R AH P IY AH N",
    "japan": "JH AH P AE N", "india": "IH N D IY AH", "korea": "K ER IY AH", "israel": "IH Z R IY AH L",
    "chip": "CH IH P", "chips": "CH IH P S", "chatbot": "CH AE T B AA T", "chatbots": "CH AE T B AA T S",
    "technology": "T EH K N AA L AH JH IY", "technical": "T EH K N IH K AH L", "chemical": "K EH M IH K AH L",
    "school": "S K UW L", "character": "K EH R AH K T ER", "architecture": "AA R K IH T EH K CH ER",
    "machine": "M AH SH IY N", "michael": "M AY K AH L", "queue": "K Y UW", "guest": "G EH S T", "guess": "G EH S",
    "guide": "G AY D", "guitar": "G IH T AA R", "build": "B IH L D", "built": "B IH L T", "biology": "B AY AA L AH JH IY",
    "receive": "R IH S IY V", "believe": "B IH L IY V", "achieve": "AH CH IY V", "ocean": "OW SH AH N",
    "open": "OW P AH N", "opens": "OW P AH N Z", "model": "M AA D AH L", "models": "M AA D AH L Z",
    "agent": "EY JH AH N T", "agents": "EY JH AH N T S", "again": "AH G EH N", "against": "AH G EH N S T",
    "beautiful": "B Y UW T AH F AH L", "experience": "IH K S P IH R IY AH N S", "science": "S AY AH N S",
    "scientist": "S AY AH N T IH S T", "scientists": "S AY AH N T IH S T S",
}
_FUNC = {"the", "a", "an", "of", "to", "and", "is", "was", "are", "were", "be", "been", "for", "in", "on", "at", "or",
         "it", "its", "as", "by", "with", "from", "that", "this", "than", "then", "but", "not", "so", "if", "he", "she",
         "we", "they", "i", "you", "his", "her", "their", "our", "my", "your", "has", "have", "had", "will", "would",
         "can", "could", "should", "do", "does", "did", "into", "up", "out", "about", "over", "who", "which", "what"}

_cmu = None
_cmu_tried = False


def _cmudict():
    global _cmu, _cmu_tried
    if not _cmu_tried:
        _cmu_tried = True
        try:                                            # optional dependency
            import cmudict                              # type: ignore
            _cmu = cmudict.dict()
        except Exception:
            _cmu = None
    return _cmu


def set_dictionary(d) -> None:
    """Inject a CMU-style dict (word -> [[ARPAbet with stress digits], ...]). Mainly for tests."""
    global _cmu, _cmu_tried
    _cmu, _cmu_tried = d, True


# ----------------------------------------------------------------------------
# letter-to-sound rules
# ----------------------------------------------------------------------------
def _is_v(c: str) -> bool:
    return bool(c) and c in _V


def _vowel_groups(w: str) -> int:
    return len(re.findall(r"[aeiouy]+", w))


def _one_syllable(stem: str) -> bool:
    return _vowel_groups(stem) == 1


def _restore_e(stem: str) -> str:
    """making -> mak -> make; hoped -> hop -> hope (only for one-syllable CVC stems)."""
    if (len(stem) >= 2 and _one_syllable(stem) and not _is_v(stem[-1]) and stem[-1] not in "wxy"
            and len(stem) >= 2 and _is_v(stem[-2]) and (len(stem) == 2 or not _is_v(stem[-3]))
            and not (len(stem) >= 3 and stem[-3] == stem[-1])):
        if len(stem) >= 3 and stem[-2] in "ae" and stem[-1] in "tdnmlp" and len(stem) == 3 and stem[-3] in "ptdbc":
            return stem + "e"
        if stem[-2] in "aiou" and stem[-1] in "ktdvnlmspzrgbcf":
            return stem + "e"
    return stem


def _rules(w: str) -> list[tuple[str, int]]:
    """Return [(phoneme, stress)] for a lowercase alphabetic word."""
    out: list[tuple[str, int]] = []
    n = len(w)
    silent_e = n > 2 and w.endswith("e") and not w.endswith(("ee", "ie", "ue", "oe", "le")) and not _is_v(w[-2])
    if w.endswith("le") and n > 3 and not _is_v(w[-3]):
        silent_e = False
    # vowel groups (for reduction)
    groups = [(m.start(), m.end()) for m in re.finditer(r"[aeiouy]+", w)]
    nvg = len(groups)
    stress_at = {0: 1}
    if nvg >= 4:
        stress_at[2] = 2

    def group_index(i: int) -> int:
        for gi, (a, b) in enumerate(groups):
            if a <= i < b:
                return gi
        return -1

    def stressed(i: int) -> bool:
        gi = group_index(i)
        return gi in stress_at or nvg == 1

    i = 0
    P = lambda *ph: out.extend((p, 0) for p in ph)

    def V(ph: str, i_: int):
        s = 1 if stressed(i_) else 0
        out.append((ph, s))

    while i < n:
        c = w[i]
        rest = w[i:]
        prev = w[i - 1] if i > 0 else ""
        nxt = w[i + 1] if i + 1 < n else ""
        nxt2 = w[i + 2] if i + 2 < n else ""

        # ---- consonant digraphs / special sequences
        if rest.startswith("tch"):
            P("CH"); i += 3; continue
        if rest.startswith("tion") or rest.startswith("sion") and prev and not _is_v(prev):
            P("SH", "AH", "N"); i += 4; continue
        if rest.startswith("sion"):
            P("ZH", "AH", "N"); i += 4; continue
        if rest.startswith(("cian", "tial", "cial")):
            P("SH", "AH", "N" if rest[3] == "n" else "L"); i += 4; continue
        if rest.startswith("ture") and i > 0:
            P("CH", "ER"); i += 4; continue
        if rest.startswith("ch"):
            if i == 0 and w.startswith("chr") or w.startswith(("chem", "char", "chaos", "chorus", "chron")):
                P("K"); i += 2; continue
            P("CH"); i += 2; continue
        if rest.startswith("sch") and i == 0:
            P("S", "K"); i += 3; continue
        if rest.startswith("sh"):
            P("SH"); i += 2; continue
        if rest.startswith("th"):
            if _is_v(prev) and _is_v(w[i + 2:i + 3]):
                P("DH")
            else:
                P("TH")
            i += 2; continue
        if rest.startswith("ph"):
            P("F"); i += 2; continue
        if rest.startswith("gh"):
            if i == 0:
                P("G")
            elif prev == "u" and i == n - 2:
                P("F")
            i += 2; continue
        if rest.startswith("ck"):
            P("K"); i += 2; continue
        if rest.startswith("ng"):
            P("NG"); i += 2; continue
        if rest.startswith("nk"):
            P("NG", "K"); i += 2; continue
        if rest.startswith("qu"):
            P("K", "W"); i += 2; continue
        if rest.startswith("wh"):
            P("W"); i += 2; continue
        if rest.startswith(("kn", "gn")) and i == 0:
            P("N"); i += 2; continue
        if rest.startswith("wr") and i == 0:
            P("R"); i += 2; continue
        if rest.startswith("mb") and i == n - 2:
            P("M"); i += 2; continue
        if c == "c":
            if nxt in "eiy":
                P("S")
            else:
                P("K")
            i += 2 if nxt == "c" and nxt2 not in "eiy" else 1
            if nxt == "c" and nxt2 in "eiy":
                P("S"); i += 0
            continue
        if c == "g":
            if nxt in "eiy" and not (w.startswith(("get", "gif", "giv", "gir", "gea"))) and i + 1 < n:
                P("JH")
            else:
                P("G")
            i += 2 if nxt == "g" else 1
            continue
        if c == "x":
            P("Z") if i == 0 else P("K", "S"); i += 1; continue
        if c == "j":
            P("JH"); i += 1; continue
        if c == "q":
            P("K"); i += 1; continue
        if c == "s":
            if prev and _is_v(prev) and _is_v(nxt) and nxt != "":
                P("Z")
            elif i == n - 1 and prev in "aeioubdglmnrvy":
                P("Z")
            else:
                P("S")
            i += 2 if nxt == "s" else 1
            continue
        if c == "h":
            if i == 0 or _is_v(prev) or prev == "":
                P("HH")
            i += 1; continue
        if c == "w":
            P("W"); i += 1; continue
        if c == "y":
            if i == 0 and _is_v(nxt):
                P("Y")
            elif i == n - 1:
                if _vowel_groups(w) == 1 and n <= 3:
                    V("AY", i)
                else:
                    out.append(("IY", 0))
            elif _is_v(nxt) and not _is_v(prev):
                P("Y")
            else:
                out.append(("IH", 0))
            i += 1; continue
        if c in "bdfklmnprtvz":
            ph = {"b": "B", "d": "D", "f": "F", "k": "K", "l": "L", "m": "M", "n": "N", "p": "P", "r": "R",
                  "t": "T", "v": "V", "z": "Z"}[c]
            P(ph)
            i += 2 if nxt == c else 1
            continue

        # ---- vowels ---------------------------------------------------------------
        last_group = group_index(i) == nvg - 1
        # word-final silent e
        if c == "e" and i == n - 1 and silent_e:
            i += 1; continue
        # r-coloured
        if c == "a" and nxt == "r":
            V("AA", i); P("R"); i += 2; continue
        if c == "o" and nxt == "r":
            V("AO", i); P("R"); i += 2; continue
        if c in "eiu" and nxt == "r" and not _is_v(nxt2):
            if rest.startswith("ear") and not rest.startswith(("earn", "earl", "earth", "early", "earc")):
                V("IH", i); P("R"); i += 3; continue
            if rest.startswith("eer") or rest.startswith("ier"):
                V("IH", i); P("R"); i += 3; continue
            if rest.startswith(("earn", "earl", "earth", "early", "earc")):
                V("ER", i); i += 3; continue
            V("ER", i); i += 2; continue
        if rest.startswith("air"):
            V("EH", i); P("R"); i += 3; continue
        if rest.startswith("are") and i + 3 == n:
            V("EH", i); P("R"); i += 3; continue
        if rest.startswith("ire") and i + 3 == n:
            V("AY", i); P("ER"); i += 3; continue
        if rest.startswith("ore") and i + 3 == n:
            V("AO", i); P("R"); i += 3; continue
        if rest.startswith("ure") and i + 3 == n:
            V("UH", i); P("R"); i += 3; continue
        # digraphs
        two = rest[:2]
        if two in ("ai", "ay"):
            V("EY", i); i += 2; continue
        if rest.startswith("eigh"):
            V("EY", i); i += 4; continue
        if rest.startswith("igh"):
            V("AY", i); i += 3; continue
        if two == "ea":
            if rest.startswith(("ead", "eath", "eas", "eal")) and not rest.startswith(("eals", "eal")) and n > 4:
                V("EH", i)
            else:
                V("IY", i)
            i += 2; continue
        if two == "ee":
            V("IY", i); i += 2; continue
        if two == "ei":
            V("IY" if prev == "c" else "EY", i); i += 2; continue
        if two in ("eu", "ew"):
            V("UW", i); i += 2; continue
        if two == "ey":
            V("IY", i); i += 2; continue
        if two == "ie":
            V("IY", i); i += 2; continue
        if two in ("oa",):
            V("OW", i); i += 2; continue
        if two == "oe":
            V("OW", i); i += 2; continue
        if two in ("oi", "oy"):
            V("OY", i); i += 2; continue
        if two == "oo":
            V("UH" if nxt2 == "k" else "UW", i); i += 2; continue
        if two == "ou":
            if nxt2 == "r":
                V("AO", i); P("R"); i += 3; continue
            if rest.startswith(("ough", "ought")):
                V("AO" if rest.startswith("ought") else "AH", i); i += 2; continue
            V("AW", i); i += 2; continue
        if two == "ow":
            if i == n - 2:
                V("OW", i)
            elif nxt2 in ("n", "l", "d", "e") and (nxt2 != "e" or rest.startswith("ower")):
                V("AW", i)
            else:
                V("OW", i)
            i += 2; continue
        if two in ("au", "aw"):
            V("AO", i); i += 2; continue
        if two == "ui":
            V("UW", i); i += 2; continue
        if two == "ue" and i == n - 2:
            V("UW", i); i += 2; continue
        if two == "ia" or two == "io":
            out.append(("IY", 0)); out.append(("AH", 0)); i += 2; continue
        # single vowels: VCe pattern, otherwise short / reduced
        vce = (i + 2 < n and not _is_v(nxt) and nxt not in "hqwxy" and
               ((i + 3 == n and w[-1] == "e") or (i + 4 == n and w[-2:] == "es") or (i + 4 == n and w[-2:] == "ed")))
        open_final = (i == n - 1)
        if c == "a":
            if vce:
                V("EY", i)
            elif open_final:
                out.append(("AH", 0))
            elif stressed(i):
                V("AE", i)
            else:
                out.append(("AH", 0))
        elif c == "e":
            if vce:
                V("IY", i)
            elif open_final:
                out.append(("IY", 0))
            elif stressed(i):
                V("EH", i)
            else:
                out.append(("AH" if (last_group and nvg > 1) else "IH", 0))
        elif c == "i":
            if vce:
                V("AY", i)
            elif open_final:
                out.append(("IY", 0))
            elif stressed(i):
                V("IH", i)
            else:
                out.append(("IH", 0))
        elif c == "o":
            if vce:
                V("OW", i)
            elif open_final:
                V("OW", i)
            elif stressed(i):
                V("AA", i)
            else:
                out.append(("AH", 0))
        elif c == "u":
            if vce:
                V("UW", i)
            elif stressed(i):
                V("AH", i)
            else:
                out.append(("AH", 0))
        else:
            pass
        i += 1
    return out


def _word_phones(word: str) -> list[tuple[str, int]]:
    """word (lowercase letters/apostrophes) -> [(ARPAbet, stress 0/1)]"""
    w = word.lower().replace("’", "'")
    if w in _EXC:
        ph = _EXC[w].split()
        out, marked = [], False
        for p in ph:
            if p in VOWELS and not marked:
                out.append((p, 0 if w in _FUNC else 1)); marked = True
            else:
                out.append((p, 0))
        return out
    d = _cmudict()
    if d:
        key = w.replace("'", "") if w.replace("'", "") in d and w not in d else w
        prons = d.get(key)
        if prons:
            out = []
            for p in prons[0]:
                m = re.match(r"([A-Z]+)(\d)?", p)
                sym, st = m.group(1), m.group(2)
                out.append((sym, 1 if st in ("1", "2") else 0))
            return out
    w = w.replace("'", "")
    if not w:
        return []
    # inflectional endings
    if len(w) > 4 and w.endswith("ing"):
        stem = w[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2] and not _is_v(stem[-1]):
            stem = stem[:-1]
        return _word_phones(_restore_e(stem)) + [("IH", 0), ("NG", 0)]
    if len(w) > 3 and w.endswith("ed") and not w.endswith(("eed", "ied")):
        stem = w[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2] and not _is_v(stem[-1]):
            stem = stem[:-1]
        stem = _restore_e(stem)
        base = _word_phones(stem)
        last = stem[-1] if stem else ""
        if last in "td":
            return base + [("IH", 0), ("D", 0)]
        if last in "pkfsxhc" or stem.endswith(("ch", "sh", "th", "ce", "se")):
            return base + [("T", 0)]
        return base + [("D", 0)]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is", "'s")):
        stem = w[:-2] if w.endswith(("ses", "xes", "zes", "ches", "shes")) else w[:-1]
        if stem in _EXC or len(stem) >= 3:
            base = _word_phones(stem)
            if w.endswith(("ses", "xes", "zes", "ches", "shes")):
                return base + [("IH", 0), ("Z", 0)]
            last = base[-1][0] if base else ""
            if last in ("P", "T", "K", "F", "TH"):
                return base + [("S", 0)]
            if last in ("S", "Z", "SH", "ZH", "CH", "JH"):
                return base + [("IH", 0), ("Z", 0)]
            return base + [("Z", 0)]
    if w.endswith("le") and len(w) > 3 and not _is_v(w[-3]):
        base = _rules(w[:-2] + "e") if False else _rules(w[:-2])
        return base + [("AH", 0), ("L", 0)]
    return _rules(w)


# ----------------------------------------------------------------------------
# public: text -> item list
# ----------------------------------------------------------------------------
@dataclass
class Item:
    kind: str            # "ph" phoneme | "wb" word break | "pause" punctuation pause ("short"/"long")
    sym: str = ""
    stress: int = 0
    word: int = -1
    pause: str = ""      # for kind == "pause": "short" | "long"


def _tokens(text: str) -> list[str]:
    """normalised text -> list of words and punctuation markers ('|' short pause, '||' long pause)"""
    t = _expand_numbers(text)
    t = t.replace("’", "'").replace("–", "-")
    t = re.sub(r"\s*[—–]\s*|\s-\s", " , ", t)
    t = re.sub(r"[\"“”()\[\]]", " ", t)
    raw = re.findall(r"[A-Za-z][A-Za-z'\-/]*|[,;:!?.]+|\S", t)
    out: list[str] = []
    for tok in raw:
        if re.fullmatch(r"[,;:]+", tok):
            out.append("|")
        elif re.fullmatch(r"[!?.]+", tok):
            out.append("||")
        elif re.match(r"[A-Za-z]", tok):
            for part in re.split(r"[-/]", tok):
                part = part.strip("'")
                if not part:
                    continue
                if part in _SPELL or (part.isupper() and 2 <= len(part) <= 3 and part not in _SAY_AS_WORD
                                      and not any(c in "AEIOU" for c in part[1:2]) and False):
                    out.append("#" + part)
                elif part.isupper() and len(part) >= 2:
                    if part in _SAY_AS_WORD:
                        out.append(part.lower())
                    elif len(part) <= 3 or not re.search(r"[AEIOUY]", part):
                        out.append("#" + part)
                    else:
                        out.append(part.lower())
                else:
                    for sub in _split_camel(part):
                        if sub.isupper() and len(sub) >= 2 and sub not in _SAY_AS_WORD:
                            out.append("#" + sub)
                        else:
                            out.append(sub.lower())
    return out


def text_to_items(text: str) -> list[Item]:
    from . import amharic_phonemes
    if amharic_phonemes.has_ethiopic(text):
        return amharic_phonemes.text_to_items(text, Item, english_word=_word_phones)
    items: list[Item] = []
    wi = 0
    for tok in _tokens(text):
        if tok == "|":
            items.append(Item("pause", pause="short"))
            continue
        if tok == "||":
            items.append(Item("pause", pause="long"))
            continue
        if items and items[-1].kind == "ph":
            items.append(Item("wb"))
        if tok.startswith("#"):                          # spelled-out acronym
            letters = [c for c in tok[1:].lower() if c.isalpha()]
            for li, ch in enumerate(letters):
                for k, p in enumerate(_LETTER[ch].split()):
                    items.append(Item("ph", p, 1 if p in VOWELS and k == len(_LETTER[ch].split()) - 1 else 0, wi))
                if li < len(letters) - 1:
                    pass
            wi += 1
            continue
        phs = _word_phones(tok)
        if not phs:
            continue
        # phrase-final / content words carry the stress the mouth should show
        for p, st in phs:
            items.append(Item("ph", p, st, wi))
        wi += 1
    # drop leading/trailing breaks
    while items and items[0].kind != "ph":
        items.pop(0)
    while items and items[-1].kind == "wb":
        items.pop()
    return items


def phoneme_string(text: str) -> str:
    """Debug helper: 'hello world' -> 'HH AH L OW | W ER L D'"""
    out = []
    for it in text_to_items(text):
        out.append(it.sym if it.kind == "ph" else ("|" if it.kind == "wb" else ","))
    return " ".join(out)
