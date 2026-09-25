"""
Amharic (Ge'ez / Ethiopic script) -> phonemes, for lip-sync.

Ge'ez is a syllabary laid out very regularly in Unicode: every consonant has a
row of 8 code points, one per vowel "order" (ä u i a e ə o, then -wa).
So a letter tells us exactly which consonant and which vowel are spoken, and we
can emit the same ARPAbet-style phonemes the English lip-sync uses. That gives
the robot exactly the same mouth behaviour as in English: lips shut on m / b / p,
round on u / o / w, open on a, a small opening on i / e.
"""
from __future__ import annotations

import re

# first code point of each consonant row -> (consonant phoneme or "" for a bare vowel, labialised row?)
_ROWS = {
    0x1200: ("HH", False), 0x1208: ("L", False), 0x1210: ("HH", False), 0x1218: ("M", False),
    0x1220: ("S", False), 0x1228: ("R", False), 0x1230: ("S", False), 0x1238: ("SH", False),
    0x1240: ("K", False), 0x1248: ("K", True), 0x1250: ("K", False), 0x1258: ("K", True),
    0x1260: ("B", False), 0x1268: ("V", False), 0x1270: ("T", False), 0x1278: ("CH", False),
    0x1280: ("HH", False), 0x1288: ("HH", True), 0x1290: ("N", False), 0x1298: ("N", False),
    0x12A0: ("", False), 0x12A8: ("K", False), 0x12B0: ("K", True), 0x12B8: ("HH", False),
    0x12C0: ("HH", True), 0x12C8: ("W", False), 0x12D0: ("", False), 0x12D8: ("Z", False),
    0x12E0: ("ZH", False), 0x12E8: ("Y", False), 0x12F0: ("D", False), 0x12F8: ("D", False),
    0x1300: ("JH", False), 0x1308: ("G", False), 0x1310: ("G", True), 0x1318: ("G", False),
    0x1320: ("T", False), 0x1328: ("CH", False), 0x1330: ("P", False), 0x1338: ("S", False),
    0x1340: ("S", False), 0x1348: ("F", False), 0x1350: ("P", False),
}
# vowel orders 1..7 (index 0..6); the 6th order (ə) is usually silent at the end of a word
_ORDER = ["AH", "UW", "IY", "AA", "EY", "IH", "OW"]
_LAB_ORDER = {0: "AH", 2: "IY", 3: "AA", 4: "EY", 5: "AH"}          # labialised rows: kwä, kwi, kwa ...
_PALATAL = {0x1298: True}                                              # ñ: n + y


def _letter(cp: int):
    """-> list of (phoneme, is_vowel) for one Ethiopic letter, or None."""
    if not (0x1200 <= cp <= 0x1357):
        return None
    base = cp - (cp - 0x1200) % 8
    order = (cp - 0x1200) % 8
    row = _ROWS.get(base)
    if row is None:
        return None
    cons, lab = row
    out = []
    if cons:
        out.append((cons, False))
        if base in _PALATAL:
            out.append(("Y", False))
    if lab:
        v = _LAB_ORDER.get(order)
        if v is None:
            return out or None
        out += [("W", False), (v, True)]
        return out
    if order == 7:                                                     # 8th slot: consonant + wa
        return out + [("W", False), ("AA", True)]
    out.append((_ORDER[order], True))
    return out


_ETHIOPIC = re.compile(r"[ሀ-፿]")


def has_ethiopic(text: str) -> bool:
    return bool(_ETHIOPIC.search(text or ""))


def text_to_items(text: str, Item, english_word=None):
    """Same output as phonemes.text_to_items (a list of Item) for Amharic text.
    Latin-script words inside (e.g. "AI") are passed to `english_word`."""
    items = []
    wi = 0
    t = text.replace("።", " || ").replace("፧", " || ").replace("!", " || ").replace("?", " || ")
    t = re.sub(r"[፣፤፥፦፡,;:]", " | ", t)
    for tok in t.split():
        if tok == "|":
            items.append(Item("pause", pause="short"))
            continue
        if tok == "||":
            items.append(Item("pause", pause="long"))
            continue
        phs: list[tuple[str, int]] = []
        if _ETHIOPIC.search(tok):
            letters = [c for c in tok if 0x1200 <= ord(c) <= 0x1357]
            first_vowel = True
            for k, ch in enumerate(letters):
                seg = _letter(ord(ch))
                if not seg:
                    continue
                last = k == len(letters) - 1
                for p, is_v in seg:
                    if is_v and p == "IH" and (last or len(seg) == 1):
                        continue                                       # word-final / bare 6th order: no vowel
                    stress = 1 if (is_v and first_vowel and p != "IH") else 0
                    if is_v and p != "IH":
                        first_vowel = False
                    phs.append((p, stress))
        elif english_word is not None and re.search(r"[A-Za-z]", tok):
            phs = english_word(re.sub(r"[^A-Za-z']", "", tok).lower())
        if not phs:
            continue
        if items and items[-1].kind == "ph":
            items.append(Item("wb"))
        for p, st in phs:
            items.append(Item("ph", p, st, wi))
        wi += 1
    while items and items[0].kind != "ph":
        items.pop(0)
    while items and items[-1].kind != "ph":
        items.pop()
    return items
