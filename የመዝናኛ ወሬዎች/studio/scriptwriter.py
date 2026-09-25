"""Turns today's headlines into a broadcast script (Gemini), with an offline fallback."""
from __future__ import annotations

import json
import re
from datetime import datetime

MODELS_FALLBACK = ["gemini-flash-latest", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-lite-latest"]
ETHIOPIC = re.compile(r"[ሀ-፿]")


def ordinal(n: int) -> str:
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def clean(text: str) -> str:
    t = re.sub(r"[*_`#>\[\]{}<>|~^]", "", text or "")
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"\([^)]*\)", "", t)
    t = t.replace("&", " and ").replace("—", ", ").replace("–", ", ").replace("\u201c", "").replace("\u201d", "")
    t = re.sub(r"[\U0001F000-\U0001FFFF\u2600-\u27BF]", "", t)
    return re.sub(r"\s+", " ", t).strip()


# words a phrase must never be left dangling on when it gets shortened
_DANGLING = {
    "a", "an", "the", "and", "or", "but", "if", "as", "at", "by", "for", "from", "in",
    "into", "of", "on", "onto", "over", "per", "so", "than", "that", "to", "up", "via",
    "with", "within", "without", "after", "before", "about", "is", "are", "was", "were",
    "be", "been", "has", "have", "had", "will", "would", "can", "could", "may", "says",
    "said", "it", "its", "his", "her", "their", "this", "these", "those", "not", "no",
}


def _short(text: str, words: int = 9) -> str:
    """Shorten to at most `words` words WITHOUT leaving a half-finished phrase.

    The old version just sliced the first N words, which is what produced on-screen
    text like "Trump says it's time to rebrand AI with a". Now we prefer to cut at a
    natural boundary (comma / dash / colon), and if we do have to cut mid-sentence we
    walk back over any trailing linking word so the line ends on something meaningful.
    """
    t = clean(text)
    w = t.split()
    if len(w) <= words:
        return t.rstrip(",;:-")

    # 1) prefer a clean clause boundary that already fits
    head = " ".join(w[:words])
    for sep in (";", ",", " - ", ":"):
        if sep in head:
            cand = head.rsplit(sep, 1)[0].strip()
            if len(cand.split()) >= max(3, words - 4):
                return cand.rstrip(",;:-")

    # 2) otherwise trim back off any dangling linking words
    cut = list(w[:words])
    while len(cut) > 3 and cut[-1].lower().strip(",;:.").strip("'\"") in _DANGLING:
        cut.pop()
    return " ".join(cut).rstrip(",;:-")


def intro_outro(studio: str, topic: str, today: datetime):
    """Sofia opens the weekly entertainment show with her signature greeting, then goes
    straight into the first story. No date is mentioned. The outro is a warm sign-off."""
    intro = dict(headline=studio, ticker="ሶፊያ • የመዝናኛ ወሬዎች", sentences=[
        dict(say="ከሳምንቱ የመዝናኛ ወሬዎች ጋር ሶፊያ ነኝ አብራችሁኝ ቆዩ።",
             highlight="ሶፊያ ነኝ • አብራችሁኝ ቆዩ"),
    ])
    outro = dict(headline="ስለተከታተሉን እናመሰግናለን", ticker="በሚቀጥለው ሳምንት እንገናኛለን", sentences=[
        dict(say="የዚህ ሳምንት የመዝናኛ ወሬዎቻችን ይህን ይመስል ነበር።", highlight="ወሬዎቹ ይህን ይመስሉ ነበር"),
        dict(say="ላይክ አድርጉልን፣ ሰብስክራይብ አድርጉ፣ በሚቀጥለው ሳምንት እንገናኛለን።",
             highlight="ላይክ • ሰብስክራይብ • እንገናኛለን"),
    ])
    return intro, outro


def _prompt(items: list[dict], today: datetime, words: int, n_stories: int, studio: str,
            topic: str, max_highlight_seconds: float, target_seconds: float = 60.0, focus: str = "world") -> str:
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(f"[{i}] {it['source']} — {it['title']} — {it['summary'][:420]}")
    length = f"{round(target_seconds / 60)}-minute"
    return f"""You are the scriptwriter for an AMHARIC-language WEEKLY US-ENTERTAINMENT news show called "{studio}", presented by a warm, natural young woman named Sofia (ሶፊያ). It is a {length} vertical video for YouTube, TikTok and Instagram.
Write the segment Sofia will read on camera, using ONLY the news items below (the items are in English — translate the facts into natural, conversational Amharic).

RULES
- WRITE EVERYTHING IN AMHARIC, in Ge'ez (Ethiopic) script: every "say", "highlight", "headline", "ticker", "title"
  and "summary". Natural, warm, friendly broadcast Amharic — the tone of a young female entertainment presenter
  chatting with her viewers, not stiff news reading. End every sentence with the Amharic full stop "።".
- This is a WEEKLY US-ENTERTAINMENT roundup: cover the {n_stories} biggest ENTERTAINMENT stories of the past week —
  famous American singers and musicians, movie actors and actresses, box-office and movie news, TV, awards
  (Oscars, Grammys, Emmys, Golden Globes), red-carpet moments, celebrity relationships and public appearances.
  Lead with the single biggest story of the week. No duplicates. NO politics, wars, disasters, sports or business.
  Prefer clearly-sourced facts about named celebrities and named movies/songs/shows.
- Do NOT include any story about Ethiopia.
- About {words} spoken Amharic words in total (each story roughly {words // n_stories} words), 3 to 6 sentences
  per story, each story well under {max_highlight_seconds:.0f} seconds to read aloud.
- Short, clear sentences (6 to 14 words). Engaging, personal presenter style — Sofia is talking WITH the audience,
  not reading AT them. It is fine to add a light reaction like "አስገራሚ ነው" or "እንኳን ደስ አላችሁ" where natural.
- Write numbers, percentages and years as Amharic words the anchor can read naturally. Write celebrity and movie
  names in Amharic script the way Ethiopian media spell them (e.g. ቢዮንሴ, ቴለር ስዊፍት, ብራድ ፒት, ማርቬል),
  but you may keep a very well-known short English title inside quotes if there is no Amharic form.
- Use only facts present in the items. Never invent numbers, quotes, dates or names.
- Credit the source naturally at most once per story, e.g. "ቫራይቲ እንደዘገበው".
- Do NOT greet the viewer or introduce Sofia — the greeting is added separately. Do NOT mention today's date or the
  day of the week. Never say "ዛሬ"; use "በዚህ ሳምንት" or "በቅርቡ" instead.
- Plain text for speech: no markdown, emojis, parentheses, URLs, hashtags or Latin letters in "say".
- For EVERY sentence give a "highlight": the on-screen key point, 2 to 6 Amharic words, factual and punchy.
- "headline": a 1 to 3 word Amharic label for the story tab. "ticker": one Amharic line, at most 10 words.
- "sources": the item numbers you used.
- "title": an Amharic title (max 70 characters) naming the top story, prefixed with "የመዝናኛ ወሬዎች፦".
  "summary": 1 to 2 Amharic sentences.

Return ONLY valid JSON in this exact shape:
{{"title": "...", "summary": "...", "stories": [{{"headline": "...", "ticker": "...", "sources": [1, 4],
  "sentences": [{{"say": "...", "highlight": "..."}}]}}]}}

NEWS ITEMS
""" + "\n".join(lines)


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    a, b = text.find("{"), text.rfind("}")
    return json.loads(text[a:b + 1])


def _validate(data: dict) -> list[dict]:
    stories = []
    for st in data.get("stories", []):
        sents = []
        prev = ""
        for s in st.get("sentences", []):
            say = clean(s.get("say", ""))
            if len(say.split()) < 3:
                continue
            hl = clean(s.get("highlight", "")) or prev or _short(say)
            hl = _short(hl, 14)
            prev = hl
            sents.append(dict(say=say, highlight=hl))
        if len(sents) >= 2:
            stories.append(dict(headline=_short(st.get("headline", "") or sents[0]["highlight"], 6),
                                ticker=_short(st.get("ticker", "") or st.get("headline", ""), 14),
                                sources=[int(x) for x in st.get("sources", []) if str(x).isdigit()],
                                sentences=sents))
    if len(stories) < 3:
        raise ValueError("script too short")
    return stories


def _call_gemini(api_key: str, model: str, prompt: str) -> dict:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=120_000))
    order = [model] + [m for m in MODELS_FALLBACK if m != model]
    err = None
    import time
    for m in order:
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=m, contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.6))
                return _extract_json(resp.text)
            except Exception as exc:
                err = exc
                msg = str(exc)
                # free plan: rate limit / busy -> wait a little before trying again
                if any(code in msg for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")) and attempt < 2:
                    time.sleep(10 * (attempt + 1))
                    continue
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    break                                   # quota for this model is used up -> next model
    raise RuntimeError(f"Gemini request failed: {err}")


def _story_from_item(it: dict, source_no: int, max_words: int = 999) -> dict | None:
    title = clean(it["title"]).rstrip("።.")
    sents = [dict(say=f"{it['source']} እንደዘገበው፣ {title}።", highlight=_short(title, 8))]
    pieces = re.split(r"(?<=[.!?።])\s*", clean(it["summary"]))
    total = len(sents[0]["say"].split())
    for piece in pieces:
        if len(piece.split()) < 5:
            continue
        if total >= max_words:
            break
        sents.append(dict(say=piece, highlight=_short(title, 8)))
        total += len(piece.split())
    if len(sents) < 2:
        return None
    return dict(headline=_short(it["title"], 5), ticker=_short(it["title"], 12),
                sources=[source_no], sentences=sents)


def _word_count(stories: list[dict]) -> int:
    return sum(len(s["say"].split()) for st in stories for s in st.get("sentences", []))


def offline_script(items: list[dict], n_stories: int, max_story_words: int = 999) -> dict:
    """No LLM available: an entertainment show cannot really run without Amharic translation,
    but we still return whatever Ethiopic-titled items exist so the caller can raise a
    helpful error rather than crash."""
    stories = []
    for idx, it in enumerate(items, 1):
        if len(stories) >= n_stories:
            break
        if not ETHIOPIC.search(it["title"]):
            continue
        st = _story_from_item(it, idx, max_words=max_story_words)
        if st:
            stories.append(st)
    return dict(title="", summary="", stories=stories)


def _top_up(stories: list[dict], items: list[dict], target_words: int, n_stories: int,
            max_story_words: int = 999, log=lambda m: None) -> list[dict]:
    """If the script (usually from the LLM) came in short on both story count AND words,
    pull in one or two extra real stories rather than leaving the show short — but never
    past n_stories, so a short-form show can't overshoot its target length."""
    have = _word_count(stories)
    if len(stories) >= n_stories or have >= target_words * 0.85:
        return stories
    used = {s for st in stories for s in st.get("sources", [])}
    log(f"Script ran short ({have}/{target_words} words); adding another story to fill the runtime.")
    for idx, it in enumerate(items, 1):
        if idx in used or not ETHIOPIC.search(it["title"]):
            continue
        if len(stories) >= n_stories:
            break
        st = _story_from_item(it, idx, max_words=max_story_words)
        if st:
            stories.append(st)
    return stories


def write(items: list[dict], cfg: dict, api_key: str, today: datetime, target_seconds: float,
          log=lambda m: None) -> dict:
    studio = cfg["studio_name"]
    topic = cfg.get("topic_label", "WORLD NEWS")
    wps = float(cfg.get("words_per_second", 1.7))          # Amharic words are long: ~100 words/min
    max_highlight_seconds = float(cfg.get("max_highlight_seconds", 18))
    lead_in = float(cfg.get("lead_in_sec", 0.6))
    tail = float(cfg.get("tail_sec", 1.2))
    intro, outro = intro_outro(studio, topic, today)
    intro_outro_words = _word_count([intro, outro])
    intro_outro_seconds = intro_outro_words / wps + 1.0    # +1s for the pauses between those sentences

    available = max(15.0, target_seconds - lead_in - tail - intro_outro_seconds)
    words = int(available * wps)
    per_story_words = max(20, int(max_highlight_seconds * wps * 0.8))
    n_stories = max(3, min(10, round(words / per_story_words)))

    data, mode = None, "gemini"
    if api_key and items:
        try:
            focus = "us" if str(cfg.get("news_focus", "world")).lower() == "us" else "world"
            data = _call_gemini(api_key, cfg["gemini_model"],
                                _prompt([i for i in items if i.get("region") != "am"], today, words, n_stories, studio, topic, max_highlight_seconds,
                                        target_seconds, focus))
            stories = _validate(data)
        except Exception as exc:
            log(f"Gemini script failed ({exc}); using headline reader instead.")
            data = None
    if data is None:
        mode = "offline"
        if not items:
            raise RuntimeError("Could not download any news. Check your internet connection.")
        data = offline_script(items, n_stories, max_story_words=per_story_words)
        try:
            stories = _validate(data)
        except ValueError:
            raise RuntimeError("Gemini could not write this week's Amharic entertainment script (it may be busy, "
                               "or the key / daily quota is the problem). Please press New broadcast again in a few "
                               "minutes.") from None
    stories = _top_up(stories, items, words, n_stories, max_story_words=per_story_words, log=log)
    title = clean(data.get("title", "")) or f"{stories[0]['headline']} | {studio}"
    return dict(title=title[:95], summary=clean(data.get("summary", "")), mode=mode,
                stories=[intro] + stories + [outro])
