"""Collect the latest world / US headlines from public RSS/Atom feeds (no API key needed)."""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests

FEEDS = [
    # US entertainment newsrooms - celebrities, movies, music. Gemini rewrites into Amharic.
    ("Variety", "https://variety.com/feed/", "entertainment"),
    ("The Hollywood Reporter", "https://www.hollywoodreporter.com/feed/", "entertainment"),
    ("Deadline", "https://deadline.com/feed/", "entertainment"),
    ("Billboard", "https://www.billboard.com/feed/", "entertainment"),
    ("Rolling Stone", "https://www.rollingstone.com/music/music-news/feed/", "entertainment"),
    ("Entertainment Weekly", "https://ew.com/feed/", "entertainment"),
    ("People", "https://people.com/feed/", "entertainment"),
    ("E! Online", "https://www.eonline.com/syndication/feeds/rssfeeds/topstories.xml", "entertainment"),
    ("TMZ", "https://www.tmz.com/rss.xml", "entertainment"),
    ("Vanity Fair Hollywood", "https://www.vanityfair.com/feed/hollywood/rss", "entertainment"),
    ("IndieWire", "https://www.indiewire.com/feed/", "entertainment"),
    ("Pitchfork", "https://pitchfork.com/rss/news/", "entertainment"),
    ("BBC Entertainment", "http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml", "entertainment"),
]
UA = {"User-Agent": "Mozilla/5.0 (compatible; MAStudioNews/1.0)"}


def _strip(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def _date(txt: str):
    if not txt:
        return None
    try:
        d = parsedate_to_datetime(txt)
    except Exception:
        try:
            d = datetime.fromisoformat(txt.replace("Z", "+00:00"))
        except Exception:
            return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _image_from_html(text: str) -> str:
    """Best-effort <img src="…"> extraction from a summary/description blob."""
    if not text:
        return ""
    m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', text, re.I)
    return m.group(1) if m else ""


def _parse(name: str, region: str, xml_bytes: bytes) -> list[dict]:
    out = []
    root = ET.fromstring(xml_bytes)
    for node in root.iter():
        tag = node.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        f = {}
        image = ""
        for ch in node:
            t = ch.tag.split("}")[-1]
            if t == "link":
                f["link"] = ch.attrib.get("href") or (ch.text or "").strip()
            elif t in ("title", "description", "summary", "encoded", "content", "pubDate", "published", "updated", "date"):
                f.setdefault(t, (ch.text or "").strip())
            # RSS/Atom image hints — Variety, THR, EW, Deadline etc. all publish one of these
            elif t in ("thumbnail", "content"):
                url = ch.attrib.get("url") or ""
                medium = (ch.attrib.get("medium") or "").lower()
                mtype = (ch.attrib.get("type") or "").lower()
                if url and (medium == "image" or mtype.startswith("image/") or url.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
                    if not image:
                        image = url
            elif t == "enclosure":
                url = ch.attrib.get("url") or ""
                mtype = (ch.attrib.get("type") or "").lower()
                if url and (mtype.startswith("image/") or url.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
                    if not image:
                        image = url
        title = _strip(f.get("title", ""))
        if not title:
            continue
        raw_html = f.get("description") or f.get("summary") or f.get("encoded") or f.get("content") or ""
        summary = _strip(raw_html)
        if not image:                                     # try nested media groups (Rolling Stone etc.)
            for d in node.iter():
                dt = d.tag.split("}")[-1]
                if dt in ("thumbnail", "content", "image"):
                    url = d.attrib.get("url") or d.attrib.get("href") or ""
                    if url and url.lower().split("?", 1)[0].endswith((".jpg", ".jpeg", ".png", ".webp")):
                        image = url
                        break
        if not image:
            image = _image_from_html(raw_html)
        out.append(dict(source=name, region=region, title=title, summary=summary[:600], link=f.get("link", ""),
                        image=image,
                        published=_date(f.get("pubDate") or f.get("published") or f.get("updated") or f.get("date"))))
    return out


def _fetch(feed):
    name, url, region = feed
    try:
        r = requests.get(url, headers=UA, timeout=12)
        r.raise_for_status()
        return _parse(name, region, r.content)
    except Exception:
        return []


# stories about Ethiopia are left out of the show (English and Amharic spellings)
ETHIOPIA_WORDS = [
    "ethiopia", "ethiopian", "addis ababa", "tigray", "tigrayan", "tplf", "amhara", "oromia", "oromo", "afar region",
    "abiy", "mekelle", "fano", "gerd", "grand ethiopian renaissance",
    "ኢትዮጵያ", "ትግራይ", "አማራ", "ኦሮሚያ", "ኦሮሞ", "አዲስ አበባ", "መቀለ", "ዐቢይ", "አቢይ", "ፋኖ", "ሕወሓት", "ህወሓት",
    "ብልጽግና", "ህዳሴ", "ሕዳሴ", "ፌዴራል መንግሥት", "ፌደራል መንግሥት", "ክልል",
]


def about_ethiopia(item: dict) -> bool:
    text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
    return any(w in text for w in ETHIOPIA_WORDS)


def _tokens(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 3}


def gather(max_age_hours: int = 72, limit: int = 24, feeds=None, us_ratio: float = 0.35,
           priority_region: str = "am") -> list[dict]:
    """Fetch, dedupe and rank headlines. US-region stories are given priority (roughly
    `us_ratio` of the picks) so the show leads with American news, with the rest of the
    world filling out the remainder — never all-US and never all-world."""
    feeds = feeds or FEEDS
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_fetch, feeds))
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max_age_hours)
    items = [i for r in results for i in r if i["published"] is None or i["published"] >= cutoff]
    items.sort(key=lambda i: i["published"] or cutoff, reverse=True)

    picked, per_source = [], {}

    def _try_add(it) -> bool:
        if per_source.get(it["source"], 0) >= 4:
            return False
        tk = _tokens(it["title"])
        if any(len(tk & _tokens(p["title"])) / max(1, len(tk | _tokens(p["title"]))) > 0.5 for p in picked):
            return False
        picked.append(it)
        per_source[it["source"]] = per_source.get(it["source"], 0) + 1
        return True

    us_target = max(1, round(limit * us_ratio))
    for it in items:                                   # pass 1: US stories first, most recent first
        if len(picked) >= us_target:
            break
        if it.get("region") == priority_region:
            _try_add(it)
    for it in items:                                   # pass 2: fill the rest (world + any leftover US)
        if len(picked) >= limit:
            break
        _try_add(it)
    return picked
