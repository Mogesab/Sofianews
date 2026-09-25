# የመዝናኛ ወሬዎች · Weekly US Entertainment News in Amharic, with Sofia

Sofia (ሶፊያ) is an AI-generated young Amharic-speaking female presenter. Every week
she reads a 3–5 minute **US entertainment news roundup** — famous singers, movie
actors and actresses, big movie releases, awards, red carpets — all in Amharic,
in a natural, warm, conversational tone. Vertical 9:16 video, ready for
**YouTube Shorts, TikTok and Instagram Reels**.

She always opens with:

> **"ከሳምንቱ የመዝናኛ ወሬዎች ጋር ሶፊያ ነኝ አብራችሁኝ ቆዩ"**

The date and the day of the week are never mentioned on screen or in the script,
so the same episode can be posted anytime that week.

## How it works
1. **News** — fresh entertainment stories from the past 7 days: Variety, The
   Hollywood Reporter, Deadline, Billboard, Rolling Stone, Entertainment
   Weekly, People, E!, TMZ, Vanity Fair Hollywood, IndieWire, Pitchfork, BBC
   Entertainment.
2. **Script** — Gemini (`gemini_model`, default `gemini-flash-latest`) rewrites
   the week's biggest celebrity / movie / music stories into warm, natural,
   conversational Amharic. She never mentions the date.
3. **Voice** — Gemini text-to-speech (`gemini-3.1-flash-tts-preview`, default
   female voice `Aoede`; try `Kore`, `Leda` or `Zephyr` for a different young
   female sound). Offline fallback: Microsoft's Amharic female voice
   `am-ET-MekdesNeural`.
4. **Sofia on camera** — the same natural-motion animated presenter engine as
   before (head/torso/hands micro-movements, blinks, breath, phoneme-accurate
   lip-sync). New photo: `assets/presenters/sofia.png`.
5. **Behind-shoulder screen** — while Sofia presents each story, the framed
   monitor above her shoulder plays footage related to that story (Pexels /
   Pixabay stock, or your own clips in `assets/broll/`). Turn it off with
   `"show_inset": false`.
6. **Weekly outfit change** — put one Sofia photo per week in
   `assets/presenters/` (e.g. `sofia.png`, `sofia-red-dress.png`,
   `sofia-yellow-blouse.png`) and set `"presenter_rotation": true` +
   `"presenter_rotation_period": "weekly"`. Sofia's outfit will change once a
   week automatically. Use `tools/add_suit.py` to copy Sofia's landmarks onto a
   new photo (same 941×1672 crop as `assets/journalist.png`).

## Run
1. Install Python 3.10+.
2. Put your Gemini key in `config.json` (`"gemini_api_key"`).
3. Windows: double-click `run.bat`. Mac/Linux: `./run.sh`
   (or: `pip install -r requirements.txt` then `python app.py`)

The browser opens at http://127.0.0.1:8767. Press **Start broadcast**. Output:
- `<file_prefix>_<date>_<time>.mp4` — 1080×1920, 30 fps, H.264 + AAC (9:16)
- `..._youtube.txt` — a shared title / description / chapters / hashtags file
  ready to paste into YouTube, TikTok and Instagram.
- `..._script.json` — the Amharic script and timing.

## Key settings (`config.json`)
| key | default | meaning |
|---|---|---|
| studio_name | የመዝናኛ ወሬዎች | shown on screen and used as a filename hint |
| topic_label | የመዝናኛ ወሬዎች | short tag shown in the studio bug + ticker |
| target_minutes | 4 | 3–5 makes a comfortable weekly roundup |
| gemini_voice | Aoede | any Gemini voice: Aoede, Kore, Leda, Zephyr, Charon, Orus, Fenrir |
| backup_voice | am-ET-MekdesNeural | Microsoft Amharic female voice, used if Gemini TTS is unavailable |
| presenter_today | sofia | which photo in `assets/presenters/` to use |
| presenter_rotation | false | true = rotate through every photo in `assets/presenters/` |
| presenter_rotation_period | daily | set `weekly` to change outfit once per ISO week |
| show_inset | true | over-the-shoulder screen with story-matched footage |
| max_age_hours | 168 | 168 = past 7 days (weekly show) |
| date_style | none | `none` = no date on screen (default for Sofia) |
| news_focus | entertainment | picks up the US entertainment feeds in `studio/news.py` |
| port | 8767 | runs alongside the old world-news app on 8766 |

Everything else is inherited from the original studio — see the section below
for advanced tuning (lip-sync, camera moves, over-the-shoulder screen).

## Adding more Sofia outfits (weekly rotation)

1. Generate or edit a new Sofia photo — same face, same framing, **exactly**
   941×1672 pixels, mouth closed, front-facing. Change only the clothes and
   background if you like.
2. Copy Sofia's landmarks onto it:
   ```
   python tools/add_suit.py path/to/new_sofia.png sofia-week-red
   ```
   That adds `assets/presenters/sofia-week-red.png` + `.json`.
3. Set in `config.json`:
   ```json
   "presenter_today": "",
   "presenter_rotation": true,
   "presenter_rotation_period": "weekly"
   ```
Sofia will now wear a different outfit every ISO week.

## Behind-the-scenes: what was reused from the Mr Robot studio
Sofia uses the same rendering pipeline (`studio/pipeline.py`), lip-sync
(`studio/phonemes.py`, `studio/align.py`, `studio/visemes.py`,
`studio/face.py`), motion (`studio/motion.py`) and over-the-shoulder screen
(`studio/inset.py`, `studio/broll.py`). Only the news source
(`studio/news.py`), the scriptwriter's prompt (`studio/scriptwriter.py`),
the presenter photo, the voice and the description hashtags were replaced for
the weekly entertainment format.
