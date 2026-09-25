# የእለቱ ዜና በሚስተር ሮቦት · Amharic news with Mister Robot

Open the app -> the robot news anchor reads today's major news **in Amharic** as a
3-minute vertical YouTube video (1080×1920): a countdown clock, big Amharic
lower-thirds that change with every key point, an Amharic ticker and the
Ethiopian-calendar date.

How it works:
1. **News** - BBC Amharic and VOA Amharic plus the big world newsrooms (BBC, Al Jazeera, Guardian, NYT, NPR, DW, Sky).
2. **Script** - Gemini (`gemini_model`, default `gemini-flash-latest`) writes the whole broadcast in Amharic.
3. **Voice** - Gemini text-to-speech (`gemini-3.1-flash-tts-preview`, voice `Charon`) reads it in Amharic.
   To stay inside the free plan's small TTS limits, the script is recorded in a few long parts
   (about 2 per minute of video) and cut back into sentences at the natural pauses.
   If Gemini's voice is unavailable (quota used up / no internet), the whole show is read by the free
   Microsoft Amharic voice `am-ET-AmehaNeural` instead, so the voice never changes mid-show.
4. **Video** - the same animated robot anchor as the English app (mechanical LED mouth driven by the
   voice, servo-style head/body/hand movement, rotating globe, daily suit & tie).

Without any Gemini key the show still runs: it reads the BBC/VOA Amharic headlines directly
with the backup voice.

Amharic settings in `config.json`: `studio_name`, `topic_label`, `labels` (on-screen words),
`gemini_voice` (e.g. Charon, Orus, Fenrir), `tts_models`, `backup_voice`, `tts_engine`
(`gemini` or `edge`), `words_per_second` (script length for Amharic), `date_style` (`ethiopian`),
`port` (8766, so it can run next to the English app).

## Run
1. Install Python 3.10+.
2. Put your Gemini key in `config.json` (`"gemini_api_key"`).
   (If this folder sits next to your `Mark-LIV-main` folder, the key from its `config/api_keys.json` is picked up automatically.)
3. Windows: double-click `run.bat`.   Mac/Linux: `./run.sh`
   (or: `pip install -r requirements.txt` then `python app.py`)

The browser opens at http://127.0.0.1:8766. Preparing the broadcast takes a
minute or two (headlines -> script -> voice -> HD render). Then press
**Start broadcast**. Files are saved in `output/`:
- `<Studio-Name>_<date>_<time>.mp4`  1080x1920, 30 fps, H.264 + AAC
- `..._youtube.txt`  title, description, chapters and source links (includes #Shorts)
- `..._script.json`  the script and timing

Opening the app again on the same day shows today's video straight away; use **New broadcast** for a fresh one.
`python app.py --new` forces a new one at start-up.

## The robot anchor
`config.json` has `"presenter_today": "robot"`, which uses `assets/presenters/robot.*`
(made from `assets/Robot.png` by `python tools/build_robot.py`; run it again if you replace the photo).

- **Voice** – a natural human male neural voice (`en-US-AndrewNeural`).
- **Mouth** – a mechanical jaw: the chin plate drops between the cheek seams and a glowing LED
  voice grille shows inside. It is driven by the same word → phoneme → forced-alignment lip-sync
  as before, so it opens on every vowel and shuts on m / b / p at exactly the right moment, but it
  moves in small mechanical steps like a machine.
- **Movement** – servo-style, like a robot: holds still, then snaps to a new pose with a small
  overshoot. The head re-poses at each sentence, squares up to camera for a new story and gives a
  short nod on stressed words. The torso leans into each new story. The clasped hands lift and tap
  on emphasis beats, tilting side to side, and their reflection in the desk follows them.
- **Eyes & LEDs** – eyes blink by dimming their lights, the pupils step around, and the glow rises
  while he speaks. Headphone LEDs pulse with the voice.
- **Studio** – the globe behind him rotates, and the "GLOBAL NEWS LIVE" lettering glows with a
  light sweep across it. The camera slowly pushes in and cuts to a closer shot between stories
  (`"camera_moves": false` to keep one fixed shot).
- **Daily outfit** – every day a new suit colour + cloth (solid, pinstripe, chalk stripe,
  windowpane, glen check, birdseye) and a new tie colour + pattern (solid, stripes, dots, check).
  `"outfit_today": "7"` forces one; `"outfit_rotation": false` keeps the original navy suit.

## Settings (`config.json`)
| key | default | meaning |
|---|---|---|
| studio_name | World News with AI | name spoken and shown on screen |
| topic_label | WORLD NEWS | short tag shown in the studio bug, ticker and on-screen text |
| target_minutes | 1 | length of the show — 1 = a 60-second Short; set higher (e.g. 5) for a long-form broadcast |
| width / height | 1080 / 1920 | vertical 9:16, ready for YouTube Shorts. Set 1920x1080 for a landscape show instead |
| voice | en-US-AndrewNeural | any Edge neural voice, e.g. en-US-ChristopherNeural, en-US-BrianNeural |
| news_focus | world | `world` = major world headlines, `us` = mostly US stories |
| outfit_rotation / outfit_today | true / "" | robot anchor: daily suit & tie change / force outfit number |
| camera_moves | true | robot anchor: push-in and closer shot between stories |
| show_countdown | true | 60→0 countdown clock at the top of the frame, timed to countdown_seconds |
| countdown_seconds | 0 | 0 = match target_minutes×60 automatically; set a number to override |
| max_highlight_seconds | 18 | guidance the scriptwriter uses to keep each on-screen headline brisk |
| lead_in_sec / tail_sec | 0.6 / 1.2 | silent studio shot before the anchor starts / after the last word |
| presenter_today | "" | set to a name in `assets/presenters/` (no extension) to force that exact photo today, instead of the daily rotation |
| presenter_rotation | true | if presenter_today is empty, rotate daily through assets/presenters/ |
| show_ai_label | true | small "AI-generated presenter" tag on screen |
| lipsync | auto | `auto` = words → phonemes → mouth shapes (default). `audio` = old loudness-only mouth |
| lipsync_intensity | 1.0 | how strongly the mouth articulates: 0.7 subtle, 1.0 natural, 1.3 very expressive |
| lipsync_lead_ms | 25 | the mouth leads the sound by this many ms (real speakers do; 0-50 is sensible) |
| render_workers | 0 | 0 = automatic (CPU cores - 1) |
| news_feeds | built-in world/US list | optional: list of `["Name", "https://.../feed"]` |

## Test without internet
`python tools/selftest.py` renders a short clip with a synthetic voice to check that video/graphics/encoding work.

## How the lip-sync works
The mouth is no longer driven by loudness. For every sentence the app knows the exact words, so:
1. `studio/phonemes.py` turns the text into phonemes (numbers, "$5 billion", acronyms like "GPT" are expanded the way the voice says them).
   Install `cmudict` (in requirements.txt) for exact dictionary pronunciations; without it built-in rules are used.
2. `studio/align.py` pins every phoneme to the real audio (forced alignment, no ML model or download needed).
3. `studio/visemes.py` converts the timed phonemes into jaw / lip width / rounding / lip-closure / teeth / tongue curves with
   coarticulation, so lips shut on m-b-p, tuck under the teeth on f-v, round on oo-w-sh and spread on ee.
4. `studio/face.py` draws those shapes on the presenter photo.

If a sentence cannot be aligned, that sentence automatically falls back to the old audio-only mouth.

Check it quickly with your real voice (needs internet, ~30 s):
`python tools/lipsync_preview.py "Nvidia spent five billion dollars on new chips."` → `output/lipsync_preview.mp4`
(add `--audio` to compare with the old mouth). `python tools/viseme_sheet.py` writes a contact sheet of every mouth shape.

## Changing the presenter photo
Replace `assets/journalist.png` (native crop, front-facing, mouth closed) and re-measure the pixel positions
of mouth corners, eyes, brows and head in `assets/face.json`.

## Over-the-shoulder screen

A framed monitor above the presenter's shoulder plays footage that matches the
story he is reading, so there is something to look at besides the anchor. It
fades in when a story starts, carries the story label along the bottom, and
dips out between stories.

Footage is chosen in this order:

1. **Your own clips** - drop them in `assets/broll/` and name them after what
   they show (`nvidia-chip-factory.mp4`). The filename is matched against the
   story keywords.
2. **Pexels Videos** - free and licensed for commercial use. Get a key at
   <https://www.pexels.com/api/> and put it in `config.json` as
   `"pexels_api_key"`.
3. **Pixabay Videos** - same idea, `"pixabay_api_key"`.

If none of those is configured the screen still appears, showing generated
motion graphics (a drifting node network in the show's colours). That is the
default, so the screen works out of the box. Set `"inset_fallback": "none"` if
you would rather hide it when there is no real footage.

To check what your setup will actually do, without rendering an episode:

```
python tools/check_broll.py
```

It reports which source will be used, tests your API key if you set one, and
writes `output/broll_preview.png` showing the screen in the frame.

Set `"show_inset": false` in `config.json` to turn it off.

**Do not use clips taken from YouTube, TV news or agency feeds.** That footage
is someone else's copyright and it is the quickest way to get strikes on your
channel. The stock sources above exist precisely for this kind of reuse.

## Changing today's photo (your workflow: new suit/tie every day)

The simplest way: generate or edit today's photo, run `add_presenter.py`
(step 2 below) to save it into `assets/presenters/` under a name like
`2026-09-22`, then set `"presenter_today": "2026-09-22"` in `config.json`.
That one photo is used for every broadcast until you change the setting
again tomorrow — no rotation guesswork, no waiting for the date to cycle
around to the right image.

If you'd rather not touch the setting every day, leave `presenter_today`
blank and drop photos into `assets/presenters/` — see "Different clothes
each day" below for automatic rotation instead.

## Different clothes each day (automatic rotation)

By default the presenter wears the same suit in every episode, since it's one
fixed photo. You can put more than one photo in `assets/presenters/` and it
will rotate through them automatically, one per day (as long as
`presenter_today` is left blank in `config.json`).

Two ways to add a photo:

1. **Same person, different suit colour (recommended, always safe)** - edit
   `assets/journalist.png` in any photo editor or AI image-edit tool so only
   the suit/tie colour changes, keep the face and framing pixel-for-pixel
   identical, then run:
   ```
   python tools/add_suit.py path/to/new_photo.png tuesday-navy-suit --today
   ```
   This copies the photo and the existing landmarks into
   `assets/presenters/` for you (no manual copying or JSON editing) and, with
   `--today`, points `presenter_today` at it so the very next broadcast uses
   it. Drop the `--today` flag to just add it to the daily rotation instead.
   It checks the new photo is the same size as `assets/journalist.png` first
   and refuses if it isn't, since a size mismatch means the crop changed and
   the copied landmarks would be wrong - use `add_presenter.py` (below) for
   that case instead.

2. **A genuinely different photo** - run:
   ```
   python tools/add_presenter.py path/to/photo.jpg some-name
   ```
   This detects the eyes in the new photo and re-derives every landmark
   (mouth, eyes, head, hands, torso) from the reference ones. It writes a
   `..._check.png` with the detected points drawn on top - **look at that
   before trusting it**. It works best when the new photo is front-facing
   with shoulders up. When the framing or aspect ratio differs a lot from
   the reference (e.g. a portrait crop for this vertical Short vs. the
   original landscape one), the scene-level boxes (torso/hands/head-ROI/any
   background screen) can come out stretched even though the face itself
   lines up — check the whole image, not just the face, and nudge those
   boxes by hand in the `.json` if needed.

See `assets/presenters/README.txt` for the full walkthrough. With nothing in
that folder, everything works exactly as before (one fixed photo). Set
`"presenter_rotation": false` in `config.json` to turn rotation off even if
the folder has photos in it (this has no effect when `presenter_today` is set).
