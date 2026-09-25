"""Tiny local web server (standard library only): status, video streaming, download."""
from __future__ import annotations

import json
import mimetypes
import re
import struct
import threading
import time
import traceback
from datetime import datetime, date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import config, pipeline, presenters


IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _png_size(data: bytes) -> tuple[int, int] | None:
    # PNG: 8-byte signature, then IHDR chunk with width/height at bytes 16..24
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    try:
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    except struct.error:
        return None


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\-_.]+", "-", s.strip())
    return s.strip("-.")[:60] or "outfit"


def _outfit_list(cfg: dict) -> list[dict]:
    """List Sofia's outfits (files in assets/presenters/), which is active this week, etc."""
    folder = config.ASSETS / "presenters"
    out = []
    for img, _ in presenters._pairs(folder):
        out.append({"name": img.stem, "file": img.name, "size": img.stat().st_size})
    active_img, _ = presenters.resolve(cfg, config.ASSETS, date.today())
    return [dict(o, active=(o["file"] == active_img.name)) for o in out]

WEB = config.ROOT / "web"
WEIGHTS = {"news": (0.00, 0.03), "script": (0.03, 0.10), "voice": (0.10, 0.28), "render": (0.28, 1.00)}
STAGE_ORDER = ["news", "script", "voice", "render"]


class Studio:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.state = "idle"
        self.stage = ""
        self.progress = 0.0
        self.message = ""
        self.error = ""
        self.started = 0.0
        self.video: Path | None = None
        self.meta: dict = {}

    # ---------------------------------------------------------------- helpers
    def _find_today(self) -> Path | None:
        day = datetime.now().strftime("%Y-%m-%d")
        vids = sorted(config.OUTPUT.glob(f"{config.file_prefix(self.cfg)}_{day}_*.mp4"))
        return vids[-1] if vids else None

    def _load(self, path: Path) -> None:
        self.video = path
        self.meta = {}
        txt = path.with_name(path.stem + "_youtube.txt")
        if txt.exists():
            self.meta["youtube"] = txt.read_text(encoding="utf-8")
        js = path.with_name(path.stem + "_script.json")
        if js.exists():
            try:
                self.meta["duration"] = json.loads(js.read_text(encoding="utf-8"))["timeline"]["duration"]
            except Exception:
                pass
        self.state, self.stage, self.progress = "ready", "done", 1.0

    def start(self, force: bool = False) -> None:
        with self.lock:
            if self.state == "running":
                return
            if not force:
                existing = self._find_today()
                if existing:
                    self._load(existing)
                    return
            self.state, self.stage, self.progress, self.error = "running", "news", 0.0, ""
            self.message, self.started, self.video = "Getting ready…", time.time(), None
        threading.Thread(target=self._work, daemon=True).start()

    def _on_progress(self, stage: str, frac: float, msg: str = "") -> None:
        lo, hi = WEIGHTS[stage]
        self.stage = stage
        self.progress = min(0.995, lo + (hi - lo) * max(0.0, min(1.0, frac)))
        if msg:
            self.message = msg

    def _work(self) -> None:
        try:
            res = pipeline.run(self.cfg, self._on_progress)
            with self.lock:
                self._load(Path(res["video"]))
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                self.state, self.error = "error", str(exc)

    def status(self) -> dict:
        return dict(state=self.state, stage=self.stage, progress=self.progress, message=self.message,
                    error=self.error, elapsed=(time.time() - self.started) if self.started else 0,
                    video=bool(self.video), filename=self.video.name if self.video else "",
                    duration=self.meta.get("duration", 0), youtube=self.meta.get("youtube", ""),
                    stages=STAGE_ORDER, studio=self.cfg["studio_name"], topic=self.cfg.get("topic_label", "WORLD NEWS"))


def make_handler(studio: Studio):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            path = u.path
            if path in ("/", "/index.html"):
                return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/status":
                return self._send(200, json.dumps(studio.status()).encode())
            if path == "/api/outfits":
                return self._send(200, json.dumps(_outfit_list(studio.cfg)).encode())
            if path == "/outfit.png":                      # <img src="/outfit.png?name=sofia">
                q = parse_qs(u.query)
                name = _slug((q.get("name") or [""])[0])
                for ext in IMG_EXTS:
                    p = config.ASSETS / "presenters" / f"{name}{ext}"
                    if p.exists():
                        return self._send(200, p.read_bytes(),
                                          mimetypes.guess_type(str(p))[0] or "image/png")
                return self._send(404, b'{"error":"not found"}')
            if path == "/still.png":
                p, _ = presenters.resolve(studio.cfg, config.ASSETS)
                return self._send(200, p.read_bytes(), mimetypes.guess_type(str(p))[0] or "image/png")
            if path in ("/video", "/download") and studio.video and studio.video.exists():
                return self._file(studio.video, attachment=(path == "/download"))
            self._send(404, b'{"error":"not found"}')

        do_HEAD = do_GET

        def do_POST(self):
            u = urlparse(self.path)
            path = u.path
            if path == "/api/generate":
                studio.start(force=True)
                return self._send(200, b'{"ok":true}')
            if path == "/api/outfit":                          # upload a new Sofia photo
                return self._upload_outfit(u.query)
            if path == "/api/outfit/select":                   # make one active
                q = parse_qs(u.query)
                name = _slug((q.get("name") or [""])[0])
                return self._select_outfit(name)
            if path == "/api/outfit/delete":                   # remove one
                q = parse_qs(u.query)
                name = _slug((q.get("name") or [""])[0])
                return self._delete_outfit(name)
            self._send(404, b'{"error":"not found"}')

        def _upload_outfit(self, query: str):
            q = parse_qs(query)
            name = _slug((q.get("name") or ["outfit"])[0])
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 20_000_000:
                return self._send(400, b'{"error":"missing or oversized body (20 MB max)"}')
            data = self.rfile.read(length)
            # Sofia's landmarks were measured on a 941x1672 PNG. We accept exactly that
            # size in PNG so the mouth / eye positions line up on the new photo.
            size = _png_size(data)
            if not size:
                return self._send(400,
                                  b'{"error":"the file must be a PNG (change format in your image editor)"}')
            if size != (941, 1672):
                return self._send(400,
                                  json.dumps({"error": f"the file must be 941x1672 pixels (got {size[0]}x{size[1]})"}).encode())
            folder = config.ASSETS / "presenters"
            folder.mkdir(parents=True, exist_ok=True)
            # keep the name unique so successive uploads accumulate as separate outfits
            base = name
            i = 1
            while (folder / f"{name}.png").exists():
                i += 1
                name = f"{base}-{i}"
            img_path = folder / f"{name}.png"
            img_path.write_bytes(data)
            # reuse the Sofia landmark JSON (all Sofia photos share the 941x1672 crop)
            src_json = folder / "sofia.json"
            if not src_json.exists():
                src_json = config.ASSETS / "face.json"
            landmarks = json.loads(src_json.read_text(encoding="utf-8"))
            landmarks["image"] = f"{name}.png"
            landmarks["_comment"] = (
                "Landmarks copied from sofia.json — all Sofia photos share the same "
                "941x1672 crop. Re-measure only if you change the framing.")
            (folder / f"{name}.json").write_text(
                json.dumps(landmarks, indent=1, ensure_ascii=False), encoding="utf-8")
            return self._send(200, json.dumps({"ok": True, "name": name}).encode())

        def _select_outfit(self, name: str):
            folder = config.ASSETS / "presenters"
            if not any((folder / f"{name}{e}").exists() for e in IMG_EXTS):
                return self._send(404, b'{"error":"no such outfit"}')
            cfg_path = config.ROOT / "config.json"
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            data["presenter_today"] = name
            data["presenter_rotation"] = False
            cfg_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            studio.cfg["presenter_today"] = name
            studio.cfg["presenter_rotation"] = False
            return self._send(200, json.dumps({"ok": True, "name": name}).encode())

        def _delete_outfit(self, name: str):
            folder = config.ASSETS / "presenters"
            removed = 0
            for e in IMG_EXTS + (".json",):
                p = folder / f"{name}{e}"
                if p.exists():
                    p.unlink()
                    removed += 1
            if removed == 0:
                return self._send(404, b'{"error":"no such outfit"}')
            return self._send(200, b'{"ok":true}')

        def _file(self, p: Path, attachment: bool):
            size = p.stat().st_size
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            code = 200
            if rng and rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                try:
                    if a:
                        start = int(a)
                        end = int(b) if b else size - 1
                    else:
                        start = max(0, size - int(b))
                    end = min(end, size - 1)
                    code = 206
                except ValueError:
                    start, end, code = 0, size - 1, 200
            length = end - start + 1
            self.send_response(code)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{p.name}"')
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                with open(p, "rb") as f:
                    f.seek(start)
                    left = length
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    return H


def serve(studio: Studio, port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(studio))
    srv.daemon_threads = True
    return srv
