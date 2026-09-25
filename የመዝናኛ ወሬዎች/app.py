"""MA Studio - AI News.  Run:  python app.py"""
import argparse
import multiprocessing
import threading
import webbrowser

from studio import config
from studio.server import Studio, serve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--new", action="store_true", help="ignore today's existing video and make a fresh broadcast")
    args = ap.parse_args()

    cfg = config.load()
    port = args.port or int(cfg["port"])
    studio = Studio(cfg)
    srv = serve(studio, port)
    url = f"http://127.0.0.1:{port}/"
    print(f"\n  {cfg['studio_name']} is live at {url}\n  (press Ctrl+C to quit)\n")
    if not config.gemini_key(cfg):
        print("  NOTE: no Gemini API key found - the show will read headlines directly.\n"
              "        Add \"gemini_api_key\" to config.json for a fully written script.\n")
    studio.start(force=args.new)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
