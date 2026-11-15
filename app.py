"""Local photo sorter. Usage: python app.py /path/to/photos  [--port 8765]
Reads photos (never modifies, moves or deletes them), labels them with a local CLIP model,
and serves a UI on http://127.0.0.1:PORT (localhost only). Export is opt-in and copy-only."""
import os, sys, json, io, shutil, threading, argparse, hashlib, time, secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from PIL import Image, ImageOps
from engine import Engine

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")
try:
    import pillow_heif; pillow_heif.register_heif_opener(); EXTS += (".heic", ".heif")
except Exception:
    pass

ap = argparse.ArgumentParser(); ap.add_argument("folder"); ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--no-cache", action="store_true")
args = ap.parse_args()
ROOT = os.path.realpath(os.path.expanduser(args.folder))
if not os.path.isdir(ROOT): sys.exit("not a folder: " + ROOT)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache-" + hashlib.md5(ROOT.encode()).hexdigest()[:10] + ".json")

files = []
for dp, _, fs in os.walk(ROOT):
    for f in sorted(fs):
        fp = os.path.join(dp, f)
        if f.lower().endswith(EXTS) and not f.startswith(".") and not os.path.islink(fp): files.append(fp)
files.sort()
STATE = {"total": len(files), "done": 0, "results": {}, "error": None, "finished": False}
LOCK = threading.Lock()
cache = {}
if not args.no_cache and os.path.exists(CACHE):
    try: cache = json.load(open(CACHE))
    except Exception: cache = {}

def worker():
    try:
        eng = Engine()
        for i, p in enumerate(files):
            key = "%s|%d|%d" % (p, os.path.getsize(p), int(os.path.getmtime(p)))
            if key in cache: res = cache[key]
            else:
                try: c, s = eng.classify(p); res = [c, round(s, 3)]
                except Exception as e: res = ["Unreadable", 0.0]
                cache[key] = res
            with LOCK:
                STATE["results"][i] = res; STATE["done"] += 1
        json.dump(cache, open(CACHE, "w"))
    except Exception as e:
        STATE["error"] = repr(e)
    STATE["finished"] = True

def thumb(i):
    im = Image.open(files[i]); im = ImageOps.exif_transpose(im).convert("RGB"); im.thumbnail((240, 240))
    b = io.BytesIO(); im.save(b, "JPEG", quality=80); return b.getvalue()

def safe_copy(src, dest_dir):
    base = os.path.basename(src); name, ext = os.path.splitext(base); out = os.path.join(dest_dir, base); n = 1
    while os.path.exists(out):  # never overwrite
        out = os.path.join(dest_dir, "%s_%d%s" % (name, n, ext)); n += 1
    shutil.copy2(src, out); return out

TOKEN = secrets.token_urlsafe(24)
ALLOWED_HOSTS = ("127.0.0.1:%d" % args.port, "localhost:%d" % args.port)
PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html"), encoding="utf-8").read().replace("__TOKEN__", TOKEN)

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, body, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.headers.get("Host") not in ALLOWED_HOSTS: return self.send(403, b"bad host", "text/plain")
        u = urlparse(self.path)
        if u.path == "/": return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            with LOCK:
                cats = {}
                for i, (c, s) in STATE["results"].items(): cats.setdefault(c, []).append([i, s, os.path.relpath(files[i], ROOT)])
                d = {"total": STATE["total"], "done": STATE["done"], "finished": STATE["finished"], "error": STATE["error"], "root": ROOT, "cats": cats}
            return self.send(200, json.dumps(d).encode(), "application/json")
        if u.path == "/thumb":
            try: i = int(parse_qs(u.query)["i"][0])
            except Exception: return self.send(400, b"bad index", "text/plain")
            if 0 <= i < len(files): return self.send(200, thumb(i), "image/jpeg")
        self.send(404, b"nope", "text/plain")
    def do_POST(self):
        err = lambda m, c=400: self.send(c, json.dumps({"error": m}).encode(), "application/json")
        if urlparse(self.path).path != "/api/export": return self.send(404, b"nope", "text/plain")
        if self.headers.get("Host") not in ALLOWED_HOSTS: return err("bad host", 403)
        origin = self.headers.get("Origin")
        if origin not in ("http://" + h for h in ALLOWED_HOSTS): return err("bad origin", 403)
        if not secrets.compare_digest(self.headers.get("X-Token", ""), TOKEN): return err("bad token", 403)
        if "application/json" not in self.headers.get("Content-Type", ""): return err("bad content type", 415)
        try: body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 10000)) or b"{}")
        except Exception: return err("bad json")
        raw = (body.get("dest") or "").strip()
        if not raw: return err("dest required")
        dest = os.path.realpath(os.path.expanduser(raw))
        inside = lambda child, parent: child == parent or child.startswith(parent + os.sep)
        if inside(dest, ROOT) or inside(ROOT, dest):
            return err("choose a destination folder that is not the photo folder, inside it, or a parent of it")
        if not os.path.isdir(os.path.dirname(dest)): return err("parent of destination does not exist")
        n = 0
        with LOCK: items = list(STATE["results"].items())
        for i, (c, s) in items:
            d = os.path.join(dest, c.replace("/", "-")); os.makedirs(d, exist_ok=True)
            if inside(os.path.realpath(d), ROOT): return err("destination subfolder resolves into the photo folder (symlink?)")
            safe_copy(files[i], d); n += 1
        self.send(200, json.dumps({"copied": n, "dest": dest}).encode(), "application/json")

threading.Thread(target=worker, daemon=True).start()
srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
print("Sorting %d photos from %s\nOpen http://127.0.0.1:%d  (Ctrl+C to stop)" % (len(files), ROOT, args.port))
try: srv.serve_forever()
except KeyboardInterrupt: pass
