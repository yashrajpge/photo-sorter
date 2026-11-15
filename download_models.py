"""One-time download of CLIP ViT-B/32 (OpenAI, MIT license) as quantized ONNX, ~155 MB,
from Hugging Face repo Xenova/clip-vit-base-patch32 at a pinned revision, with SHA-256 checks.
This is the only network access the app's own code makes."""
import os, sys, hashlib, urllib.request
REV = "6b7c523703ae26e39be4b03774966f8571c7f523"
BASE = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/%s/" % REV
FILES = {  # remote -> (local, sha256)
    "onnx/vision_model_quantized.onnx": ("vision.onnx", "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299"),
    "onnx/text_model_quantized.onnx": ("text.onnx", "73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a"),
    "tokenizer.json": ("tokenizer.json", "f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd"),
}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(dest, exist_ok=True)
for remote, (local, want) in FILES.items():
    path = os.path.join(dest, local)
    if os.path.exists(path) and sha(path) == want:
        print("ok (already have)", local); continue
    print("downloading", local)
    urllib.request.urlretrieve(BASE + remote, path + ".part")
    if sha(path + ".part") != want:
        os.remove(path + ".part"); sys.exit("CHECKSUM MISMATCH for " + local + " - file discarded")
    os.replace(path + ".part", path)
print("done ->", dest)
