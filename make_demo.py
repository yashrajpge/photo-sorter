"""Creates a small demo folder: copies samples and draws a fake receipt/screenshot."""
import sys, os, shutil
from PIL import Image, ImageDraw
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f.lower().endswith((".jpg", ".png")): shutil.copy(os.path.join(src, f), dst)
im = Image.new("RGB", (800, 600), "white"); d = ImageDraw.Draw(im)
for i, t in enumerate(["CAFE RECEIPT", "Latte        120.00", "Sandwich     240.00", "Tax           18.00", "TOTAL        378.00", "Thank you!"]):
    d.text((60, 60 + i * 60), t, fill="black")
im.save(os.path.join(dst, "receipt_drawn.png"))
