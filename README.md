# Photo Sorter

A local web app that sorts a folder of photos into categories (Pets & animals,
People, Food & drink, Nature & landscapes, Cities & buildings, Vehicles,
Documents & screenshots, Other) using zero-shot CLIP labelling — ONNX, CPU only.

Everything runs on your own machine after a one-time ~155 MB model download.
Your photos never leave your computer. Originals are only read — never moved,
renamed or deleted. Export is opt-in and copy-only.

## How it works

- `engine.py` embeds each photo and each category's text prompts with CLIP
  ViT-B/32 (ONNX) and assigns the best-matching category.
- `app.py` serves a local UI on `http://127.0.0.1:8765` (localhost only) with a
  live sorted counter, progress bar, category tabs and thumbnails. Results are
  cached so re-runs are instant.
- Labels are zero-shot guesses from a general model. They can be wrong — treat
  the sorting as a convenience, not a judgment. People and faces are not
  identified. Categories are plain text in `engine.py` and easy to edit.

## Setup (Mac / Linux; Windows noted below)

Python 3 required. On a Mac, if `python3` prompts to install developer tools,
accept — or install Python from python.org.

```bash
cd photo-sorter
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_models.py        # one-time ~155 MB, checksum-verified
python app.py "/path/to/your/photo folder"
```

Open http://127.0.0.1:8765 in your browser. Stop with Ctrl+C. Start with a
small folder to see how the labels look on your photos.

- Windows: use `.venv\Scripts\activate` instead of `source .venv/bin/activate`.
- iPhone .HEIC photos: also `pip install pillow-heif` (untested).

## Export

Type a destination folder (outside your photo folder) and click **Copy into
folders**. Files are copied into per-category subfolders; nothing is ever
overwritten and originals are untouched.

## Tested status

Tested on Linux with a small set of sample images (8 in the original test run).
Mac and HEIC support are untested. No accuracy figures are claimed — CLIP
match scores are shown per photo as a hint, not a measure of correctness.

## Demo data

`make_demo.py` builds a small demo folder from any source images you point it
at, plus a drawn receipt image, so you can try the app without touching real
photos.

## Licenses

- This project: MIT — see [LICENSE](LICENSE). Author: Yashraj Pangave.
- The CLIP model (ViT-B/32, OpenAI; ONNX conversion by Xenova on Hugging Face)
  is MIT-licensed and downloaded at setup time, not bundled — see
  [MODEL_LICENSE.txt](MODEL_LICENSE.txt).
