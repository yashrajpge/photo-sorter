import os, numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# category -> list of prompts (zero-shot). Edit freely.
CATEGORIES = {
    "Pets & animals": ["a photo of a cat", "a photo of a dog", "a photo of a pet animal", "a photo of a bird", "a photo of wildlife"],
    "People": ["a photo of a person", "a selfie", "a group photo of friends", "a portrait of a person", "a photo of a crowd"],
    "Food & drink": ["a photo of food", "a photo of a meal on a plate", "a photo of a drink", "a photo of pizza"],
    "Nature & landscapes": ["a photo of a mountain landscape", "a photo of a beach", "a photo of a forest", "a photo of the sky and clouds", "a photo of flowers"],
    "Cities & buildings": ["a photo of a city skyline", "a photo of a building", "a photo of a street", "a photo of a monument"],
    "Vehicles": ["a photo of a car", "a photo of a motorcycle", "a photo of a bus or train", "a photo of an airplane"],
    "Documents & screenshots": ["a screenshot of a phone or computer screen", "a photo of a document", "a photo of a receipt", "a photo of text on paper", "a whiteboard or notes"],
    "Other": ["a photo of an object on a table", "a blurry photo", "a photo of a room interior"],
}

class Engine:
    def __init__(self, categories=None):
        self.cats = categories or CATEGORIES
        so = ort.SessionOptions(); so.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 0)
        self.v = ort.InferenceSession(os.path.join(MODELS, "vision.onnx"), so, providers=["CPUExecutionProvider"])
        self.t = ort.InferenceSession(os.path.join(MODELS, "text.onnx"), so, providers=["CPUExecutionProvider"])
        tok = Tokenizer.from_file(os.path.join(MODELS, "tokenizer.json"))
        tok.enable_truncation(77); tok.enable_padding(length=77, pad_id=49407, pad_token="<|endoftext|>")
        self.names, prompts = [], []
        for c, ps in self.cats.items():
            for p in ps: self.names.append(c); prompts.append(p)
        ids = np.array([e.ids for e in tok.encode_batch(prompts)], dtype=np.int64)
        te = np.concatenate([self.t.run(None, {"input_ids": ids[i:i+8]})[0] for i in range(0, len(ids), 8)])
        self.text = te / np.linalg.norm(te, axis=1, keepdims=True)
        self.order = list(self.cats)

    @staticmethod
    def prep(path):
        im = Image.open(path); im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size; s = 224 / min(w, h)
        im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
        w, h = im.size; l, t = (w - 224) // 2, (h - 224) // 2
        a = np.asarray(im.crop((l, t, l + 224, t + 224)), dtype=np.float32) / 255.0
        return ((a - MEAN) / STD).transpose(2, 0, 1)

    def classify(self, path):
        x = self.prep(path)[None]
        e = self.v.run(None, {"pixel_values": x})[0]
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        logits = 100.0 * (e @ self.text.T)[0]
        # softmax over prompts, then sum probability per category
        p = np.exp(logits - logits.max()); p /= p.sum()
        score = {c: 0.0 for c in self.order}
        for n, pi in zip(self.names, p): score[n] += float(pi)
        best = max(score, key=score.get)
        return best, score[best]
