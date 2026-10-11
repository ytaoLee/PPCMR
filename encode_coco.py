"""Frozen local CLIP ViT-B/32 inference; no dataset-specific training."""

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from research_common import ARTIFACTS, MODEL, file_sha256, write_json


class Images(Dataset):
    def __init__(self, manifest, preprocess):
        self.manifest, self.preprocess = manifest, preprocess

    def __len__(self):
        return len(self.manifest["images"])

    def __getitem__(self, i):
        item = self.manifest["images"][i]
        path = Path(self.manifest["data_root"]) / item["image_path"]
        if file_sha256(path) != item["image_sha256"]:
            raise ValueError(f"Image bytes changed: {path}")
        with Image.open(path) as image:
            image.load()
            return self.preprocess(image.convert("RGB"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", type=Path, default=ARTIFACTS / "val")
    p.add_argument("--weights", type=Path, default=MODEL)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    manifest = json.loads((args.folder / "manifest.json").read_text(encoding="utf-8"))
    torch.manual_seed(manifest["seed"])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly")
    # The OpenAI checkpoint uses QuickGELU. A local filename does not trigger
    # OpenCLIP's pretrained-tag activation override, so set it explicitly.
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32",
        pretrained=str(args.weights.resolve()), force_quick_gelu=True,
        precision="fp32", device=args.device)
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    loader = DataLoader(Images(manifest, preprocess), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False)
    start = perf_counter()
    image_batches = []
    with torch.inference_mode():
        for batch_id, batch in enumerate(loader):
            encoded = model.encode_image(batch.to(args.device), normalize=True)
            image_batches.append(encoded.float().cpu().numpy())
            if batch_id % 10 == 0:
                print(f"Images {min((batch_id + 1) * args.batch_size, len(loader.dataset))}/{len(loader.dataset)}", flush=True)
        image_seconds = perf_counter() - start
        captions = [c["text"] for x in manifest["images"] for c in x["captions"]]
        start = perf_counter()
        text_batches = []
        for i in range(0, len(captions), args.batch_size * 4):
            tokens = tokenizer(captions[i:i + args.batch_size * 4]).to(args.device)
            encoded = model.encode_text(tokens, normalize=True)
            text_batches.append(encoded.float().cpu().numpy())
            if i % (args.batch_size * 40) == 0:
                print(f"Texts {min(i + args.batch_size * 4, len(captions))}/{len(captions)}", flush=True)
        text_seconds = perf_counter() - start
    images, texts = np.concatenate(image_batches), np.concatenate(text_batches)
    if images.shape[1] != 512 or not np.isfinite(images).all() or not np.isfinite(texts).all():
        raise ValueError("Invalid CLIP features")
    np.savez_compressed(args.folder / "features.npz", images=images, texts=texts)
    provenance = {"model": "OpenAI pretrained CLIP ViT-B/32, frozen, QuickGELU",
        "weights_path": str(args.weights.resolve()), "weights_sha256": file_sha256(args.weights),
        "weights_source": "https://huggingface.co/timm/vit_base_patch32_clip_224.openai",
        "manifest_sha256": file_sha256(args.folder / "manifest.json"),
        "features_sha256": file_sha256(args.folder / "features.npz"),
        "training_on_local_data": False, "precision": "fp32, TF32 disabled", "dimensions": 512,
        "normalization": "l2 per embedding", "preprocess": str(preprocess),
        "tokenizer": "OpenCLIP ViT-B-32 BPE, context length 77 (long captions truncated)",
        "image_shape": list(images.shape), "text_shape": list(texts.shape),
        "image_seconds": image_seconds, "text_seconds": text_seconds,
        "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "python": sys.executable, "python_version": sys.version,
        "versions": {x: importlib.metadata.version(x) for x in ("torch", "torchvision", "open_clip_torch", "numpy", "pillow", "safetensors")}}
    write_json(args.folder / "encoder.json", provenance)
    print(f"Features saved: {images.shape}, {texts.shape}; images {image_seconds:.1f}s, texts {text_seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
