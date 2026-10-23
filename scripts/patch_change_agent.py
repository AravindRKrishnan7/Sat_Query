#!/usr/bin/env python3
"""Apply the minimal SatQuery CPU/MPS safety patch to a pinned Change-Agent checkout."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


EXPECTED_REVISION = "68cbaa7f388b36e4fc10872f7a2911482d26ae5b"
ORIGINAL_HASHES = {
    "model/model_encoder_att.py": "33dba6e33a295f84ed4fb5ec07065dfd36a681969a69d6296380d6244758ddd3",
    "model/model_decoder.py": "ae81dcb241e02b1aa9c58308d7d8e01cc84185c3f1b721de1b0c9abd314232a4",
    "model/segformer.py": "73a6c45c1c5d74708d39bc3fec2c5576116ab96a30b2c89ae95d43fa0257bb75",
}
PATCHED_HASHES = {
    "model/model_encoder_att.py": "c590af28f9fa580c82082a6fa493eabd9f3458307a84336f9c5fd35bca3e840b",
    "model/model_decoder.py": "a47f9b5be8f00b18038cffc9efac73477c6a68c76de1ce0448571dc09fa61aa2",
    "model/segformer.py": "719877d00ca236c2e70cf2f27fc27c9671b1a5a63ca6051065e716d1eade6158",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label} occurrence, found {count}")
    return text.replace(old, new, 1)


def remove_once(text: str, old: str, label: str) -> str:
    lines = text.splitlines(keepends=True)
    count = lines.count(old)
    if count == 0:
        return text
    if count != 1:
        raise RuntimeError(f"expected at most one {label} occurrence, found {count}")
    lines.remove(old)
    return "".join(lines)


def patch_encoder(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = remove_once(text, "import torchvision.models as models\n", "eager torchvision import")
    text = replace_once(
        text,
        "        self.network = network\n",
        "        self.network = network\n        if 'segformer' not in self.network:\n            import torchvision.models as models\n",
        "lazy torchvision insertion",
    )
    text = text.replace("torch.arange(h).cuda()", "torch.arange(h, device=x.device)")
    text = text.replace("torch.arange(w).cuda()", "torch.arange(w, device=x.device)")
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def patch_decoder(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        "torch.ones(word_length, word_length)": "torch.ones(word_length, word_length, device=x.device)",
        "        mask = mask.cuda()\n": "",
        "torch.zeros(batch, self.max_lengths).to(torch.int64).cuda()": "torch.zeros(batch, self.max_lengths, dtype=torch.int64, device=x.device)",
        "torch.ones(self.max_lengths, self.max_lengths)": "torch.ones(self.max_lengths, self.max_lengths, device=x.device)",
        "torch.LongTensor([self.word_vocab['<START>']] *batch).cuda()": "torch.tensor([self.word_vocab['<START>']] * batch, dtype=torch.long, device=x.device)",
        "torch.LongTensor([[self.word_vocab['<START>']]] *batch).cuda()": "torch.tensor([[self.word_vocab['<START>']]] * batch, dtype=torch.long, device=x.device)",
        "torch.zeros(k*batch, self.max_lengths).to(torch.int64).cuda()": "torch.zeros(k*batch, self.max_lengths, dtype=torch.int64, device=x.device)",
        "torch.LongTensor([self.word_vocab['<START>']] *batch*k).cuda()": "torch.tensor([self.word_vocab['<START>']] * batch * k, dtype=torch.long, device=x.device)",
        "torch.LongTensor([[self.word_vocab['<START>']]] *batch*k).cuda()": "torch.tensor([[self.word_vocab['<START>']]] * batch * k, dtype=torch.long, device=x.device)",
        "torch.zeros(k*batch, 1).cuda()": "torch.zeros(k*batch, 1, device=x.device)",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    active_cuda = [line for line in text.splitlines() if ".cuda()" in line and not line.lstrip().startswith("#")]
    if active_cuda:
        raise RuntimeError(f"active CUDA calls remain in {path}: {active_cuda}")
    path.write_text(text, encoding="utf-8")


def patch_segformer(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    old_imports = """import os

from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch
from torch import nn
"""
    new_imports = """import os

from functools import partial
import math
import torch
from torch import nn


def to_2tuple(value):
    return value if isinstance(value, tuple) else (value, value)


trunc_normal_ = nn.init.trunc_normal_


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep_prob) * random_tensor.floor()


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
"""
    text = replace_once(text, old_imports, new_imports, "legacy framework imports")
    text = replace_once(
        text,
        "            logger = get_root_logger()\n            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)",
        "            checkpoint = torch.load(pretrained, map_location='cpu', weights_only=True)\n            self.load_state_dict(checkpoint, strict=False)",
        "MMCV checkpoint helper",
    )
    for model in ("b0", "b1", "b2", "b3", "b4", "b5"):
        old = f'torch.load(rf"{{script_dir}}/pretrained/mit_{model}.pth")'
        new = f'torch.load(rf"{{script_dir}}/pretrained/mit_{model}.pth", map_location="cpu", weights_only=True)'
        text = text.replace(old, new)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Change-Agent repository root")
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != EXPECTED_REVISION:
        raise RuntimeError(f"expected revision {EXPECTED_REVISION}, got {revision}")

    root = source / "Multi_change"
    files = {relative: root / relative for relative in ORIGINAL_HASHES}
    for relative, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        current = sha256(path)
        allowed = {ORIGINAL_HASHES[relative], PATCHED_HASHES[relative]}
        if current not in allowed:
            raise RuntimeError(
                f"refusing to patch {relative}: expected an original or already-patched file, got {current}"
            )
        if current == PATCHED_HASHES[relative]:
            print(f"note: {relative} is already patched")

    patch_encoder(files["model/model_encoder_att.py"])
    patch_decoder(files["model/model_decoder.py"])
    patch_segformer(files["model/segformer.py"])
    for relative, path in files.items():
        actual = sha256(path)
        if actual != PATCHED_HASHES[relative]:
            raise RuntimeError(
                f"patched hash mismatch for {relative}: expected {PATCHED_HASHES[relative]}, got {actual}"
            )
    print(f"Patched Change-Agent {revision} for device-neutral restricted inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
