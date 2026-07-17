"""Shared utilities: reproducibility, cosine LR schedule, a memmap data loader,
and a CheckpointManager that persists to the Hugging Face Hub so training
survives Kaggle session resets.

Why HF Hub for checkpoints (not GitHub)? Model weights are binary and change
every few minutes; the Hub is built for exactly this (versioned, resumable
uploads) whereas GitHub is for the code + small logs. We push a *rolling* latest
checkpoint. On startup we look local-first, then pull the latest from the Hub —
so a fresh kernel resumes from wherever the last one died.
"""
import os, sys, json, time, math, glob
import numpy as np
import torch

# make `import tiny_inkling` work whether run from repo root or scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def set_seed(s=0):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def cosine_lr(step, *, warmup, total, base_lr, min_ratio=0.1):
    """Linear warmup then cosine decay to min_ratio*base_lr.

    Note: the optimiser couples weight decay to lr² internally, so as lr anneals
    the decay pull shrinks quadratically — see muon.py."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return base_lr * min_ratio
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


class BinDataset:
    """Random fixed-length windows from a uint16 token memmap.

    We sample a random start each time (with replacement) rather than iterating in
    order — for a 500M-token stream this is effectively shuffling and keeps the
    loader stateless (nice for resume: no epoch bookkeeping to restore)."""
    def __init__(self, bin_path, seq_len):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n = len(self.data)

    def batch(self, bs, device, generator=None):
        # +1 so we can form (input, label) by shifting
        ix = torch.randint(0, self.n - self.seq_len - 1, (bs,), generator=generator)
        x = torch.stack([torch.from_numpy(self.data[i:i+self.seq_len].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i+1:i+1+self.seq_len].astype(np.int64)) for i in ix])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


class CheckpointManager:
    def __init__(self, local_dir, hf_repo=None, hf_token=None, is_main=True):
        self.local_dir = local_dir
        self.hf_repo = hf_repo
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.is_main = is_main
        os.makedirs(local_dir, exist_ok=True)
        self._api = None
        if hf_repo and self.hf_token and is_main:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.hf_token)
            try:
                self._api.create_repo(hf_repo, repo_type="model", exist_ok=True, private=False)
            except Exception as e:
                print("hf create_repo warn:", e)

    def save(self, state: dict, name="latest.pt", push=True):
        if not self.is_main:
            return
        path = os.path.join(self.local_dir, name)
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)   # atomic: a crash mid-save never corrupts latest.pt
        if push and self._api is not None:
            try:
                self._api.upload_file(path_or_fileobj=path, path_in_repo=name,
                                      repo_id=self.hf_repo, repo_type="model")
            except Exception as e:
                print("hf upload warn:", e)
        return path

    def load_latest(self, name="latest.pt", map_location="cpu"):
        """Local-first, then Hub. Returns state dict or None."""
        path = os.path.join(self.local_dir, name)
        if not os.path.exists(path) and self.hf_repo and self.hf_token:
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(self.hf_repo, name, repo_type="model",
                                       token=self.hf_token, local_dir=self.local_dir)
            except Exception as e:
                print("no hub checkpoint to resume from:", e)
                return None
        if os.path.exists(path):
            print("resuming from", path)
            return torch.load(path, map_location=map_location, weights_only=False)
        return None


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
