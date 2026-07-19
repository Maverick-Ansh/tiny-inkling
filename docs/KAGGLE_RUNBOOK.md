# Running tiny-inkling on Kaggle (2× T4)

This is the exact orchestration used to train the models end-to-end on Kaggle's
free 2×T4, driven from a notebook. The design goal: **survive session resets**
(9-hour cap, idle timeouts) with zero lost work, by checkpointing to the Hugging
Face Hub and resuming automatically.

## Prerequisites (Kaggle notebook settings)
- Accelerator: **GPU T4 ×2**
- Internet: **On**
- Secrets (Add-ons → Secrets): `HF_TOKEN` (write), `GITHUB_TOKEN` (optional, code only)

## Why fp16 (not bf16)
The T4 is Turing — **no bf16 tensor cores**. We force fp16 autocast + a GradScaler
in pretrain, and do RMSNorm/softmax reductions in fp32 for stability. bf16 on a T4
silently falls back to slow emulation.

## 0. Setup
```python
import os, subprocess
os.chdir('/kaggle/working')
subprocess.run(['git','clone','https://github.com/Maverick-Ansh/tiny-inkling.git'])
os.chdir('/kaggle/working/tiny-inkling')
subprocess.run(['pip','install','-q','peft>=0.11'])          # transformers/tokenizers preinstalled
from kaggle_secrets import UserSecretsClient
os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
```

## 1. Data (once, ~20 min, CPU)
```python
# download TinyStoriesV2 — WAIT for it to finish before training the tokenizer
!python scripts/train_tokenizer.py --sample_lines 1500000     # ~40 s, 8k BPE
!python scripts/prepare_data.py                               # tokenize -> uint16 memmap (~540M tokens)
# the tokenizer IS part of the checkpoint — push it with the weights, always:
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj='/kaggle/working/tok/tiny8k.json',
    path_in_repo='tiny8k.json', repo_id='<user>/tiny-inkling-pretrain')
```
> **Hard-learned:** a checkpoint without its tokenizer can never encode new text
> again. BPE retraining is only reproducible if the corpus file is *complete and
> identical* — training the tokenizer while the download is still running bakes an
> arbitrary file prefix into the merges, unrecoverably (see REPORT §6).

## 2. Track A — pretrain Inkling-mini (both GPUs, DDP fp16)
```python
# time-boxed + checkpointed to HF every 6 min; resumes automatically on restart
!torchrun --nproc_per_node=2 scripts/pretrain.py \
    --micro_bs 12 --grad_accum 3 --seq_len 512 \
    --total_steps 8000 --lr 3e-3 --max_minutes 95 \
    --hf_repo <user>/tiny-inkling-pretrain
```
Watch: `loss` ↓, `load_max_ratio` → ~1.2 (aux-loss-free balancing), `dead_frac` = 0.

**Resume after a reset:** just re-run the same command. `CheckpointManager.load_latest`
pulls `latest.pt` from the Hub and restores model + optimiser + scaler + step.

## 3. Track A — SFT the tool-use format, then quantize
```python
!python scripts/sft.py --hf_repo <user>/tiny-inkling-sft        # gold traces, loss-masked
!python scripts/quantize_bonsai.py                              # 1.58-bit ternary side-experiment
```

## 4. Track B — the agentic RL centerpiece (both GPUs)
```python
# learner (fp32) on cuda:0, async actor (fp16) on cuda:1; GRPO + IcePop
!python scripts/rl_agentic.py --base Qwen/Qwen2.5-0.5B-Instruct \
    --sync_mode async --group_size 8 --groups_per_step 2 \
    --icepop_c 2.0 --sync_every 4 --total_steps 600 \
    --hf_repo <user>/tiny-inkling-rl-qwen
```
Watch: `reward`/`acc`/`tool_use` ↑, `staleness` (async lag), `icepop` (masked frac).

## 5. Evaluate + plot
```python
!python scripts/eval_agentic.py --n 50                                  # BEFORE (base)
!python scripts/eval_agentic.py --n 50 --adapter /kaggle/working/checkpoints/rl_qwen  # AFTER
!python scripts/make_plots.py                                           # figures -> assets/
```

## Keeping the session alive
Active training keeps the GPU busy, so the idle-timeout never fires *during* a run.
The risk is the hard 9-hour session cap — that's what the HF-Hub checkpoint/resume
is for. Each phase is independently resumable; a fresh session re-pulls the latest
checkpoint and continues.

## MCP-driven variant
This project was actually driven from a laptop via an MCP bridge to the Kaggle
kernel (each cell added + executed remotely). The kernel resets on reconnect, so
the setup cell (clone, secrets, env) must be re-run first after any reconnect —
the training checkpoints on the Hub make that cheap.
