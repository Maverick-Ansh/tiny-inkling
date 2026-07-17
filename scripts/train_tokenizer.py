"""Train an 8k byte-level BPE tokenizer on TinyStories.

Small vocab (8192) is plenty for TinyStories' simple English and keeps the
embedding table tiny (8192×384 ≈ 3M params, tied with the LM head). We also add
the agentic special tokens now so tool-call markup becomes single tokens later
(important: we want `<tool>` to be one atomic token the policy can emit/attend to,
not several byte-pieces).
"""
import os, sys, argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# agentic markup — kept as atomic special tokens shared across pretrain/SFT/RL
SPECIAL = [
    "<|endoftext|>", "<|pad|>",
    "<tool>", "</tool>", "<result>", "</result>", "<answer>", "</answer>",
    "<think>", "</think>", "<user>", "</user>", "<assistant>", "</assistant>",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/working/data/TinyStoriesV2-GPT4-train.txt")
    ap.add_argument("--out", default="/kaggle/working/tok/tiny8k.json")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--sample_lines", type=int, default=2_000_000,
                    help="train the tokenizer on a sample for speed")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # full byte coverage => no UNK
        show_progress=True,
    )

    # stream a sample of lines to keep tokenizer training fast/cheap
    def gen():
        with open(args.data, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= args.sample_lines:
                    break
                yield line

    tok.train_from_iterator(gen(), trainer=trainer)
    tok.save(args.out)
    print(f"saved tokenizer -> {args.out}  (vocab={tok.get_vocab_size()})")
    # quick check
    enc = tok.encode("Once upon a time, a <tool>calc(2+2)</tool> gave <result>4</result>.")
    print("sample ids:", enc.ids[:20])
    print("roundtrip :", tok.decode(enc.ids))

if __name__ == "__main__":
    main()
