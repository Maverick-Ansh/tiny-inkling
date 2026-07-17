"""Tokenize TinyStories and pack the token ids into a flat uint16 memmap.

uint16 because vocab (8192) < 65536, so each token is 2 bytes — half the disk/RAM
of int32 and it memmaps cleanly for the data loader. Documents are separated by
the <|endoftext|> id so the model learns document boundaries.

Output:
    train.bin / val.bin  — raw uint16 token streams
    meta.json            — vocab size, token counts, special ids
"""
import os, sys, json, argparse
import numpy as np
from tokenizers import Tokenizer


def encode_file(tok, path, eot_id, chunk_lines=100_000):
    """Yield arrays of token ids, inserting eot between documents.

    TinyStoriesV2 separates stories with a line that is exactly '<|endoftext|>'.
    We treat that as a document boundary and emit the eot token there.
    """
    buf_lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == "<|endoftext|>":
                if buf_lines:
                    text = "".join(buf_lines)
                    ids = tok.encode(text).ids
                    ids.append(eot_id)
                    yield np.array(ids, dtype=np.uint16)
                    buf_lines = []
            else:
                buf_lines.append(line)
    if buf_lines:
        ids = tok.encode("".join(buf_lines)).ids
        ids.append(eot_id)
        yield np.array(ids, dtype=np.uint16)


def write_bin(tok, in_path, out_path, eot_id):
    total = 0
    with open(out_path, "wb") as f:
        for i, arr in enumerate(encode_file(tok, in_path, eot_id)):
            f.write(arr.tobytes())
            total += len(arr)
            if i % 20000 == 0:
                print(f"  {out_path}: {i} docs, {total/1e6:.1f}M tokens", flush=True)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default="/kaggle/working/tok/tiny8k.json")
    ap.add_argument("--train", default="/kaggle/working/data/TinyStoriesV2-GPT4-train.txt")
    ap.add_argument("--val", default="/kaggle/working/data/TinyStoriesV2-GPT4-valid.txt")
    ap.add_argument("--out", default="/kaggle/working/data")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tok)
    eot_id = tok.token_to_id("<|endoftext|>")
    assert eot_id is not None

    n_train = write_bin(tok, args.train, os.path.join(args.out, "train.bin"), eot_id)
    n_val = write_bin(tok, args.val, os.path.join(args.out, "val.bin"), eot_id)

    meta = dict(vocab_size=tok.get_vocab_size(), eot_id=eot_id,
                pad_id=tok.token_to_id("<|pad|>"),
                n_train_tokens=n_train, n_val_tokens=n_val)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("DONE", meta)


if __name__ == "__main__":
    main()
