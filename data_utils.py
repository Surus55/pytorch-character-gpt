from pathlib import Path
import torch

def load_text(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Nem találom a(z) {p} fájlt. Másold a learning_dataset.txt fájlt a projekt főmappájába."
        )
    text = p.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise ValueError(f"A(z) {p} fájl üres.")
    return text

def build_char_vocab(text):
    # Stable, sorted character vocabulary; no external tokenizer required.
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    unknown = sorted(set(text) - set(stoi))
    if unknown:
        raise ValueError(f"A prompt olyan karaktert tartalmaz, ami nincs a tanítóadatban: {unknown!r}")
    return [stoi[ch] for ch in text]

def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)

def split_data(text, stoi, train_fraction=0.90, block_size=256):
    ids = torch.tensor(encode(text, stoi), dtype=torch.long)
    split = int(len(ids) * train_fraction)
    # Keep a full context window between train and validation to avoid cross-boundary windows.
    train = ids[:max(0, split - block_size)]
    val = ids[split:]
    if len(train) <= block_size or len(val) <= block_size:
        raise ValueError(
            f"Túl kevés adat a splithez: train={len(train)}, val={len(val)}, block_size={block_size}. "
            "Ellenőrizd, hogy valóban a Tiny Shakespeare teljes szövegét használod-e."
        )
    return train, val

def get_batch(data, batch_size, block_size, device):
    max_start = len(data) - block_size - 1
    ix = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
