import math, os, time
import torch
from torch.optim import AdamW
from tqdm import trange
from config import Config
from model import GPTLanguageModel
from data_utils import load_text, build_char_vocab, split_data, get_batch

@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, amp_dtype):
    model.eval()
    result = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(cfg.eval_batches)
        for k in range(cfg.eval_batches):
            x, y = get_batch(data, cfg.batch_size, cfg.block_size, cfg.device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=(cfg.device == "cuda" and cfg.use_amp)):
                _, loss = model(x, y)
            losses[k] = loss.item()
        result[name] = losses.mean().item()
    model.train()
    return result

def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    if cfg.device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        print("GPU:", torch.cuda.get_device_name(0))
    print("Eszköz:", cfg.device)

    text = load_text(cfg.data_file)
    stoi, itos = build_char_vocab(text)
    train_data, val_data = split_data(text, stoi, cfg.train_fraction, cfg.block_size)
    vocab_size = len(stoi)
    print(f"Karakterek: {len(text):,} | Egyedi karakterek: {vocab_size}")
    print(f"Train karakterek: {len(train_data):,} | Val karakterek: {len(val_data):,}")
    print(f"Modell: {cfg.n_layer} réteg, {cfg.n_head} fej, {cfg.n_embd} dimenzió, block={cfg.block_size}")

    model = GPTLanguageModel(vocab_size, cfg).to(cfg.device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Paraméterek: {params:,}")

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate,
                      betas=(0.9, 0.99), weight_decay=cfg.weight_decay)
    amp_dtype = torch.bfloat16 if (cfg.device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.device == "cuda" and amp_dtype == torch.float16))
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    best_val = float("inf")
    best_step = 0
    patience, bad_evals = 12, 0
    start = time.time()

    for step in trange(1, cfg.max_steps + 1, desc="Tanítás"):
        if step <= cfg.warmup_steps:
            lr = cfg.learning_rate * step / cfg.warmup_steps
        else:
            progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
            lr = cfg.min_learning_rate + 0.5 * (cfg.learning_rate - cfg.min_learning_rate) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, cfg.batch_size, cfg.block_size, cfg.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                            enabled=(cfg.device == "cuda" and cfg.use_amp)):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % cfg.eval_interval == 0 or step == 1:
            losses = estimate_loss(model, train_data, val_data, cfg, amp_dtype)
            print(f"\n{step:4d} lépés | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")
            checkpoint = {
                "model": model.state_dict(),
                "config": vars(cfg),
                "stoi": stoi,
                "itos": itos,
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "vocab_size": vocab_size,
            }
            torch.save(checkpoint, os.path.join(cfg.checkpoint_dir, "pretrain_latest.pt"))
            if losses["val"] < best_val:
                best_val = losses["val"]
                best_step = step
                bad_evals = 0
                torch.save(checkpoint, os.path.join(cfg.checkpoint_dir, "pretrain_best.pt"))
                print("Új legjobb modell mentve.")
            else:
                bad_evals += 1
                print(f"Nincs javulás: {bad_evals}/{patience}")
            if bad_evals >= patience:
                print("Korai leállítás: a validációs loss már nem javul.")
                break

    print(f"Kész. Idő: {(time.time()-start)/60:.1f} perc.")
    print(f"Legjobb validációs loss: {best_val:.4f} (lépés: {best_step})")
    print("Checkpoint:", os.path.join(cfg.checkpoint_dir, "pretrain_best.pt"))

if __name__ == "__main__":
    main()
