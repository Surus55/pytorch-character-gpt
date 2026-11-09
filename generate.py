import argparse
import torch
from model import GPTLanguageModel

def main():
    parser = argparse.ArgumentParser(description="Tiny Shakespeare karakteralapú szöveggenerátor")
    parser.add_argument("--checkpoint", default="checkpoints/tiny_shakespeare/pretrain_best.pt")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except FileNotFoundError:
        raise SystemExit(f"Nem található a checkpoint: {args.checkpoint}. Előbb futtasd a python pretrain.py parancsot.")

    from types import SimpleNamespace
    cfg = SimpleNamespace(**ckpt["config"])
    model = GPTLanguageModel(ckpt["vocab_size"], cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    stoi = ckpt["stoi"]
    itos = {int(k): v for k, v in ckpt["itos"].items()}

    print(f"Betöltve: {args.checkpoint} | lépés: {ckpt['step']} | val loss: {ckpt['val_loss']:.4f}")
    print("Kilépés: /exit")
    while True:
        prompt = args.prompt if args.prompt is not None else input("\nKezdő szöveg: ")
        args.prompt = None
        if prompt.strip().lower() == "/exit":
            break
        if not prompt:
            prompt = "\n"
        unknown = [ch for ch in prompt if ch not in stoi]
        if unknown:
            print("A promptban nem tanult karakter van:", repr(sorted(set(unknown))))
            continue
        idx = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(idx, args.tokens, args.temperature, args.top_k)
        generated = "".join(itos[int(i)] for i in out[0].tolist())
        print("\nGenerált szöveg:\n")
        print(generated)

if __name__ == "__main__":
    main()
