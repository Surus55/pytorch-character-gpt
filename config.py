from dataclasses import dataclass
import torch

@dataclass
class Config:
    # nanoGPT-style character-level Shakespeare baseline
    block_size: int = 256
    batch_size: int = 64
    max_steps: int = 5000
    eval_interval: int = 250
    eval_batches: int = 100
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    train_fraction: float = 0.90
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    checkpoint_dir: str = "checkpoints/tiny_shakespeare"
    data_file: str = "learning_dataset.txt"
