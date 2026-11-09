Karakteralapú GPT – szöveggenerátor

Kísérleti, nanoGPT-stílusú, karakteralapú nyelvi modell PyTorch használatával. A projekt egy Transformer-alapú autoregresszív modellt tanít a `learning_dataset.txt` állományon, majd a tanult karaktereloszlás alapján új szöveget generál.

Nem előtanított modell. A modell véletlenszerű súlyokkal indul, és kizárólag a megadott szövegkorpuszból tanul. 

Funkciók

- Karakterenkénti tokenizálás, külső tokenizer nélkül.
- GPT-szerű Transformer blokkok:
  -oksági (causal) önfigyelem,
  -több figyelmi fej,
  - MLP és GELU aktiváció,
  - LayerNorm és dropout,
  - megosztott token embedding és kimeneti súlymátrix.
- Train/validation adatok elkülönítése.
- AdamW optimalizáló, learning-rate warmup és cosine decay.
- Gradiens clipping, opcionális CUDA mixed precision és korai leállítás.
- Legjobb és legutóbbi checkpoint mentése.
- Interaktív szöveggenerálás hőmérséklet- és top-k mintavételezéssel.

Követelmények:

- Python 3.10 vagy újabb ajánlott.
- PyTorch 2.2 vagy újabb.
- A tanítás CPU-n is futtatható, de GPU-val lényegesen gyorsabb.
- CUDA-s futtatáshoz megfelelő NVIDIA driver és CUDA-kompatibilis PyTorch telepítés szükséges.

Telepítés:

Virtuális környezet használata ajánlott.

python -m venv .venv

Aktiválás Windows PowerShell alatt:

.venv\Scripts\Activate.ps1

Függőségek telepítése:

python -m pip install --upgrade pip
pip install -r requirements.txt

Tanítás:

python pretrain.py

A program a `learning_dataset.txt` fájlt olvassa be, felépíti a karakter-szótárat, majd a `config.py` beállításai szerint tanítja a modellt.

A tanítás fő paraméterei a `config.py` fájlban módosíthatók, például:

- max_steps: maximális tanítási lépésszám;
- batch_size: batch méret;
- block_size: kontextusablak hossza karakterekben;
- n_layer,n_head,n_embd: a Transformer mérete;
- learning_rate, min_learning_rate: tanulási ráta;
- eval_interval, eval_batches: validáció gyakorisága és mérete.

Szöveggenerálás:

A tanítás után:

python generate.py

Adatkészlet:

A modell külső szöveges adatkészleten tanítható.
A tanítóadat alapértelmezett fájlneve: learning_dataset.txt.
A dataset nem része a GitHub-repositorynak, azt a felhasználónak külön kell biztosítania a projekt futtatásához.
A modell karakteralapú tokenizálást alkalmaz, ezért UTF-8 kódolású szövegfájl használata ajánlott.