# Review guide — Step 2: data loader v1

Purpose: a checklist for my line-by-line review of the AI-written data pipeline.
For each function: what it does, what to check, and one way to try to break it.
Tick the box only when I can explain the function in my own words.

Rule for the whole review: open a Python shell from the repo root
(`python` then `from src.dataset import *`) and actually run the pieces — don't just read.

---

## File 1: `src/dataset.py`  (8 functions + 1 class)

### 1.1 Module header / paths block  (`REPO_ROOT`, `RAW_DIR`, `ARCHIVE`, `BATCHES_DIR`, `SPLIT_DIR`, `CIFAR10_MEAN/STD`)
- [ ] What it does: defines where everything lives, relative to the repo root, plus the normalisation constants.
- [ ] Check: `Path(__file__).resolve().parents[1]` — why `[1]` and not `[0]`? Print `REPO_ROOT` and confirm.
- [ ] Check: why are `CIFAR10_MEAN/STD` three numbers each? (one per colour channel). Where did the values come from?
- [ ] Break it: run the script from a different folder (`cd /tmp && python /mnt/c/.../scripts/verify_data.py`). Does it still find the data?

### 1.2 `extract_if_needed()`
- [ ] What it does: unpacks the tar.gz into `data/raw/cifar-10-batches-py/` only if that folder is missing.
- [ ] Check: what happens on the second run? (nothing — folder exists). Is this safe if the archive was replaced?
- [ ] Check: what error do you get if the archive is missing, and is the message useful?
- [ ] Break it: rename `data/raw/cifar-10-batches-py` to something else and run again. Then rename the tar.gz and run again.

### 1.3 `_read_batch(path)`
- [ ] What it does: opens one pickle file, returns (images uint8 [N,32,32,3], labels int64 [N]).
- [ ] Check: `encoding="latin1"` — why is it needed? (the pickles were written with Python 2)
- [ ] Check: `.reshape(-1, 3, 32, 32)` — what does `-1` mean? Why 3 first, then 32, 32?
- [ ] Check: `.transpose(0, 2, 3, 1)` — which axis goes where? Draw it: [N, C, H, W] -> [N, H, W, C].
- [ ] Break it: remove the `.transpose(...)` and look at the sample grid — what do the images look like and why?
- [ ] Break it: call it on `batches.meta` (not an image batch). What error?

### 1.4 `load_cifar10()`
- [ ] What it does: reads 5 train batches + 1 test batch + class names; returns one dict.
- [ ] Check: `zip(*(...))` — explain the unzip trick that turns a list of (x, y) pairs into xs and ys.
- [ ] Check: `np.concatenate` — along which axis? Resulting shape?
- [ ] Check: dtype of `x_train` (uint8) — why not float yet? (memory: 180 MB vs 720 MB)
- [ ] Break it: print `x_train.shape, y_train.shape, x_train.dtype, x_train.nbytes / 1e6`.

### 1.5 `make_stratified_split(labels, val_per_class=500, seed=42)`  ← the most important one
- [ ] What it does: per class, shuffle that class's indices with a seeded RNG, first 500 -> val, rest -> train.
- [ ] Check: why loop per class instead of shuffling all 50k at once? (stratified = exactly 500 per class)
- [ ] Check: `np.flatnonzero(labels == c)` — what does it return?
- [ ] Check: `np.random.default_rng(seed)` vs `np.random.seed()` — why the newer API?
- [ ] Check: why `np.sort` at the end? (nicer CSV, deterministic order; shuffling happens later in the DataLoader anyway)
- [ ] Alternative discussed: `sklearn.model_selection.train_test_split(stratify=labels)`. Pros/cons?
- [ ] Break it: call it twice with the same seed -> identical? With seed 43 -> different? With `val_per_class=6000` -> what happens (more than exist)?

### 1.6 `_write_split_csv(path, indices, labels, class_names)` and `_read_split_csv(path)`
- [ ] What it does: save/load the chosen indices as CSV (index, label, class_name).
- [ ] Check: why store `label` and `class_name` in the CSV if `index` alone is enough? (human-readable, self-checking)
- [ ] Check: `newline=""` in `open(...)` — why does the csv module want this?
- [ ] Check: `_read_split_csv` only reads `index` — so the label column is never trusted from disk. Is that the right choice?
- [ ] Break it: edit one row of `val_idx.csv` so an index appears in both files, run `verify_data.py` -> which assert fires?

### 1.7 `get_or_create_split(y_train, class_names, val_per_class, seed, force=False)`
- [ ] What it does: if the two CSVs exist, load them; otherwise create + save them.
- [ ] Check: what does `force=True` do and when would you use it?
- [ ] Check: danger — if the CSVs exist but were made with a different seed, the code silently uses them. Is that acceptable? Should the seed be stored in the CSV?
- [ ] Break it: delete `data/splits/`, rerun, `md5sum data/splits/*.csv` -> same hashes as before? (cc0bef88… / a57d306e…)

### 1.8 `class CIFAR10Arrays(Dataset)`  — `__init__`, `__len__`, `__getitem__`
- [ ] What it does: wraps numpy arrays so PyTorch can index them like a list of (image, label).
- [ ] Check: the `assert len(images) == len(labels)` in `__init__` — what bug does this catch?
- [ ] Check: `torch.from_numpy(...).permute(2, 0, 1)` — HWC -> CHW. Why does PyTorch want channels first?
- [ ] Check: `.float() / 255.0` only runs when there is no transform. What happens when we later pass a torchvision transform — who does the scaling then?
- [ ] Check: `int(self.labels[i])` — why cast to Python int?
- [ ] Compare: how is this different from `ImageFolder` in the GTSRB assignment? What does `ImageFolder` do that we don't need?
- [ ] Break it: `ds[-1]`, `ds[45000]` — what happens? Pass `labels[:100]` with 45000 images -> which line fails?

### 1.9 `build_datasets(train_transform, eval_transform, val_per_class, seed)`
- [ ] What it does: load -> split -> three `CIFAR10Arrays` (train/val/test) + metadata in one dict.
- [ ] Check: train gets `train_transform`, val AND test get `eval_transform` — why must val/test never get augmentation?
- [ ] Check: `data["x_train"][train_idx]` — fancy indexing makes a copy. Memory cost? Is it a problem?
- [ ] Check: test set is loaded here but we promised not to touch it until the end — where is that promise actually enforced? (it isn't yet — it's discipline, not code)

### 1.10 `build_dataloaders(batch_size=128, num_workers=0, **kwargs)`
- [ ] What it does: wraps the three datasets in DataLoaders; returns datasets + loaders in one dict.
- [ ] Check: `shuffle=True` for train only — why?
- [ ] Check: `num_workers=0` — the reasoning (data already in RAM; WSL multiprocessing). What would change on a GPU box reading PNGs?
- [ ] Check: the bug the AI hit — keys `train`/`train_loader`. Explain why `{**ds, **loaders}` silently overwrote before the rename.
- [ ] Check: `batch_size=128` — what's the last batch size for 45,000 / 128? Does it matter (`drop_last`)?
- [ ] Break it: `batch_size=45000` on CPU — what happens to RAM?

---

## File 2: `scripts/verify_data.py`  (5 functions)

### 2.1 Header: `sys.path.insert(0, REPO_ROOT)` and the imports
- [ ] Check: why is the `sys.path` line needed, and why must it come before `from src.dataset import ...`? (`# noqa: E402` — what is that?)
- [ ] Break it: comment the `sys.path` line out and run -> `ModuleNotFoundError`?

### 2.2 `check_counts(d)`
- [ ] What it does: prints per-split, per-class counts; asserts 45k/5k/10k and val == 500 per class.
- [ ] Check: `Counter(...)` — what does it return? Why `.tolist()` first?
- [ ] Check: are the asserts too strict? What if we later change `val_per_class`?

### 2.3 `check_no_overlap(d)`
- [ ] What it does: set intersection must be empty; set union must be all 50k.
- [ ] Check: `&` and `|` on Python sets — what do they mean?
- [ ] Check: why both tests? What bug does each one catch that the other misses?

### 2.4 `check_batch_shapes(d)`
- [ ] What it does: pulls one batch per loader; asserts shape [B,3,32,32], float, range [0,1].
- [ ] Check: `next(iter(loader))` — why not `loader[0]`?
- [ ] Check: the `[0,1]` assert will FAIL once we add normalisation (values go negative). Is that a bug in the test or a reminder to update it?

### 2.5 `show_random_samples(d, n_show=8, seed=None, show=False)`
- [ ] What it does: your GTSRB visualisation cell adapted: 8 random train images, title = `name (index)`, saved to `data/output/`.
- [ ] Check: `rng.choice(len(ds), n_show, replace=False)` — what does `replace=False` prevent?
- [ ] Check: `img.permute(1, 2, 0)` — CHW -> HWC for `imshow`. (Note: in `__getitem__` we went the other way. Why both?)
- [ ] Check: `matplotlib.use("Agg")` only when `--show` is not given — why, and why did `--show` still warn in WSL?
- [ ] Check: `--seed` lets you re-pick the 8 images reproducibly — why does this have its own RNG, separate from the split seed?

### 2.6 `__main__` block
- [ ] Check: order of the checks — why counts first, visual last?
- [ ] Check: what is the exit code when an assert fails? (useful later for CI/pytest)

---

## After the review
- [ ] Update README status row 2: "verified" (date).
- [ ] Write 3–5 lines in `doc/decision-log.md`: Option 1 (arrays in memory) vs Option 2 (PNGs); 90/10 stratified; seed 42; CSV split files.
- [ ] Note in `doc/ai-findings.md`: the `train`/`train_loader` key-collision bug and how the verify script caught it.
- [ ] List any questions I still have -> ask before the model step.
