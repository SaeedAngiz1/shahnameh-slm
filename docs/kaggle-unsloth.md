# Running Unsloth on Kaggle's free two GPU tier

**Language:** English · [فارسی](kaggle-unsloth.fa.md) · [Deutsch](kaggle-unsloth.de.md)

---

This is the operational companion to [Unsloth and LoRA](unsloth-lora.md). That page
explains *why* the method fits in 16 GB; this one explains *how to actually run it* on
Kaggle: the setup, the filesystem, the installs, driving both GPUs, automating it from a
terminal, and the specific things that broke.

Everything here was done on the free tier. No paid compute was used anywhere in this
project.

---

## What the free tier actually gives you

| Resource | Allowance |
|---|---|
| GPU | **2× NVIDIA Tesla T4**, 16 GB each |
| GPU quota | **30 hours per week**, resets weekly |
| Max session length | 9 hours (GPU), 12 hours (CPU/TPU) |
| TPU quota | 20 hours per week |
| Output storage (`/kaggle/working`) | 20 GB, persisted as the notebook's output |
| Internet access | available, but requires a **phone verified account** |

This entire project (three training runs, a decoding sweep, an evaluation notebook)
consumed **under three hours** of that 30 hour weekly budget.

Two constraints matter more than the rest:

**Internet requires phone verification.** Without it, `pip install unsloth` fails and the
model cannot be downloaded from Hugging Face. This is the single most common reason a
notebook that "should work" does not.

**The T4 is Turing generation: no bfloat16.** Everything runs in fp16. This is not a
detail. It eliminates entire model families that are prone to NaN loss in fp16, and it is
why this project chose Qwen3 over Gemma.

---

## Setting up a notebook

In the right hand panel of the Kaggle notebook editor:

1. **Session options → Accelerator → GPU T4 x2.** Not "GPU P100", which is a single card.
2. **Session options → Internet → On.**
3. **Add Input → Datasets** → attach your training data.

Then *Run All*. Every tunable setting in this project's notebook lives in one cell at the
top, so nothing else needs editing.

### The filesystem

| Path | Purpose |
|---|---|
| `/kaggle/input/<dataset-slug>/` | attached datasets, **read only** |
| `/kaggle/working/` | your output, **persisted** when the notebook is saved |
| `/kaggle/temp/` | scratch, discarded at session end |

Datasets mount at unpredictable depths, so this project never hardcodes a path. It
searches for a known filename:

```python
def find_data_dir(root):
    """Kaggle mounts datasets at unknown depths; find the folder by a known file."""
    matches = sorted(Path(root).rglob("all_train.jsonl"))
    if not matches:
        raise FileNotFoundError(f"no all_train.jsonl under {root}; did you attach the dataset?")
    return matches[0].parent
```

That habit paid off repeatedly. See [Gotchas](#gotchas-that-cost-real-time).

---

## Installing Unsloth on Kaggle

Kaggle images ship their own PyTorch and Transformers, which frequently conflict with what
Unsloth needs. The installs are therefore **pinned, and differ per model family**:

```python
if MODEL_PRESET == "gemma-4-e4b":
    !pip install -q unsloth
    !pip install -q --no-deps transformers==5.10.1 "tokenizers>=0.22.0,<=0.23.0"
    !pip install -q "huggingface_hub>=1.5.0,<2.0"
    !pip install -q torchcodec
    !pip install -q --no-deps --upgrade timm
else:                                    # qwen3-4b
    !pip install -q pip3-autoremove
    !pip install -q torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
    !pip install -q unsloth
    !pip install -q --no-deps --upgrade "torchao>=0.16.0"
    !pip install -q transformers==4.56.2
    !pip install -q --no-deps trl==0.22.2
```

Three things worth understanding:

**The pins come from Unsloth's own Kaggle notebooks**, not from guesswork. Each model
family has a combination that is known to work on this image; deviating from it produces
import errors or silent version conflicts.

**`--no-deps` is deliberate.** It installs a package without letting pip "helpfully"
resolve again and downgrade neighbouring packages. Without it, installing `trl` can drag
`transformers` to a different version and break the Unsloth patches.

**Gemma and Qwen3 need incompatible stacks**: `transformers 5.10.1` versus `4.56.2`. They
cannot coexist, which is why the preset switch changes the install cell too. This alone is
a good reason to pick one model family per notebook.

Installation takes about 5 minutes and must run once per session.

### Verifying the environment

Unsloth prints a banner on import. Read it: it confirms what actually happened:

```
==((====))==  Unsloth 2026.9.4: Fast Qwen3 patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 2. Max memory: 14.562 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.10.0+cu128. CUDA: 7.5. CUDA Toolkit: 12.8. Triton: 3.6.0
\        /    Bfloat16 = FALSE. FA [Xformers = 0.0.35. FA2 = False]
```

`Num GPUs = 2` confirms the accelerator setting took effect. `Bfloat16 = FALSE` confirms
the T4 constraint. `Fast Qwen3 patching` confirms Unsloth recognised the architecture.
If any of these is wrong, stop there rather than debugging training later.

---

## Using both GPUs

There are two ways to spend a second T4, and the default here is the less obvious one.

### Default: two different jobs in parallel

Fine tuning a 4B model with QLoRA fits on **one** T4. Splitting one job across both cards
adds coordination risk for little gain at this scale. Instead, GPU 0 trains while GPU 1
generates baseline answers from the untuned model, answers you need anyway, to prove the
fine tune changed anything.

The mechanism is plain subprocesses with a pinned device:

```python
def run(cmd, gpus, name, background=False):
    """Run a script on the given GPUs, in its own folder."""
    cwd = f"{WORK}/run_{name}"
    os.makedirs(cwd, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus, "PYTHONPATH": SCRIPTS}
    if background:
        log = open(f"{cwd}/log.txt", "w")
        return subprocess.Popen(cmd, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    ...
```

```python
baseline = run(baseline_cmd, "1", "baseline", background=True)   # GPU 1
run([PY, *train_args], "0", "train")                             # GPU 0
code = baseline.wait()
```

Two details that are easy to miss:

**Each process gets its own working directory.** Unsloth writes a compiled kernel cache
into the current directory. Two processes sharing a cwd will race on that cache. This is
not theoretical: it is why `cwd=f"{WORK}/run_{name}"` exists.

**`CUDA_VISIBLE_DEVICES` is set per subprocess, not globally.** Each process sees exactly
one card and calls it `cuda:0`. No code in the training script is GPU aware at all.

### Optional: true distributed training

```python
run([PY, "-m", "torch.distributed.run", "--nproc_per_node=2", *train_args], "0,1", "train")
```

This works, and this project used it for the SmolLM2 full fine tune. It also produced the
project's worst failure. See below.

---

## The failure worth planning around

The SmolLM2 run completed **all 51 training steps across both T4s successfully**, then
crashed:

```
NCCL ALLGATHER sequence number mismatch
```

It died *after* training, while loading the best checkpoint for export. Training was fine.
Collective coordination during the export path was not.

What saved the run was that **Kaggle preserves `/kaggle/working` as notebook output even
when the notebook fails.** All 41 output files, including checkpoints, survived. The fix
was a separate single process recovery notebook that loaded the preserved checkpoint,
evaluated it and exported it, with no collectives involved. It ran in 218 seconds.

**Three rules that follow from this:**

1. **Do not put export in the same process as distributed training.** Train, save
   checkpoints, exit. Export separately.
2. **Checkpoint every epoch.** It is what makes recovery possible at all.
3. **A failed Kaggle notebook is not a lost run.** Check the output before you run
   anything: the expensive part may already be on disk.

---

## Automating it from a terminal

The web UI is fine for one run. For a sequence of experiments, the Kaggle API is far
better, and lets you drive everything without touching a browser.

```bash
pip install kaggle
# token from kaggle.com/settings -> API -> Create New Token
# saved to ~/.kaggle/access_token
```

### Uploading a dataset

```bash
# dataset-metadata.json in the folder alongside the data files:
# { "title": "Shahnameh Style SFT",
#   "id": "<username>/shahnameh-style-sft",
#   "licenses": [{"name": "CC0-1.0"}] }

kaggle datasets create -p .        # private by default
```

### Pushing and running a notebook

`kernel-metadata.json` controls everything the right hand panel would:

```json
{
  "id": "<username>/shahnameh-qwen3-4b-unsloth",
  "title": "Shahnameh Qwen3-4B Unsloth",
  "code_file": "shahnameh-qwen3-4b-unsloth.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["<username>/shahnameh-style-sft"],
  "kernel_sources": [],
  "machine_shape": "NvidiaTeslaT4"
}
```

**`"machine_shape": "NvidiaTeslaT4"` is what selects T4 × 2.** `enable_gpu: true` alone
can give you a single card. This field is poorly documented; the reliable way to find the
right value is to pull the metadata of a notebook already configured correctly:

```bash
kaggle kernels pull <username>/<existing-notebook> -m -p .
```

Then:

```bash
kaggle kernels push -p .                       # pushing also starts the run
kaggle kernels status <username>/<notebook>    # PENDING / RUNNING / COMPLETE / ERROR
kaggle kernels output <username>/<notebook> -p ./out
```

A poll loop is usually enough:

```bash
while true; do
  S=$(kaggle kernels status <username>/<notebook> | grep -o 'KernelWorkerStatus\.[A-Z_]*')
  echo "$(date +%H:%M:%S) $S"
  case "$S" in *COMPLETE*|*ERROR*) break;; esac
  sleep 120
done
```

### Chaining notebooks

One notebook's output can be another's input, via `kernel_sources`. This project used it to
run a decoding sweep against an already trained adapter without retraining:

```json
"kernel_sources": ["<username>/shahnameh-qwen3-4b-fa-only"]
```

---

## Gotchas that cost real time

All of these were hit in this project. None are in the official documentation.

**Notebook output mounts under a longer path than you expect.** A kernel attached via
`kernel_sources` does *not* appear at `/kaggle/input/<slug>/`. The real path is:

```
/kaggle/input/notebooks/<username>/<slug>/...
```

Hardcoding the short path produces `Can't find 'adapter_config.json'` and a failed run at
the ~160 second mark. Search instead of assuming:

```python
found = [os.path.dirname(p)
         for p in glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
         if "checkpoint" not in p]
```

**The Kaggle CLI mangles absolute Windows paths.** `kaggle datasets create -p "C:/long/path"`
builds a broken temp path and fails with `[Errno 2] No such file or directory`. Work around
it by running from inside the folder with `-p .`.

**Notebooks with characters outside ASCII fail to upload from Windows.** Pushing a notebook containing Persian
text raises `'charmap' codec can't decode byte 0x81`. Force UTF-8:

```bash
PYTHONUTF8=1 kaggle kernels push -p .
```

**`enable_gpu: true` is not the same as two GPUs.** Without `machine_shape`, a two GPU
notebook may silently get one card, and any code pinning `CUDA_VISIBLE_DEVICES=1` fails.

**A saved notebook page is read only.** To interact with a notebook (type a question into
a widget, run one cell) you must click **Edit** to start a live session. Viewing the saved
version and expecting interactivity is a common confusion.

**Interactive sessions expire.** Install and model load cells must be run again after a
session times out. Budget ~7 minutes before the first useful output.

---

## Cost summary for this project

| Run | Wall time | What it produced |
|---|---|---|
| SmolLM2 full fine tune (2× T4, DDP) | ~25 min | trained, then crashed on export |
| SmolLM2 recovery/export | 218 s | the usable model |
| Qwen3-4B multilingual LoRA | 411 s train | 33% Persian rhyme |
| Qwen3-4B Persian only LoRA | 724 s train | 65% Persian rhyme |
| Persian decoding sweep | ~12 min | 5 configs measured |
| **Total** | **under 3 hours** | of a 30 hour weekly quota |

Fine tuning a 4 billion parameter model is well inside what free compute can do. The
limiting factor in this project was never the GPUs. It was having only 90 Persian training
examples. See the [main README](../README.md) for what would actually fix that.
