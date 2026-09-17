# Unsloth and LoRA: how the training actually works

**Language:** English · [فارسی](unsloth-lora.fa.md) · [Deutsch](unsloth-lora.de.md)

---

This project fine tunes a 4 billion parameter model on a 16 GB consumer class GPU, for
free. That is possible because of two techniques stacked together: **LoRA** (train a tiny
fraction of the weights) and **4 bit quantization** (store the rest compressed), wrapped
by **Unsloth** (make it fast and make the two work together correctly).

---

## The problem

Qwen3-4B has about 4 billion parameters. Conventional full fine tuning needs to hold, per
parameter:

| What | Precision | Memory for 4B |
|---|---|---|
| Model weights | fp16 | ~8 GB |
| Gradients | fp16 | ~8 GB |
| Adam optimiser state (2 moments) | fp32 | ~32 GB |
| **Total before activations** | | **~48 GB** |

A Tesla T4 has **16 GB**. Full fine tuning is not close to fitting. It is off by a
factor of three, before activations.

---

## LoRA: train a small correction instead of the whole model

**Low-Rank Adaptation** freezes the original weights entirely and learns a small additive
correction to each targeted weight matrix.

For an original weight matrix `W` of shape `d × k`, instead of learning an update `ΔW`
of the same size, LoRA learns two thin matrices:

```
W_effective = W + (α / r) · B · A

    A :  r × k      (r is the "rank", 16 or 64 here)
    B :  d × r
    W :  frozen, never updated
```

Because `r` is small, `B · A` has far fewer parameters than `W`, but still produces a
`d × k` update. Only `A` and `B` receive gradients. The optimiser state shrinks in
proportion, which is where the real saving is.

### The settings this project uses

From `adapter_config.json`, identical in structure across both runs:

```json
{
  "r": 64,                    // 16 in the multilingual run
  "lora_alpha": 64,           // always equal to r
  "lora_dropout": 0,
  "bias": "none",
  "use_rslora": false,
  "task_type": "CAUSAL_LM",
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]
}
```

**`lora_alpha` equals `r`, deliberately.** The scaling factor is `α / r`, so setting them
equal pins it at exactly 1.0 at every rank. Raising rank from 16 to 64 then adds capacity
*without* also multiplying the size of the update: one variable changes, not two. If
`alpha` were left fixed while `r` moved, a rank change would silently become a learning
rate change too, and the comparison between the two runs would be meaningless.

**All seven projection matrices are targeted**, not just attention. Targeting only
`q_proj`/`v_proj` is a common default and is cheaper, but style transfer changes *how* the
model writes, which lives substantially in the MLP blocks (`gate_proj`, `up_proj`,
`down_proj`). Excluding them would have meant fighting for the same effect with less of
the network available.

**`lora_dropout = 0`.** Dropout regularises against overfitting. This project *wants* to
fit the form hard (see the overfitting discussion in the main README), so it is off.

### What rank costs

| Rank | Adapter size | Persian rhyme | Side effect |
|---|---|---|---|
| r=16 | 132 MB | 33% | none, clean script |
| r=64 | 520 MB | **65%** | leaks foreign characters |

Higher rank captured the poetic form much better. It also, combined with 12 epochs,
degraded the model enough to start emitting Chinese and Cyrillic characters inside Persian
verse. **Capacity is not free.**

---

## 4 bit quantization: shrink the frozen part

LoRA removes the optimiser and gradient cost, but the frozen base weights still have to
sit in memory: ~8 GB in fp16, most of a T4 before any activations.

The base model is therefore loaded in **4 bit** (`load_in_4bit=True`), using the
already quantized `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`. Weights drop to roughly
2.5 GB. They are dequantized on the fly during the forward pass; because they are frozen
and never updated, quantization error does not compound through training.

LoRA over a 4 bit frozen base is what the name **QLoRA** refers to.

Using the already quantized checkpoint rather than quantizing at load time also removes a
step that otherwise runs on every single training start.

### The resulting budget on one T4

| | Memory |
|---|---|
| Base weights, 4 bit | ~2.5 GB |
| LoRA parameters + gradients + optimiser | ~0.3 GB |
| Activations (with gradient checkpointing, seq 1024, batch 2) | ~2 to 4 GB |
| **Total** | **comfortably inside 16 GB** |

---

## Unsloth: what it adds

[Unsloth](https://github.com/unslothai/unsloth) is the layer that makes the above fast and
correct without handwritten CUDA.

**Fused Triton kernels.** Handwritten kernels for attention, RoPE, RMSNorm, the MLP block
and cross entropy, replacing the stock PyTorch path. Roughly 2× faster training with lower
memory, the practical result here being a 12 epoch run in **724 seconds**.

**Manual autograd for the LoRA path.** Rather than relying on the generic autograd graph,
Unsloth implements backward passes that avoid materialising large intermediates, the main
source of its memory savings.

**Correct patching per architecture.** Qwen3, Gemma and Llama each need different handling.
Unsloth detects the architecture and patches accordingly; the startup banner reports what
it did:

```
==((====))==  Unsloth 2026.9.4: Fast Qwen3 patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 2. Max memory: 14.562 GB.
O^O/ \_/ \    Torch: 2.10.0+cu128. CUDA: 7.5. Triton: 3.6.0
\        /    Bfloat16 = FALSE. FA [Xformers = 0.0.35. FA2 = False]
```

That `Bfloat16 = FALSE` line is the T4 constraint made visible. Turing generation GPUs have
no bf16 support, so everything runs in fp16, which is precisely why this project chose
Qwen3 over Gemma, whose family is more prone to NaN loss in fp16.

**Model hosting, already quantized.** Unsloth publishes `-unsloth-bnb-4bit` variants, avoiding
a quantization pass at every load.

### The API

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=None,                    # None = pick correctly for the GPU (fp16 on T4)
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64, lora_alpha=64, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

Then training proceeds with an ordinary HuggingFace `Trainer`/`SFTTrainer`. Unsloth
replaces the internals, not the interface.

**One inference detail that matters:**

```python
FastLanguageModel.for_inference(model)   # ~2x faster generation
```

Omitting this is not an error, just needlessly slow. Some versions do not expose
it, so the scripts here call it defensively.

---

## Why LoRA was right for this project specifically

Beyond fitting in memory, LoRA suits the task:

**The dataset is tiny.** 90 to 270 examples. Updating 4 billion parameters on 90 examples
would overwrite general language ability with noise. Restricting the update to a low rank
correction is itself a regulariser.

**The goal is style, not knowledge.** The model already knows Persian. It needs to be told
*how to write*: rhyming couplets in a classical register. That is a change in output
distribution, not an injection of facts, and it is what LoRA is best at.

**Adapters are small and swappable.** 132 MB and 520 MB versus 8 GB for a full model. Both
adapters in this project share one base model, so comparing them means swapping a small
file rather than storing two complete models.

**The base model stays intact.** The frozen weights cannot be damaged. The worst outcome
of a bad run is a bad adapter, which is deleted.

### Where LoRA did not save us

LoRA restricts *which* weights change, not *how far* they drift. Run 2 (rank 64, 12
epochs) still overfit badly enough to break script consistency. **The adapter being small
does not make overfitting impossible.** Rank, epochs and learning rate still need the same
care as in full fine tuning.

---

## Comparison: the full fine tune in this project

The SmolLM2-360M model was **fully** fine tuned, as a deliberate contrast:

| | SmolLM2-360M | Qwen3-4B |
|---|---|---|
| Method | full fine tune | LoRA + 4 bit |
| Trainable parameters | all 360M | ~1 to 2% of 4B |
| Precision | fp16 mixed | 4 bit base, fp16 compute |
| Output artefact | 1.3 GB model | 132 / 520 MB adapter |
| GPU use | 2× T4 via DDP | single T4 |
| Learning rate | 2e-5 | 2e-4 / 3e-4 |

Note the learning rates differ by **an order of magnitude**. That is expected, not a
mistake: LoRA updates a small low rank correction and tolerates, and needs, a much larger
learning rate than full fine tuning, where 2e-4 across all weights would be destructive.

At 360M, full fine tuning fits and is simple. At 4B it does not fit at all on this
hardware. The crossover is roughly where LoRA stops being a convenience and becomes the
only option.

---

## Running this on Kaggle

This page covers the method. For the operational side (accelerator and internet setup,
the pinned installs and why they differ per model family, driving both T4s, automating
runs through the Kaggle API, and the mount path and encoding gotchas that cost real
time), see **[Running Unsloth on Kaggle](kaggle-unsloth.md)**.

---

## Further reading

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685): Hu et al., 2021
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314): Dettmers et al., 2023
- [Unsloth documentation](https://docs.unsloth.ai/)
