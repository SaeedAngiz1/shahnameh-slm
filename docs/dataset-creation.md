# How the dataset was built

**Language:** English · [فارسی](dataset-creation.fa.md) · [Deutsch](dataset-creation.de.md)

---

The training dataset for this project, 300 parallel prompt/answer pairs across Persian,
English and German, was researched, written, structured and validated using
**Claude Code running Opus 5 at extra high reasoning effort**.

This page documents how that was done and what it produced, because the dataset is the
strongest artefact in the repository and the method is reproducible.

---

## What was produced

| | |
|---|---|
| Parallel prompt sets | 100 |
| Languages | Persian, English, German |
| Total training examples | **300** |
| Split | 90 train / 10 validation per language |
| Verse lines per answer | 4 to 16, mean exactly 10.0 |
| Couplet integrity | **300/300** answers have an even line count |
| System prompts | 3, one per language, identical across every example |

Each of the 100 prompts exists in all three languages with a matched answer, so the
three language subsets are genuinely parallel rather than three unrelated datasets.

---

## Why an AI authored dataset was the right approach here

The target is a specific, demanding literary form: Ferdowsi's *Shahnameh* is written in
*motaqāreb* meter, in rhyming couplets (*masnavi*), with a classical vocabulary distinct
from modern Persian. Writing 100 such passages by hand, then again in English, then in
German, keeping the meaning aligned across all three, is weeks of specialist work.

Three properties made the automated approach worth it:

**1. Trilingual parallelism is nearly free.** Producing the same answer in three
languages at once, keeping the meaning aligned, is a task where a language model has a
structural advantage. Done manually, each language is a separate translation pass with
drift at every step.

**2. Formal consistency is enforced, not hoped for.** Every answer targets the same
shape: rhyming couplets, an even line count, ~10 lines, the same register. The result is
visible in the numbers: 300 out of 300 answers have an even line count, and the mean is
exactly 10.0 lines. A handwritten corpus drifts.

**3. Extra high reasoning effort matters for verse.** Producing rhyming, metrical verse
is not a fluent text task; it requires holding a rhyme target and a syllable count while
composing. Higher reasoning effort measurably improves that, and the rhyme rates below
are the evidence.

---

## Measured quality of the result

Rhyme is checkable by machine. In *masnavi* the lines rhyme in pairs (AA, BB, CC), so
stripping everything but letters and comparing the end of each line pair gives a number:

| Language | Rhyming couplets | Even line count | Script purity |
|---|---|---|---|
| **Persian** | **447/500 (89%)** | 100/100 | no foreign characters |
| German | 363/500 (73%) | 100/100 | n/a |
| English | 338/500 (68%) | 100/100 | n/a |

Persian, the hardest of the three and the primary target, scores highest. That is the
opposite of what happens when verse is machine translated from an English original, and
it reflects the Persian being composed natively rather than derived.

For comparison, the fine tuned models reproduce **33% to 65%** Persian rhyme. **The dataset
is substantially better than anything trained on it**, which means the dataset is not the
bottleneck in this project.

---

## The pipeline, and where reliability comes from

Speed only helps if the output is trustworthy. Reliability here comes from making the
dataset *checkable at every stage* rather than from trusting the generation step.

```
shahnameh_style_dataset.xlsx     ← source of truth (human editable)
         │  glossary · style guide · reference verses · sources · stats
         ▼
   export_jsonl.py               ← validation gate
         │  schema · role order · non-empty · encoding · split integrity
         ▼
   training_data/*.jsonl         ← 8 files, train/val per language + combined
         │
         ▼
   tokenizer round-trip check    ← run against the real tokenizer before training
```

**Spreadsheet as source of truth.** The dataset lives in a workbook with a glossary, a
style guide, reference verses, sources and a statistics sheet: 369 formulas, 0 errors.
It stays human editable: a Persian teacher can correct a line without touching JSON.

**Validation is a gate, not a report.** `export_jsonl.py` refuses to emit JSONL if the
message roles are out of order, a field is empty, or a split is malformed. Bad data
cannot reach training silently.

**Round trip verification with the real tokenizer.** Before any training run, all 300
examples were checked against the actual model tokenizer for two things: that the text
survives encode→decode unchanged, and that response only loss masking covers exactly the
answer tokens and nothing else. All 300 passed. The longest sequence is 635 tokens against
a 1024 limit, so nothing is silently truncated.

That last check is what later made it possible to prove a decoding bug was *not* a data
problem: the masking and EOS supervision were already verified correct, so the fault had
to lie elsewhere. **Verification earlier saved a wasted retraining cycle later.**

---

## Time saved

| Task | By hand | This approach |
|---|---|---|
| 100 Persian passages in classical meter | weeks | one working session |
| English + German parallel versions | weeks more | same session |
| Schema, validation, JSONL export tooling | 1 to 2 days | minutes |
| Spreadsheet with glossary, style guide, stats | 1 to 2 days | minutes |

The dataset, the export tooling, the validation, the training notebooks and the
evaluation harness were all produced in a small number of working sessions. Consistency
across 300 examples in three languages is the part that is genuinely hard to achieve
manually, and it is the part that came out strongest.

---

## Honest limitations

The dataset is the best artefact here, and it still carries real caveats:

- **No native speaker has reviewed the Persian verse.** Rhyme and script purity are
  measured mechanically. **Meter (عروض), grammar and classical register are unverified.**
  89% rhyme says the form is largely correct; it does not prove the verse scans.
- **The verse is newly written in Ferdowsi's style. It is not his text.** No line is a
  quotation from the *Shahnameh*.
- **100 prompts is small.** Enough to teach a voice; not enough to teach prosody. See
  the main README for what would fix that.
- **English and German rhyme less consistently** (68% and 73%) than Persian, partly
  because rhyme is less structurally central in those traditions and partly because
  Persian was the priority.

---

## Reproducing it

The workbook is the input; everything downstream regenerates:

```bash
python export_jsonl.py      # xlsx -> training_data/*.jsonl, with validation
python make_visuals.py      # regenerate every chart from the training logs
```

Edit the spreadsheet, export again, retrain. The verse can be corrected by a human expert
without touching any code, which is exactly the workflow that a native review of the
Persian would need.
