# Shahnameh SLM Fine Tuning: Session Handoff

> ## ⚠️ Superseded
>
> **This file describes an earlier state of the project and is kept for history only.**
> The Qwen3 training runs, the published GitHub repository, the three Hugging Face models
> and the multilingual documentation all happened after it was written.
>
> **The README and the guides in `docs/` describe the current state.**


**Last updated:** 2026-09-15
**Written for:** the next Claude Code session, or any other AI assistant picking up this project.
**Project folder:** `C:\Users\angiz\Desktop\AI workers\Claude Code\Fine tuning`

---

## Latest update (end of session, 2026-09-15)
- **Codex update:** User now requests `HuggingFaceTB/SmolLM2-360M`. Created `kaggle/train_smollm2.py`, `kaggle/build_smollm2_notebook.py`, and `kaggle/smollm2_360m_shahnameh_kaggle.ipynb`. This is a separate full fine tuning recipe with Transformers 4.56.2, Accelerate 1.10.1, two GPU DDP, FP16 mixed precision, assistant only loss, 3 epochs, and best validation checkpoint selection. All 8 original JSONL files are embedded byte for byte.
- **Training and recovery export completed (2026-09-16):** Account verified, Internet enabled, both T4 GPUs trained all 51 steps / 3 epochs. Private saved version: https://www.kaggle.com/code/saeed010/smollm2-360m-shahnameh-fine-tuning?scriptVersionId=350181097 . Final epoch validation loss 2.65028524. The job subsequently failed on mismatched NCCL ALLGATHER sequence numbers after checkpoint loading, before final export. Its 41 output files, including checkpoints, are preserved. Do not treat the original notebook as a verified end to end recipe.
- **Recovery succeeded:** `kaggle/recover_smollm2.py`, `kaggle/build_recovery_notebook.py`, and `kaggle/recover_smollm2.ipynb` evaluated/exported checkpoint-51 in a single process. Private successful run: https://www.kaggle.com/code/saeed010/smollm2-shahnameh-model-export?scriptVersionId=350278641 . Kaggle reports 217.7 seconds, successful, subprocess return code 0, 13 output files. The ZIP is 1,342,296,888 bytes and is visible under Output. Includes full model/tokenizer/chat template, metrics and samples. **Local download confirmed (2026-09-16):** `smollm2-shahnameh-model.zip` in the project root is 1,342,296,888 bytes, an exact byte count match with the Kaggle artifact, and opens as a valid archive with all 11 expected entries.
- **Measured quality:** Combined validation loss improved 3.030281 to 2.650258 (~12.5%); each language improved. Idle interactive training session was stopped. See `kaggle/SMOLLM2_RESULTS.md` for details.
- **Repetition diagnosed and fixed (2026-09-16):** The "severe repetition" reported earlier was a **decoding fault, not a model defect**. The export script used pure greedy decoding with no repetition control; greedy enters a self reinforcing line loop and never reaches EOS, so all samples ran to the token cap. The model had in fact learned to stop: teacher forced on gold answers, `<|endoftext|>` is the most likely prediction at the correct end position in 17/30 cases (mean P(EOS)=0.55). With `repetition_penalty=1.15`, 25/30 held out prompts terminate on their own and no language repeats a line (greedy: 0/30 terminate).
- **Revised quality verdict, three different answers:** **English is usable**: coherent Shahnameh style verse with correct Rostam/Sohrab content. **German** terminates and stops repeating but has broken grammar and drifts off domain. **Persian remains poor and is a separate, unsolved problem**: repetition control does not rescue it, and Persian was the only language where EOS was never the most likely token (0/10). Likely cause is the base model: SmolLM2-360M is centred on English with weak Persian pretraining, so no decoding change will fix it.
- **SmolLM2 validation:** Notebook schema and code syntax pass; all 300 examples pass response only masking and exact text round trip checks with the real SmolLM2 tokenizer. Longest sequence is 635 tokens (limit 1024). Pinned package installation and training on both Kaggle T4s succeeded. The model is primarily English; generated multilingual quality still requires review.
- **Done:** LibreOffice formula check. 369 formulas, 0 errors, Stats counts correct. Caveat: `line_count` shows 1 in LibreOffice (text functions ignore in cell line breaks); the line breaks themselves and the training data are fine. Details in section 7.
- **Done:** this handoff file.
- **Not done:** native review of the Persian verse; user has not yet confirmed whether Stats B14 shows 10.0 or 1.0 in the LibreOffice window. (The SmolLM2 Kaggle training run *is* done, see above. The separate Unsloth recipe in `kaggle/shahnameh_unsloth_kaggle.ipynb` has still never been run; it is an alternative path, not a blocker.)
- **Next action:** decide how to attack the Persian problem, which is now the main open quality issue. Decoding is fixed and English is usable; Persian needs either a base model with real Persian pretraining (SmolLM2-360M is centred on English) or substantially more Persian training data. Running the Unsloth notebook with a multilingual base such as Gemma 4 E4B or Qwen3 4B is the obvious candidate.

---

## 1. Goal

The user wants to fine tune a **small language model (SLM)** to answer in the voice of **Ferdowsi's Shahnameh** (the Persian *Book of Kings*), in **three languages: Persian, English and German**.

Two phases are done:
1. **Dataset:** a spreadsheet based training set plus JSONL exports. Done and verified.
2. **Training setup:** a Kaggle notebook using **Unsloth** on **2x NVIDIA T4** GPUs. Built, but **not yet run on Kaggle**.

---

## 2. Current state at a glance

| Item | Status |
|---|---|
| 100 parallel prompts x 3 languages = 300 SFT examples | Done |
| Excel workbook with data, glossary, style guide, reference verses, sources, stats | Done |
| JSONL export script with validation | Done, tested |
| JSONL files (train/val per language + combined) | Done |
| Native speaker review of the verse (especially Persian meter) | **Not done** (user to arrange) |
| LibreOffice formula recalculation check of the workbook | Done: 369 formulas, 0 errors. One caveat about `line_count`; see section 7 |
| Kaggle notebook for Unsloth training (2x T4) | Built; helper code tested locally; **never run** (alternative path) |
| SmolLM2-360M training run on Kaggle | **Done**: 51 steps / 3 epochs, val loss 3.030 to 2.650 |
| Trained model exported and downloaded | **Done**: `smollm2-shahnameh-model.zip`, 1,342,296,888 bytes, verified |
| Repetition in generated output | **Fixed**: decoding fault, not a model defect; `repetition_penalty=1.15` |
| English output quality | **Usable**: coherent verse, correct content, terminates |
| German output quality | Terminates, but broken grammar and off domain drift |
| Persian output quality | **Poor, unsolved**: base model lacks Persian; needs different base or more data |

---

## 3. Files in the project folder

```
Fine tuning/
├── HANDOFF.md                         this file
├── shahnameh_style_dataset.xlsx       SOURCE OF TRUTH for the dataset
├── export_jsonl.py                    xlsx -> JSONL exporter with validation
├── smollm2-shahnameh-model.zip        Kaggle artifact, UNMODIFIED (greedy defaults, degenerate samples)
├── smollm2-shahnameh-model/           unpacked USABLE copy; generation_config fixed with
│                                      repetition_penalty 1.15 + max_new_tokens 200
├── training_data/
│   ├── fa_train.jsonl  (90)   fa_val.jsonl  (10)
│   ├── en_train.jsonl  (90)   en_val.jsonl  (10)
│   ├── de_train.jsonl  (90)   de_val.jsonl  (10)
│   ├── all_train.jsonl (270)  all_val.jsonl (30)
└── kaggle/
    └── shahnameh_unsloth_kaggle.ipynb SOURCE OF TRUTH for training code (26 cells)
```

**Important:** the helper scripts used to *generate* these files lived in a temporary session scratchpad and **no longer exist**. They were: content text files, the workbook builder, the lint script, the tests and the notebook builder. From now on:
- Edit the **dataset** in `shahnameh_style_dataset.xlsx`, then run `python export_jsonl.py`.
- Edit the **training code** directly in the notebook cells. The scripts are embedded there with `%%writefile`.

---

## 4. The dataset

### 4.1 Design decisions
- **Format:** supervised fine tuning (SFT), chat `messages` format: system, then user, then assistant.
- **Answers are original verse written in the Shahnameh style.** They are *not* quotations from the poem, which avoids fabricated "quotes" and copyright questions.
- **Form:** rhymed couplets, one verse line per line (two lines = one couplet).
  - Persian aims at the Shahnameh meter, **motaqarib** (فعولن فعولن فعولن فعل, 11 syllables per half line). Diction is pure Persian with few Arabic loanwords, and archaic forms such as همی، ز، چو، کجا، به … اندرون. Couplets may use *radif* (a repeated refrain word after the rhyme).
  - English and German use rhymed couplets in an archaic register modelled on the 19th century translations (Warner; Rückert).
- **Persona:** "a storyteller in Ferdowsi's manner", never Ferdowsi himself.
- **Helpfulness:** practical questions (tea, code bugs, sleep, saving money, apology emails, anxiety, job interviews) get real, correct advice inside the verse. The anxiety answer suggests seeing a doctor if it gets worse, and the bullying answer says to tell a teacher or parent.
- **Story accuracy:** the retold tales follow the canonical plots (Rostam and Sohrab, the Seven Labours, Zal and the Simorgh, Kaveh and Zahhak, Siyavash's fire ordeal, Rostam and Esfandiyar, Bizhan and Manizheh, Rostam's death by Shaghad, Fereydun's three sons, Kay Kavus's flying throne, Kay Khosrow's renunciation, Jamshid's fall, and others).

### 4.2 Categories (10 prompts each)
`heroes, wisdom, fate, courage, love, nature, everyday, leadership, battle, persona`

IDs run `SH-001` to `SH-100`. **The same ID is the same prompt in all three languages.** Categories repeat in that order within every block of ten IDs.

### 4.3 Train/validation split
Items are grouped in blocks of 10. In block *k* (0 to 9), the item at position *k* is validation. That gives exactly one validation item per category:
`SH-001, SH-012, SH-023, SH-034, SH-045, SH-056, SH-067, SH-078, SH-089, SH-100` are val. Everything else is train (90/10 per language).

### 4.4 System prompts (sheet "System Prompts")
- **FA:** تو داستان‌سرایی به شیوه‌ی شاهنامه‌ی فردوسی هستی. به هر پرسش، به پارسی سره و زبان حماسی، در بیت‌های هم‌قافیه پاسخ بده.
- **EN:** You are a storyteller in the manner of Ferdowsi's Shahnameh, the Persian Book of Kings. Answer every request in an elevated, heroic, archaic voice, in rhymed couplets.
- **DE:** Du bist ein Erzähler im Stil von Firdausis Schahname, dem persischen Königsbuch. Antworte auf jede Bitte in erhabener, heldenhafter, altertümlicher Sprache, in gereimten Verspaaren.

### 4.5 Workbook: `shahnameh_style_dataset.xlsx`

| Sheet | Contents |
|---|---|
| README | How to use; notes on quality, export, model choice and recipe |
| System Prompts | `lang`, `sheet`, `system_prompt` (read by the exporter) |
| Persian (FA) / English (EN) / German (DE) | The data. Columns: `id, category, split, user_prompt, assistant_response, line_count, review_status, notes`. The Persian sheet is right to left. |
| Glossary | 39 names and terms: Persian, English, German, meaning |
| Style Guide | 11 style features with examples per language |
| Reference Verses | 12 famous real Shahnameh couplets with literal EN/DE renderings. Each has a verification note: well attested, check the wording in a critical edition, or authenticity debated. |
| Sources | Public domain full texts and URLs (section 9) |
| Stats | `COUNTIFS` per category and split per language, totals, average lines per answer |

Formula details:
- `line_count` is `=IF(E2="",0,LEN(E2)-LEN(SUBSTITUTE(E2,CHAR(10),""))+1)`.
- Stats formulas cover rows 2 to 5000, so rows the user adds are counted.
- `fullCalcOnLoad = True`, so Excel recalculates everything on open. The file was written by openpyxl, which stores no cached values.
- `review_status` is `draft` for all rows. The user should set it to `approved` after review.

### 4.6 Exporter: `export_jsonl.py`
```
python export_jsonl.py                 # all languages, with system prompt
python export_jsonl.py --no-system     # omit the system message
python export_jsonl.py --langs fa en   # subset
python export_jsonl.py --xlsx PATH --out DIR
```
- **Validation (it writes nothing if any row fails):**
  - required columns are present;
  - IDs match `SH-\d{3,}` and are unique per sheet;
  - `split` is `train` or `val`;
  - prompt and response are not empty;
  - Persian responses contain Persian script;
  - English and German rows contain no Persian script.
- **Warnings only:** an odd number of verse lines, or ID sets that differ between language sheets.
- **Output:** `training_data/{lang}_{train,val}.jsonl` and `all_{train,val}.jsonl`, with UTF-8 text written unescaped (`ensure_ascii=False`). Combined files are sorted by (id, lang).
- **Example line:** `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`

### 4.7 How the dataset was made and checked
1. Researched public domain source texts (section 9).
2. Wrote the content by hand in small batches, as plain text blocks (`### SH-001 | heroes`, then `FA-Q:`, `FA-A:`, `EN-Q:`, …). Persian lines were scanned syllable by syllable against the motaqarib pattern while writing. That scan was **manual, not machine verified**.
3. A build script parsed the blocks into the workbook with openpyxl (Arial font, styled headers, frozen panes, autofilter, RTL for Persian).
4. **Lint checks on all 100 items** (they pass):
   - no odd line counts;
   - no couplet rhyming a word with itself (for Persian, a repeated last word is allowed if the word before it differs, i.e. radif);
   - no duplicate lines or prompts within a language.
   The lint caught 5 English/German self rhymes, all fixed (SH-003, SH-009, SH-047, SH-055, SH-081). Two more (SH-017, SH-018) were fixed by hand before the lint existed.
5. **Tests (6, all passing):** block parser, split rule, export with and without system prompt, and rejection of bad rows (nothing written).
6. Checked the formulas by computing the same values in Python, e.g. heroes had 9 train and 1 val, and the average Persian answer length was 10.0 lines.

### 4.8 Known quality caveats
- **Persian meter is unverified.** Some half lines will be off meter. A native speaker who knows prosody should review before training.
- English and German rhymes are sometimes slant rhymes; the register is archaic and occasionally forced.
- 300 examples is a **seed**. For a robust style, aim for 1,000 or more rows: add prompts, rephrase existing ones, or mine the public domain translations.
- The workbook README still names older model suggestions (Qwen3, Gemma 3, Llama 3.2). The training notebook now defaults to **Gemma 4 E4B**.

---

## 5. Training setup: Kaggle + Unsloth + 2x T4

### 5.1 Research findings (checked 2026-09-15)
- **Unsloth multi GPU:** "supported but not yet officially released". It is launched with `torchrun --nproc_per_node=2 script.py` or `accelerate launch`, and DDP is enabled automatically with more than one GPU. Unsloth's Kaggle CI multi GPU test leg only started running after PR #10270 (merged 2026-09-04), which suggests it is still fragile.
- **Known issue #5178:** Unsloth *Studio* on Kaggle T4 x2 detects only one GPU. This does not affect scripts.
- **Gemma 4 E4B:** 140 languages; about 10 GB VRAM for QLoRA, so it fits one T4 (about 15 GB). It is loaded with `FastModel`. Chat template `gemma-4`. Turn markers are `<|turn>system/user/model … <turn|>`, and the system role is supported natively.
- **Qwen3-4B-Instruct-2507:** loaded with `FastLanguageModel`, chat template `qwen3-instruct`. Its Hugging Face repo template inserts an empty `<think>\n\n</think>\n\n` into assistant turns.
- **Install pins come from Unsloth's official Kaggle notebooks** (`nb/Kaggle-Gemma4_(12B)_Text.ipynb` and `nb/Kaggle-Qwen3_(4B)-Instruct.ipynb` in `unslothai/notebooks`):
  - Gemma 4: `pip install unsloth`; `--no-deps transformers==5.10.1 "tokenizers>=0.22.0,<=0.23.0"`; `"huggingface_hub>=1.5.0,<2.0"`; `torchcodec`; `--no-deps --upgrade timm`.
  - Qwen3: `pip3-autoremove`; torch/torchvision/torchaudio/xformers from cu128; `unsloth`; `--no-deps --upgrade "torchao>=0.16.0"`; `transformers==4.56.2`; `--no-deps trl==0.22.2`.
- **The official pattern** is:
  1. `get_chat_template`;
  2. a `text` field from `apply_chat_template(...).removeprefix('<bos>')`;
  3. `SFTTrainer(tokenizer=..., args=SFTConfig(dataset_text_field="text", ...))`;
  4. `train_on_responses_only(trainer)`, which detects automatically the markers;
  5. `torch._dynamo.config.recompile_limit = 64`.
- **GGUF for Gemma 4:** only Q8_0, BF16 or F16 are supported for now.

### 5.2 Design decisions
- **Scripts run as subprocesses** from the notebook, so DDP is possible and both GPUs can work at once.
- **Default GPU mode `train+baseline`** (most reliable): GPU 0 trains while GPU 1 generates baseline answers from the base model.
- **Optional GPU mode `ddp`:** the baseline runs first on GPU 0, then `python -m torch.distributed.run --nproc_per_node=2`.
- **Presets use Unsloth's already quantized 4 bit repos:** `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` (default) and `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` (fallback). The model is downloaded in advance with `snapshot_download` in the notebook, so two processes never download at once.
- **Separate working directory per subprocess** (`/kaggle/working/run_<name>`), because Unsloth writes a compile cache into the cwd.
- **fp16 on T4** (no bf16), chosen with `torch.cuda.is_bf16_supported()`.
- **Hyperparameters (defaults):** LoRA r=16, alpha=16, dropout 0; lr 2e-4, linear schedule, 5 warmup steps; `adamw_8bit`; weight decay 0.001; 3 epochs; per device batch 2 with gradient accumulation 4; max sequence length 1024; seed 3407.
- **Checkpoints:** validation loss is evaluated each epoch; `load_best_model_at_end=True` with `metric_for_best_model="eval_loss"` and `save_total_limit=2`. Checkpoints go to a **sibling** folder `shahnameh-lora-checkpoints`, so the adapter zip stays small.
- **Mask safety check:** after `train_on_responses_only`, `verify_masking` decodes the loss bearing tokens of the first 3 training rows. It raises an error if nothing is trained, if the assistant reply is missing, or if the user prompt leaked in.
- **Generation:** batched (6), left padding, `max_new_tokens=512`, sampling with the recommended settings per model:
  - Gemma: temperature 1.0, top_p 0.95, top_k 64;
  - Qwen: temperature 0.7, top_p 0.8, top_k 20.
- **Two generation fixes:**
  - Gemma prompts always get exactly one `<bos>`, because sampling tokenizes with `add_special_tokens=False` and the repo template emits no BOS.
  - A leading `<think>…</think>` block is stripped from answers.

### 5.3 Notebook structure (`kaggle/shahnameh_unsloth_kaggle.ipynb`, 26 cells)
1. Intro markdown with setup steps and an outputs table
2. **Settings:** `MODEL_PRESET`, `GPU_MODE`, `LANGS`, `USE_SYSTEM_PROMPT`, `EPOCHS`, `LEARNING_RATE`, `LORA_R`, `BATCH_SIZE`, `GRAD_ACCUM`, `MAX_SEQ_LEN`, `EXPORT_MERGED_16BIT`, `EXPORT_GGUF`, `HF_REPO`
3. GPU check with `nvidia-smi`; stops if fewer than 2 GPUs
4. Pinned install, branched by preset
5. `%%writefile` of 4 scripts to `/kaggle/working/scripts/`:
   - `shahnameh_common.py`: standard library only. Presets, `find_data_dir` (recursive search for `all_train.jsonl` under `/kaggle/input`), `load_examples`, `render_chat` (folds the system prompt into the user turn if a template rejects the system role), `generation_prompt`, `check_response_mask`, `clean_answer`, `checkpoint_dir`, `custom_example`, `comparison_markdown`.
   - `train_shahnameh.py`: args `--preset --data-root --out --langs --no-system --epochs --lr --lora-r --batch-size --grad-accum --max-seq-len --seed`. Writes `train_log.json` with metrics, best checkpoint, best eval loss and log history.
   - `sample_shahnameh.py`: args `--preset --adapter --data-root --out --prompt --langs --no-system --max-new-tokens --batch-size --max-seq-len`. Writes JSONL rows of `{lang, prompt, reference, answer}`.
   - `export_shahnameh.py`: args `--preset --adapter --merged --gguf --hf-repo --out`. `HF_TOKEN` comes from the environment; the notebook reads it from Kaggle Secrets.
6. Setup: find the data, download in advance the model, define `run()` and `tail()` helpers
7. Train, with the baseline in parallel or first depending on `GPU_MODE`
8. Loss summary (eval loss per epoch, first and last train loss)
9. Fine tuned answers to the validation prompts
10. `comparison.md`: baseline vs fine tuned, side by side
11. "Try your own question" cell
12. Optional exports and a zip of the adapter

**Outputs in `/kaggle/working`:** `shahnameh-lora/` (with `train_log.json`), `shahnameh-lora.zip`, `samples_baseline.jsonl`, `samples_finetuned.jsonl`, `comparison.md`, `custom_answer.jsonl`, plus `shahnameh-merged-16bit/` and `shahnameh-gguf/` if exports are switched on.

### 5.4 What was tested locally (Windows, no GPU)
- **14 pytest tests pass** for `shahnameh_common.py`, using the **real tokenizers** downloaded from both 4 bit repos:
  - data loading and the missing data error;
  - chat rendering contains every turn;
  - BOS stripping, and a single BOS in generation prompts;
  - the mask check accepts a correct mask and rejects a leaked prompt or an empty mask (on Persian text);
  - `<think>` cleanup, system prompt folding, custom question building, comparison markdown, bad preset or language.
- All 4 scripts compile (`py_compile`).
- The notebook validates with nbformat, and every code cell parses once IPython magics are stubbed out.
- Rendered template inspection confirmed the Gemma 4 and Qwen3 formats shown in section 5.1.

### 5.5 NOT tested (only runs on Kaggle with GPUs)
- The pinned installs working together on today's Kaggle image
- Model loading, LoRA setup, `SFTTrainer` with `eval_dataset` plus `load_best_model_at_end` under Unsloth's patched trainer
- `train_on_responses_only` automatic detection on these templates (the mask check will catch a failure)
- Gemma 4 E4B stability in fp16 on T4
- Batched generation, adapter reload, exports and Hub push
- DDP mode as a whole

---

## 6. How the user runs it on Kaggle
1. Create a new Kaggle Dataset and upload the 8 files from `training_data/`.
2. Import `kaggle/shahnameh_unsloth_kaggle.ipynb`. In settings, set the accelerator to **GPU T4 x2** and turn **Internet on** (this needs a phone verified account).
3. Use **Add Input** to attach the dataset. The notebook finds it automatically.
4. Optional: add the Kaggle Secret `HF_TOKEN` and set `HF_REPO`.
5. **Run All.**

---

## 7. LibreOffice formula check (done 2026-09-15, with one caveat)

**Setup.**
- LibreOffice 26.8.0.3 is at `C:\Program Files\LibreOffice\program\soffice.exe`.
- The xlsx skill's `recalc.py` only works on Linux and macOS (hardcoded macro paths, expects `soffice` on PATH), so it was not used.

**The crash (exit `0xC0000409`).** The Windows Application log shows `soffice.bin` failing inside `ucrtbase.dll`. A step by step test matrix established the following. Early guesses blaming the console launcher or the xlsx output format were **wrong**.
- Crashes happened on the **first headless launch with a brand new private profile**, and on every launch of the profile whose `Module1.xba` macro file had been overwritten.
- On an already initialised, untouched profile, headless `--convert-to csv|ods|xlsx` all work with `soffice.exe`.
- The user's own LibreOffice window runs on the default profile and was never touched.

**Method that works.**
1. Copy the workbook.
2. `soffice.exe -env:UserInstallation=<used profile> --headless --norestore --convert-to ods`
3. Parse the ODS `content.xml`: every cell with `table:formula` and its computed `office:value`.
4. Compare against values computed in Python.

**Results.**
- **369 formula cells, 0 errors, 0 without a value.**
- **Stats `COUNTIFS` and `SUM` are all correct** (per category 9 train / 1 val per language; totals 90/10).
- **Caveat: `line_count` (column F) evaluates to 1 in LibreOffice** for every row (real values are 4 to 16), so the Stats "avg lines" row shows 1.0 instead of 10.0.
  - Probe: `"a\nb\nc"` gives LEN = 3 and the second character is `b`. This is the same for inline strings, shared strings, CRLF, `&#10;`, and `xml:space="preserve"`, so it is not a storage problem.
  - **The line breaks themselves are preserved.** LibreOffice holds them as separate `<text:p>` paragraphs, and a LibreOffice save back to xlsx returns `a\nb\nc` intact, so editing in LibreOffice does not damage the verse.
  - Conclusion: LibreOffice's text functions ignore the paragraph breaks in this headless evaluation. Excel's `LEN`/`SUBSTITUTE(…,CHAR(10),…)` counts them as expected (not verified here; Excel is not installed).
- **The training data is unaffected.** `export_jsonl.py` reads the text with openpyxl, where the breaks are present.

**Open question for the user.** In the LibreOffice window, check whether Stats cell B14 shows 10.0 or 1.0. If 1.0, either ignore the `line_count` column (it is informational only) or replace it with static numbers.

---

## 8. Recommended next steps (in priority order)
1. **Run the notebook on Kaggle** in the default mode. Report any traceback, and the mask check output ("mask check ok…").
2. If the Gemma 4 loss is `nan`/0 or the run is unstable, **switch to `qwen3-4b`** and restart the Kaggle session, because the install pins differ.
3. **Read `comparison.md`.** Watch validation loss: if it rises after epoch 1, lower `EPOCHS` (small data overfits fast).
4. **Native review of the Persian verse,** then mark rows `approved` in the workbook.
5. **Grow the dataset** toward 1,000+ rows, then export again and train again.
6. Try `ddp` mode only after the default works.
7. Optional: export GGUF (Q8_0 for Gemma 4) for Ollama or llama.cpp, watching Kaggle's 20 GB output limit.

### Troubleshooting quick reference
| Symptom | Likely cause and fix |
|---|---|
| "Only 1 GPU found" | Accelerator not set to T4 x2 |
| `no all_train.jsonl found under /kaggle/input` | Dataset not attached, or JSONL files not uploaded |
| pip resolver errors or import errors | Kaggle image drift; compare against Unsloth's latest Kaggle notebook for that model and update the pins in cell 4 |
| `AssertionError` from the mask check | `train_on_responses_only` did not detect the template markers. Pass `instruction_part`/`response_part` explicitly (Gemma 4: `"<|turn>user\n"` / `"<|turn>model\n"`; Qwen: `"<|im_start|>user\n"` / `"<|im_start|>assistant\n"`) |
| Loss `nan` with Gemma 4 | fp16 instability on T4; use the `qwen3-4b` preset |
| Out of memory | `BATCH_SIZE = 1`, and/or `MAX_SEQ_LEN = 768` |
| DDP hangs or crashes | Use `GPU_MODE = "train+baseline"` |
| Answers contain `<think>` tags | `clean_answer` should strip them; check the template |

---

## 9. Sources used
- Persian text: Ganjoor, https://ganjoor.net/ferdousi/shahname
- English: Warner & Warner, *The Shahnama of Firdausi* (1905 to 1925), https://archive.org/details/in.ernet.dli.2015.461512 ; Zimmern, *The Epic of Kings* (1883), https://classics.mit.edu/Ferdowsi/kings.html
- German: Rückert, *Firdosi's Königsbuch* (1890 to 1895), https://archive.org/details/firdosisknigsbu01bayegoog ; Schack, *Heldensagen von Firdusi* (1851), https://books.google.com/books/about/Heldensagen_von_Firdusi_metrisch_%C3%BCbers.html?id=97kOAAAAQAAJ
- Modern translations such as Dick Davis (2006) are copyrighted; they were not used.
- Unsloth docs:
  - https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth
  - https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp
  - https://unsloth.ai/docs/models/gemma-4/train
  - https://unsloth.ai/docs/get-started/unsloth-notebooks
- Unsloth notebooks repo: https://github.com/unslothai/notebooks
- Issues and PRs: https://github.com/unslothai/unsloth/issues/5178 , https://github.com/unslothai/unsloth/pull/10270
- Model: https://huggingface.co/unsloth/gemma-4-E4B-it-unsloth-bnb-4bit
- Community 2x T4 example: https://www.kaggle.com/code/nguyenit67/unsloth-finetuning-multiple-gpus-2x-t4-on-kaggle

---

## 10. Environment and working notes
- **Machine:** Windows 11 Home, PowerShell (plus a Git Bash tool). Python 3.13.3, openpyxl 3.1.5, pandas 2.2.3, transformers 4.48.3 locally. No local GPU training.
- **Safety hook:** a hook blocks recursive deletes (`Remove-Item -Recurse`, `rm -r`); delete specific files instead. Set `PYTHONDONTWRITEBYTECODE=1` when running tests, to avoid `__pycache__` folders in the project.
- **User preferences seen this session:**
  - uses the "superpowers" and ECC global rules (TDD, research before coding, verify before claiming done);
  - turned on the terse "caveman" reply style partway through;
  - writes informal English.
- **Claude Code memory files:** `C:\Users\angiz\.claude\projects\C--Users-angiz-Desktop-AI-workers-Claude-Code-Fine-tuning\memory\` (`shahnameh-slm-dataset.md`, `shahnameh-training-plan.md`).
