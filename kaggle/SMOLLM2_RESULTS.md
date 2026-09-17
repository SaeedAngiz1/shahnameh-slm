# SmolLM2-360M Shahnameh training

## Training

- Exact base model: `HuggingFaceTB/SmolLM2-360M`.
- 270 training examples; 30 validation examples across Persian, English, German.
- Full fine-tuning, two Tesla T4 GPUs, three epochs, 51 optimizer steps.
- Best checkpoint: `checkpoint-51`.
- Original private run: https://www.kaggle.com/code/saeed010/smollm2-360m-shahnameh-fine-tuning?scriptVersionId=350181097

Training completed, but the original distributed job timed out during an ALLGATHER after loading the best checkpoint. Kaggle preserved the checkpoint. The original training notebook has not passed end-to-end export verification; use the recovery export below for this trained model.

## Recovery and evaluation

Recovery notebook: https://www.kaggle.com/code/saeed010/smollm2-shahnameh-model-export?scriptVersionId=350278641

Verified successful: 217.7-second run, subprocess exit code 0, 13 saved output files. Archive size: 1,342,296,888 bytes. The persistent Kaggle download is available; a completed local download has not been verified.

The recovery process loads the saved best checkpoint, checks that all 51 steps and three epochs completed, verifies tied embedding/output weights, evaluates baseline and trained models with the same response-only loss, and saves the full model, tokenizer, chat template, metrics, and sample answers.

| Validation split | Base loss | Trained loss |
| --- | ---: | ---: |
| All 30 examples | 3.030281 | 2.650258 |
| Persian, 10 examples | 2.096059 | 1.751347 |
| English, 10 examples | 3.319725 | 2.955827 |
| German, 10 examples | 3.675059 | 3.243600 |

Loss is the mean of per-example assistant-token cross entropy, measured with FP16 autocast. Lower is better. Combined validation loss decreased about 12.5%.

## Generated answer quality

**Correction (2026-09-16).** An earlier version of this file reported severe repetition in all three languages and concluded the checkpoint was not useful. That conclusion was wrong. The repetition was a decoding fault in the export script, not a defect in the trained model.

### Root cause of the repetition

`sample_answers.json` was produced with pure greedy decoding and no repetition control. Greedy decoding on this checkpoint enters a self-reinforcing line loop and never reaches a state where EOS is likely, so every sample ran to the token cap.

The model did learn to stop. Teacher-forcing each gold held-out answer and reading the next-token distribution at the position where EOS belongs gives mean P(EOS)=0.55, with `<|endoftext|>` the top-1 prediction in 17 of 30 cases. EOS is also correctly supervised in training (`train_smollm2.py`, `labels = [-100]*prompt + answer + [eos]`), and the saved chat template matches the training prompt format exactly.

### Measured effect of decoding, all 30 held-out prompts

`stopped` counts generations that emitted EOS before the 200-token cap. `distinct` is the ratio of unique non-empty lines.

| Language | greedy: stopped | greedy: distinct | rep_penalty 1.15: stopped | rep_penalty 1.15: distinct |
| --- | ---: | ---: | ---: | ---: |
| English | 0/10 | 0.62 | 10/10 | 1.00 |
| German | 0/10 | 0.29 | 8/10 | 1.00 |
| Persian | 0/10 | 0.59 | 7/10 | 1.00 |

With `repetition_penalty=1.15`, 25 of 30 prompts terminate on their own and no language produces a repeated line.

### Quality per language, after the decoding fix

Three different verdicts, not one:

- **English: usable.** Coherent Shahnameh-style verse with correct content. For "Who was Rostam?": *"Rostam was a hero who lived long ago; / He fought against evil and won his way. … When Sohrab came from far away land, / With poison in his heart, Rostam went with him."*
- **German: shape right, substance wrong.** Terminates and stops repeating, but grammar is broken (`Rostam war der Herr des Weltes`) and content drifts off-domain (`Herrn des Weltkrieges`, `Kriegsgefangenen`). Names are hallucinated.
- **Persian: still poor, and this is a separate problem.** Decoding control does not rescue it. Outputs are flat non-verse lines or still-looping fragments. This is consistent with the EOS probe, where Persian was the only language with EOS never ranked top-1 (0/10, always rank 2 behind a newline). The likely cause is the base model: SmolLM2-360M is English-centric with weak Persian pretraining. No decoding change will fix it; it needs a different base model or substantially more Persian data.

The checkpoint is a useful English result and an unsolved Persian one. Keep the supplied validation set out of training.

## Files

The recovery notebook produces `smollm2-shahnameh-model.zip`, containing the full Hugging Face model and tokenizer, `training_summary.json`, and `sample_answers.json`. Use the saved tokenizer's `apply_chat_template(..., add_generation_prompt=True)` when constructing prompts.

The zip is preserved unmodified: 1,342,296,888 bytes, byte-for-byte the Kaggle artifact. Its `generation_config.json` still carries the greedy defaults that caused the repetition, and its `sample_answers.json` still holds the degenerate samples.

`smollm2-shahnameh-model/` in the project root is the unpacked, usable copy. Its `generation_config.json` adds `repetition_penalty: 1.15` and `max_new_tokens: 200`, so a bare `model.generate(...)` terminates correctly with no extra arguments. Verified: loads the defaults and stops on EOS after 59 tokens for the first English held-out question.

Anything generating from the zip directly must pass `repetition_penalty=1.15` explicitly.
