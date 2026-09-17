"""Fine-tune SmolLM2-360M with response-only loss on two Kaggle T4 GPUs."""
import json
import math
import os
from pathlib import Path

MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
MAX_LENGTH = 1024


def prompt_text(messages):
    return "".join(f"### {m['role'].capitalize()}:\n{m['content']}\n\n" for m in messages) + "### Assistant:\n"


def encode_example(row, tokenizer):
    messages = row["messages"]
    if [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("Expected system/user/assistant messages")
    prompt = tokenizer.encode(prompt_text(messages[:-1]), add_special_tokens=False)
    answer = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
    tokens = prompt + answer + [tokenizer.eos_token_id]
    if len(tokens) > MAX_LENGTH:
        raise ValueError(f"Example has {len(tokens)} tokens; raise MAX_LENGTH rather than silently truncate verse")
    if not answer:
        raise ValueError("Empty assistant response")
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens),
            "labels": [-100] * len(prompt) + answer + [tokenizer.eos_token_id]}


def main():
    import shutil
    import torch
    import torch.distributed as dist
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, DataCollatorForSeq2Seq, set_seed)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available() or world_size != 2:
        raise RuntimeError("Select GPU T4 x2 and launch with torchrun --nproc_per_node=2")
    torch.cuda.set_device(local_rank)
    set_seed(42)
    work = Path(os.environ.get("SMOLLM_WORK", "/kaggle/working"))
    data = work / "shahnameh-data"
    output = work / "smollm2-shahnameh-model"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # Save the exact training format for downstream apply_chat_template calls.
    tokenizer.chat_template = "{% for message in messages %}{{ '### ' + message['role'].capitalize() + ':\n' + message['content'] }}{% if message['role'] == 'assistant' %}{{ eos_token }}{% else %}{{ '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '### Assistant:\n' }}{% endif %}"
    rows = {split: [json.loads(line) for line in (data / f"all_{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            for split in ("train", "val")}
    encoded = {split: [encode_example(row, tokenizer) for row in items] for split, items in rows.items()}
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32,
                                               attn_implementation="sdpa")
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    args = TrainingArguments(
        output_dir=str(work / "smollm2-checkpoints"), num_train_epochs=3,
        learning_rate=2e-5, per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=8,
        fp16=True, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, warmup_ratio=0.1, weight_decay=0.01,
        logging_steps=1, logging_nan_inf_filter=False, max_grad_norm=1.0,
        optim="adamw_torch", report_to="none", seed=42, data_seed=42,
        ddp_find_unused_parameters=False, dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=encoded["train"],
                      eval_dataset=encoded["val"], processing_class=tokenizer,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True,
                                                          pad_to_multiple_of=8, label_pad_token_id=-100))
    baseline = trainer.evaluate(metric_key_prefix="baseline")
    if not math.isfinite(baseline["baseline_loss"]):
        raise RuntimeError(f"Non-finite baseline loss: {baseline}")
    trainer.train()
    metrics = trainer.evaluate()
    if not math.isfinite(metrics["eval_loss"]):
        raise RuntimeError(f"Non-finite final loss: {metrics}")
    for lang in ("fa", "en", "de"):
        lang_rows = [json.loads(line) for line in (data / f"{lang}_val.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        metrics.update(trainer.evaluate([encode_example(row, tokenizer) for row in lang_rows], metric_key_prefix=f"eval_{lang}"))
    trainer.save_model(str(output))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output)
        (output / "training_summary.json").write_text(json.dumps({
            "model": MODEL_ID, "world_size": world_size,
            "train_rows": len(rows["train"]), "validation_rows": len(rows["val"]),
            "baseline": baseline, "final": metrics, "history": trainer.state.log_history,
            "best_checkpoint": trainer.state.best_model_checkpoint,
        }, indent=2), encoding="utf-8")
        shutil.make_archive(str(output), "zip", output)
        print(f"Saved model and tokenizer: {output}.zip", flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
