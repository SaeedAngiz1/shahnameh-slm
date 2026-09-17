"""Evaluate and export the completed Kaggle checkpoint without distributed collectives."""
import gc
import json
import math
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
REPETITION_PENALTY = 1.15
WORK = Path('/kaggle/working')
states = list(Path('/kaggle/input').rglob('trainer_state.json'))
assert states, 'Attach the output of the completed SmolLM2 training notebook.'
state_path = max(states, key=lambda p: json.loads(p.read_text())['global_step'])
state = json.loads(state_path.read_text())
assert state['global_step'] == 51 and state['epoch'] == 3.0, state
checkpoint = state_path.parent.parent / Path(state['best_model_checkpoint']).name
assert (checkpoint / 'model.safetensors').exists(), checkpoint
data = next(Path('/kaggle/input').rglob('all_val.jsonl')).parent
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
assert tokenizer.chat_template, 'Missing saved training template'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Recovered:', checkpoint, 'steps:', state['global_step'], 'device:', device, flush=True)


def evaluate(model, rows):
    losses = []
    for row in rows:
        messages = row['messages']
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        answer = tokenizer.encode(messages[-1]['content'], add_special_tokens=False) + [tokenizer.eos_token_id]
        ids = torch.tensor([prefix + answer], device=device)
        labels = torch.tensor([[-100] * len(prefix) + answer], device=device)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=device.startswith('cuda')):
            loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss.item()
        assert math.isfinite(loss), loss
        losses.append(loss)
    return sum(losses) / len(losses)


rows = {lang: [json.loads(line) for line in (data / f'{lang}_val.jsonl').read_text().splitlines()]
        for lang in ('all', 'fa', 'en', 'de')}
baseline_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device).eval()
baseline = {lang: evaluate(baseline_model, items) for lang, items in rows.items()}
print('Baseline:', baseline, flush=True)
del baseline_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32).to(device).eval()
assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
final = {lang: evaluate(model, items) for lang, items in rows.items()}
print('Final:', final, flush=True)
output = WORK / 'smollm2-shahnameh-model'
output.mkdir(exist_ok=True)
model.config.use_cache = True
model.save_pretrained(output, safe_serialization=True)
tokenizer.save_pretrained(output)
summary = {'model': MODEL_ID, 'training_gpus': 2, 'train_rows': 270, 'validation_rows': 30,
           'epochs': state['epoch'], 'global_step': state['global_step'],
           'best_checkpoint': checkpoint.name, 'baseline_loss': baseline, 'final_loss': final,
           'metric': 'mean of per-example assistant-token cross entropy; FP16 autocast',
           'recovery': 'Training completed; original DDP job timed out after checkpoint loading. Exported preserved best checkpoint.',
           'history': state['log_history']}
(output / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
samples = []
for lang in ('fa', 'en', 'de'):
    row = rows[lang][0]
    prompt = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(device)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=device.startswith('cuda')):
        # Greedy decoding degenerates into a self-reinforcing line loop on this
        # checkpoint and never reaches EOS (0/30 held-out prompts terminated).
        # The model has learned to stop: teacher-forced on gold answers, EOS is
        # the top-1 prediction at the end position in 17/30 cases. Repetition
        # control is what lets generation reach that state (25/30 terminate).
        result = model.generate(**inputs, max_new_tokens=300, do_sample=False,
                                repetition_penalty=REPETITION_PENALTY,
                                pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(result[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    samples.append({'language': lang, 'question': row['messages'][-2]['content'], 'answer': answer})
    print(json.dumps(samples[-1], ensure_ascii=False), flush=True)
(output / 'sample_answers.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding='utf-8')
archive = shutil.make_archive(str(output), 'zip', output)
print('EXPORTED:', archive, Path(archive).stat().st_size, 'bytes', flush=True)
