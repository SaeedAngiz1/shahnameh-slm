"""Build a self-contained Kaggle notebook using the existing JSONL exports."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cells = []


def cell(kind, text):
    item = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True),
            "id": f"cell-{len(cells):02d}"}
    if kind == "code":
        ast.parse(text)
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell("markdown", """# SmolLM2-360M · Shahnameh fine-tuning
Train the requested **HuggingFaceTB/SmolLM2-360M base model** on the existing Persian, English, and German examples.

1. Complete Kaggle phone verification, select **Settings → Accelerator → GPU T4 x2**, and enable Internet.
2. Click **Run All**. Both GPUs train together using PyTorch distributed data parallel.
3. Download `/kaggle/working/smollm2-shahnameh-model.zip` after training.

The eight original JSONL files are embedded below, so no separate dataset upload is needed.
This performs full fine-tuning with assistant-only loss, three epochs, FP16 mixed precision,
and selects the checkpoint with the lowest validation loss. It evaluates before training and after training,
including separate validation losses for all three languages.

SmolLM2 is primarily English; Persian and German quality must be judged from generated answers.
The base model has no instruction-tuned chat format, so we train and save an explicit role-based template.

Sources: [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M),
[Trainer documentation](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/trainer).
""")
cell("code", """import subprocess, sys
import torch
assert torch.cuda.device_count() == 2, "Select GPU T4 x2 first. Kaggle may require phone verification."
for i in range(2):
    print(i, torch.cuda.get_device_name(i))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'transformers==4.56.2', 'accelerate==1.10.1'], check=True)
""")
payload = {}
hashes = {}
for path in sorted((ROOT / "training_data").glob("*.jsonl")):
    raw = path.read_bytes()
    payload[path.name] = raw.decode("utf-8")
    hashes[path.name] = hashlib.sha256(raw).hexdigest()
cell("markdown", "## Prepared dataset\n270 training examples and 30 validation examples. Embedded text below reproduces the existing JSONL files exactly.")
cell("code", "from pathlib import Path\nimport hashlib, json\nWORK = Path('/kaggle/working')\nDATA = WORK / 'shahnameh-data'\nDATA.mkdir(exist_ok=True)\n" +
     "DATA_FILES = " + repr(payload) + "\nEXPECTED_HASHES = " + repr(hashes) + "\n" +
     "for name, text in DATA_FILES.items():\n    raw = text.encode('utf-8')\n    assert hashlib.sha256(raw).hexdigest() == EXPECTED_HASHES[name]\n    (DATA / name).write_bytes(raw)\n    print(name, len(text.splitlines()), 'examples')\n")
script = (ROOT / "kaggle" / "train_smollm2.py").read_text(encoding="utf-8")
ast.parse(script)
cell("markdown", "## Training program\nUses both T4s, preserves complete responses, masks prompts and padding, and saves the best model with its tokenizer.")
cell("code", "TRAINING_SCRIPT = " + repr(script) + "\n(WORK / 'train_smollm2.py').write_text(TRAINING_SCRIPT, encoding='utf-8')\n")
cell("code", """import os
env = dict(os.environ, TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--nproc_per_node=2', str(WORK / 'train_smollm2.py')], env=env, check=True)
""")
cell("markdown", "## Inspect results and try a question\nThe ZIP contains the complete model, tokenizer, saved chat template, and training metrics.")
cell("code", """from IPython.display import FileLink, display
summary = json.loads((WORK / 'smollm2-shahnameh-model' / 'training_summary.json').read_text())
print(json.dumps(summary['final'], indent=2))
display(FileLink('/kaggle/working/smollm2-shahnameh-model.zip'))
""")
cell("code", """# Inference runs in a fresh process so it uses the installed Transformers version.
INFERENCE = r'''
import json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
path = '/kaggle/working/smollm2-shahnameh-model'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).to('cuda:0').eval()
for lang in ('fa', 'en', 'de'):
    row = json.loads(Path(f'/kaggle/working/shahnameh-data/{lang}_val.jsonl').read_text().splitlines()[0])
    prompt = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=400, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    print('\\nLANGUAGE:', lang, '\\nQUESTION:', row['messages'][-2]['content'])
    print(tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True))
'''
subprocess.run([sys.executable, '-c', INFERENCE], check=True)
""")
notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "kaggle" / "smollm2_360m_shahnameh_kaggle.ipynb"
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(out)
