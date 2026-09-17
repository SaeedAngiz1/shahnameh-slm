"""Ask the fine-tuned SmolLM2 Shahnameh model a question, on CPU.

    python ask_smollm2.py "Who was Rostam?"
    python ask_smollm2.py --lang de "Wer war Rostam?"

English output is the usable one; Persian and German are known to be weak.
Uses the corrected generation defaults (repetition_penalty), without which the
model loops forever instead of stopping.
"""
import argparse
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "smollm2-shahnameh-model"

SYSTEM = {
    "en": "You are a poet who answers in the voice of Ferdowsi's Shahnameh, in English verse.",
    "fa": "تو شاعری هستی که به شیوه‌ی شاهنامه‌ی فردوسی، به شعر فارسی پاسخ می‌دهی.",
    "de": "Du bist ein Dichter, der in der Stimme von Ferdowsis Schahname in deutschen Versen antwortet.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+", help="the question to ask")
    ap.add_argument("--lang", default="en", choices=sorted(SYSTEM), help="system-prompt language")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = deterministic")
    args = ap.parse_args()

    if not MODEL_DIR.exists():
        sys.exit(f"model not found at {MODEL_DIR}\nUnpack smollm2-shahnameh-model.zip first.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("loading model (a few seconds on CPU)...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()

    question = " ".join(args.question)
    messages = [{"role": "system", "content": SYSTEM[args.lang]},
                {"role": "user", "content": question}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)

    sampling = ({"do_sample": False} if args.temperature <= 0
                else {"do_sample": True, "temperature": args.temperature, "top_p": 0.9})
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                             pad_token_id=tok.eos_token_id, **sampling)

    answer = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nQ: {question}\n")
    print(answer.strip())


if __name__ == "__main__":
    main()
