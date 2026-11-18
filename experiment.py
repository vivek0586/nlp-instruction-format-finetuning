"""LoRA SFT experiment for prompt-format sensitivity on DAIR.AI Emotion.

This script deliberately saves every prediction, including invalid generations,
so evaluation cannot hide parsing failures. No test-set tuning is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DATASET_ID = "dair-ai/emotion"
SST2_DATASET_ID = "nyu-mll/glue"
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
TASK_LABELS = {"emotion": LABELS, "sentiment": ["negative", "positive"]}
FORMATS = {
    "natural": "Classify the {task} expressed in the text. Choose exactly one label from: {labels}.\nText: {text}\n{task_title}:",
    "compact": "Task: {task}-classification | Labels: {labels}\nInput: {text}\nLabel:",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prompt_for(text: str, fmt: str, task: str = "emotion") -> str:
    labels = ", ".join(TASK_LABELS[task])
    return FORMATS[fmt].format(text=text.strip(), task=task, task_title=task.title(), labels=labels)


def extract_label(generation: str, labels=LABELS) -> str | None:
    normalized = generation.lower().strip().replace("<|endoftext|>", "")
    # Exact first-line matching prevents accidental substring matches such as
    # "joy" inside an explanation. A second pass accepts a label followed by punctuation.
    first = normalized.splitlines()[0].strip() if normalized.splitlines() else ""
    first = first.strip(" .,:;!?`*_\"'()[]{}")
    if first in labels:
        return first
    for label in labels:
        if first.startswith(label + " ") or first.startswith(label + ".") or first.startswith(label + ","):
            return label
    return None


def load_model_and_tokenizer(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model.to(device)
    return model, tokenizer


def build_features(rows, tokenizer, fmt: str, max_length: int):
    features = []
    for row in rows:
        prompt = prompt_for(row["text"], fmt)
        target = LABELS[int(row["label"])]
        prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length - 2)["input_ids"]
        target_ids = tokenizer(" " + target, add_special_tokens=False)["input_ids"]
        ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
        ids = ids[:max_length]
        labels = [-100] * min(len(prompt_ids), len(ids)) + ids[len(prompt_ids):]
        labels = labels[:len(ids)]
        features.append({"input_ids": ids, "labels": labels})
    return features


def emotion_rows(split):
    return [{"text": row["text"], "gold": LABELS[int(row["label"])]} for row in split]


def sentiment_rows(split):
    # SST-2 test labels are hidden; use its labeled validation split as an
    # untouched secondary-task evaluation, never for model optimization.
    return [{"text": row["sentence"], "gold": ["negative", "positive"][int(row["label"])]} for row in split]


def collate(batch, pad_id: int):
    width = max(len(item["input_ids"]) for item in batch)
    ids, masks, labels = [], [], []
    for item in batch:
        pad = width - len(item["input_ids"])
        ids.append(item["input_ids"] + [pad_id] * pad)
        masks.append([1] * len(item["input_ids"]) + [0] * pad)
        labels.append(item["labels"] + [-100] * pad)
    return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks), "labels": torch.tensor(labels)}


@torch.inference_mode()
def predict(model, tokenizer, rows, fmt: str, device: torch.device, batch_size: int, task: str = "emotion"):
    model.eval()
    # Decoder-only generation should left-pad variable-length prompts so the
    # final input position is a real prompt token for every batch item.
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    records = []
    labels = TASK_LABELS[task]
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        prompts = [prompt_for(r["text"], fmt, task) for r in batch_rows]
        tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        generated = model.generate(**tokens, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        prefix_width = tokens["input_ids"].shape[1]
        for row, output in zip(batch_rows, generated):
            decoded = tokenizer.decode(output[prefix_width:], skip_special_tokens=True).strip()
            gold = row["gold"]
            records.append({"text": row["text"], "gold": gold, "prediction": extract_label(decoded, labels), "raw_generation": decoded})
    tokenizer.padding_side = old_padding_side
    return records


def summarize(records, labels=LABELS):
    y_true = [r["gold"] for r in records]
    y_pred = [r["prediction"] if r["prediction"] in labels else "__invalid__" for r in records]
    acc = accuracy_score(y_true, y_pred)
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels + ["__invalid__"])
    class_metrics = {label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])} for i, label in enumerate(labels)}
    result = {"n": len(records), "accuracy": float(acc), "macro_f1": float(np.mean(f)), "invalid_rate": float(sum(v == "__invalid__" for v in y_pred) / len(y_pred)), "per_class": class_metrics}
    return result, cm


def write_evaluation(records, out_dir: Path, tag: str, labels=LABELS):
    metrics, cm = summarize(records, labels)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out_dir / f"{tag}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (out_dir / f"{tag}_confusion.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gold/predicted"] + labels + ["__invalid__"])
        for label, row in zip(labels + ["__invalid__"], cm):
            writer.writerow([label] + row.tolist())
    return metrics


def train(args, device, out_dir: Path):
    dataset = load_dataset(DATASET_ID, "split")
    train_rows = list(dataset["train"])
    if args.max_train:
        train_rows = train_rows[:args.max_train]
    model, tokenizer = load_model_and_tokenizer(device)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    features = build_features(train_rows, tokenizer, args.format, args.max_length)
    loader = DataLoader(features, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    model.train()
    step = 0
    for epoch in range(args.epochs):
        total = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            total += float(loss.detach())
        print(f"epoch={epoch + 1} mean_loss={total / max(1, len(loader)):.4f} steps={step}")
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    model = model.merge_and_unload()
    emotion_test = emotion_rows(dataset["test"])
    sst2 = load_dataset(SST2_DATASET_ID, "sst2")
    sentiment_eval = sentiment_rows(sst2["validation"])
    adapted_emotion = predict(model, tokenizer, emotion_test, args.format, device, args.eval_batch_size, "emotion")
    adapted_sentiment = predict(model, tokenizer, sentiment_eval, args.format, device, args.eval_batch_size, "sentiment")
    # Release accelerator memory before loading the separate base-model baseline.
    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    base_model, base_tokenizer = load_model_and_tokenizer(device)
    base_emotion = predict(base_model, base_tokenizer, emotion_test, args.format, device, args.eval_batch_size, "emotion")
    base_sentiment = predict(base_model, base_tokenizer, sentiment_eval, args.format, device, args.eval_batch_size, "sentiment")
    for task, rows, base_records, adapted_records, labels in (
        ("emotion", emotion_test, base_emotion, adapted_emotion, LABELS),
        ("sentiment", sentiment_eval, base_sentiment, adapted_sentiment, TASK_LABELS["sentiment"]),
    ):
        print(task, "base", json.dumps(write_evaluation(base_records, out_dir, f"{task}_base", labels), indent=2))
        print(task, "adapted", json.dumps(write_evaluation(adapted_records, out_dir, f"{task}_adapted", labels), indent=2))
        write_paired_analysis(base_records, adapted_records, out_dir, args.seed, task, labels)
        if task == "emotion":
            write_slice_metrics(rows, base_records, adapted_records, out_dir)


def write_slice_metrics(rows, base_records, adapted_records, out_dir):
    lengths = np.array([len(r["text"].split()) for r in rows])
    cuts = np.quantile(lengths, [0.25, 0.5, 0.75])
    slices = {"q1_short": lengths <= cuts[0], "q2": (lengths > cuts[0]) & (lengths <= cuts[1]), "q3": (lengths > cuts[1]) & (lengths <= cuts[2]), "q4_long": lengths > cuts[2]}
    out = {}
    for name, mask in slices.items():
        indices = np.flatnonzero(mask)
        out[name] = {"n": int(len(indices)), "mean_words": float(lengths[mask].mean()) if len(indices) else None}
        for tag, records in (("base", base_records), ("adapted", adapted_records)):
            subset = [records[i] for i in indices]
            metrics, _ = summarize(subset, LABELS)
            out[name][tag] = {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "invalid_rate": metrics["invalid_rate"]}
    (out_dir / "length_slices.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def write_paired_analysis(base_records, adapted_records, out_dir, seed, task, labels, output_stem=None):
    if len(base_records) != len(adapted_records):
        raise ValueError("Paired predictions have different lengths")
    gold = np.array([r["gold"] for r in base_records])
    base_ok = np.array([r["prediction"] == r["gold"] for r in base_records])
    tuned_ok = np.array([r["prediction"] == r["gold"] for r in adapted_records])
    delta = tuned_ok.astype(float) - base_ok.astype(float)
    rng = np.random.default_rng(seed)
    boot = np.empty(10000, dtype=float)
    for i in range(len(boot)):
        idx = rng.integers(0, len(delta), len(delta))
        boot[i] = delta[idx].mean()
    out = {
        "n": int(len(gold)),
        "accuracy_delta_adapted_minus_base": float(delta.mean()),
        "paired_bootstrap_95pct": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
        "wrong_to_correct": int(np.sum(~base_ok & tuned_ok)),
        "correct_to_wrong": int(np.sum(base_ok & ~tuned_ok)),
        "unchanged_correct": int(np.sum(base_ok & tuned_ok)),
        "unchanged_wrong": int(np.sum(~base_ok & ~tuned_ok)),
        "per_class_flips": {},
    }
    for label in labels:
        mask = gold == label
        out["per_class_flips"][label] = {
            "n": int(mask.sum()),
            "wrong_to_correct": int(np.sum(mask & ~base_ok & tuned_ok)),
            "correct_to_wrong": int(np.sum(mask & base_ok & ~tuned_ok)),
        }
    output_stem = output_stem or task
    (out_dir / f"{output_stem}_paired_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Keep a deterministic audit sample in both directions for qualitative analysis.
    flips = []
    for i, (base, tuned) in enumerate(zip(base_records, adapted_records)):
        direction = "wrong_to_correct" if not base_ok[i] and tuned_ok[i] else "correct_to_wrong" if base_ok[i] and not tuned_ok[i] else None
        if direction:
            flips.append({"direction": direction, "gold": gold[i], "text": base["text"], "base_prediction": base["prediction"], "base_raw": base["raw_generation"], "adapted_prediction": tuned["prediction"], "adapted_raw": tuned["raw_generation"], "stable_order": hashlib.sha256(base["text"].encode("utf-8")).hexdigest()})
    flips.sort(key=lambda x: x["stable_order"])
    (out_dir / f"{output_stem}_flip_examples.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in flips[:40]) + ("\n" if flips else ""), encoding="utf-8")


def cross_eval(args, device, out_dir: Path):
    """Evaluate an existing adapter with the prompt format it did not see in training."""
    adapter_dir = Path(args.adapter_path) if args.adapter_path else out_dir / "adapter"
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Adapter missing at {adapter_dir}. The results ZIP excludes weights. "
            "Restore the saved adapter folder or retrain this run before cross-evaluation."
        )
    test_format = "compact" if args.format == "natural" else "natural"
    emotion_test = emotion_rows(load_dataset(DATASET_ID, "split")["test"])
    sentiment_test = sentiment_rows(load_dataset(SST2_DATASET_ID, "sst2")["validation"])
    model, tokenizer = load_model_and_tokenizer(device)
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    adapted = {}
    for task, rows, labels in (
        ("emotion", emotion_test, LABELS),
        ("sentiment", sentiment_test, TASK_LABELS["sentiment"]),
    ):
        records = predict(model, tokenizer, rows, test_format, device, args.eval_batch_size, task)
        adapted[task] = records
        print(task, "cross adapted", json.dumps(write_evaluation(records, out_dir, f"{task}_cross_adapted", labels), indent=2))
    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    base_model, base_tokenizer = load_model_and_tokenizer(device)
    for task, rows, labels in (
        ("emotion", emotion_test, LABELS),
        ("sentiment", sentiment_test, TASK_LABELS["sentiment"]),
    ):
        base_records = predict(base_model, base_tokenizer, rows, test_format, device, args.eval_batch_size, task)
        print(task, "cross base", json.dumps(write_evaluation(base_records, out_dir, f"{task}_cross_base", labels), indent=2))
        write_paired_analysis(base_records, adapted[task], out_dir, args.seed, task, labels, f"{task}_cross")
    (out_dir / "cross_eval_config.json").write_text(json.dumps({
        "training_format": args.format, "evaluation_format": test_format,
        "seed": args.seed, "adapter_path": str(adapter_dir.resolve()),
        "model": MODEL_ID, "emotion_split": "test", "sentiment_split": "validation",
        "eval_batch_size": args.eval_batch_size, "generation": "greedy, max_new_tokens=8",
    }, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "train", "cross-eval"], required=True)
    p.add_argument("--format", choices=sorted(FORMATS), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--adapter-path", type=str, default=None,
                   help="Saved adapter folder for --mode cross-eval; defaults to results/<format>_seed<seed>/adapter")
    args = p.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.mode == "train":
        raise RuntimeError("Training on CPU is not practical for this experiment. Use a CUDA GPU.")
    out_dir = Path("results") / f"{args.format}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode != "cross-eval":
        (out_dir / "run_config.json").write_text(json.dumps({"model": MODEL_ID, "training_dataset": DATASET_ID, "secondary_evaluation_dataset": f"{SST2_DATASET_ID}/sst2 validation", "format": args.format, "seed": args.seed, "max_train": args.max_train or "all", "epochs": args.epochs, "batch_size": args.batch_size, "max_length": args.max_length, "lora_r": args.lora_r, "lr": args.lr, "device": str(device)}, indent=2), encoding="utf-8")
    if args.mode == "baseline":
        dataset = load_dataset(DATASET_ID, "split")
        model, tokenizer = load_model_and_tokenizer(device)
        emotion_eval = emotion_rows(dataset["test"])
        sst2 = load_dataset(SST2_DATASET_ID, "sst2")
        sentiment_eval = sentiment_rows(sst2["validation"])
        for task, rows, labels in (("emotion", emotion_eval, LABELS), ("sentiment", sentiment_eval, TASK_LABELS["sentiment"])):
            records = predict(model, tokenizer, rows, args.format, device, args.eval_batch_size, task)
            print(task, json.dumps(write_evaluation(records, out_dir, f"{task}_base", labels), indent=2))
    elif args.mode == "cross-eval":
        cross_eval(args, device, out_dir)
    else:
        train(args, device, out_dir)


if __name__ == "__main__":
    main()
