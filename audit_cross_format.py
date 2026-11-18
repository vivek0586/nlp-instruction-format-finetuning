"""Audit crossed prompt evaluations against the existing matched-format results."""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
original = zipfile.ZipFile(ROOT / "colab_results.zip")
crossed = zipfile.ZipFile(ROOT / "cross_eval_results.zip")
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
TASKS = {"emotion": (EMOTIONS, 2000), "sentiment": (["negative", "positive"], 872)}


def records(archive, name):
    return [json.loads(line) for line in archive.read(name).decode("utf-8").splitlines()]


def metrics(rows, labels):
    n = len(rows)
    class_f1 = {}
    for label in labels:
        tp = sum(r["gold"] == label and r["prediction"] == label for r in rows)
        fp = sum(r["gold"] != label and r["prediction"] == label for r in rows)
        fn = sum(r["gold"] == label and r["prediction"] != label for r in rows)
        class_f1[label] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "n": n,
        "accuracy": sum(r["gold"] == r["prediction"] for r in rows) / n,
        "macro_f1": sum(class_f1.values()) / len(labels),
        "invalid_rate": sum(r["prediction"] not in labels for r in rows) / n,
        "per_class_f1": class_f1,
    }


summary = {}
for fmt in ("natural", "compact"):
    opposite = "compact" if fmt == "natural" else "natural"
    for seed in (17, 42, 73):
        run = f"{fmt}_seed{seed}"
        config_path = f"results/{run}/cross_eval_config.json"
        if config_path not in crossed.namelist():
            print("MISSING", run)
            continue
        config = json.loads(crossed.read(config_path))
        assert config["training_format"] == fmt and config["evaluation_format"] == opposite
        assert config["seed"] == seed
        summary[run] = {}
        for task, (labels, expected) in TASKS.items():
            matched = records(original, f"results/{run}/{task}_adapted_predictions.jsonl")
            cross_adapted = records(crossed, f"results/{run}/{task}_cross_adapted_predictions.jsonl")
            cross_base = records(crossed, f"results/{run}/{task}_cross_base_predictions.jsonl")
            other_base = records(original, f"results/{opposite}_seed{seed}/{task}_base_predictions.jsonl")
            assert len(matched) == len(cross_adapted) == len(cross_base) == expected
            identities = lambda rows: [(r["text"], r["gold"]) for r in rows]
            assert identities(matched) == identities(cross_adapted) == identities(cross_base)
            # Base prompts should reproduce the existing baseline for that prompt style.
            assert [(r["prediction"], r["raw_generation"]) for r in cross_base] == [
                (r["prediction"], r["raw_generation"]) for r in other_base
            ], (run, task, "cross baseline differs from prior same-prompt baseline")
            row = {}
            for name, rows in (("matched_adapted", matched), ("cross_adapted", cross_adapted), ("cross_base", cross_base)):
                computed = metrics(rows, labels)
                if name != "matched_adapted":
                    saved = json.loads(crossed.read(f"results/{run}/{task}_{name}_metrics.json"))
                    for key in ("accuracy", "macro_f1", "invalid_rate"):
                        assert abs(computed[key] - saved[key]) < 1e-10, (run, task, name, key)
                row[name] = computed
            base_ok = [r["prediction"] == r["gold"] for r in cross_base]
            adapted_ok = [r["prediction"] == r["gold"] for r in cross_adapted]
            paired = json.loads(crossed.read(f"results/{run}/{task}_cross_paired_analysis.json"))
            assert paired["wrong_to_correct"] == sum(not a and b for a, b in zip(base_ok, adapted_ok))
            assert paired["correct_to_wrong"] == sum(a and not b for a, b in zip(base_ok, adapted_ok))
            row["delta_cross_minus_matched_pp"] = 100 * (row["cross_adapted"]["accuracy"] - row["matched_adapted"]["accuracy"])
            row["delta_cross_minus_base_pp"] = 100 * (row["cross_adapted"]["accuracy"] - row["cross_base"]["accuracy"])
            summary[run][task] = row

out = ROOT / "cross_format_audit.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("Audited", len(summary), "crossed adapter runs; wrote", out)
for run, tasks in summary.items():
    for task, row in tasks.items():
        print(run, task,
              f'matched={100*row["matched_adapted"]["accuracy"]:.2f}%',
              f'crossed={100*row["cross_adapted"]["accuracy"]:.2f}%',
              f'cross-prompt base={100*row["cross_base"]["accuracy"]:.2f}%')
