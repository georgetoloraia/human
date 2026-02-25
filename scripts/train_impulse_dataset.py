#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manager.neural_core import ImpulseNeuralCore
from manager.neural_io import WavImpulseEncoder, ImpulseSpeechDecoder


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_no} is not an object")
            rows.append(row)
    return rows


def resolve_wav_path(dataset_path: Path, wav_value: str) -> Path:
    p = Path(wav_value)
    if p.is_absolute():
        return p
    candidate = (dataset_path.parent / p).resolve()
    if candidate.exists():
        return candidate
    # fallback to repo-relative
    return (ROOT / p).resolve()


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def token_edit_distance(a: List[str], b: List[str]) -> int:
    n = len(a)
    m = len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # delete
                dp[i][j - 1] + 1,      # insert
                dp[i - 1][j - 1] + cost,  # substitute/match
            )
    return dp[n][m]


def token_accuracy(pred: List[str], target: List[str]) -> float:
    denom = max(len(pred), len(target), 1)
    matches = sum(1 for p, t in zip(pred, target) if p == t)
    return matches / denom


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch train impulse neural core + decoder on a JSONL dataset of {wav, target_text}.",
        epilog=(
            "Dataset format (JSONL): one object per line with keys: "
            '{"wav":"path/to/file.wav","target_text":"token sequence","id":"optional"}'
        ),
    )
    ap.add_argument("dataset", help="Path to dataset JSONL")
    ap.add_argument("--out-dir", default="artifacts/impulse_batch_train", help="Output directory for model artifacts and reports")
    ap.add_argument("--epochs", type=int, default=3, help="Dataset passes")
    ap.add_argument("--core-output-epochs", type=int, default=4, help="Per-sample output-layer epochs")
    ap.add_argument("--core-output-lr", type=float, default=0.05, help="Per-sample output-layer learning rate")
    ap.add_argument("--hidden-evo-iters", type=int, default=0, help="Per-sample hidden/recurrent evolutionary iterations (0 disables)")
    ap.add_argument("--hidden-evo-pop", type=int, default=4, help="Population for hidden/recurrent evolutionary search")
    ap.add_argument("--hidden-evo-sigma", type=float, default=0.05, help="Mutation sigma for hidden/recurrent search")
    ap.add_argument("--align-mode", choices=["ctc", "dynamic", "anchors"], default="ctc", help="Target/time alignment mode")
    ap.add_argument("--save-every-epoch", action="store_true", help="Write checkpoints after each dataset epoch")
    args = ap.parse_args()

    dataset_path = Path(args.dataset).resolve()
    rows = load_jsonl(dataset_path)
    if not rows:
        raise SystemExit("Dataset is empty")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = WavImpulseEncoder()
    core_path = out_dir / "neural_core.json"
    decoder_map_path = out_dir / "decoder_mapping.json"

    core = ImpulseNeuralCore.load(core_path) if core_path.exists() else ImpulseNeuralCore(input_channels=encoder.input_channels)
    decoder = (
        ImpulseSpeechDecoder.load_mapping(decoder_map_path, output_channels=core.output_channels)
        if decoder_map_path.exists()
        else ImpulseSpeechDecoder(output_channels=core.output_channels)
    )

    report: Dict[str, Any] = {
        "dataset": str(dataset_path),
        "samples": len(rows),
        "epochs": int(args.epochs),
        "align_mode": args.align_mode,
        "history": [],
    }

    for epoch in range(1, int(args.epochs) + 1):
        sample_reports: List[Dict[str, Any]] = []
        output_mse_values: List[float] = []
        hidden_mse_values: List[float] = []
        token_acc_values: List[float] = []
        token_edit_values: List[int] = []
        token_edit_norm_values: List[float] = []
        exact_match_count = 0
        for idx, row in enumerate(rows, start=1):
            wav_value = row.get("wav")
            target_text = str(row.get("target_text", "") or "").strip()
            sample_id = str(row.get("id", f"sample{idx}"))
            if not wav_value or not target_text:
                sample_reports.append({"id": sample_id, "skipped": True, "reason": "missing wav or target_text"})
                continue
            wav_path = resolve_wav_path(dataset_path, str(wav_value))
            if not wav_path.exists():
                sample_reports.append({"id": sample_id, "skipped": True, "reason": f"missing wav file: {wav_path}"})
                continue

            in_impulses = encoder.encode_wav(wav_path)
            target_channels = decoder.target_channels_for_text(target_text)
            hidden_info = None
            if int(args.hidden_evo_iters) > 0 and target_channels:
                hidden_info = core.train_hidden_evolutionary(
                    in_impulses,
                    target_channels,
                    iterations=int(args.hidden_evo_iters),
                    population=int(args.hidden_evo_pop),
                    sigma=float(args.hidden_evo_sigma),
                    align_mode=args.align_mode,
                )
                hidden_mse_values.append(float(hidden_info.get("mse_end", 0.0)))

            output_info = None
            if target_channels:
                output_info = core.train_output_supervised(
                    in_impulses,
                    target_channels,
                    epochs=int(args.core_output_epochs),
                    lr=float(args.core_output_lr),
                    align_mode=args.align_mode,
                )
                output_mse_values.append(float(output_info.get("mse_end", 0.0)))

            out_impulses = core.process(in_impulses)
            decoder_info = decoder.train_from_pair(out_impulses, target_text)
            decoded = decoder.decode(out_impulses)
            pred_tokens = [str(t) for t in (decoded.get("tokens") or [])]
            target_tokens = [t for t in target_text.split() if t]
            edit = token_edit_distance(pred_tokens, target_tokens)
            acc = token_accuracy(pred_tokens, target_tokens)
            norm_edit = edit / max(1, len(target_tokens))
            token_acc_values.append(acc)
            token_edit_values.append(edit)
            token_edit_norm_values.append(norm_edit)
            if pred_tokens == target_tokens:
                exact_match_count += 1

            sample_reports.append(
                {
                    "id": sample_id,
                    "wav": str(wav_path),
                    "target_text": target_text,
                    "input_impulses": in_impulses.summary(),
                    "output_impulses": out_impulses.summary(),
                    "hidden_train": hidden_info,
                    "output_train": output_info,
                    "decoder_train": decoder_info,
                    "pred_tokens": pred_tokens,
                    "token_accuracy": acc,
                    "token_edit_distance": edit,
                    "token_edit_distance_norm": norm_edit,
                    "decoded_text": decoded.get("text"),
                }
            )

        epoch_report = {
            "epoch": epoch,
            "samples": sample_reports,
            "avg_output_mse_end": mean(output_mse_values),
            "avg_hidden_mse_end": mean(hidden_mse_values) if hidden_mse_values else None,
            "avg_token_accuracy": mean(token_acc_values),
            "avg_token_edit_distance": mean([float(x) for x in token_edit_values]) if token_edit_values else 0.0,
            "avg_token_edit_distance_norm": mean(token_edit_norm_values),
            "exact_match_rate": (exact_match_count / max(1, len(token_acc_values))) if token_acc_values else 0.0,
        }
        report["history"].append(epoch_report)

        decoder.save_mapping(decoder_map_path)
        core.save(core_path)
        if args.save_every_epoch:
            (out_dir / f"report_epoch_{epoch}.json").write_text(json.dumps(epoch_report, indent=2), encoding="utf-8")

        print(
            f"[epoch {epoch}] samples={len(sample_reports)} "
            f"avg_output_mse_end={epoch_report['avg_output_mse_end']:.4f} "
            f"avg_hidden_mse_end={epoch_report['avg_hidden_mse_end'] if epoch_report['avg_hidden_mse_end'] is not None else 'n/a'} "
            f"token_acc={epoch_report['avg_token_accuracy']:.3f} "
            f"edit_norm={epoch_report['avg_token_edit_distance_norm']:.3f}"
        )

    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    decoder.save_mapping(decoder_map_path)
    core.save(core_path)

    print("=== IMPULSE DATASET TRAINING COMPLETE ===")
    print("Dataset:", dataset_path)
    print("Samples:", len(rows))
    print("Core saved:", core_path)
    print("Decoder mapping saved:", decoder_map_path)
    print("Report:", out_dir / "training_report.json")


if __name__ == "__main__":
    main()
