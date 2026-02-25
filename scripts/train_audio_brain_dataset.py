#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manager.audio_brain import AudioBrain
from manager.neural_io import Impulse, ImpulseStream


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Dataset line {i} is not an object")
            rows.append(obj)
    return rows


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
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def token_accuracy(pred: List[str], target: List[str]) -> float:
    denom = max(len(pred), len(target), 1)
    return sum(1 for p, t in zip(pred, target) if p == t) / denom


def resolve_wav_path(dataset_path: Path, wav_value: str) -> Path:
    p = Path(wav_value)
    if p.is_absolute():
        return p
    candidate = (dataset_path.parent / p).resolve()
    if candidate.exists():
        return candidate
    return (ROOT / p).resolve()


def reply_seed_impulses_for_text(text: str, decoder, in_channels: int) -> ImpulseStream:
    tokens = [t for t in text.strip().split() if t]
    if not tokens:
        return ImpulseStream()
    token_to_ch = decoder.ensure_tokens_mapped(tokens)
    blank = getattr(decoder, "blank_channel", decoder.output_channels - 1)
    stream = ImpulseStream()
    t = 0.0
    for tok in tokens[:64]:
        ch = int(token_to_ch.get(tok, 0)) % max(1, in_channels)
        stream.add(Impulse(t=t, ch=ch, v=0.95, kind="train_reply_seed"))
        t += 0.05
        if 0 <= blank < in_channels:
            stream.add(Impulse(t=t, ch=blank, v=0.6, kind="train_reply_blank"))
        t += 0.05
    stream.sort()
    return stream


def project_target_channels(target: List[int], out_channels: int) -> List[int]:
    if out_channels <= 0:
        return []
    return [int(ch) % out_channels for ch in target]


def maybe_train_core(
    core,
    inputs: ImpulseStream,
    target_channels: List[int],
    *,
    align_mode: str,
    hidden_evo_iters: int,
    hidden_evo_pop: int,
    hidden_evo_sigma: float,
    out_epochs: int,
    out_lr: float,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not target_channels:
        return result
    if hidden_evo_iters > 0:
        result["hidden_train"] = core.train_hidden_evolutionary(
            inputs,
            target_channels,
            iterations=hidden_evo_iters,
            population=hidden_evo_pop,
            sigma=hidden_evo_sigma,
            align_mode=align_mode,
        )
    if out_epochs > 0:
        result["output_train"] = core.train_output_supervised(
            inputs,
            target_channels,
            epochs=out_epochs,
            lr=out_lr,
            align_mode=align_mode,
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train a multi-core AudioBrain dataset model (auditory, association, vocal cores + decoder mapping)."
    )
    ap.add_argument("dataset", help="Dataset JSONL with {wav, target_text}.")
    ap.add_argument("--out-dir", default="artifacts/audio_brain_train", help="Output directory.")
    ap.add_argument("--epochs", type=int, default=1, help="Dataset passes.")
    ap.add_argument("--align-mode", choices=["ctc", "dynamic", "anchors"], default="ctc")
    ap.add_argument("--core-output-epochs", type=int, default=2)
    ap.add_argument("--core-output-lr", type=float, default=0.05)
    ap.add_argument("--hidden-evo-iters", type=int, default=1)
    ap.add_argument("--hidden-evo-pop", type=int, default=2)
    ap.add_argument("--hidden-evo-sigma", type=float, default=0.05)
    ap.add_argument("--hidden-size", type=int, default=24)
    ap.add_argument("--auditory-out-channels", type=int, default=16)
    ap.add_argument("--association-channels", type=int, default=16)
    ap.add_argument("--vocal-out-channels", type=int, default=12)
    args = ap.parse_args()

    dataset_path = Path(args.dataset).resolve()
    rows = load_jsonl(dataset_path)
    if not rows:
        raise SystemExit("Dataset is empty.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    core_paths = {
        "auditory": str(out_dir / "auditory_core.json"),
        "association": str(out_dir / "association_core.json"),
        "vocal": str(out_dir / "vocal_core.json"),
    }
    decoder_map_path = str(out_dir / "decoder_mapping.json")

    brain = AudioBrain(
        hidden_size=int(args.hidden_size),
        auditory_out_channels=int(args.auditory_out_channels),
        assoc_channels=int(args.association_channels),
        vocal_out_channels=int(args.vocal_out_channels),
        core_paths=core_paths,
        decoder_map_path=decoder_map_path,
    )

    report: Dict[str, Any] = {
        "dataset": str(dataset_path),
        "samples": len(rows),
        "epochs": int(args.epochs),
        "align_mode": args.align_mode,
        "history": [],
    }

    for epoch in range(1, int(args.epochs) + 1):
        samples_report: List[Dict[str, Any]] = []
        hearing_accs: List[float] = []
        hearing_edit_norms: List[float] = []
        vocal_accs: List[float] = []
        vocal_edit_norms: List[float] = []
        assoc_mses: List[float] = []
        vocal_mses: List[float] = []
        auditory_mses: List[float] = []

        for idx, row in enumerate(rows, start=1):
            wav_value = str(row.get("wav", "") or "").strip()
            target_text = str(row.get("target_text", "") or "").strip()
            sample_id = str(row.get("id", f"sample{idx}"))
            if not wav_value or not target_text:
                samples_report.append({"id": sample_id, "skipped": True, "reason": "missing wav or target_text"})
                continue
            wav_path = resolve_wav_path(dataset_path, wav_value)
            if not wav_path.exists():
                samples_report.append({"id": sample_id, "skipped": True, "reason": f"missing wav file: {wav_path}"})
                continue

            audio_imp = brain.encoder.encode_wav(wav_path)
            target_hear = brain.hearing_decoder.target_channels_for_text(target_text)
            target_vocal = brain.decoder.target_channels_for_text(target_text)
            # Auditory is trained to emit a coarse projected version of the same target.
            target_auditory = project_target_channels(target_hear, brain.auditory_core.output_channels)

            auditory_train = maybe_train_core(
                brain.auditory_core,
                audio_imp,
                target_auditory,
                align_mode=args.align_mode,
                hidden_evo_iters=int(args.hidden_evo_iters),
                hidden_evo_pop=int(args.hidden_evo_pop),
                hidden_evo_sigma=float(args.hidden_evo_sigma),
                out_epochs=int(args.core_output_epochs),
                out_lr=float(args.core_output_lr),
            )
            aud_loss = brain.auditory_core.evaluate_supervised_loss(audio_imp, target_auditory, align_mode=args.align_mode)
            auditory_mses.append(float(aud_loss.get("mse", 0.0)))

            auditory_out = brain.auditory_core.process(audio_imp)
            assoc_in = brain._project_impulses(auditory_out, brain.association_core.input_channels, "assoc_in_train")
            assoc_train = maybe_train_core(
                brain.association_core,
                assoc_in,
                target_hear,
                align_mode=args.align_mode,
                hidden_evo_iters=int(args.hidden_evo_iters),
                hidden_evo_pop=int(args.hidden_evo_pop),
                hidden_evo_sigma=float(args.hidden_evo_sigma),
                out_epochs=int(args.core_output_epochs),
                out_lr=float(args.core_output_lr),
            )
            assoc_loss = brain.association_core.evaluate_supervised_loss(assoc_in, target_hear, align_mode=args.align_mode)
            assoc_mses.append(float(assoc_loss.get("mse", 0.0)))

            assoc_out = brain.association_core.process(assoc_in)
            brain.hearing_decoder.train_from_pair(assoc_out, target_text)
            heard = brain.hearing_decoder.decode(assoc_out)
            pred_hear_tokens = [str(t) for t in (heard.get("tokens") or [])]
            gold_tokens = [t for t in target_text.split() if t]
            hearing_acc = token_accuracy(pred_hear_tokens, gold_tokens)
            hearing_edit_norm = token_edit_distance(pred_hear_tokens, gold_tokens) / max(1, len(gold_tokens))
            hearing_accs.append(hearing_acc)
            hearing_edit_norms.append(hearing_edit_norm)

            vocal_in = reply_seed_impulses_for_text(target_text, brain.decoder, brain.vocal_core.input_channels)
            vocal_train = maybe_train_core(
                brain.vocal_core,
                vocal_in,
                target_vocal,
                align_mode=args.align_mode,
                hidden_evo_iters=int(args.hidden_evo_iters),
                hidden_evo_pop=int(args.hidden_evo_pop),
                hidden_evo_sigma=float(args.hidden_evo_sigma),
                out_epochs=int(args.core_output_epochs),
                out_lr=float(args.core_output_lr),
            )
            vocal_loss = brain.vocal_core.evaluate_supervised_loss(vocal_in, target_vocal, align_mode=args.align_mode)
            vocal_mses.append(float(vocal_loss.get("mse", 0.0)))

            vocal_out = brain.vocal_core.process(vocal_in)
            brain.decoder.train_from_pair(vocal_out, target_text)
            spoken = brain.decoder.decode(vocal_out)
            pred_vocal_tokens = [str(t) for t in (spoken.get("tokens") or [])]
            vocal_acc = token_accuracy(pred_vocal_tokens, gold_tokens)
            vocal_edit_norm = token_edit_distance(pred_vocal_tokens, gold_tokens) / max(1, len(gold_tokens))
            vocal_accs.append(vocal_acc)
            vocal_edit_norms.append(vocal_edit_norm)

            samples_report.append(
                {
                    "id": sample_id,
                    "wav": str(wav_path),
                    "target_text": target_text,
                    "input_impulses": audio_imp.summary(),
                    "auditory_train": auditory_train,
                    "association_train": assoc_train,
                    "vocal_train": vocal_train,
                    "heard_text": heard.get("text"),
                    "hearing_token_accuracy": hearing_acc,
                    "hearing_token_edit_distance_norm": hearing_edit_norm,
                    "spoken_text": spoken.get("text"),
                    "vocal_token_accuracy": vocal_acc,
                    "vocal_token_edit_distance_norm": vocal_edit_norm,
                }
            )

        epoch_report = {
            "epoch": epoch,
            "samples": samples_report,
            "avg_auditory_mse_end": mean(auditory_mses),
            "avg_association_mse_end": mean(assoc_mses),
            "avg_vocal_mse_end": mean(vocal_mses),
            "avg_hearing_token_accuracy": mean(hearing_accs),
            "avg_hearing_token_edit_distance_norm": mean(hearing_edit_norms),
            "avg_vocal_token_accuracy": mean(vocal_accs),
            "avg_vocal_token_edit_distance_norm": mean(vocal_edit_norms),
            # Compatibility keys used by sound_world_loop auto-retrain comparator:
            "avg_token_accuracy": mean(hearing_accs) if hearing_accs else 0.0,
            "avg_token_edit_distance_norm": mean(hearing_edit_norms) if hearing_edit_norms else 999.0,
            "avg_output_mse_end": mean(assoc_mses) if assoc_mses else 999.0,
            "ts": time.time(),
        }
        report["history"].append(epoch_report)

        brain.save_cores(out_dir)
        brain.decoder.save_mapping(out_dir / "decoder_mapping.json")
        (out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(
            f"[epoch {epoch}] hearing_acc={epoch_report['avg_hearing_token_accuracy']:.3f} "
            f"hearing_edit_norm={epoch_report['avg_hearing_token_edit_distance_norm']:.3f} "
            f"assoc_mse={epoch_report['avg_association_mse_end']:.4f} "
            f"vocal_acc={epoch_report['avg_vocal_token_accuracy']:.3f}"
        )

    print("=== AUDIO BRAIN DATASET TRAINING COMPLETE ===")
    print("Dataset:", dataset_path)
    print("Out dir:", out_dir)
    print("Saved:", out_dir / "auditory_core.json")
    print("Saved:", out_dir / "association_core.json")
    print("Saved:", out_dir / "vocal_core.json")
    print("Saved:", out_dir / "decoder_mapping.json")
    print("Saved:", out_dir / "training_report.json")


if __name__ == "__main__":
    main()
