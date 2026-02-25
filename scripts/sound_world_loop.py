#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manager.audio_brain import AudioBrain


def available_recorders() -> list[str]:
    return [name for name in ("arecord", "ffmpeg") if shutil.which(name)]


def available_players() -> list[str]:
    return [name for name in ("aplay", "ffplay") if shutil.which(name)]


def record_with_arecord(out_wav: Path, seconds: float, sample_rate: int, device: str | None) -> None:
    cmd = [
        "arecord", "-q",
        "-d", str(max(1, int(round(seconds)))),
        "-f", "S16_LE",
        "-r", str(sample_rate),
        "-c", "1",
        str(out_wav),
    ]
    if device:
        cmd[1:1] = ["-D", device]
    subprocess.run(cmd, check=True)


def record_with_ffmpeg(out_wav: Path, seconds: float, sample_rate: int) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-f", "pulse", "-i", "default",
        "-t", f"{seconds:.2f}",
        "-ac", "1",
        "-ar", str(sample_rate),
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def record_audio(out_wav: Path, seconds: float, sample_rate: int, recorder: str | None, device: str | None) -> str:
    choices = [recorder] if recorder else available_recorders()
    last_err: Exception | None = None
    for name in choices:
        if not name:
            continue
        try:
            if name == "arecord":
                record_with_arecord(out_wav, seconds, sample_rate, device)
                return "arecord"
            if name == "ffmpeg":
                record_with_ffmpeg(out_wav, seconds, sample_rate)
                return "ffmpeg"
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not record audio: {last_err}")


def play_wav(path: Path, player: str | None = None) -> str | None:
    choices = [player] if player else available_players()
    for name in choices:
        try:
            if name == "aplay":
                subprocess.run(["aplay", "-q", str(path)], check=True)
                return "aplay"
            if name == "ffplay":
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return "ffplay"
        except Exception:
            continue
    return None


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def next_index(dataset_path: Path) -> int:
    if not dataset_path.exists():
        return 1
    n = 0
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n + 1


def choose_split(index: int, val_every: int) -> str:
    if val_every > 0 and index % val_every == 0:
        return "val"
    return "train"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def extract_eval_metrics(report: dict[str, Any] | None) -> dict[str, float] | None:
    if not report:
        return None
    history = report.get("history", [])
    if not isinstance(history, list) or not history:
        return None
    last = history[-1] if isinstance(history[-1], dict) else {}
    return {
        "avg_token_accuracy": float(last.get("avg_token_accuracy", 0.0) or 0.0),
        "avg_token_edit_distance_norm": float(last.get("avg_token_edit_distance_norm", 999.0) or 999.0),
        "avg_output_mse_end": float(last.get("avg_output_mse_end", 999.0) or 999.0),
    }


def is_improved_metrics(new: dict[str, float] | None, best: dict[str, float] | None) -> bool:
    if not new:
        return False
    if not best:
        return True
    # Primary: higher token accuracy. Secondary: lower normalized edit distance. Tertiary: lower MSE.
    if new["avg_token_accuracy"] > best["avg_token_accuracy"] + 1e-6:
        return True
    if abs(new["avg_token_accuracy"] - best["avg_token_accuracy"]) <= 1e-6:
        if new["avg_token_edit_distance_norm"] < best["avg_token_edit_distance_norm"] - 1e-6:
            return True
        if abs(new["avg_token_edit_distance_norm"] - best["avg_token_edit_distance_norm"]) <= 1e-6:
            if new["avg_output_mse_end"] < best["avg_output_mse_end"] - 1e-6:
                return True
    return False


def run_retrainer(
    *,
    trainer_kind: str,
    dataset_path: Path,
    out_dir: Path,
    epochs: int,
    core_output_epochs: int,
    core_output_lr: float,
    hidden_evo_iters: int,
    hidden_evo_pop: int,
    hidden_evo_sigma: float,
    align_mode: str,
) -> tuple[bool, str]:
    script_name = "train_audio_brain_dataset.py" if trainer_kind == "audio_brain" else "train_impulse_dataset.py"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / script_name),
        str(dataset_path),
        "--out-dir",
        str(out_dir),
        "--epochs",
        str(int(epochs)),
        "--core-output-epochs",
        str(int(core_output_epochs)),
        "--core-output-lr",
        str(float(core_output_lr)),
        "--hidden-evo-iters",
        str(int(hidden_evo_iters)),
        "--hidden-evo-pop",
        str(int(hidden_evo_pop)),
        "--hidden-evo-sigma",
        str(float(hidden_evo_sigma)),
        "--align-mode",
        str(align_mode),
    ]
    try:
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as e:
        return False, f"failed to launch trainer: {e}"
    if res.returncode != 0:
        tail = (res.stderr or res.stdout or "").strip()[-800:]
        return False, f"trainer exited {res.returncode}: {tail}"
    tail = (res.stdout or "").strip().splitlines()[-4:]
    return True, " | ".join(tail)


def try_hot_swap_models(brain: AudioBrain, retrain_dir: Path) -> list[str]:
    notes: list[str] = []
    core_path = retrain_dir / "neural_core.json"
    aud_path = retrain_dir / "auditory_core.json"
    assoc_path = retrain_dir / "association_core.json"
    vocal_path = retrain_dir / "vocal_core.json"
    dec_path = retrain_dir / "decoder_mapping.json"

    # Decoder mappings can be reloaded into hearing/vocal decoders with channel overrides.
    if dec_path.exists():
        try:
            brain.decoder = brain.decoder.load_mapping(dec_path, output_channels=brain.decoder.output_channels)
            notes.append("swapped vocal decoder mapping")
        except Exception as e:
            notes.append(f"vocal decoder swap failed: {e}")
        try:
            brain.hearing_decoder = brain.hearing_decoder.load_mapping(dec_path, output_channels=brain.hearing_decoder.output_channels)
            notes.append("swapped hearing decoder mapping")
        except Exception as e:
            notes.append(f"hearing decoder swap failed: {e}")

    # Preferred path: dedicated AudioBrain trainer outputs all three cores.
    if aud_path.exists():
        try:
            cand = brain.auditory_core.load(aud_path)
            if (
                cand.input_channels == brain.auditory_core.input_channels
                and cand.output_channels == brain.auditory_core.output_channels
                and cand.hidden_size == brain.auditory_core.hidden_size
            ):
                brain.auditory_core = cand
                notes.append("swapped auditory core")
            else:
                notes.append("auditory core shape mismatch")
        except Exception as e:
            notes.append(f"auditory core swap failed: {e}")
    if assoc_path.exists():
        try:
            cand = brain.association_core.load(assoc_path)
            if (
                cand.input_channels == brain.association_core.input_channels
                and cand.output_channels == brain.association_core.output_channels
                and cand.hidden_size == brain.association_core.hidden_size
            ):
                brain.association_core = cand
                notes.append("swapped association core")
            else:
                notes.append("association core shape mismatch")
        except Exception as e:
            notes.append(f"association core swap failed: {e}")
    if vocal_path.exists():
        try:
            cand = brain.vocal_core.load(vocal_path)
            if (
                cand.input_channels == brain.vocal_core.input_channels
                and cand.output_channels == brain.vocal_core.output_channels
                and cand.hidden_size == brain.vocal_core.hidden_size
            ):
                brain.vocal_core = cand
                notes.append("swapped vocal core")
            else:
                notes.append("vocal core shape mismatch")
        except Exception as e:
            notes.append(f"vocal core swap failed: {e}")

    # Backward-compatible path: single core trainer output.
    if core_path.exists():
        try:
            cand = brain.auditory_core.load(core_path)
            if (
                cand.input_channels == brain.auditory_core.input_channels
                and cand.output_channels == brain.auditory_core.output_channels
                and cand.hidden_size == brain.auditory_core.hidden_size
            ):
                brain.auditory_core = cand
                notes.append("swapped auditory core")
            else:
                notes.append(
                    "trainer core shape mismatch for auditory core "
                    f"({cand.input_channels}->{cand.output_channels}, h={cand.hidden_size})"
                )
        except Exception as e:
            notes.append(f"auditory core swap failed: {e}")
    return notes


def capture_correction(
    *,
    turn: int,
    wav_path: Path,
    heard_text: str,
    dataset_path: Path,
    audio_dir: Path | None,
    speaker: str,
    session: str,
    sample_prefix: str,
    sample_rate: int,
    duration_s: float,
    val_every: int,
) -> tuple[str, str | None]:
    guess = (heard_text or "").strip() or "..."
    print(f"[turn {turn}] Did you mean: {guess}")
    print("[turn {turn}] correction> Enter = yes | type correction text | /skip | /repeat".replace("{turn}", str(turn)))
    user = input().strip()
    if user.lower() in {"/skip", "skip"}:
        return guess, None
    if user.lower() in {"/repeat", "repeat"}:
        # Use empty text so caller can choose behavior (e.g., skip reply or ask to repeat)
        return "", None
    final_text = guess if user == "" else user

    out_wav = wav_path
    if audio_dir is not None:
        audio_dir.mkdir(parents=True, exist_ok=True)
        sample_id = f"{sample_prefix}_{session}_{next_index(dataset_path):05d}"
        copied = audio_dir / f"{sample_id}.wav"
        try:
            shutil.copy2(wav_path, copied)
            out_wav = copied
        except Exception:
            sample_id = f"{sample_prefix}_{session}_{next_index(dataset_path):05d}"
    else:
        sample_id = f"{sample_prefix}_{session}_{next_index(dataset_path):05d}"

    idx = next_index(dataset_path)
    row = {
        "id": sample_id,
        "wav": str(out_wav.as_posix()),
        "target_text": final_text,
        "model_guess": guess,
        "speaker": speaker,
        "session": session,
        "sample_rate": int(sample_rate),
        "duration_s": float(duration_s),
        "split": choose_split(idx, val_every),
        "source": "sound_world_loop_correction",
        "turn": int(turn),
        "ts": time.time(),
    }
    append_jsonl(dataset_path, row)
    return final_text, str(dataset_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sound-in / sound-out loop through connected neural cores (auditory -> association -> vocal)."
    )
    ap.add_argument("--seconds", type=float, default=3.0, help="Recording duration per turn.")
    ap.add_argument("--sample-rate", type=int, default=16000, help="Recording sample rate.")
    ap.add_argument("--recorder", choices=["arecord", "ffmpeg"], help="Force recorder backend.")
    ap.add_argument("--device", help="Optional `arecord` device (e.g. hw:0,0).")
    ap.add_argument("--player", choices=["aplay", "ffplay"], help="Force playback backend.")
    ap.add_argument("--no-play", action="store_true", help="Do not play reply placeholder audio.")
    ap.add_argument("--no-llm", action="store_true", help="Use no-LLM cognitive reply generation.")
    ap.add_argument("--push-to-talk", action="store_true", help="Press Enter before each recording turn.")
    ap.add_argument("--max-turns", type=int, default=0, help="Stop after N turns (0 = unlimited).")
    ap.add_argument("--out-dir", default="artifacts/sound_world_loop", help="Artifacts directory.")
    ap.add_argument("--wav", help="Use an existing WAV (for testing); loop mode reuses same file.")
    ap.add_argument("--decoder-map", help="Load decoder mapping JSON for hearing/vocal decoders.")
    ap.add_argument("--auditory-core", help="Load auditory core JSON.")
    ap.add_argument("--association-core", help="Load association core JSON.")
    ap.add_argument("--vocal-core", help="Load vocal core JSON.")
    ap.add_argument("--save-cores", action="store_true", help="Save current cores under out-dir/cores at exit.")
    ap.add_argument("--capture-corrections", action="store_true", help="Ask 'Did you mean...?' and save corrected audio+text to a dataset.")
    ap.add_argument("--corrections-dataset", default="datasets/impulse_dataset.corrections.jsonl", help="JSONL dataset file for captured corrections.")
    ap.add_argument("--corrections-audio-dir", default="datasets/audio_corrections", help="Directory to copy corrected audio clips into.")
    ap.add_argument("--speaker", default="user1", help="Speaker ID for correction dataset rows.")
    ap.add_argument("--session", default=None, help="Session tag for correction dataset rows (default timestamp).")
    ap.add_argument("--val-every", type=int, default=10, help="Mark every Nth correction as validation sample (0 disables).")
    ap.add_argument("--auto-retrain-every", type=int, default=0, help="After every N saved corrections, retrain and hot-swap improved models.")
    ap.add_argument("--auto-retrain-out-dir", default="artifacts/sound_world_auto_retrain", help="Where auto-retrain artifacts/checkpoints are written.")
    ap.add_argument("--auto-retrain-epochs", type=int, default=1, help="Dataset epochs per auto-retrain run.")
    ap.add_argument("--auto-retrain-core-output-epochs", type=int, default=2, help="Output-layer epochs per sample during auto-retrain.")
    ap.add_argument("--auto-retrain-core-output-lr", type=float, default=0.05, help="Output-layer learning rate for auto-retrain.")
    ap.add_argument("--auto-retrain-hidden-evo-iters", type=int, default=1, help="Hidden evolution iterations per sample during auto-retrain.")
    ap.add_argument("--auto-retrain-hidden-evo-pop", type=int, default=2, help="Hidden evolution population per iteration.")
    ap.add_argument("--auto-retrain-hidden-evo-sigma", type=float, default=0.05, help="Hidden evolution mutation sigma.")
    ap.add_argument("--auto-retrain-align-mode", choices=["ctc", "dynamic", "anchors"], default="ctc", help="Alignment mode for auto-retrain.")
    ap.add_argument("--auto-retrain-trainer", choices=["audio_brain", "single_core"], default="audio_brain", help="Auto-retrain backend.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    correction_session = args.session or time.strftime("%Y%m%d-%H%M%S")
    corrections_dataset = Path(args.corrections_dataset)
    corrections_audio_dir = Path(args.corrections_audio_dir) if args.corrections_audio_dir else None
    auto_retrain_dir = Path(args.auto_retrain_out_dir)
    corrections_since_retrain = 0
    best_auto_metrics = extract_eval_metrics(load_json(auto_retrain_dir / "training_report.json"))

    brain = AudioBrain(
        core_paths={
            "auditory": args.auditory_core,
            "association": args.association_core,
            "vocal": args.vocal_core,
        },
        decoder_map_path=args.decoder_map,
    )

    print("=== SOUND WORLD LOOP (scaffold) ===")
    print("Incoming information: sound -> impulses -> connected neural cores")
    print("Outgoing information: connected neural cores -> impulses -> sound (placeholder)")
    print("Stop with Ctrl+C.")
    if args.push_to_talk:
        print("Push-to-talk is ON: press Enter before each turn.")
    if int(args.auto_retrain_every) > 0:
        print(f"Auto-retrain enabled: every {int(args.auto_retrain_every)} saved corrections.")

    turn = 0
    try:
        while True:
            turn += 1
            if args.max_turns and turn > int(args.max_turns):
                break
            if args.push_to_talk:
                gate = input(f"[turn {turn}] Press Enter to record (or 'q' to quit): ").strip().lower()
                if gate in {"q", "quit", "exit"}:
                    break

            if args.wav:
                wav_path = Path(args.wav)
                if not wav_path.exists():
                    raise SystemExit(f"WAV not found: {wav_path}")
                print(f"[turn {turn}] Using existing WAV: {wav_path}")
            else:
                wav_path = out_dir / f"turn_{turn:03d}_in.wav"
                print("Available recorders:", ", ".join(available_recorders()) or "(none)")
                print(f"[turn {turn}] Recording {args.seconds:.1f}s ...")
                used = record_audio(wav_path, float(args.seconds), int(args.sample_rate), args.recorder, args.device)
                print(f"[turn {turn}] Recorder used: {used}")

            turn_dir = out_dir / f"turn_{turn:03d}"
            result = brain.react_to_wav(
                wav_path,
                out_dir=turn_dir,
                use_llm=not args.no_llm,
                save_audio=True,
            )
            heard_text = str(result.get("heard_text") or "")
            corrected_text = heard_text
            if args.capture_corrections:
                try:
                    corrected_text, saved_ds = capture_correction(
                        turn=turn,
                        wav_path=wav_path,
                        heard_text=heard_text,
                        dataset_path=corrections_dataset,
                        audio_dir=corrections_audio_dir,
                        speaker=args.speaker,
                        session=correction_session,
                        sample_prefix="corr",
                        sample_rate=int(args.sample_rate),
                        duration_s=float(args.seconds),
                        val_every=int(args.val_every),
                    )
                    if saved_ds:
                        print(f"[turn {turn}] Correction saved -> {saved_ds}")
                        corrections_since_retrain += 1
                except EOFError:
                    pass

            if corrected_text == "":
                print(f"[turn {turn}] User requested repeat. Skipping reply this turn.")
                continue

            if corrected_text != heard_text:
                # Re-run only cognitive reply stage on corrected text while preserving sound traces.
                spoken = brain.think_and_speak(corrected_text, use_llm=not args.no_llm)
                result["reply_text_cognitive"] = spoken.get("reply_text_cognitive")
                result["decoded_reply"] = spoken.get("decoded_reply")
                result["vocal_out"] = spoken.get("vocal_out")
                reply_wav = turn_dir / "reply_placeholder.wav"
                brain.decoder.synthesize_placeholder_wav(spoken["vocal_out"], reply_wav)
            reply_text = str(result.get("reply_text_cognitive") or "")
            reply_decoded = str((result.get("decoded_reply") or {}).get("text") or "")

            print(f"you(sound)->brain: {heard_text}")
            if corrected_text != heard_text:
                print(f"you(correction): {corrected_text}")
            print(f"brain(cognitive)->text: {reply_text}")
            print(f"brain(vocal net)->sound tokens: {reply_decoded}")
            reply_wav = turn_dir / "reply_placeholder.wav"
            if not args.no_play and reply_wav.exists():
                played = play_wav(reply_wav, player=args.player)
                print(f"[turn {turn}] Playback: {played or 'not available'}")
            print(f"[turn {turn}] Artifacts: {turn_dir}")

            if (
                int(args.auto_retrain_every) > 0
                and corrections_since_retrain >= int(args.auto_retrain_every)
                and args.capture_corrections
                and corrections_dataset.exists()
            ):
                print(
                    f"[turn {turn}] Auto-retrain triggered after {corrections_since_retrain} new corrections..."
                )
                ok, msg = run_retrainer(
                    trainer_kind=str(args.auto_retrain_trainer),
                    dataset_path=corrections_dataset,
                    out_dir=auto_retrain_dir,
                    epochs=int(args.auto_retrain_epochs),
                    core_output_epochs=int(args.auto_retrain_core_output_epochs),
                    core_output_lr=float(args.auto_retrain_core_output_lr),
                    hidden_evo_iters=int(args.auto_retrain_hidden_evo_iters),
                    hidden_evo_pop=int(args.auto_retrain_hidden_evo_pop),
                    hidden_evo_sigma=float(args.auto_retrain_hidden_evo_sigma),
                    align_mode=str(args.auto_retrain_align_mode),
                )
                print(f"[turn {turn}] Auto-retrain status: {'ok' if ok else 'error'}")
                if msg:
                    print(f"[turn {turn}] Auto-retrain log: {msg}")
                if ok:
                    new_metrics = extract_eval_metrics(load_json(auto_retrain_dir / "training_report.json"))
                    if new_metrics:
                        print(f"[turn {turn}] Auto-retrain metrics: {new_metrics}")
                    if is_improved_metrics(new_metrics, best_auto_metrics):
                        notes = try_hot_swap_models(brain, auto_retrain_dir)
                        best_auto_metrics = new_metrics or best_auto_metrics
                        print(f"[turn {turn}] Hot-swap applied: {', '.join(notes) if notes else 'no compatible components'}")
                    else:
                        print(f"[turn {turn}] Model not improved; keeping current live models.")
                corrections_since_retrain = 0
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if args.save_cores:
            core_dir = out_dir / "cores"
            paths = brain.save_cores(core_dir)
            print("Saved cores:", paths)


if __name__ == "__main__":
    main()
