#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def available_recorders() -> list[str]:
    return [name for name in ("arecord", "ffmpeg") if shutil.which(name)]


def available_players() -> list[str]:
    return [name for name in ("aplay", "ffplay") if shutil.which(name)]


def record_with_arecord(out_wav: Path, seconds: float, sample_rate: int, device: str | None) -> None:
    cmd = [
        "arecord",
        "-q",
        "-d",
        str(max(1, int(round(seconds)))),
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate),
        "-c",
        "1",
        str(out_wav),
    ]
    if device:
        cmd[1:1] = ["-D", device]
    subprocess.run(cmd, check=True)


def record_with_ffmpeg(out_wav: Path, seconds: float, sample_rate: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "pulse",
        "-i",
        "default",
        "-t",
        f"{seconds:.2f}",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def record_mic(out_wav: Path, seconds: float, sample_rate: int, recorder: str | None, device: str | None) -> str:
    choices = [recorder] if recorder else available_recorders()
    if not choices:
        raise RuntimeError("No recorder found (need `arecord` or `ffmpeg`).")
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
    raise RuntimeError(f"Recording failed: {last_err}")


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Record microphone clips and label them as JSONL dataset rows for impulse speech training."
    )
    ap.add_argument("--dataset", default="datasets/impulse_dataset.user.jsonl", help="Output JSONL dataset file.")
    ap.add_argument("--audio-dir", default="datasets/audio", help="Where recorded WAV clips are stored.")
    ap.add_argument("--seconds", type=float, default=2.5, help="Recording duration per clip.")
    ap.add_argument("--sample-rate", type=int, default=16000, help="Recording sample rate.")
    ap.add_argument("--recorder", choices=["arecord", "ffmpeg"], help="Force recorder backend.")
    ap.add_argument("--device", help="Optional `arecord` device (e.g. hw:0,0).")
    ap.add_argument("--player", choices=["aplay", "ffplay"], help="Playback backend for review.")
    ap.add_argument("--speaker", default="user1", help="Speaker ID metadata.")
    ap.add_argument("--session", default=None, help="Session tag (default timestamp).")
    ap.add_argument("--prefix", default="clip", help="Filename/id prefix.")
    ap.add_argument("--val-every", type=int, default=10, help="Mark every Nth sample as validation (0 disables).")
    ap.add_argument("--playback", action="store_true", help="Play each recording before labeling.")
    ap.add_argument("--push-to-talk", action="store_true", help="Press Enter before each recording.")
    ap.add_argument("--max-clips", type=int, default=0, help="Stop after N saved clips (0 = unlimited).")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    recs = available_recorders()
    if not recs:
        raise SystemExit("No recorder found. Install/use `arecord` or `ffmpeg`.")

    session = args.session or time.strftime("%Y%m%d-%H%M%S")
    idx = next_index(dataset_path)
    saved = 0

    print("=== RECORD DATASET ===")
    print("Dataset:", dataset_path)
    print("Audio dir:", audio_dir)
    print("Available recorders:", ", ".join(recs))
    print("Commands after recording:")
    print("  transcript text -> save sample")
    print("  /retry -> record again")
    print("  /skip -> discard clip")
    print("  /play -> replay clip")
    print("  /quit -> stop")
    if args.push_to_talk:
        print("Push-to-talk is ON: press Enter before each recording.")

    while True:
        if args.max_clips and saved >= int(args.max_clips):
            break
        if args.push_to_talk:
            gate = input("Press Enter to record next clip (or 'q' to quit): ").strip().lower()
            if gate in {"q", "quit", "exit"}:
                break

        sample_id = f"{args.prefix}_{session}_{idx:05d}"
        wav_path = audio_dir / f"{sample_id}.wav"
        print(f"[{sample_id}] Recording {args.seconds:.1f}s ...")
        try:
            used = record_mic(
                wav_path,
                seconds=float(args.seconds),
                sample_rate=int(args.sample_rate),
                recorder=args.recorder,
                device=args.device,
            )
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print("Recording error:", e)
            continue
        print(f"[{sample_id}] Recorder: {used}")

        if args.playback:
            played = play_wav(wav_path, player=args.player)
            print(f"[{sample_id}] Playback: {played or 'not available'}")

        while True:
            text = input(f"[{sample_id}] transcript> ").strip()
            if text.lower() in {"/quit", "quit", "exit"}:
                print("Stopping.")
                return
            if text.lower() == "/play":
                played = play_wav(wav_path, player=args.player)
                print(f"[{sample_id}] Playback: {played or 'not available'}")
                continue
            if text.lower() == "/retry":
                try:
                    wav_path.unlink(missing_ok=True)
                except Exception:
                    pass
                print(f"[{sample_id}] Retrying recording...")
                break
            if text.lower() == "/skip" or text == "":
                try:
                    wav_path.unlink(missing_ok=True)
                except Exception:
                    pass
                print(f"[{sample_id}] Skipped.")
                idx += 1
                break

            split = choose_split(idx, int(args.val_every))
            row = {
                "id": sample_id,
                "wav": str(wav_path.as_posix()),
                "target_text": text,
                "speaker": args.speaker,
                "session": session,
                "sample_rate": int(args.sample_rate),
                "duration_s": float(args.seconds),
                "split": split,
                "ts": time.time(),
            }
            append_jsonl(dataset_path, row)
            saved += 1
            print(f"[{sample_id}] Saved -> {dataset_path} (split={split})")
            idx += 1
            break

    print(f"Done. Saved {saved} clips to {dataset_path}")


if __name__ == "__main__":
    main()
