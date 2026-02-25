#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manager.neural_core import ImpulseNeuralCore
from manager.neural_io import WavImpulseEncoder, ImpulseSpeechDecoder
from manager.neural_io.impulses import Impulse, ImpulseStream
from talk_to_human import respond_to_teacher


def available_recorders() -> list[str]:
    tools = []
    for name in ("arecord", "ffmpeg"):
        if shutil.which(name):
            tools.append(name)
    return tools


def available_players() -> list[str]:
    tools = []
    for name in ("aplay", "ffplay"):
        if shutil.which(name):
            tools.append(name)
    return tools


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
    # Linux pulseaudio/default source path. User can adapt if needed.
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
    raise RuntimeError(f"Recording failed with available recorders: {last_err}")


def play_wav(path: Path, player: str | None = None) -> str | None:
    choices = [player] if player else available_players()
    for name in choices:
        if not name:
            continue
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


def reply_text_to_impulses(reply_text: str, decoder: ImpulseSpeechDecoder, step_s: float = 0.12) -> ImpulseStream:
    tokens = [t for t in reply_text.strip().split() if t]
    if not tokens:
        return ImpulseStream()
    token_to_ch = decoder.ensure_tokens_mapped(tokens)
    blank = decoder.blank_channel if hasattr(decoder, "blank_channel") else (decoder.output_channels - 1)
    stream = ImpulseStream()
    t = 0.0
    for tok in tokens[:32]:
        ch = int(token_to_ch.get(tok, 0))
        stream.add(Impulse(t=t, ch=ch, v=0.9, kind="reply_token"))
        # insert blank between tokens to preserve repeats in CTC-style collapse
        t += step_s * 0.5
        if 0 <= blank < decoder.output_channels:
            stream.add(Impulse(t=t, ch=blank, v=0.8, kind="reply_blank"))
        t += step_s * 0.5
    stream.sort()
    return stream


def process_one_audio_turn(
    wav_path: Path,
    encoder: WavImpulseEncoder,
    core: ImpulseNeuralCore,
    decoder: ImpulseSpeechDecoder,
    out_dir: Path,
    turn_idx: int,
) -> dict:
    in_impulses = encoder.encode_wav(wav_path)
    out_impulses = core.process(in_impulses)
    decoded = decoder.decode(out_impulses)
    turn_dir = out_dir / f"turn_{turn_idx:03d}"
    turn_dir.mkdir(parents=True, exist_ok=True)
    in_imp_path = turn_dir / "input_impulses.jsonl"
    out_imp_path = turn_dir / "output_impulses.jsonl"
    in_impulses.save_jsonl(in_imp_path)
    out_impulses.save_jsonl(out_imp_path)
    return {
        "turn_dir": turn_dir,
        "in_impulses": in_impulses,
        "out_impulses": out_impulses,
        "decoded": decoded,
        "input_imp_path": in_imp_path,
        "output_imp_path": out_imp_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Record microphone audio, convert to impulses, and decode (first live audio step).")
    ap.add_argument("--seconds", type=float, default=3.0, help="Recording duration in seconds.")
    ap.add_argument("--sample-rate", type=int, default=16000, help="Recording sample rate.")
    ap.add_argument("--recorder", choices=["arecord", "ffmpeg"], help="Force a recorder backend.")
    ap.add_argument("--device", help="Optional `arecord` device (e.g. hw:0,0).")
    ap.add_argument("--out-dir", default="artifacts/mic_session", help="Directory to write wav/impulses/decoded files.")
    ap.add_argument("--wav", help="Use an existing WAV file instead of recording.")
    ap.add_argument("--core", help="Path to trained neural_core.json to load.")
    ap.add_argument("--decoder-map", help="Path to decoder mapping JSON to load.")
    ap.add_argument("--save-placeholder-audio", action="store_true", help="Write placeholder decoded tone WAV.")
    ap.add_argument("--loop", action="store_true", help="Turn-taking conversation loop: record -> decode -> reply -> playback.")
    ap.add_argument("--push-to-talk", action="store_true", help="In loop mode, press Enter before each recording turn.")
    ap.add_argument("--max-turns", type=int, default=0, help="Stop loop after N turns (0 = unlimited).")
    ap.add_argument("--player", choices=["aplay", "ffplay"], help="Force audio player for reply playback.")
    ap.add_argument("--no-play", action="store_true", help="Do not play synthesized reply audio.")
    ap.add_argument("--no-llm", action="store_true", help="Use no-LLM reply generation for the mind.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = WavImpulseEncoder()

    if args.core:
        core = ImpulseNeuralCore.load(args.core)
    else:
        core = ImpulseNeuralCore(input_channels=encoder.input_channels, hidden_size=24, output_channels=12)

    if args.decoder_map and Path(args.decoder_map).exists():
        decoder = ImpulseSpeechDecoder.load_mapping(args.decoder_map, output_channels=core.output_channels)
    else:
        decoder = ImpulseSpeechDecoder(output_channels=core.output_channels)

    def acquire_wav(turn_idx: int) -> tuple[Path, str]:
        if args.wav:
            wav_path = Path(args.wav)
            if not wav_path.exists():
                raise SystemExit(f"WAV not found: {wav_path}")
            return wav_path, "existing_wav"
        wav_path = out_dir / f"mic_{turn_idx:03d}_{int(time.time())}.wav"
        print("Available recorders:", ", ".join(available_recorders()) or "(none)")
        print(f"[turn {turn_idx}] Recording {args.seconds:.1f}s to {wav_path} ...")
        recorder_used = record_mic(
            wav_path,
            seconds=float(args.seconds),
            sample_rate=int(args.sample_rate),
            recorder=args.recorder,
            device=args.device,
        )
        print(f"[turn {turn_idx}] Recorder used:", recorder_used)
        return wav_path, recorder_used

    if not args.loop:
        wav_path, _rec = acquire_wav(1)
        result = process_one_audio_turn(wav_path, encoder, core, decoder, out_dir, 1)
        if args.save_placeholder_audio:
            decoder.synthesize_placeholder_wav(result["out_impulses"], out_dir / "decoded_placeholder.wav")

        print("=== MIC -> IMPULSES ===")
        print("WAV:", wav_path)
        print("Input impulse summary:", result["in_impulses"].summary())
        print("Neural core summary:", core.summary())
        print("Output impulse summary:", result["out_impulses"].summary())
        print("Decoded tokens:", result["decoded"].get("tokens"))
        print("Decoded text:", result["decoded"].get("text"))
        print("Wrote:", result["input_imp_path"])
        print("Wrote:", result["output_imp_path"])
        if args.save_placeholder_audio:
            print("Wrote:", out_dir / "decoded_placeholder.wav")
        return

    if args.wav:
        print("`--loop` with `--wav` will replay the same WAV each turn.")
    print("=== TALK LOOP (scaffold) ===")
    print("Speak, wait for recording to stop, then the program will decode and reply.")
    print("Stop with Ctrl+C.")
    if args.push_to_talk:
        print("Push-to-talk is ON: press Enter to record each turn (type 'q' then Enter to quit).")
    turn = 0
    try:
        while True:
            turn += 1
            if args.max_turns and turn > int(args.max_turns):
                break
            if args.push_to_talk:
                gate = input(f"[turn {turn}] Press Enter to record, or 'q' to quit: ").strip().lower()
                if gate in {"q", "quit", "exit"}:
                    break
            wav_path, _rec = acquire_wav(turn)
            result = process_one_audio_turn(wav_path, encoder, core, decoder, out_dir, turn)
            heard_text = str(result["decoded"].get("text") or "").strip()
            if not heard_text or heard_text == "(no speech output impulses)":
                heard_text = "..."
            print(f"you(audio->{turn}): {heard_text}")
            reply_text = respond_to_teacher(heard_text, use_llm=not args.no_llm)
            print(f"mind: {reply_text}")

            reply_imp = reply_text_to_impulses(reply_text, decoder)
            reply_wav = result["turn_dir"] / "mind_reply_placeholder.wav"
            decoder.synthesize_placeholder_wav(reply_imp, reply_wav)
            print("Wrote:", reply_wav)
            if not args.no_play:
                used = play_wav(reply_wav, player=args.player)
                if used:
                    print(f"[turn {turn}] played reply with {used}")
                else:
                    print(f"[turn {turn}] no player available (use `--no-play` to silence this)")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
