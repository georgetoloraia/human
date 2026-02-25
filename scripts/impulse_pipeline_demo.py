#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import struct
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manager.neural_core import ImpulseNeuralCore
from manager.neural_io import ImpulseStream, WavImpulseEncoder, ImpulseSpeechDecoder


def generate_demo_wav(path: str | Path, sample_rate: int = 16000, seconds: float = 1.8) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    total = int(sample_rate * seconds)
    samples = bytearray()
    for n in range(total):
        t = n / sample_rate
        # simple voiced-like synthetic signal: changing tones + envelope
        env = 0.5 + 0.5 * math.sin(2 * math.pi * 0.7 * t)
        sig = (
            0.5 * math.sin(2 * math.pi * (180 + 40 * math.sin(2 * math.pi * 0.5 * t)) * t)
            + 0.25 * math.sin(2 * math.pi * 480 * t)
            + 0.15 * math.sin(2 * math.pi * 1100 * t)
        )
        x = max(-1.0, min(1.0, env * sig))
        samples += struct.pack("<h", int(x * 32767))
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(samples))
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Scaffold brain-like impulse pipeline demo.")
    ap.add_argument("--wav", help="Input 16-bit PCM WAV file (mono/stereo). If omitted, a demo WAV is generated.")
    ap.add_argument("--out-dir", default="artifacts/impulse_demo", help="Output directory for impulses and placeholder audio.")
    ap.add_argument("--save-network", action="store_true", help="Save generated neural core weights to JSON.")
    ap.add_argument("--target-text", help="Supervise decoder mapping using this space-separated token/phoneme string.")
    ap.add_argument("--decoder-map", help="Path to decoder mapping JSON (load if exists, save after training).")
    ap.add_argument("--train-core", action="store_true", help="Train the neural core output layer against target text channels.")
    ap.add_argument("--train-hidden-evo", action="store_true", help="Train hidden/recurrent weights with evolutionary search.")
    ap.add_argument("--epochs", type=int, default=12, help="Epochs for neural core supervised output-layer training.")
    ap.add_argument("--lr", type=float, default=0.05, help="Learning rate for neural core supervised output-layer training.")
    ap.add_argument("--evo-iters", type=int, default=30, help="Evolutionary iterations for hidden/recurrent search.")
    ap.add_argument("--evo-pop", type=int, default=6, help="Population per evolutionary iteration.")
    ap.add_argument("--evo-sigma", type=float, default=0.06, help="Mutation scale for evolutionary search.")
    ap.add_argument("--align-mode", choices=["ctc", "dynamic", "anchors"], default="ctc", help="Target/time alignment mode.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_path = Path(args.wav) if args.wav else generate_demo_wav(out_dir / "demo_input.wav")
    encoder = WavImpulseEncoder()
    in_impulses = encoder.encode_wav(wav_path)

    core = ImpulseNeuralCore(input_channels=encoder.input_channels, hidden_size=24, output_channels=12)

    decoder_map_path = Path(args.decoder_map) if args.decoder_map else (out_dir / "decoder_mapping.json")
    if decoder_map_path.exists():
        decoder = ImpulseSpeechDecoder.load_mapping(decoder_map_path, output_channels=12)
    else:
        decoder = ImpulseSpeechDecoder(output_channels=12)

    core_train_info = None
    target_channels = []
    if args.target_text:
        target_channels = decoder.target_channels_for_text(args.target_text)
    if args.train_core:
        if not args.target_text:
            raise SystemExit("--train-core requires --target-text")
        core_train_info = core.train_output_supervised(
            in_impulses,
            target_channels,
            epochs=args.epochs,
            lr=args.lr,
            align_mode=args.align_mode,
        )
    hidden_train_info = None
    if args.train_hidden_evo:
        if not args.target_text:
            raise SystemExit("--train-hidden-evo requires --target-text")
        hidden_train_info = core.train_hidden_evolutionary(
            in_impulses,
            target_channels,
            iterations=args.evo_iters,
            population=args.evo_pop,
            sigma=args.evo_sigma,
            align_mode=args.align_mode,
        )

    out_impulses = core.process(in_impulses)
    decoded_before = decoder.decode(out_impulses)
    train_info = None
    if args.target_text:
        train_info = decoder.train_from_pair(out_impulses, args.target_text)
        decoder.save_mapping(decoder_map_path)
    decoded = decoder.decode(out_impulses)
    decoder.synthesize_placeholder_wav(out_impulses, out_dir / "decoded_placeholder.wav")

    in_impulses.save_jsonl(out_dir / "input_impulses.jsonl")
    out_impulses.save_jsonl(out_dir / "output_impulses.jsonl")
    if args.save_network:
        core.save(out_dir / "neural_core.json")

    print("=== IMPULSE PIPELINE DEMO ===")
    print("Input WAV:", wav_path)
    print("Input impulse summary:", in_impulses.summary())
    print("Neural core summary:", core.summary())
    if target_channels:
        print("Target channels:", target_channels[:24], "(len=", len(target_channels), ")")
    if core_train_info:
        print("Neural core training:", core_train_info)
    if hidden_train_info:
        print("Hidden/recurrent evolutionary training:", hidden_train_info)
    print("Output impulse summary:", out_impulses.summary())
    print("Decoded text (before training):", decoded_before["text"])
    if train_info:
        print("Decoder training:", train_info)
        print("Decoder mapping saved:", decoder_map_path)
    print("Decoded text:", decoded["text"])
    print("Wrote:", out_dir / "input_impulses.jsonl")
    print("Wrote:", out_dir / "output_impulses.jsonl")
    print("Wrote:", out_dir / "decoded_placeholder.wav")
    if args.save_network:
        print("Wrote:", out_dir / "neural_core.json")


if __name__ == "__main__":
    main()
