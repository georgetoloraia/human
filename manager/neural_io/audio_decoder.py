from __future__ import annotations

import math
import json
import struct
import wave
from pathlib import Path
from typing import Any, Dict, List

from .impulses import ImpulseStream


class ImpulseSpeechDecoder:
    """
    Scaffold decoder:
    - output impulses -> pseudo tokens (text)
    - optional tone-based WAV synthesis placeholder

    Replace later with phoneme decoder + real TTS/vocoder.
    """

    def __init__(self, output_channels: int = 12, blank_channel: int | None = None) -> None:
        self.output_channels = output_channels
        self.blank_channel = (output_channels - 1) if blank_channel is None else int(blank_channel)
        base_vocab = [
            "ha", "la", "mi", "no", "ta", "ri", "sa", "ko", "ve", "di", "zu", "pa",
            "I", "hear", "signal", "pattern", "learning", "focus",
        ]
        self.channel_vocab = {i: base_vocab[i % len(base_vocab)] for i in range(max(1, output_channels))}
        if 0 <= self.blank_channel < self.output_channels:
            self.channel_vocab[self.blank_channel] = "<blank>"
        self.channel_token_counts: Dict[int, Dict[str, float]] = {i: {} for i in range(max(1, output_channels))}

    def _channel_sequence(self, stream: ImpulseStream, max_tokens: int = 64) -> List[int]:
        seq: List[int] = []
        prev_raw = None
        for imp in sorted(stream.impulses, key=lambda x: (x.t, -x.v)):
            ch = int(imp.ch)
            if ch == prev_raw:
                continue
            prev_raw = ch
            if ch == self.blank_channel:
                continue
            seq.append(ch)
            if len(seq) >= max_tokens:
                break
        return seq

    def _preferred_token_to_channel(self) -> Dict[str, int]:
        inv: Dict[str, int] = {}
        for ch in sorted(self.channel_vocab.keys()):
            tok = self.channel_vocab[ch]
            if tok not in inv:
                inv[tok] = ch
        return inv

    def ensure_tokens_mapped(self, tokens: List[str]) -> Dict[str, int]:
        token_to_ch = self._preferred_token_to_channel()
        for tok in tokens:
            if tok in token_to_ch:
                continue
            # choose channel with smallest learned mass, or first available
            best_ch = None
            best_score = None
            for ch in range(self.output_channels):
                if ch == self.blank_channel:
                    continue
                counts = self.channel_token_counts.setdefault(ch, {})
                score = sum(float(v) for v in counts.values())
                if best_score is None or score < best_score:
                    best_score = score
                    best_ch = ch
            if best_ch is None:
                best_ch = 0
            self.channel_vocab[best_ch] = tok
            self.channel_token_counts.setdefault(best_ch, {})
            self.channel_token_counts[best_ch][tok] = float(self.channel_token_counts[best_ch].get(tok, 0.0)) + 0.5
            token_to_ch[tok] = best_ch
        return token_to_ch

    def target_channels_for_text(self, text: str, max_len: int = 64) -> List[int]:
        tokens = [t for t in text.strip().split() if t]
        if not tokens:
            return []
        token_to_ch = self.ensure_tokens_mapped(tokens)
        return [int(token_to_ch[t]) for t in tokens[:max_len]]

    def impulses_to_tokens(self, stream: ImpulseStream, max_tokens: int = 24) -> List[str]:
        tokens: List[str] = []
        for ch in self._channel_sequence(stream, max_tokens=max_tokens):
            tokens.append(self.channel_vocab.get(ch, f"ch{ch}"))
        return tokens

    def train_from_pair(
        self,
        output_stream: ImpulseStream,
        target_text: str,
        *,
        weight: float = 1.0,
        max_align: int = 64,
    ) -> Dict[str, Any]:
        target_tokens = [t for t in target_text.strip().split() if t]
        ch_seq = self._channel_sequence(output_stream, max_tokens=max_align)
        n = min(len(ch_seq), len(target_tokens))
        if n == 0:
            return {"aligned": 0, "updated_channels": 0}
        updated_channels = set()
        for i in range(n):
            ch = ch_seq[i]
            tok = target_tokens[i]
            bucket = self.channel_token_counts.setdefault(ch, {})
            bucket[tok] = float(bucket.get(tok, 0.0)) + float(weight)
            self.channel_token_counts[ch] = bucket
            updated_channels.add(ch)
        self._refresh_channel_vocab_from_counts()
        return {
            "aligned": n,
            "target_tokens": len(target_tokens),
            "output_tokens": len(ch_seq),
            "updated_channels": len(updated_channels),
        }

    def _refresh_channel_vocab_from_counts(self) -> None:
        for ch, counts in self.channel_token_counts.items():
            if not counts:
                continue
            best_token = max(counts.items(), key=lambda kv: kv[1])[0]
            self.channel_vocab[int(ch)] = str(best_token)

    def save_mapping(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "output_channels": self.output_channels,
            "blank_channel": self.blank_channel,
            "channel_vocab": {str(k): v for k, v in self.channel_vocab.items()},
            "channel_token_counts": {
                str(ch): {tok: float(score) for tok, score in counts.items()}
                for ch, counts in self.channel_token_counts.items()
            },
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_mapping(cls, path: str | Path, output_channels: int | None = None) -> "ImpulseSpeechDecoder":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        channels = int(output_channels or data.get("output_channels", 12))
        dec = cls(output_channels=channels, blank_channel=data.get("blank_channel"))
        vocab = data.get("channel_vocab", {}) or {}
        dec.channel_vocab = {int(k): str(v) for k, v in vocab.items()}
        counts = data.get("channel_token_counts", {}) or {}
        dec.channel_token_counts = {
            int(ch): {str(tok): float(score) for tok, score in (bucket or {}).items()}
            for ch, bucket in counts.items()
        }
        for i in range(channels):
            dec.channel_token_counts.setdefault(i, {})
            dec.channel_vocab.setdefault(i, f"ch{i}")
        if 0 <= dec.blank_channel < channels:
            dec.channel_vocab[dec.blank_channel] = "<blank>"
        dec._refresh_channel_vocab_from_counts()
        return dec

    def impulses_to_text(self, stream: ImpulseStream) -> str:
        tokens = self.impulses_to_tokens(stream)
        if not tokens:
            return "(no speech output impulses)"
        return " ".join(tokens)

    def synthesize_placeholder_wav(
        self,
        stream: ImpulseStream,
        out_path: str | Path,
        sample_rate: int = 16000,
        seconds_cap: float = 4.0,
    ) -> None:
        tokens = self.impulses_to_tokens(stream, max_tokens=32)
        duration_per = 0.12
        total_dur = min(seconds_cap, max(0.25, len(tokens) * duration_per))
        total_samples = int(sample_rate * total_dur)
        buf = [0.0] * total_samples
        for idx, tok in enumerate(tokens):
            start = int(idx * duration_per * sample_rate)
            end = min(total_samples, start + int(duration_per * sample_rate))
            ch = next((k for k, v in self.channel_vocab.items() if v == tok), idx % self.output_channels)
            freq = 220.0 + (ch % 12) * 40.0
            for n in range(start, end):
                t = n / sample_rate
                env = 1.0 - ((n - start) / max(1, (end - start)))
                buf[n] += 0.15 * env * math.sin(2.0 * math.pi * freq * t)

        pcm = bytearray()
        for x in buf:
            s = max(-1.0, min(1.0, x))
            pcm += struct.pack("<h", int(s * 32767))
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(pcm))

    def decode(self, stream: ImpulseStream) -> Dict[str, object]:
        return {
            "tokens": self.impulses_to_tokens(stream),
            "text": self.impulses_to_text(stream),
            "output_impulses": len(stream.impulses),
        }
