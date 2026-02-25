from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import List, Tuple

from .impulses import Impulse, ImpulseStream


def _read_wav_pcm16_mono(path: str | Path) -> Tuple[int, List[float]]:
    p = Path(path)
    with wave.open(str(p), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported in this scaffold (got sampwidth={sampwidth})")
    total_samples = len(raw) // 2
    ints = struct.unpack("<" + "h" * total_samples, raw)
    if channels == 1:
        mono = [s / 32768.0 for s in ints]
    elif channels == 2:
        mono = []
        for i in range(0, len(ints), 2):
            mono.append(((ints[i] + ints[i + 1]) / 2.0) / 32768.0)
    else:
        raise ValueError(f"Only mono/stereo WAV supported in scaffold (got channels={channels})")
    return rate, mono


def _goertzel_power(samples: List[float], sample_rate: int, target_freq: float) -> float:
    if not samples:
        return 0.0
    n = len(samples)
    k = int(0.5 + (n * target_freq / sample_rate))
    omega = (2.0 * math.pi * k) / n
    coeff = 2.0 * math.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0
    for x in samples:
        s = x + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    power = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
    return max(0.0, power / max(1, n))


class WavImpulseEncoder:
    """
    Scaffold encoder: WAV audio -> event impulses.

    Uses frame RMS + simple Goertzel band energies to emit spikes on channels:
    - 0..(band_count-1): spectral band spikes
    - band_count: loudness spike
    - band_count+1: zero-crossing activity spike
    """

    def __init__(
        self,
        frame_ms: float = 20.0,
        hop_ms: float = 10.0,
        threshold: float = 0.35,
        band_freqs: List[float] | None = None,
    ) -> None:
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.threshold = threshold
        self.band_freqs = band_freqs or [150, 300, 600, 900, 1200, 1800, 2600, 3400]

    @property
    def input_channels(self) -> int:
        return len(self.band_freqs) + 2

    def encode_wav(self, path: str | Path) -> ImpulseStream:
        sample_rate, samples = _read_wav_pcm16_mono(path)
        return self.encode_samples(samples, sample_rate)

    def encode_samples(self, samples: List[float], sample_rate: int) -> ImpulseStream:
        stream = ImpulseStream()
        if not samples:
            return stream
        frame = max(8, int(sample_rate * (self.frame_ms / 1000.0)))
        hop = max(4, int(sample_rate * (self.hop_ms / 1000.0)))
        if len(samples) < frame:
            pad = [0.0] * (frame - len(samples))
            samples = list(samples) + pad

        for start in range(0, max(1, len(samples) - frame + 1), hop):
            win = samples[start : start + frame]
            if len(win) < frame:
                win = win + [0.0] * (frame - len(win))
            t = start / float(sample_rate)
            rms = math.sqrt(sum(x * x for x in win) / max(1, len(win)))
            if rms > self.threshold * 0.5:
                stream.add(Impulse(t=t, ch=len(self.band_freqs), v=min(1.0, rms / (self.threshold + 1e-6)), kind="audio_rms"))

            zc = 0
            for i in range(1, len(win)):
                if (win[i - 1] <= 0 < win[i]) or (win[i - 1] >= 0 > win[i]):
                    zc += 1
            zc_ratio = zc / max(1, len(win) - 1)
            if zc_ratio > 0.08:
                stream.add(Impulse(t=t, ch=len(self.band_freqs) + 1, v=min(1.0, zc_ratio * 4.0), kind="audio_zcr"))

            band_powers = [_goertzel_power(win, sample_rate, f) for f in self.band_freqs]
            peak = max(band_powers) if band_powers else 0.0
            if peak <= 0:
                continue
            for ch, power in enumerate(band_powers):
                rel = power / peak
                if rel >= self.threshold:
                    stream.add(
                        Impulse(
                            t=t,
                            ch=ch,
                            v=min(1.0, rel),
                            kind="audio_band",
                            meta={"freq_hz": self.band_freqs[ch]},
                        )
                    )
        stream.sort()
        return stream
