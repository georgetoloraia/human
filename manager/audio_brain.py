from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from manager.neural_core import ImpulseNeuralCore
from manager.neural_io import Impulse, ImpulseStream, WavImpulseEncoder, ImpulseSpeechDecoder
from talk_to_human import respond_to_teacher


class AudioBrain:
    """
    Sound-in / sound-out scaffold using multiple connected neural cores.

    Pipeline:
      sound -> encoder impulses -> auditory core -> association core
      -> heard-text decode -> cognitive reply text
      -> reply text impulses -> vocal core -> sound (placeholder synthesis)
    """

    def __init__(
        self,
        *,
        hidden_size: int = 24,
        auditory_out_channels: int = 16,
        assoc_channels: int = 16,
        vocal_out_channels: int = 12,
        seed: int = 7,
        core_paths: Dict[str, str] | None = None,
        decoder_map_path: str | None = None,
    ) -> None:
        self.encoder = WavImpulseEncoder()
        core_paths = core_paths or {}

        self.auditory_core = self._load_or_make_core(
            core_paths.get("auditory"),
            input_channels=self.encoder.input_channels,
            hidden_size=hidden_size,
            output_channels=auditory_out_channels,
            seed=seed,
        )
        self.association_core = self._load_or_make_core(
            core_paths.get("association"),
            input_channels=auditory_out_channels,
            hidden_size=hidden_size,
            output_channels=assoc_channels,
            seed=seed + 1,
        )
        self.vocal_core = self._load_or_make_core(
            core_paths.get("vocal"),
            input_channels=assoc_channels,
            hidden_size=hidden_size,
            output_channels=vocal_out_channels,
            seed=seed + 2,
        )
        if decoder_map_path and Path(decoder_map_path).exists():
            self.decoder = ImpulseSpeechDecoder.load_mapping(decoder_map_path, output_channels=vocal_out_channels)
            self.hearing_decoder = ImpulseSpeechDecoder.load_mapping(decoder_map_path, output_channels=assoc_channels)
        else:
            self.decoder = ImpulseSpeechDecoder(output_channels=vocal_out_channels)
            self.hearing_decoder = ImpulseSpeechDecoder(output_channels=assoc_channels)

        self.last_trace: Dict[str, Any] = {}

    def _load_or_make_core(
        self,
        path: str | None,
        *,
        input_channels: int,
        hidden_size: int,
        output_channels: int,
        seed: int,
    ) -> ImpulseNeuralCore:
        if path and Path(path).exists():
            return ImpulseNeuralCore.load(path)
        return ImpulseNeuralCore(
            input_channels=input_channels,
            hidden_size=hidden_size,
            output_channels=output_channels,
            seed=seed,
        )

    @staticmethod
    def _project_impulses(stream: ImpulseStream, out_channels: int, kind: str) -> ImpulseStream:
        projected = ImpulseStream()
        if out_channels <= 0:
            return projected
        for imp in stream.impulses:
            projected.add(
                Impulse(
                    t=float(imp.t),
                    ch=int(imp.ch) % out_channels,
                    v=float(imp.v),
                    kind=kind,
                    meta=dict(imp.meta or {}),
                )
            )
        projected.sort()
        return projected

    def hear(self, wav_path: str | Path) -> Dict[str, Any]:
        audio_imp = self.encoder.encode_wav(wav_path)
        auditory_out = self.auditory_core.process(audio_imp)
        assoc_in = self._project_impulses(auditory_out, self.association_core.input_channels, "assoc_in")
        assoc_out = self.association_core.process(assoc_in)
        heard = self.hearing_decoder.decode(assoc_out)
        return {
            "audio_impulses": audio_imp,
            "auditory_out": auditory_out,
            "assoc_in": assoc_in,
            "assoc_out": assoc_out,
            "heard": heard,
        }

    def think_and_speak(
        self,
        heard_text: str,
        *,
        use_llm: bool = False,
    ) -> Dict[str, Any]:
        reply_text = respond_to_teacher(heard_text, use_llm=use_llm)
        reply_tokens = [t for t in reply_text.strip().split() if t]

        # Convert reply text into impulses using decoder token-channel mapping.
        token_to_ch = self.decoder.ensure_tokens_mapped(reply_tokens or ["..."])
        blank = getattr(self.decoder, "blank_channel", self.decoder.output_channels - 1)
        reply_seed = ImpulseStream()
        t = 0.0
        for tok in (reply_tokens[:48] or ["..."]):
            ch = int(token_to_ch.get(tok, 0))
            reply_seed.add(Impulse(t=t, ch=ch % self.vocal_core.input_channels, v=0.95, kind="reply_seed"))
            t += 0.05
            if 0 <= blank < self.vocal_core.input_channels:
                reply_seed.add(Impulse(t=t, ch=blank, v=0.5, kind="reply_blank"))
            t += 0.05
        reply_seed.sort()

        vocal_in = self._project_impulses(reply_seed, self.vocal_core.input_channels, "vocal_in")
        vocal_out = self.vocal_core.process(vocal_in)
        if not vocal_out.impulses:
            # Fallback so the system still emits sound impulses when the random vocal core is silent.
            vocal_out = self._project_impulses(reply_seed, self.decoder.output_channels, "vocal_fallback")
        decoded_reply = self.decoder.decode(vocal_out)
        return {
            "reply_text_cognitive": reply_text,
            "reply_seed_impulses": reply_seed,
            "vocal_in": vocal_in,
            "vocal_out": vocal_out,
            "decoded_reply": decoded_reply,
        }

    def react_to_wav(
        self,
        wav_path: str | Path,
        *,
        out_dir: str | Path | None = None,
        use_llm: bool = False,
        save_audio: bool = True,
    ) -> Dict[str, Any]:
        heard = self.hear(wav_path)
        heard_text = str((heard.get("heard") or {}).get("text") or "").strip() or "..."
        spoken = self.think_and_speak(heard_text, use_llm=use_llm)

        result = {
            "heard_text": heard_text,
            **heard,
            **spoken,
            "ts": time.time(),
        }
        self.last_trace = {
            "heard_text": heard_text,
            "reply_text_cognitive": spoken.get("reply_text_cognitive"),
            "decoded_reply": (spoken.get("decoded_reply") or {}).get("text"),
            "auditory_core": self.auditory_core.summary(),
            "association_core": self.association_core.summary(),
            "vocal_core": self.vocal_core.summary(),
        }

        if out_dir:
            root = Path(out_dir)
            root.mkdir(parents=True, exist_ok=True)
            # Save impulses for inspection
            heard["audio_impulses"].save_jsonl(root / "audio_impulses.jsonl")
            heard["auditory_out"].save_jsonl(root / "auditory_out.jsonl")
            heard["assoc_out"].save_jsonl(root / "association_out.jsonl")
            spoken["vocal_out"].save_jsonl(root / "vocal_out.jsonl")
            if save_audio:
                self.decoder.synthesize_placeholder_wav(spoken["vocal_out"], root / "reply_placeholder.wav")
            summary = {
                "heard_text": heard_text,
                "heard_tokens": (heard.get("heard") or {}).get("tokens"),
                "reply_text_cognitive": spoken.get("reply_text_cognitive"),
                "reply_text_from_vocal": (spoken.get("decoded_reply") or {}).get("text"),
                "auditory_core": self.auditory_core.summary(),
                "association_core": self.association_core.summary(),
                "vocal_core": self.vocal_core.summary(),
            }
            (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return result

    def save_cores(self, out_dir: str | Path) -> Dict[str, str]:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        paths = {
            "auditory": str(d / "auditory_core.json"),
            "association": str(d / "association_core.json"),
            "vocal": str(d / "vocal_core.json"),
        }
        self.auditory_core.save(paths["auditory"])
        self.association_core.save(paths["association"])
        self.vocal_core.save(paths["vocal"])
        return paths
