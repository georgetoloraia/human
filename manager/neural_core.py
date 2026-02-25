from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from manager.neural_io.impulses import Impulse, ImpulseStream


class ImpulseNeuralCore:
    """
    Small recurrent impulse-processing core (scaffold).

    This is not a biological brain model. It is a compact stateful network that:
    - consumes input impulses by time bucket
    - updates hidden recurrent state
    - emits output impulses when output activations cross thresholds
    """

    def __init__(
        self,
        input_channels: int,
        hidden_size: int = 24,
        output_channels: int = 12,
        seed: int = 7,
    ) -> None:
        self.input_channels = max(1, input_channels)
        self.hidden_size = max(4, hidden_size)
        self.output_channels = max(2, output_channels)
        self.seed = seed
        rng = random.Random(seed)

        self.w_in = [
            [(rng.random() * 2 - 1) * 0.35 for _ in range(self.input_channels)]
            for _ in range(self.hidden_size)
        ]
        self.w_rec = [
            [(rng.random() * 2 - 1) * 0.12 for _ in range(self.hidden_size)]
            for _ in range(self.hidden_size)
        ]
        self.w_out = [
            [(rng.random() * 2 - 1) * 0.30 for _ in range(self.hidden_size)]
            for _ in range(self.output_channels)
        ]
        self.b_h = [(rng.random() * 2 - 1) * 0.05 for _ in range(self.hidden_size)]
        self.b_o = [(rng.random() * 2 - 1) * 0.05 for _ in range(self.output_channels)]
        self.state = [0.0 for _ in range(self.hidden_size)]
        self.last_stats: Dict[str, Any] = {}

    def reset_state(self) -> None:
        self.state = [0.0 for _ in range(self.hidden_size)]

    def _bucketize(self, stream: ImpulseStream) -> List[tuple[float, List[Impulse]]]:
        by_t: Dict[float, List[Impulse]] = {}
        for imp in stream.impulses:
            key = round(float(imp.t), 3)
            by_t.setdefault(key, []).append(imp)
        return sorted(by_t.items(), key=lambda x: x[0])

    def _vector_from_bucket(self, items: List[Impulse]) -> List[float]:
        x = [0.0] * self.input_channels
        for imp in items:
            if 0 <= imp.ch < self.input_channels:
                x[imp.ch] += float(imp.v)
        # light normalization
        total = sum(abs(v) for v in x)
        if total > 4.0:
            x = [v / total * 4.0 for v in x]
        return x

    def process(self, inputs: ImpulseStream) -> ImpulseStream:
        outputs = ImpulseStream()
        buckets = self._bucketize(inputs)
        emitted = 0
        peak_out = 0.0
        for t, items in buckets:
            x = self._vector_from_bucket(items)
            old = self.state[:]
            new_h = [0.0] * self.hidden_size
            for i in range(self.hidden_size):
                s = self.b_h[i]
                s += sum(self.w_in[i][j] * x[j] for j in range(self.input_channels))
                s += sum(self.w_rec[i][j] * old[j] for j in range(self.hidden_size))
                new_h[i] = math.tanh(s)
            self.state = new_h

            out_act = [0.0] * self.output_channels
            for k in range(self.output_channels):
                s = self.b_o[k] + sum(self.w_out[k][j] * self.state[j] for j in range(self.hidden_size))
                out_act[k] = math.tanh(s)
                peak_out = max(peak_out, abs(out_act[k]))
                if abs(out_act[k]) > 0.45:
                    outputs.add(
                        Impulse(
                            t=t,
                            ch=k,
                            v=abs(out_act[k]),
                            kind="nn_out",
                            meta={"sign": "pos" if out_act[k] >= 0 else "neg"},
                        )
                    )
                    emitted += 1

        self.last_stats = {
            "input_impulses": len(inputs.impulses),
            "time_buckets": len(buckets),
            "output_impulses": emitted,
            "peak_output_activation": peak_out,
        }
        outputs.sort()
        return outputs

    def _forward_trace(self, inputs: ImpulseStream) -> List[Dict[str, Any]]:
        """
        Forward pass with per-time-step trace for lightweight supervised training.
        Leaves self.state at the end of the sequence.
        """
        trace: List[Dict[str, Any]] = []
        buckets = self._bucketize(inputs)
        for t, items in buckets:
            x = self._vector_from_bucket(items)
            old = self.state[:]
            new_h = [0.0] * self.hidden_size
            for i in range(self.hidden_size):
                s = self.b_h[i]
                s += sum(self.w_in[i][j] * x[j] for j in range(self.input_channels))
                s += sum(self.w_rec[i][j] * old[j] for j in range(self.hidden_size))
                new_h[i] = math.tanh(s)
            self.state = new_h
            z_out = [0.0] * self.output_channels
            y_out = [0.0] * self.output_channels
            for k in range(self.output_channels):
                z = self.b_o[k] + sum(self.w_out[k][j] * self.state[j] for j in range(self.hidden_size))
                z_out[k] = z
                y_out[k] = math.tanh(z)
            trace.append({"t": t, "hidden": self.state[:], "out": y_out, "inp_count": len(items)})
        return trace

    def _target_schedule(self, n_steps: int, target_channels: List[int]) -> List[int | None]:
        """
        Map target token channels onto time buckets.
        Uses evenly spaced anchor points; other buckets are unlabeled (None).
        """
        if n_steps <= 0:
            return []
        schedule: List[int | None] = [None] * n_steps
        if not target_channels:
            return schedule
        if len(target_channels) == 1:
            schedule[n_steps // 2] = int(target_channels[0])
            return schedule
        for i, ch in enumerate(target_channels):
            pos = round(i * (n_steps - 1) / max(1, (len(target_channels) - 1)))
            schedule[int(pos)] = int(ch)
        return schedule

    def _dynamic_target_schedule(self, trace: List[Dict[str, Any]], target_channels: List[int]) -> List[int | None]:
        """
        Monotonic dynamic alignment (CTC-like idea, simplified):
        align target tokens to a monotonic subsequence of time steps maximizing target-channel activation.
        """
        n_steps = len(trace)
        n_toks = len(target_channels)
        schedule: List[int | None] = [None] * n_steps
        if n_steps <= 0 or n_toks <= 0:
            return schedule
        if n_toks == 1:
            best_i = max(range(n_steps), key=lambda i: trace[i]["out"][target_channels[0]])
            schedule[best_i] = int(target_channels[0])
            return schedule
        if n_steps < n_toks:
            # fallback to anchors if too few steps
            return self._target_schedule(n_steps, target_channels)

        neg_inf = -1e18
        dp = [[neg_inf] * n_toks for _ in range(n_steps)]
        back = [[-1] * n_toks for _ in range(n_steps)]
        for i in range(n_steps):
            score = float(trace[i]["out"][target_channels[0]])
            if i == 0:
                dp[i][0] = score
            else:
                # can start token0 at any step, keep best so far
                if dp[i - 1][0] >= score:
                    dp[i][0] = dp[i - 1][0]
                    back[i][0] = back[i - 1][0]
                else:
                    dp[i][0] = score
                    back[i][0] = -2  # start here
        for j in range(1, n_toks):
            for i in range(j, n_steps):
                score = float(trace[i]["out"][target_channels[j]])
                best_prev = neg_inf
                best_idx = -1
                # monotonic predecessor over earlier steps
                for p in range(j - 1, i):
                    if dp[p][j - 1] > best_prev:
                        best_prev = dp[p][j - 1]
                        best_idx = p
                if best_idx >= 0 and best_prev > neg_inf / 2:
                    dp[i][j] = best_prev + score
                    back[i][j] = best_idx

        # end at best step for final token
        end_i = max(range(n_toks - 1, n_steps), key=lambda i: dp[i][n_toks - 1])
        if dp[end_i][n_toks - 1] <= neg_inf / 2:
            return self._target_schedule(n_steps, target_channels)

        aligned_steps = [-1] * n_toks
        i = end_i
        j = n_toks - 1
        while j >= 0 and i >= 0:
            aligned_steps[j] = i
            prev = back[i][j]
            if j == 0:
                break
            if prev < 0:
                return self._target_schedule(n_steps, target_channels)
            i = prev
            j -= 1

        if any(idx < 0 for idx in aligned_steps):
            return self._target_schedule(n_steps, target_channels)
        for idx, ch in zip(aligned_steps, target_channels):
            schedule[idx] = int(ch)
        return schedule

    def _schedule_for_training(
        self,
        trace: List[Dict[str, Any]],
        target_channels: List[int],
        align_mode: str = "dynamic",
    ) -> List[int | None]:
        if align_mode == "ctc":
            return self._ctc_viterbi_schedule(trace, target_channels)
        if align_mode == "dynamic":
            return self._dynamic_target_schedule(trace, target_channels)
        return self._target_schedule(len(trace), target_channels)

    def _ctc_viterbi_schedule(self, trace: List[Dict[str, Any]], target_channels: List[int]) -> List[int | None]:
        """
        CTC-style Viterbi alignment with a reserved blank label (last output channel).
        Handles repeated tokens via blank-separated extended targets.
        Returns sparse per-time schedule labels (includes blank labels at aligned blank states).
        """
        n_steps = len(trace)
        schedule: List[int | None] = [None] * n_steps
        if n_steps <= 0:
            return schedule
        if not target_channels:
            return schedule

        blank = self.output_channels - 1
        labels = [int(ch) for ch in target_channels if int(ch) != blank]
        if not labels:
            return schedule

        # Extended target with blanks: [b, l1, b, l2, ..., b, lN, b]
        ext: List[int] = [blank]
        for ch in labels:
            ext.append(ch)
            ext.append(blank)
        s_len = len(ext)
        neg_inf = -1e18
        dp = [[neg_inf] * s_len for _ in range(n_steps)]
        back = [[-1] * s_len for _ in range(n_steps)]

        def score_at(ti: int, state_idx: int) -> float:
            lab = ext[state_idx]
            return float(trace[ti]["out"][lab])

        # init at t=0: can be at state 0 (blank) or state 1 (first label)
        dp[0][0] = score_at(0, 0)
        back[0][0] = -2
        if s_len > 1:
            dp[0][1] = score_at(0, 1)
            back[0][1] = -2

        for t in range(1, n_steps):
            for s in range(s_len):
                best_prev = dp[t - 1][s]  # stay
                best_state = s

                if s - 1 >= 0 and dp[t - 1][s - 1] > best_prev:
                    best_prev = dp[t - 1][s - 1]
                    best_state = s - 1

                # skip over blank / repeated constraint (CTC rule)
                if s - 2 >= 0:
                    can_skip = ext[s] != blank and ext[s] != ext[s - 2]
                    if can_skip and dp[t - 1][s - 2] > best_prev:
                        best_prev = dp[t - 1][s - 2]
                        best_state = s - 2

                if best_prev <= neg_inf / 2:
                    continue
                dp[t][s] = best_prev + score_at(t, s)
                back[t][s] = best_state

        # valid end states are final blank or final label
        end_candidates = [s_len - 1]
        if s_len - 2 >= 0:
            end_candidates.append(s_len - 2)
        end_s = max(end_candidates, key=lambda s: dp[n_steps - 1][s])
        if dp[n_steps - 1][end_s] <= neg_inf / 2:
            return self._dynamic_target_schedule(trace, labels)

        path_states = [0] * n_steps
        t = n_steps - 1
        s = end_s
        while t >= 0:
            path_states[t] = s
            prev = back[t][s]
            if t == 0:
                break
            if prev < 0:
                # fallback if traceback breaks
                return self._dynamic_target_schedule(trace, labels)
            s = prev
            t -= 1

        # Emit labels only when state changes (sparse schedule), keeping blanks too.
        prev_state = None
        for ti, s_idx in enumerate(path_states):
            if s_idx != prev_state:
                schedule[ti] = ext[s_idx]
            prev_state = s_idx
        return schedule

    def _supervised_loss_from_trace(
        self,
        trace: List[Dict[str, Any]],
        target_channels: List[int],
        *,
        align_mode: str = "dynamic",
        positive_target: float = 0.9,
        negative_target: float = -0.15,
    ) -> Dict[str, Any]:
        schedule = self._schedule_for_training(trace, target_channels, align_mode=align_mode)
        loss_sum = 0.0
        count = 0
        labeled = 0
        for step_idx, step in enumerate(trace):
            target_ch = schedule[step_idx]
            if target_ch is None:
                continue
            labeled += 1
            y = step["out"]
            for k in range(self.output_channels):
                target = positive_target if k == target_ch else negative_target
                err = target - y[k]
                loss_sum += err * err
                count += 1
        return {
            "mse": loss_sum / max(1, count),
            "schedule": schedule,
            "labeled_steps": labeled,
        }

    def evaluate_supervised_loss(
        self,
        inputs: ImpulseStream,
        target_channels: List[int],
        *,
        align_mode: str = "dynamic",
        positive_target: float = 0.9,
        negative_target: float = -0.15,
    ) -> Dict[str, Any]:
        self.reset_state()
        trace = self._forward_trace(inputs)
        out = self._supervised_loss_from_trace(
            trace,
            target_channels,
            align_mode=align_mode,
            positive_target=positive_target,
            negative_target=negative_target,
        )
        self.reset_state()
        out["time_buckets"] = len(trace)
        return out

    def train_output_supervised(
        self,
        inputs: ImpulseStream,
        target_channels: List[int],
        *,
        epochs: int = 8,
        lr: float = 0.05,
        positive_target: float = 0.9,
        negative_target: float = -0.15,
        reset_each_epoch: bool = True,
        align_mode: str = "dynamic",
    ) -> Dict[str, Any]:
        """
        Lightweight supervised training of the output layer (w_out, b_o).

        This is a real training loop (gradient-based updates), but intentionally simple:
        - hidden dynamics are frozen
        - only output layer is updated
        - targets are aligned to evenly spaced time anchors
        """
        epochs = max(1, int(epochs))
        lr = float(lr)
        history: List[float] = []
        labeled_steps = 0
        for _epoch in range(epochs):
            if reset_each_epoch:
                self.reset_state()
            trace = self._forward_trace(inputs)
            if not trace:
                history.append(0.0)
                continue
            schedule = self._schedule_for_training(trace, target_channels, align_mode=align_mode)
            loss_sum = 0.0
            count = 0
            for step_idx, step in enumerate(trace):
                target_ch = schedule[step_idx]
                if target_ch is None:
                    continue
                h = step["hidden"]
                y = step["out"]
                labeled_steps += 1
                for k in range(self.output_channels):
                    target = positive_target if k == target_ch else negative_target
                    err = target - y[k]
                    # d/dz tanh(z) = 1 - tanh(z)^2
                    grad = err * (1.0 - y[k] * y[k])
                    self.b_o[k] += lr * grad
                    row = self.w_out[k]
                    for j in range(self.hidden_size):
                        row[j] += lr * grad * h[j]
                    loss_sum += err * err
                    count += 1
            history.append(loss_sum / max(1, count))
        if reset_each_epoch:
            self.reset_state()
        return {
            "epochs": epochs,
            "lr": lr,
            "align_mode": align_mode,
            "target_len": len(target_channels),
            "labeled_steps_per_epoch": sum(1 for x in self.evaluate_supervised_loss(inputs, target_channels, align_mode=align_mode)["schedule"] if x is not None),
            "history_mse": history,
            "mse_start": history[0] if history else 0.0,
            "mse_end": history[-1] if history else 0.0,
        }

    def _hidden_params_snapshot(self) -> Dict[str, Any]:
        return {
            "w_in": [row[:] for row in self.w_in],
            "w_rec": [row[:] for row in self.w_rec],
            "b_h": self.b_h[:],
        }

    def _restore_hidden_params(self, snap: Dict[str, Any]) -> None:
        self.w_in = [row[:] for row in snap["w_in"]]
        self.w_rec = [row[:] for row in snap["w_rec"]]
        self.b_h = snap["b_h"][:]

    def _mutate_hidden_params(self, rng: random.Random, scale: float) -> Dict[str, Any]:
        snap = self._hidden_params_snapshot()
        # sparse perturbations for speed/stability
        num_in = max(1, (self.hidden_size * self.input_channels) // 10)
        num_rec = max(1, (self.hidden_size * self.hidden_size) // 20)
        num_b = max(1, self.hidden_size // 4)
        for _ in range(num_in):
            i = rng.randrange(self.hidden_size)
            j = rng.randrange(self.input_channels)
            self.w_in[i][j] += (rng.random() * 2 - 1) * scale
        for _ in range(num_rec):
            i = rng.randrange(self.hidden_size)
            j = rng.randrange(self.hidden_size)
            self.w_rec[i][j] += (rng.random() * 2 - 1) * scale * 0.6
        for _ in range(num_b):
            i = rng.randrange(self.hidden_size)
            self.b_h[i] += (rng.random() * 2 - 1) * scale * 0.4
        return snap

    def train_hidden_evolutionary(
        self,
        inputs: ImpulseStream,
        target_channels: List[int],
        *,
        iterations: int = 40,
        population: int = 6,
        sigma: float = 0.06,
        sigma_decay: float = 0.98,
        align_mode: str = "dynamic",
        seed: int | None = None,
    ) -> Dict[str, Any]:
        """
        Evolutionary search over hidden/recurrent weights (w_in, w_rec, b_h).
        Uses supervised loss as fitness.
        """
        rng = random.Random(self.seed if seed is None else seed)
        best_loss_info = self.evaluate_supervised_loss(inputs, target_channels, align_mode=align_mode)
        best_loss = float(best_loss_info["mse"])
        accepted = 0
        history = [best_loss]
        for it in range(max(1, int(iterations))):
            iter_best_loss = best_loss
            iter_best_snap: Dict[str, Any] | None = None
            for _ in range(max(1, int(population))):
                snap = self._mutate_hidden_params(rng, sigma)
                cand = self.evaluate_supervised_loss(inputs, target_channels, align_mode=align_mode)
                cand_loss = float(cand["mse"])
                if cand_loss < iter_best_loss:
                    iter_best_loss = cand_loss
                    iter_best_snap = self._hidden_params_snapshot()
                self._restore_hidden_params(snap)
            if iter_best_snap is not None and iter_best_loss < best_loss:
                self._restore_hidden_params(iter_best_snap)
                best_loss = iter_best_loss
                accepted += 1
            history.append(best_loss)
            sigma = max(0.005, sigma * sigma_decay)
        return {
            "method": "evolutionary_hidden",
            "iterations": int(iterations),
            "population": int(population),
            "align_mode": align_mode,
            "accepted_generations": accepted,
            "sigma_final": sigma,
            "mse_start": history[0] if history else best_loss,
            "mse_end": history[-1] if history else best_loss,
            "history_best_mse": history,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "input_channels": self.input_channels,
            "hidden_size": self.hidden_size,
            "output_channels": self.output_channels,
            "last_stats": dict(self.last_stats),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_channels": self.input_channels,
            "hidden_size": self.hidden_size,
            "output_channels": self.output_channels,
            "seed": self.seed,
            "w_in": self.w_in,
            "w_rec": self.w_rec,
            "w_out": self.w_out,
            "b_h": self.b_h,
            "b_o": self.b_o,
            "state": self.state,
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ImpulseNeuralCore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        core = cls(
            input_channels=int(data["input_channels"]),
            hidden_size=int(data["hidden_size"]),
            output_channels=int(data["output_channels"]),
            seed=int(data.get("seed", 7)),
        )
        core.w_in = data["w_in"]
        core.w_rec = data["w_rec"]
        core.w_out = data["w_out"]
        core.b_h = data["b_h"]
        core.b_o = data["b_o"]
        core.state = data.get("state", core.state)
        return core
