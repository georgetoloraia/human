#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter, defaultdict

TRACE_PATH = Path("manager/traces/trace.jsonl")
METRICS_PATH = Path("manager/metrics.json")

def load_metrics():
    if not METRICS_PATH.exists():
        print("metrics.json not found")
        return {}
    return json.loads(METRICS_PATH.read_text())

def load_traces(max_lines=None):
    if not TRACE_PATH.exists():
        print("trace.jsonl not found at", TRACE_PATH)
        return []
    traces = []
    with TRACE_PATH.open() as f:
        for i, line in enumerate(f):
            try:
                traces.append(json.loads(line))
            except Exception:
                continue
            if max_lines is not None and i >= max_lines:
                break
    return traces

def analyze_traces(traces, window=200):
    if not traces:
        print("No traces to analyze.")
        return

    # basic reward stats and domain mix
    rewards = []
    domains = Counter()
    streak = 0
    worst_streak = 0

    for step in traces:
        domain = step.get("domain", "unknown")
        domains[domain] += 1
        step_rewards = []
        for action in step.get("actions", []):
            r = action.get("reward", 0.0)
            rewards.append(r)
            step_rewards.append(r)
        # fallback if no actions recorded
        if not step_rewards:
            rewards.append(step.get("reward", 0.0))
        step_reward = sum(step_rewards) / len(step_rewards) if step_rewards else step.get("reward", 0.0)
        if step_reward <= 0:
            streak -= 1
        else:
            streak = 0
        worst_streak = min(worst_streak, streak)

    avg_reward = sum(rewards) / len(rewards)
    total_steps = len(traces)
    total_reward = sum(rewards) if rewards else 0.0
    avg_reward = (total_reward / len(rewards)) if rewards else 0.0
    print(f"Steps in trace: {total_steps}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Worst negative streak: {worst_streak}")
    print("Domain counts:", dict(domains))

    # moving average of reward (last window steps)
    if len(rewards) >= window:
        recent = rewards[-window:]
        recent_avg = sum(recent) / len(recent)
        print(f"Average reward over last {window} steps: {recent_avg:.3f}")

    # plugin/strategy frequency from trace
    plugin_counter = Counter()
    strategy_counter = Counter()
    unsafe_counter = 0

    for step in traces:
        for action in step.get("actions", []):
            plugin = action.get("plugin")
            strategy = action.get("pattern") or action.get("strategy")
            if plugin:
                plugin_counter[plugin] += 1
            if strategy:
                strategy_counter[strategy] += 1
            if action.get("error_type") == "Unsafe" or action.get("result") == "unsafe":
                unsafe_counter += 1

    print("\nTop 10 plugins (from trace):")
    for name, cnt in plugin_counter.most_common(10):
        print(" ", name, "=>", cnt)

    print("\nStrategies (from trace):")
    for name, cnt in strategy_counter.most_common():
        print(" ", name, "=>", cnt)

    print(f"\nApprox. Unsafe-related steps: {unsafe_counter}")

def main():
    metrics = load_metrics()
    if metrics:
        print("=== METRICS SUMMARY ===")
        print("Total steps:", metrics.get("total_steps"))
        print("Total reward:", metrics.get("total_reward"))
        print("Average reward:", metrics.get("average_reward"))
        print("Domains:", metrics.get("domains", {}))
        print()

    print("=== TRACE ANALYSIS ===")
    traces = load_traces()
    analyze_traces(traces)

if __name__ == "__main__":
    main()
