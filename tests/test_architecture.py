from __future__ import annotations

import manager.metrics as metrics_module
import manager.value_function as vf_module
from manager.neuron_graph import NeuronGraph
from manager.reward import compute_reward


def test_neuron_graph_roundtrip(tmp_path):
    path = tmp_path / "graph.json"
    g = NeuronGraph(path)
    g.add_node("plugin:a", "plugin", [0.1, 0.2], {"label": "a"})
    g.add_node("error:x", "error", [0.3, 0.4], {"label": "x"})
    g.add_edge("plugin:a", "error:x", weight=0.5)
    g.update_edge_weight("plugin:a", "error:x", 0.5)
    g.save()

    reloaded = NeuronGraph(path)
    assert "plugin:a" in reloaded.nodes
    neighbors = reloaded.get_neighbors("plugin:a", top_k=1)
    assert neighbors and neighbors[0]["id"] == "error:x"
    assert neighbors[0]["weight"] >= 1.0


def test_value_function_scores_with_updates(tmp_path):
    vf_module.VALUE_STATE_FILE = tmp_path / "vf.json"
    g = NeuronGraph(tmp_path / "g.json")
    g.add_node("plugin:test", "plugin", [1.0, 0.0], {})
    vf = vf_module.ValueFunction(g, metrics=None, alpha=0.0)
    vf.update_plugin("plugin:test", 1.0)
    score = vf.score("plugin:test", candidate_type="plugin")
    assert score > 0


def test_reward_and_metrics(tmp_path):
    metrics_module.METRICS_FILE = tmp_path / "metrics.json"
    metrics = metrics_module.Metrics()
    metrics.record_plugin_outcome("plugin:test", True, 1.0)
    metrics.record_plugin_outcome("plugin:test", False, -1.0)
    assert metrics.average_reward("plugin:test") == 0.0
    assert compute_reward({"tests_ok": True, "eval_ok": True}) > 0
    assert compute_reward({"tests_ok": False, "eval_ok": False}) < 0


def test_multi_agent_smoke(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HUMAN_MULTI_AGENT", "1")
    monkeypatch.setenv("HUMAN_ENABLE_ENVS", "0")
    monkeypatch.setenv("USE_VALUE_FUNCTION", "0")
    (tmp_path / "manager").mkdir(parents=True, exist_ok=True)
    (tmp_path / "plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tasks").mkdir(parents=True, exist_ok=True)
    vf_module.VALUE_STATE_FILE = tmp_path / "vf.json"
    metrics_module.METRICS_FILE = tmp_path / "metrics.json"
    from manager.mind import Mind  # import after chdir to avoid touching repo files

    mind = Mind()
    monkeypatch.setattr(mind, "_act_and_learn", lambda active_task_names, forced_targets=None, strategy=None: None)
    mind._step_multi_agent([])
    assert hasattr(mind, "planner_agent")
