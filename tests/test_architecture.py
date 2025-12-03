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


def test_reward_shaping_v2():
    reward, details = compute_reward(
        {"tests_ok": True, "eval_ok": True, "tests_delta": 1, "regressions": 0, "domain": "code"},
        return_details=True,
    )
    assert reward > 1.0
    assert details["domain"] == "code"
    clipped, _ = compute_reward({"reward": 10}, return_details=True)
    assert clipped <= 2.0
    negative, _ = compute_reward({"tests_ok": False, "eval_ok": False, "regressions": 2}, return_details=True)
    assert negative < 0


def test_explain_action(tmp_path):
    metrics_module.METRICS_FILE = tmp_path / "metrics.json"
    vf_module.VALUE_STATE_FILE = tmp_path / "vf.json"
    metrics = metrics_module.Metrics()
    g = NeuronGraph(tmp_path / "g.json")
    g.add_node("plugin:test", "plugin", [0.5, 0.1], {})
    g.add_node("error:x", "error", [0.2, 0.2], {})
    g.add_edge("plugin:test", "error:x", 0.8)
    vf = vf_module.ValueFunction(g, metrics=metrics, alpha=0.1)
    metrics.record_plugin_outcome("plugin:test", True, 0.5, {"base": 1.0})
    vf.update_plugin("plugin:test", 0.5)
    explanation = vf.explain_action("plugin:test")
    assert explanation["score"] != 0
    assert explanation["components"]["historical"] != 0 or explanation["components"]["metric"] != 0
    assert explanation["neighbors"]


def test_export_for_viz(tmp_path):
    g = NeuronGraph(tmp_path / "viz.json")
    g.add_node("plugin:a", "plugin", [0.1, 0.1], {})
    g.add_node("plugin:b", "plugin", [0.2, 0.3], {})
    g.add_edge("plugin:a", "plugin:b", weight=1.2)
    subgraph = g.export_for_viz(max_nodes=10, min_weight=0.1)
    assert subgraph["nodes"]
    assert subgraph["edges"]


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


def test_async_step_runs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HUMAN_ASYNC", "1")
    monkeypatch.setenv("HUMAN_ENABLE_ENVS", "0")
    (tmp_path / "manager").mkdir(parents=True, exist_ok=True)
    (tmp_path / "plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tasks").mkdir(parents=True, exist_ok=True)
    metrics_module.METRICS_FILE = tmp_path / "manager" / "metrics.json"
    vf_module.VALUE_STATE_FILE = tmp_path / "manager" / "vf.json"
    from manager.mind import Mind

    mind = Mind()

    def fake_act(tasks, forced_targets=None, strategy=None):
        mind._current_step_actions.append(
            {"plugin": "fake", "pattern": "noop", "result": "accepted", "tests_ok": True, "eval_ok": True, "reward": 1.0}
        )

    monkeypatch.setattr(mind, "_act_and_learn", fake_act)
    mind.step()
    assert metrics_module.METRICS_FILE.exists()


def test_env_and_code_schedule(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HUMAN_ENABLE_ENVS", "1")
    monkeypatch.setenv("HUMAN_ENV_RATIO", "1.0")
    (tmp_path / "manager").mkdir(parents=True, exist_ok=True)
    (tmp_path / "plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tasks").mkdir(parents=True, exist_ok=True)
    metrics_module.METRICS_FILE = tmp_path / "manager" / "metrics.json"
    vf_module.VALUE_STATE_FILE = tmp_path / "manager" / "vf.json"
    from manager.mind import Mind

    mind = Mind()
    monkeypatch.setattr(mind, "_act_and_learn", lambda active_task_names, forced_targets=None, strategy=None: None)
    mind.step()  # env-focused step

    def fake_code(tasks, forced_targets=None, strategy=None):
        mind._current_step_actions.append(
            {"plugin": "fake", "pattern": "noop", "result": "accepted", "tests_ok": True, "eval_ok": True, "reward": 0.5}
        )

    mind.enable_envs = False
    monkeypatch.setattr(mind, "_act_and_learn", fake_code)
    mind.step()  # code-focused step
    domains = mind.metrics.state.get("recent_domains", [])
    assert any(d in {"env", "mixed"} for d in domains)
    assert "code" in domains or "mixed" in domains
