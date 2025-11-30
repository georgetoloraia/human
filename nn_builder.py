"""
Command-line neural network builder with chaining and text-to-neuron graph mode.

- Numeric mode: asks for model settings (input size, hidden layers, output size,
  activation, learning rate) via flags or interactive prompts and builds one or
  more feed-forward networks in NumPy. Each additional network is automatically
  connected to the previous network's outputs. A tiny training demo on random
  data can be triggered with --demo-epochs.
- Text mode: pass --text-files or --interactive-text to feed files; the script
  breaks content into tokens, instantiates them as "neurons", and connects each
  new batch to all previously seen neurons (subject to --bridge-limit). Can
  stream neurons/edges directly into PostgreSQL with --pg-url or SQLite with
  --sqlite-out to avoid holding everything in RAM.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


# ----- Activation functions -------------------------------------------------
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_grad(x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return 1.0 - t * t


def linear(x: np.ndarray) -> np.ndarray:
    return x


def linear_grad(_: np.ndarray) -> np.ndarray:
    return 1.0


ACTIVATIONS = {
    "relu": (relu, relu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
    "linear": (linear, linear_grad),
}


# ----- Core network pieces --------------------------------------------------
@dataclass
class DenseLayer:
    input_dim: int
    output_dim: int
    activation: str

    def __post_init__(self) -> None:
        self.act_fn, self.act_grad = ACTIVATIONS[self.activation]
        # He initialization works well with ReLU; it is still stable for the rest.
        self.W = np.random.randn(self.input_dim, self.output_dim) * np.sqrt(
            2.0 / self.input_dim
        )
        self.b = np.zeros((1, self.output_dim))
        self._last_z: np.ndarray | None = None
        self._last_x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_x = x
        z = x @ self.W + self.b
        self._last_z = z
        return self.act_fn(z)

    def backward(self, grad_out: np.ndarray, lr: float) -> np.ndarray:
        if self._last_x is None or self._last_z is None:
            raise RuntimeError("Call forward before backward.")
        grad_activation = grad_out * self.act_grad(self._last_z)
        grad_w = self._last_x.T @ grad_activation / self._last_x.shape[0]
        grad_b = grad_activation.mean(axis=0, keepdims=True)
        grad_input = grad_activation @ self.W.T

        self.W -= lr * grad_w
        self.b -= lr * grad_b
        return grad_input

    def summary_row(self) -> str:
        params = self.W.size + self.b.size
        return f"Dense({self.input_dim}->{self.output_dim}, act={self.activation}) | params: {params}"


class NeuralNetwork:
    def __init__(
        self, input_dim: int, hidden_layers: List[int], output_dim: int, activation: str
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        sizes = [input_dim] + hidden_layers + [output_dim]
        self.layers = [
            DenseLayer(sizes[i], sizes[i + 1], activation=activation)
            for i in range(len(sizes) - 1)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
        return grad

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        # Mean squared error loss to keep things simple.
        preds = self.forward(x)
        loss = np.mean((preds - y) ** 2)
        grad = 2.0 * (preds - y) / y.shape[0]
        self.backward(grad, lr)
        return float(loss)

    def summary(self) -> str:
        lines = ["Neural network layout:"]
        for idx, layer in enumerate(self.layers):
            lines.append(f"  [{idx}] {layer.summary_row()}")
        total_params = sum(layer.W.size + layer.b.size for layer in self.layers)
        lines.append(f"Total parameters: {total_params}")
        return "\n".join(lines)


class NetworkChain:
    def __init__(self, networks: List[NeuralNetwork]) -> None:
        if not networks:
            raise ValueError("At least one network is required.")
        self.networks = networks
        self._validate_chain()

    def _validate_chain(self) -> None:
        for idx in range(1, len(self.networks)):
            prev = self.networks[idx - 1]
            curr = self.networks[idx]
            if curr.input_dim != prev.output_dim:
                raise ValueError(
                    f"Network {idx} expects {curr.input_dim} inputs but previous network outputs {prev.output_dim}."
                )

    def forward(self, x: np.ndarray) -> np.ndarray:
        for net in self.networks:
            x = net.forward(x)
        return x

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        preds = self.forward(x)
        loss = np.mean((preds - y) ** 2)
        grad = 2.0 * (preds - y) / y.shape[0]
        for net in reversed(self.networks):
            grad = net.backward(grad, lr)
        return float(loss)

    def summary(self) -> str:
        lines = ["Neural network chain:"]
        for idx, net in enumerate(self.networks):
            lines.append(f"[Net {idx}] in={net.input_dim} out={net.output_dim}")
            for layer_idx, layer in enumerate(net.layers):
                lines.append(f"  L{layer_idx} {layer.summary_row()}")
        total_params = sum(
            layer.W.size + layer.b.size for net in self.networks for layer in net.layers
        )
        lines.append(f"Total parameters across chain: {total_params}")
        return "\n".join(lines)


# ----- Text-to-neuron graph builder ----------------------------------------
@dataclass
class Neuron:
    id: int
    label: str
    source: str


class GraphSink:
    def add_neurons(self, records: list[tuple[int, str, str]]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def add_edges(self, records: list[tuple[int, int, str]]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        pass


class NeuronGraph:
    def __init__(
        self,
        bridge_limit: int | None = None,
        sink: GraphSink | None = None,
        store_neurons: bool = True,
        store_edges: bool = True,
    ) -> None:
        # None bridge_limit means unlimited
        self.bridge_limit = None if bridge_limit is None or bridge_limit < 0 else bridge_limit
        self.neurons: list[Neuron] | None = [] if store_neurons else None
        self.edges: list[tuple[int, int, str]] | None = [] if store_edges else None
        self.sink = sink
        self._bridge_capped = False
        self.neuron_count = 0
        self.edge_count = 0
        self.notes: list[str] = []
        self.per_source: dict[str, int] = {}

    def add_tokens(self, tokens: list[str], source: str) -> None:
        if not tokens:
            return
        start = self.neuron_count
        new_ids = list(range(start, start + len(tokens)))

        if self.neurons is not None:
            for tok in tokens:
                self.neurons.append(Neuron(id=len(self.neurons), label=tok, source=source))
        if self.sink:
            self.sink.add_neurons([(nid, tok, source) for nid, tok in zip(new_ids, tokens)])
        self.per_source[source] = self.per_source.get(source, 0) + len(tokens)
        self.neuron_count += len(tokens)

        # Connect tokens within the file in order.
        seq_edges = [(a, b, "sequence") for a, b in zip(new_ids[:-1], new_ids[1:])]
        if seq_edges:
            if self.edges is not None:
                self.edges.extend(seq_edges)
            if self.sink:
                self.sink.add_edges(seq_edges)
            self.edge_count += len(seq_edges)

        # Connect all prior neurons to the new ones (bridge fan-in).
        if start > 0 and new_ids:
            prev_ids = list(range(start))
            total_bridges = len(prev_ids) * len(new_ids)
            bridge_edges: list[tuple[int, int, str]] = []
            if self.bridge_limit is None or total_bridges <= self.bridge_limit:
                for p in prev_ids:
                    for n in new_ids:
                        bridge_edges.append((p, n, "bridge"))
            else:
                # Cap by only connecting the most recent chunk of previous neurons.
                span = max(1, self.bridge_limit // max(1, len(new_ids)))
                subset_prev = prev_ids[-span:]
                for p in subset_prev:
                    for n in new_ids:
                        bridge_edges.append((p, n, "bridge"))
                self._bridge_capped = True
            if bridge_edges:
                if self.edges is not None:
                    self.edges.extend(bridge_edges)
                if self.sink:
                    self.sink.add_edges(bridge_edges)
                self.edge_count += len(bridge_edges)

    def summary(self) -> str:
        per_source = self.per_source
        lines = [
            "Neuron graph summary:",
            f"- Total neurons: {self.neuron_count}",
            f"- Total edges:   {self.edge_count if self.edge_count else len(self.edges or [])}",
        ]
        for src, count in per_source.items():
            lines.append(f"- {src}: {count} neurons")
        if self._bridge_capped:
            lines.append(
                "- Bridge edges capped by --bridge-limit (set to -1 for full connectivity)."
            )
        for note in self.notes:
            lines.append(f"- {note}")
        return "\n".join(lines)


# ----- CLI helpers ----------------------------------------------------------
def prompt_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    return int(raw) if raw else default


def prompt_hidden(default: list[int]) -> list[int]:
    raw = input(
        f"Hidden layer sizes separated by spaces (empty for none) {default}: "
    ).strip()
    if not raw:
        return default
    return [int(part) for part in raw.split()]


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def tokenize_text(text: str) -> list[str]:
    # Simple word-level tokenizer; keeps alphanumeric and apostrophes.
    return re.findall(r"[A-Za-z0-9']+", text.lower())


class PostgresSink(GraphSink):
    def __init__(self, url: str, schema: str = "public", batch_size: int = 5000) -> None:
        self.url = url
        self.schema = schema
        self.batch_size = batch_size
        try:
            try:
                import psycopg  # type: ignore

                self._db = "psycopg"
                self.conn = psycopg.connect(self.url)
            except ModuleNotFoundError:
                import psycopg2  # type: ignore

                self._db = "psycopg2"
                self.conn = psycopg2.connect(self.url)
        except Exception as exc:  # pragma: no cover - connectivity/environment
            raise SystemExit(f"Failed to connect to PostgreSQL: {exc}")
        self.cur = self.conn.cursor()
        self._setup_schema()
        self._neurons_buffer: list[tuple[int, str, str]] = []
        self._edges_buffer: list[tuple[int, int, str]] = []

    def _execute(self, sql: str, params: tuple | None = None) -> None:
        self.cur.execute(sql, params or ())

    def _setup_schema(self) -> None:
        self._execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS "{self.schema}".neurons (
                id BIGINT PRIMARY KEY,
                label TEXT,
                source TEXT
            )
            """
        )
        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS "{self.schema}".edges (
                src BIGINT,
                dst BIGINT,
                kind TEXT
            )
            """
        )
        self._execute(
            f'CREATE INDEX IF NOT EXISTS edges_src_idx ON "{self.schema}".edges(src)'
        )
        self._execute(
            f'CREATE INDEX IF NOT EXISTS edges_dst_idx ON "{self.schema}".edges(dst)'
        )
        self.conn.commit()

    def _flush_neurons(self) -> None:
        if not self._neurons_buffer:
            return
        self.cur.executemany(
            f'INSERT INTO "{self.schema}".neurons (id, label, source) VALUES (%s, %s, %s)',
            self._neurons_buffer,
        )
        self._neurons_buffer.clear()

    def _flush_edges(self) -> None:
        if not self._edges_buffer:
            return
        self.cur.executemany(
            f'INSERT INTO "{self.schema}".edges (src, dst, kind) VALUES (%s, %s, %s)',
            self._edges_buffer,
        )
        self._edges_buffer.clear()

    def add_neurons(self, records: list[tuple[int, str, str]]) -> None:
        self._neurons_buffer.extend(records)
        if len(self._neurons_buffer) >= self.batch_size:
            self._flush_neurons()
            self.conn.commit()

    def add_edges(self, records: list[tuple[int, int, str]]) -> None:
        self._edges_buffer.extend(records)
        if len(self._edges_buffer) >= self.batch_size:
            self._flush_edges()
            self.conn.commit()

    def close(self) -> None:
        self._flush_neurons()
        self._flush_edges()
        self.conn.commit()
        self.cur.close()
        self.conn.close()


class SQLiteSink(GraphSink):
    def __init__(self, path: str, batch_size: int = 5000) -> None:
        self.path = path
        self.batch_size = batch_size
        try:
            self.conn = sqlite3.connect(self.path)
        except Exception as exc:  # pragma: no cover - connectivity/environment
            raise SystemExit(f"Failed to open SQLite database: {exc}")
        self.cur = self.conn.cursor()
        self._setup_schema()
        self._neurons_buffer: list[tuple[int, str, str]] = []
        self._edges_buffer: list[tuple[int, int, str]] = []

    def _setup_schema(self) -> None:
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS neurons (
                id INTEGER PRIMARY KEY,
                label TEXT,
                source TEXT
            )
            """
        )
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                src INTEGER,
                dst INTEGER,
                kind TEXT
            )
            """
        )
        self.cur.execute("CREATE INDEX IF NOT EXISTS edges_src_idx ON edges(src)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS edges_dst_idx ON edges(dst)")
        self.conn.commit()

    def _flush_neurons(self) -> None:
        if not self._neurons_buffer:
            return
        self.cur.executemany(
            "INSERT OR REPLACE INTO neurons (id, label, source) VALUES (?, ?, ?)",
            self._neurons_buffer,
        )
        self._neurons_buffer.clear()

    def _flush_edges(self) -> None:
        if not self._edges_buffer:
            return
        self.cur.executemany(
            "INSERT INTO edges (src, dst, kind) VALUES (?, ?, ?)",
            self._edges_buffer,
        )
        self._edges_buffer.clear()

    def add_neurons(self, records: list[tuple[int, str, str]]) -> None:
        self._neurons_buffer.extend(records)
        if len(self._neurons_buffer) >= self.batch_size:
            self._flush_neurons()
            self.conn.commit()

    def add_edges(self, records: list[tuple[int, int, str]]) -> None:
        self._edges_buffer.extend(records)
        if len(self._edges_buffer) >= self.batch_size:
            self._flush_edges()
            self.conn.commit()

    def close(self) -> None:
        self._flush_neurons()
        self._flush_edges()
        self.conn.commit()
        self.cur.close()
        self.conn.close()


def collect_text_files(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()

    def add_path(p: Path) -> None:
        if p in seen:
            return
        seen.add(p)
        files.append(p)

    if args.text_files:
        for raw in args.text_files:
            path = Path(raw)
            if path.is_dir():
                for sub in path.rglob("*"):
                    if sub.is_file():
                        add_path(sub)
            else:
                add_path(path)

    if args.interactive_text:
        while True:
            raw = input("Path to text file (blank to finish): ").strip()
            if not raw:
                break
            path = Path(raw)
            if path.is_dir():
                for sub in path.rglob("*"):
                    if sub.is_file():
                        add_path(sub)
            else:
                add_path(path)
    return files


def build_graph_from_files(
    files: list[Path],
    bridge_limit: int,
    max_tokens_per_file: int,
    max_neurons: int,
    sink: GraphSink | None = None,
) -> NeuronGraph:
    graph = NeuronGraph(
        bridge_limit=bridge_limit,
        sink=sink,
        store_neurons=sink is None,
        store_edges=sink is None,
    )
    total_neurons = 0
    per_file_cap = None if max_tokens_per_file < 0 else max_tokens_per_file
    global_cap = None if max_neurons < 0 else max_neurons

    for path in files:
        if global_cap is not None and total_neurons >= global_cap:
            graph.notes.append(
                f"Stopped ingesting because --max-neurons ({global_cap}) was reached."
            )
            break
        if not path.exists():
            print(f"Skipping missing file: {path}", file=sys.stderr)
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            print(f"Failed to read {path}: {exc}", file=sys.stderr)
            continue
        tokens = tokenize_text(text)
        if per_file_cap is not None and len(tokens) > per_file_cap:
            graph.notes.append(f"Truncated {path.name} to {per_file_cap} tokens.")
            tokens = tokens[:per_file_cap]
        if global_cap is not None and total_neurons + len(tokens) > global_cap:
            remaining = max(0, global_cap - total_neurons)
            tokens = tokens[:remaining]
            graph.notes.append(
                f"Truncated {path.name} to fit --max-neurons ({global_cap})."
            )
        if not tokens:
            continue
        graph.add_tokens(tokens, source=path.name)
        total_neurons += len(tokens)
        print(f"Added {len(tokens)} neurons from {path}")
    if sink:
        sink.close()
    return graph


def build_single_from_input(
    default_input: int | None, args: argparse.Namespace
) -> tuple[int, list[int], int, str]:
    input_dim = prompt_int("Number of input features", default_input or args.inputs or 4)
    output_dim = prompt_int("Number of outputs", args.outputs or 1)
    hidden = prompt_hidden(args.hidden or [8, 8])
    activation = (
        input(f"Activation {list(ACTIVATIONS.keys())} [{args.activation}]: ").strip()
        or args.activation
    )
    if activation not in ACTIVATIONS:
        print(f"Unknown activation '{activation}'. Options: {list(ACTIVATIONS)}")
        sys.exit(1)
    return input_dim, hidden, output_dim, activation


def build_networks(args: argparse.Namespace) -> tuple[List[NeuralNetwork], float]:
    networks: list[NeuralNetwork] = []
    learning_rate = args.learning_rate
    if args.interactive:
        prev_output: int | None = None
        while True:
            input_dim, hidden, output_dim, activation = build_single_from_input(
                default_input=prev_output, args=args
            )
            networks.append(
                NeuralNetwork(
                    input_dim=input_dim,
                    hidden_layers=hidden,
                    output_dim=output_dim,
                    activation=activation,
                )
            )
            prev_output = output_dim
            if not prompt_yes_no("Add another network connected to this output?"):
                break
        lr_raw = input(f"Learning rate for demo [{learning_rate}]: ").strip()
        learning_rate = float(lr_raw) if lr_raw else learning_rate
    else:
        if args.inputs is None or args.outputs is None:
            print("Specify --inputs and --outputs or use --interactive.", file=sys.stderr)
            sys.exit(1)
        activation = args.activation
        if activation not in ACTIVATIONS:
            print(f"Unknown activation '{activation}'. Options: {list(ACTIVATIONS)}")
            sys.exit(1)
        networks.append(
            NeuralNetwork(
                input_dim=args.inputs,
                hidden_layers=args.hidden or [],
                output_dim=args.outputs,
                activation=activation,
            )
        )
    return networks, learning_rate


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build feed-forward neural networks from terminal-provided settings, or turn text files into connected neurons. "
            "Pass --interactive for numeric prompts, or --interactive-text/--text-files for text graph mode."
        )
    )
    parser.add_argument("--inputs", type=int, help="Number of input features.")
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="*",
        default=None,
        help="Sizes of hidden layers, e.g. --hidden 16 8 4.",
    )
    parser.add_argument("--outputs", type=int, help="Number of output neurons.")
    parser.add_argument(
        "--activation",
        choices=list(ACTIVATIONS.keys()),
        default="relu",
        help="Activation function for all layers.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.01, help="Learning rate for demo training."
    )
    parser.add_argument(
        "--demo-epochs",
        type=int,
        default=0,
        help="Run a quick training demo on random data for this many epochs.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for settings instead of using flags.",
    )
    parser.add_argument(
        "--text-files",
        nargs="*",
        help=(
            "Paths to text files to break into 'neurons' and connect together. "
            "When provided (or when using --interactive-text), the numeric network flags are ignored."
        ),
    )
    parser.add_argument(
        "--interactive-text",
        action="store_true",
        help="Prompt for one or more text file paths to stitch into connected neurons.",
    )
    parser.add_argument(
        "--bridge-limit",
        type=int,
        default=50000,
        help=(
            "Maximum number of bridge edges between previously seen neurons and the new file's neurons. "
            "Set to -1 to allow all bridges."
        ),
    )
    parser.add_argument(
        "--max-tokens-per-file",
        type=int,
        default=50000,
        help=(
            "Cap tokens per file to this number to avoid runaway memory use. "
            "Set to -1 for no per-file cap."
        ),
    )
    parser.add_argument(
        "--max-neurons",
        type=int,
        default=300000,
        help=(
            "Stop ingesting new tokens once this many neurons have been added. "
            "Set to -1 for no global cap."
        ),
    )
    parser.add_argument(
        "--pg-url",
        help=(
            "Optional PostgreSQL connection string (e.g. postgres://user:pass@host:port/db) "
            "to stream neurons/edges into a database instead of keeping them in RAM."
        ),
    )
    parser.add_argument(
        "--pg-schema",
        default="public",
        help="Schema to use for PostgreSQL output (default: public). Tables will be named neurons and edges.",
    )
    parser.add_argument(
        "--sqlite-out",
        help="Path to a SQLite database file to stream neurons/edges without keeping them in RAM.",
    )
    return parser.parse_args(argv)


def demo_training(chain: NetworkChain, epochs: int, lr: float) -> None:
    if epochs <= 0:
        return
    rng = np.random.default_rng(seed=42)
    x = rng.normal(size=(256, chain.networks[0].input_dim))
    target_w = rng.normal(
        size=(chain.networks[0].input_dim, chain.networks[-1].output_dim)
    )
    y = np.maximum(0.0, x @ target_w)  # simple ReLU transform as fake ground truth

    for epoch in range(1, epochs + 1):
        loss = chain.train_step(x, y, lr)
        if epoch % max(1, epochs // 5) == 0 or epoch == 1:
            print(f"[demo] epoch {epoch:3d}/{epochs} | loss={loss:.4f}")


def run_text_mode(args: argparse.Namespace) -> None:
    files = collect_text_files(args)
    if not files:
        print(
            "No text files provided. Use --text-files or --interactive-text to supply inputs.",
            file=sys.stderr,
        )
        return
    sink: GraphSink | None = None
    if args.pg_url:
        sink = PostgresSink(args.pg_url, schema=args.pg_schema)
        print("Streaming neurons and edges to PostgreSQL...")
    elif args.sqlite_out:
        sink = SQLiteSink(args.sqlite_out)
        print(f"Streaming neurons and edges to SQLite at {args.sqlite_out} ...")
    graph = build_graph_from_files(
        files,
        bridge_limit=args.bridge_limit,
        max_tokens_per_file=args.max_tokens_per_file,
        max_neurons=args.max_neurons,
        sink=sink,
    )
    print(graph.summary())


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    if args.text_files or args.interactive_text:
        run_text_mode(args)
        return
    networks, lr = build_networks(args)
    chain = NetworkChain(networks)
    print(chain.summary())
    if args.demo_epochs:
        demo_training(chain, epochs=args.demo_epochs, lr=lr)


if __name__ == "__main__":
    main(sys.argv[1:])
