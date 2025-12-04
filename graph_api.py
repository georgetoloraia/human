from __future__ import annotations

import argparse

from manager.graph_client import create_app


def main():
    parser = argparse.ArgumentParser(description="Serve a simple neuron-graph HTTP API")
    parser.add_argument("--graph", default="manager/neuron_graph.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(args.graph)
    print(f"Serving neuron graph API on http://{args.host}:{args.port} using {args.graph}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
