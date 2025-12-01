from pathlib import Path


def observe_codebase():
    observations = []
    plugins_dir = Path("plugins")
    if not plugins_dir.exists():
        return observations
    for p in plugins_dir.glob("*.py"):
        txt = p.read_text()
        observations.append(
            {
                "file": p.name,
                "lines": len(txt.splitlines()),
                "has_functions": "def " in txt,
                "has_return": "return" in txt,
            }
        )
    return observations
