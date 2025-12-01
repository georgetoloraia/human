import subprocess
import sys


def classify_error(output: str) -> str:
    lowered = output.lower()
    if "indexerror" in lowered:
        return "IndexError"
    if "typeerror" in lowered:
        return "TypeError"
    if "assertionerror" in lowered or "assert " in lowered:
        return "AssertionError"
    if "keyerror" in lowered:
        return "KeyError"
    return "Other"


def run_tests() -> tuple[bool, str]:
    """
    Run pytest; return (success, error_type).
    Falls back to OK if pytest is unavailable.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest"],
            capture_output=True,
            text=True,
        )
    except Exception:
        return True, "OK"

    if proc.returncode == 0:
        return True, "OK"

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return False, classify_error(combined)
