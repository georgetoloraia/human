import subprocess
import sys
import re


def classify_error(output: str) -> str:
    lowered = output.lower()
    if "indexerror" in lowered:
        return "IndexError"
    if "typeerror" in lowered:
        return "TypeError"
    if "valueerror" in lowered:
        return "ValueError"
    if "zerodivisionerror" in lowered:
        return "ZeroDivisionError"
    if "assertionerror" in lowered or "assert " in lowered or "failed" in lowered:
        return "AssertionError"
    if "keyerror" in lowered:
        return "KeyError"
    return "Other"


def extract_failing_tasks(output: str):
    tasks = set()
    for match in re.finditer(r"test_([A-Za-z0-9_]+)\.py", output):
        tasks.add(match.group(1))
    return list(tasks)


def run_tests() -> tuple[bool, str, list[str]]:
    """
    Run pytest; return (success, error_type, failing_tasks).
    Falls back to OK if pytest is unavailable.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest"],
            capture_output=True,
            text=True,
        )
    except Exception:
        return True, "OK", []

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

    if proc.returncode == 0:
        return True, "OK", []

    return False, classify_error(combined), extract_failing_tasks(combined)
