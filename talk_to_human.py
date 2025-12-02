#!/usr/bin/env python3
import sys
import json
from pathlib import Path
from manager.guidance import append_guidance


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 talk_to_human.py \"your message\"", file=sys.stderr)
        sys.exit(1)
    message = sys.argv[1]
    append_guidance(author="human_teacher", message=message)
    print(f"Recorded guidance: {message}")
    diary_path = Path("manager/mind_diary.json")
    if diary_path.exists():
        try:
            data = json.loads(diary_path.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                last = data[-1]
                reflection = last.get("reflection")
                if reflection:
                    print("Latest reflection:", reflection)
        except Exception:
            pass


if __name__ == "__main__":
    main()
