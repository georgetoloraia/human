#!/usr/bin/env python3
import sys
from manager.guidance import append_guidance


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 talk_to_human.py \"your message\"", file=sys.stderr)
        sys.exit(1)
    message = sys.argv[1]
    append_guidance(author="human_teacher", message=message)
    print(f"Recorded guidance: {message}")


if __name__ == "__main__":
    main()
