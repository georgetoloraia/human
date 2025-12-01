import time
from manager.mind import Mind


def run_forever():
    mind = Mind()
    while True:
        print("[LIFE] age=", mind.brain.state.get("age"), "skill=", mind.brain.get_skill_level())
        mind.step()
        time.sleep(5)


if __name__ == "__main__":
    run_forever()
