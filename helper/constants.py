from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MOVES = {
    "NORTH": (-1, 0),
    "SOUTH": (1, 0),
    "EAST": (0, 1),
    "WEST": (0, -1),
}