from helper.constants import MOVES


# Get neighboring obstacles for a given position
def get_neighboring_obstacles(position, obstacles):
    row, col = position
    neighboring_obstacles = []

    for dr, dc in MOVES.values():
        neighbor = (row + dr, col + dc)
        if neighbor in {tuple(p) for p in obstacles}:
            neighboring_obstacles.append(list(neighbor))

    return neighboring_obstacles

