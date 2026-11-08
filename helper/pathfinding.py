from collections import deque
from helper.constants import MOVES

# Get the shortest path from start to goal using BFS
def get_shortest_path(start, goal, obstacles, grid_size=5):
    queue = deque([(start, [])])
    visited = set()
    visited.add(tuple(start))

    while queue:
        current_position, path = queue.popleft()

        if current_position == goal:
            return path

        for action, (dr, dc) in MOVES.items():
            next_position = [current_position[0] + dr, current_position[1] + dc]

            if (
                0 <= next_position[0] < grid_size
                and 0 <= next_position[1] < grid_size
                and tuple(next_position) not in visited
                and next_position not in obstacles
            ):
                visited.add(tuple(next_position))
                queue.append((next_position, path + [action]))

    return None

