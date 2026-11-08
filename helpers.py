import re
import json
import time
import ollama
from collections import deque


# Robot moves in the grid
MOVES = {
    "NORTH": (-1, 0),
    "SOUTH": (1, 0),
    "EAST": (0, 1),
    "WEST": (0, -1),
}

# Ask the SLM model
def ask_slm(messages, model):
    # print("Asking SLM...")
    response = ollama.chat(
        model=model,
        messages=messages,
        think=False,
        options={"temperature": 0.0, "seed": 42}
    )
    
    return response["message"]["content"]

# Generate system prompt for zero-shot reasoning
def generate_zero_shot_system_prompt(grid_size=5):
    SYSTEM_PROMPT = f"""
    You are a robot path planner operating in a {grid_size}×{grid_size} grid world.

    Environment
        - Coordinates are represented as [row, column].
        - Rows and columns are indexed from 0 to {grid_size - 1}.
        - Row indices increase from top to bottom.
        - Column indices increase from left to right.

    Movement rules:
        - NORTH: [row - 1, column]
        - SOUTH: [row + 1, column]
        - EAST:  [row, column + 1]
        - WEST:  [row, column - 1]
        
        Each action moves the robot exactly one cell. Only the specified coordinate changes for each action.

    Constraints
        - Never enter an obstacle cell.
        - Never leave the grid boundaries.
        - Each action moves exactly one cell.
        - Return the possible next action.
        - Any valid action is acceptable.

    Planning procedure:
        1. Start at the given starting position.
        2. Select an action that moves toward the goal while avoiding obstacles.
        3. Never choose an action that leaves the grid or enters an obstacle.
        4. A feedback will be provided after each action to indicate whether the action was valid or invalid.
        5. If the action was valid, plan the next action based on the robot position.
        6. If the action was invalid, plan a different action.

    Output format
        - Return only a valid JSON object.
        - The move direction must be one of the following: "NORTH", "SOUTH", "EAST", or "WEST".
        - Example response: {{"action": "SOUTH"}}
        - Do not include explanations, Markdown fences, intermediate positions, or additional text.
    """

    return SYSTEM_PROMPT.strip()


def solve_grids(grids, model, generate_system_prompt, is_local_context, is_self_correcting):
    grid_size = len(grids[0]["grid"]) if grids else 0
    solved_grids = 0
    result_details = []

    for grid_info in grids:
        grid = grid_info["grid"]

        start_position = grid_info["start"]
        current_position = start_position
        goal_position = grid_info["goal"]
        obstacles = grid_info["obstacles"]
        moves_sequence = []
        total_moves = 0
        attempted_moves = 0
        path_difference = -1
        is_solved = False
        has_invalid_move = False
        total_inference_time = 0.0
        inference_times = []

    
        print(f"Grid:\n{'\n'.join(' '.join(row) for row in grid)}")

        initial_prompt = f"Starting position: {start_position}\nGoal position: {goal_position}\nObstacles: {obstacles}"

        system_prompt = generate_system_prompt(grid_size=len(grid))

        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_prompt}
            ]


        while attempted_moves < (grid_size * grid_size):
            print(f"Attempted moves: {attempted_moves}, Total executed moves: {total_moves}, Current position: {current_position}")
            attempted_moves += 1
            system_message = messages[0]

            if is_local_context:
                recent_messages = messages[1:][-4:]
                                
                context = [
                    system_message,
                    *recent_messages
                ]
            else:
                initial_user_message = messages[1]
                recent_messages = messages[2:][-4:]

                context = [
                    system_message,
                    initial_user_message,
                    *recent_messages
                ]
                
            start_time = time.perf_counter()
            response = ask_slm(context, model)
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            total_inference_time += inference_time
            inference_times.append(inference_time)

            print(f"SLM response: {response}")
            
            is_valid_json, action = validate_json(response)

            if not is_valid_json:
                print(f"Invalid JSON response: {response}. Retrying...")


                messages.append({
                    "role": "user",
                    "content": (
                        f"The action was invalid.\n"
                        f"Current position: {current_position}\n"
                        "Please provide a valid JSON object with an action."
                    ),
                })

                continue

            validation_result = validate_move(current_position, action, obstacles)

            if validation_result["valid"]:
                total_moves += 1
                moves_sequence.append(action)
                current_position = validation_result['next_position']
                is_solved = current_position == goal_position

                if is_solved:
                    solved_grids += 1
                    break

                if is_local_context:
                    neighboring_obstacles = get_neighboring_obstacles(current_position, obstacles)
                    
                    messages.append({
                        "role": "user",
                        "content": (
                            f"The action was valid.\n"
                            f"Current position: {current_position}\n"
                            f"Goal position: {goal_position}\n"
                            f"Neighboring Obstacles: {neighboring_obstacles}\n"
                        ),
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"The action was valid.\n"
                            f"Current position: {current_position}\n"
                            f"Goal position: {goal_position}\n"
                            f"Obstacles: {obstacles}\n"
                        ),
                    })

                
            else:
                has_invalid_move = True
                messages.append({
                    "role": "user",
                    "content": f"Invalid move: {action}.\nCurrent position: {current_position}.\n Reason: {validation_result['reason']}. Retrying..."
                })

                if not is_self_correcting:
                    print(f"Invalid move: {action}. Reason: {validation_result['reason']}.")
                    break

        # print(f"Response:\n{response}\n")
        print("SLM Path Planning Results:")

        if is_solved:
            print("Goal reached!")
        elif has_invalid_move and not is_self_correcting:
            print("Encountered an invalid move. Stopping further attempts.")
        else:
            print("Failed to reach the goal within 25 attempts.")

        print(f"Total attempted moves: {attempted_moves}")
        print(f"Total executed moves: {total_moves}")
        # print(f"Moves sequence: {moves_sequence}")

        print("Shortest Path Planning Results:")
        shortest_path = get_shortest_path(start_position, goal_position, obstacles)
        if shortest_path is not None:
            print(f"Shortest path to goal: {shortest_path}")
        else:
            print("Failed to find a path to the goal.")

        if shortest_path is not None and is_solved:
            print("Different between SLM generated path and the shortest path:")
            path_difference = len(moves_sequence) - len(shortest_path)
            print(f"SLM took {path_difference} more moves than the shortest path.")
        
        result_details.append({
            # grid information
            "start": start_position,
            "goal": goal_position,
            "obstacles": obstacles,

            # path planning results
            "is_solved": is_solved,
            "slm_moves": moves_sequence,
            "bfs_moves": shortest_path,
            "has_invalid_move": has_invalid_move,
            "total_attempted_moves": attempted_moves,
            "total_executed_moves": total_moves,
            "path_difference": path_difference,

            # inference time
            "total_inference_time": total_inference_time,
            "inference_times": inference_times
        })

    print(f"Total solved grids: {solved_grids} out of {len(grids)}")

    return result_details


# Get neighboring obstacles for a given position
def get_neighboring_obstacles(position, obstacles):
    row, col = position
    neighboring_obstacles = []

    for dr, dc in MOVES.values():
        neighbor = (row + dr, col + dc)
        if neighbor in {tuple(p) for p in obstacles}:
            neighboring_obstacles.append(list(neighbor))

    return neighboring_obstacles


# Validate robot move
def validate_move(position, action, obstacles, grid_size=5):
    action = action.upper()

    if action not in MOVES:
        return {
            "valid": False,
            "reason": "Unknown action"
        }

    row, col = position
    dr, dc = MOVES[action]
    next_position = (row + dr, col + dc)

    if not (
        0 <= next_position[0] < grid_size
        and 0 <= next_position[1] < grid_size
    ):
        return {
            "valid": False,
            "reason": "Move leaves grid boundaries"
        }

    if next_position in {tuple(p) for p in obstacles}:
        return {
            "valid": False,
            "reason": "Move enters an obstacle"
        }

    return {
        "valid": True,
        "next_position": list(next_position)
    }


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


# Load data from a JSON file
def load_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

# Write data to a JSON file
def write_json_file(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

# Validate the JSON string    
def validate_json(json_str):
    json_str = json_str.strip()
    json_str = re.sub(r"^```(?:json)?\s*", "", json_str)
    json_str = re.sub(r"\s*```$", "", json_str)
    try:
        data = json.loads(json_str)
        if isinstance(data, dict) and "action" in data:
            return True, data['action']
        else:
            return False, "INVALID_JSON"
    except json.JSONDecodeError:
        return False, "INVALID_JSON"


# Create a summary of the result details
def create_result_summary(result_details):
    total_grids = len(result_details)
    solved_grids = 0
    solved_path_differences = 0
    solved_path_length = 0
    unsolved_grids = 0
    unsolved_path_length = 0
    total_solved_inference_time = 0
    total_unsolved_inference_time = 0
    for detail in result_details:
        if detail["is_solved"]:
            solved_grids += 1
            solved_path_differences += detail["path_difference"]
            solved_path_length += len(detail["slm_moves"])
            total_solved_inference_time += detail["total_inference_time"]
        else:
            unsolved_grids += 1
            unsolved_path_length += len(detail["slm_moves"])
            total_unsolved_inference_time += detail["total_inference_time"]

    return {
        "total_grids": total_grids,
        "solved_grids": solved_grids,
        "unsolved_grids": unsolved_grids,
        "average_solved_path_difference": solved_path_differences / solved_grids if solved_grids > 0 else -1,
        "average_solved_path_length": solved_path_length  / solved_grids if solved_grids > 0 else -1,
        "average_unsolved_path_length": unsolved_path_length / unsolved_grids if unsolved_grids > 0 else -1,
        "average_solved_inference_time": total_solved_inference_time / solved_grids if solved_grids > 0 else -1,
        "average_unsolved_inference_time": total_unsolved_inference_time / unsolved_grids if unsolved_grids > 0 else -1
    }
