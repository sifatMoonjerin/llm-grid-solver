import time
from helper.llm import ask_slm
from helper.pathfinding import get_shortest_path
from helper.validation import validate_json, validate_move
from helper.grid_environment import get_neighboring_obstacles

# Solve the grids using the SLM model
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

