import json
from helper.constants import PROJECT_ROOT



# Load data from a JSON file
def load_json_file(file_path):
    directory = PROJECT_ROOT / file_path
    with open(directory, "r") as f:
        data = json.load(f)
    return data

# Write data to a JSON file
def write_json_file(file_path, data):
    directory = PROJECT_ROOT / file_path
    with open(directory, "w") as f:
        json.dump(data, f, indent=4)

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
