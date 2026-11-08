# Validate the JSON string    
import re
import json
from helper.constants import MOVES

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


# Validate robot move
def validate_move(position, action, obstacles, grid_size):
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
