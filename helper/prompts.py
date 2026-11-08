# Generate system prompt for zero-shot reasoning
def generate_zero_shot_system_prompt(grid_size):
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



# Generate system prompt for few-shot reasoning
def generate_few_shot_system_prompt(grid_size):
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

    Few-shot demonstrations:

    The following examples demonstrate how to select the next
    action in a grid world.

    These are illustrative examples only, not the actual environment.
    Apply the demonstrated behavior to the actual task.

    Example 1:
    Starting position: [1, 4]
    Goal position: [3, 1]
    Current position: [2, 4]
    Obstacles: [[0, 1], [4, 4], [2, 1], [2, 3], [3, 2]]
    Action: SOUTH

    Example 2:
    Starting position: [2, 0]
    Goal position: [3, 1]
    Current position: [4, 2]
    Obstacles: [[0, 1], [4, 4], [2, 1], [2, 3], [3, 2]]
    Action: WEST
    """

    return SYSTEM_PROMPT.strip()
