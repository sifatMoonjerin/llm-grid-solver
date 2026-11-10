
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time

# Import your existing functions here
from helper.llm import ask_slm
from helper.prompts import generate_zero_shot_system_prompt
from helper.validation import validate_json, validate_move
from helper.grid_environment import get_neighboring_obstacles
from helper.pathfinding import get_shortest_path


class GridNavigationGUI:

    def __init__(
        self,
        root,
        grid_info,
        models,
        is_local_context=False,
        max_attempts=25
    ):
        self.root = root
        self.root.title("SLM Grid Navigation")
        self.root.resizable(False, False)

        # Environment configuration
        self.grid = grid_info["grid"]
        self.grid_size = len(self.grid)

        self.start_position = list(grid_info["start"])
        self.current_position = self.start_position.copy()
        self.goal_position = list(grid_info["goal"])
        self.obstacles = [list(pos) for pos in grid_info["obstacles"]]

        # Experiment configuration
        self.is_local_context = is_local_context
        self.max_attempts = max_attempts

        # Navigation state
        self.moves_sequence = []
        self.total_moves = 0
        self.attempted_moves = 0
        self.total_inference_time = 0.0

        self.is_solved = False
        self.has_invalid_move = False
        self.is_busy = False

        # Model selection
        self.models = models
        self.selected_model = tk.StringVar(value=models[0])

        # Conversation initialization
        self.system_prompt = generate_zero_shot_system_prompt(
            grid_size=self.grid_size
        )

        if self.is_local_context:
            neighboring_obstacles = get_neighboring_obstacles(
                self.current_position,
                self.obstacles
            )

            initial_prompt = (
                f"Starting position: {self.current_position}\n"
                f"Goal position: {self.goal_position}\n"
                f"Neighboring Obstacles: {neighboring_obstacles}"
            )
        else:
            initial_prompt = (
                f"Starting position: {self.current_position}\n"
                f"Goal position: {self.goal_position}\n"
                f"Obstacles: {self.obstacles}"
            )

        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": initial_prompt}
        ]

        # Build interface
        self.create_widgets()
        self.draw_grid()
        self.update_status()

    # --------------------------------------------------
    # GUI construction
    # --------------------------------------------------

    def create_widgets(self):

        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack()

        # Model selection
        model_frame = ttk.Frame(main_frame)
        model_frame.pack(fill="x", pady=(0, 12))

        ttk.Label(
            model_frame,
            text="Select SLM Model:"
        ).pack(side="left", padx=(0, 10))

        self.model_dropdown = ttk.Combobox(
            model_frame,
            textvariable=self.selected_model,
            values=self.models,
            state="readonly",
            width=20
        )
        self.model_dropdown.pack(side="left")

        # Main content
        content_frame = ttk.Frame(main_frame)
        content_frame.pack()

        # Grid visualization
        self.canvas = tk.Canvas(
            content_frame,
            width=self.grid_size * 65,
            height=self.grid_size * 65,
            bg="white",
            highlightthickness=0
        )
        self.canvas.pack(side="left", padx=(0, 20))

        # Information panel
        info_frame = ttk.LabelFrame(
            content_frame,
            text="Navigation Information",
            padding=12
        )
        info_frame.pack(side="left", fill="y")

        self.position_label = ttk.Label(info_frame)
        self.position_label.pack(anchor="w", pady=5)

        self.goal_label = ttk.Label(info_frame)
        self.goal_label.pack(anchor="w", pady=5)

        self.last_action_label = ttk.Label(info_frame)
        self.last_action_label.pack(anchor="w", pady=5)

        self.executed_label = ttk.Label(info_frame)
        self.executed_label.pack(anchor="w", pady=5)

        self.attempted_label = ttk.Label(info_frame)
        self.attempted_label.pack(anchor="w", pady=5)

        self.time_label = ttk.Label(info_frame)
        self.time_label.pack(anchor="w", pady=5)

        # Status
        ttk.Separator(info_frame).pack(fill="x", pady=10)

        ttk.Label(
            info_frame,
            text="Status:",
            font=("Arial", 10, "bold")
        ).pack(anchor="w")

        self.status_label = ttk.Label(
            info_frame,
            text="Ready",
            wraplength=180
        )
        self.status_label.pack(anchor="w", pady=5)

        # Last response
        ttk.Label(
            info_frame,
            text="Last SLM response:",
            font=("Arial", 10, "bold")
        ).pack(anchor="w", pady=(12, 5))

        self.response_label = ttk.Label(
            info_frame,
            text="-",
            wraplength=180
        )
        self.response_label.pack(anchor="w")

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(15, 0))

        self.next_button = ttk.Button(
            button_frame,
            text="Next Move",
            command=self.next_move
        )
        self.next_button.pack(side="left", padx=(0, 10))

        self.reset_button = ttk.Button(
            button_frame,
            text="Reset",
            command=self.reset
        )
        self.reset_button.pack(side="left")

    # --------------------------------------------------
    # Grid visualization
    # --------------------------------------------------

    def draw_grid(self):

        self.canvas.delete("all")

        cell_size = 65

        for row in range(self.grid_size):
            for col in range(self.grid_size):

                x1 = col * cell_size
                y1 = row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                position = [row, col]

                fill_color = "white"
                symbol = ""

                if position in self.obstacles:
                    fill_color = "#52525B"
                    symbol = "■"

                if position == self.goal_position:
                    fill_color = "#70AD47"
                    symbol = "G"

                if position == self.current_position:
                    fill_color = "#4472C4"
                    symbol = "R"

                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=fill_color,
                    outline="#BFBFBF",
                    width=1
                )

                if symbol:
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        text=symbol,
                        fill="white",
                        font=("Arial", 18, "bold")
                    )

    # --------------------------------------------------
    # Context construction
    # --------------------------------------------------

    def get_context(self):

        system_message = self.messages[0]

        if self.is_local_context:

            recent_messages = self.messages[1:][-4:]

            context = [
                system_message,
                *recent_messages
            ]

        else:

            initial_user_message = self.messages[1]
            recent_messages = self.messages[2:][-4:]

            context = [
                system_message,
                initial_user_message,
                *recent_messages
            ]

        return context

    # --------------------------------------------------
    # Generate next action
    # --------------------------------------------------

    def next_move(self):

        if self.is_busy or self.is_solved:
            return

        if self.attempted_moves >= self.max_attempts:
            self.finish_navigation(
                "Maximum number of attempts reached."
            )
            return

        self.is_busy = True
        self.next_button.config(state="disabled")

        self.status_label.config(text="Generating next action...")
        
        # Run inference in a background thread so that
        # the Tkinter interface remains responsive.
        threading.Thread(
            target=self.generate_action,
            daemon=True
        ).start()

    def generate_action(self):

        model = self.selected_model.get()
        context = self.get_context()

        start_time = time.perf_counter()

        try:
            response = ask_slm(context, model)
            error = None
        except Exception as e:
            response = None
            error = str(e)

        inference_time = time.perf_counter() - start_time

        # Update Tkinter only from the main thread
        self.root.after(
            0,
            lambda: self.process_response(
                response,
                inference_time,
                error
            )
        )

    # --------------------------------------------------
    # Validate and execute action
    # --------------------------------------------------

    def process_response(self, response, inference_time, error):

        self.is_busy = False
        self.attempted_moves += 1
        self.total_inference_time += inference_time

        self.time_label.config(
            text=f"Last inference: {inference_time:.2f}s"
        )

        if error is not None:

            self.status_label.config(
                text=f"Inference error: {error}"
            )

            self.next_button.config(state="normal")
            self.update_status()
            return

        self.response_label.config(text=str(response))

        # Store model response in conversation
        self.messages.append({
            "role": "assistant",
            "content": response
        })

        is_valid_json, action = validate_json(response)

        # Invalid JSON
        if not is_valid_json:

            feedback = (
                f"The action was invalid.\n"
                f"Current position: {self.current_position}\n"
                "Please provide a valid JSON object with an action."
            )

            self.messages.append({
                "role": "user",
                "content": feedback
            })

            self.status_label.config(
                text="Invalid JSON. Feedback sent to SLM."
            )

            self.complete_attempt()
            return

        # Validate movement
        validation_result = validate_move(
            self.current_position,
            action,
            self.obstacles,
            self.grid_size
        )

        if validation_result["valid"]:

            self.total_moves += 1
            self.moves_sequence.append(action)

            self.current_position = list(
                validation_result["next_position"]
            )

            # Check whether goal is reached
            if self.current_position == self.goal_position:

                self.is_solved = True
                self.draw_grid()
                self.update_status()

                self.finish_navigation("Goal reached!")
                return

            # Prepare next state information
            if self.is_local_context:

                neighboring_obstacles = get_neighboring_obstacles(
                    self.current_position,
                    self.obstacles
                )

                feedback = (
                    f"The action was valid.\n"
                    f"Current position: {self.current_position}\n"
                    f"Goal position: {self.goal_position}\n"
                    f"Neighboring Obstacles: {neighboring_obstacles}"
                )

            else:

                feedback = (
                    f"The action was valid.\n"
                    f"Current position: {self.current_position}\n"
                    f"Goal position: {self.goal_position}\n"
                    f"Obstacles: {self.obstacles}"
                )

            self.messages.append({
                "role": "user",
                "content": feedback
            })

            self.status_label.config(
                text=f"Executed action: {action}"
            )

            self.draw_grid()

        else:

            self.has_invalid_move = True

            feedback = (
                f"Invalid move: {action}.\n"
                f"Current position: {self.current_position}.\n"
                f"Reason: {validation_result['reason']}.\n"
                "Please select another valid action."
            )

            self.messages.append({
                "role": "user",
                "content": feedback
            })

            self.status_label.config(
                text=f"Invalid move: {validation_result['reason']}"
            )

        self.complete_attempt()

    # --------------------------------------------------
    # Update interface
    # --------------------------------------------------

    def update_status(self):

        self.position_label.config(
            text=f"Current position: {self.current_position}"
        )

        self.goal_label.config(
            text=f"Goal position: {self.goal_position}"
        )

        last_action = (
            self.moves_sequence[-1]
            if self.moves_sequence else "-"
        )

        self.last_action_label.config(
            text=f"Last executed action: {last_action}"
        )

        self.executed_label.config(
            text=f"Executed moves: {self.total_moves}"
        )

        self.attempted_label.config(
            text=f"Attempted moves: {self.attempted_moves}"
        )

    def complete_attempt(self):

        self.update_status()

        if self.attempted_moves >= self.max_attempts:

            self.finish_navigation(
                "Maximum number of attempts reached."
            )
            return

        if not self.is_solved:
            self.next_button.config(state="normal")

    # --------------------------------------------------
    # Navigation completion
    # --------------------------------------------------

    def finish_navigation(self, message):

        self.next_button.config(state="disabled")

        shortest_path = get_shortest_path(
            self.start_position,
            self.goal_position,
            self.obstacles,
            self.grid_size
        )

        if shortest_path is not None and self.is_solved:

            path_difference = (
                len(self.moves_sequence) - len(shortest_path)
            )

            result = (
                f"{message}\n\n"
                f"LLM path length: {len(self.moves_sequence)}\n"
                f"BFS path length: {len(shortest_path)}\n"
                f"Path difference: {path_difference}\n"
                f"Total inference time: {self.total_inference_time:.2f}s"
            )

        else:

            result = (
                f"{message}\n\n"
                f"Executed moves: {self.total_moves}\n"
                f"Attempted moves: {self.attempted_moves}\n"
                f"Total inference time: {self.total_inference_time:.2f}s"
            )

        self.status_label.config(text=message)

        messagebox.showinfo(
            "Navigation Results",
            result
        )

    # --------------------------------------------------
    # Reset environment
    # --------------------------------------------------

    def reset(self):

        if self.is_busy:
            return

        self.current_position = self.start_position.copy()

        self.moves_sequence = []
        self.total_moves = 0
        self.attempted_moves = 0
        self.total_inference_time = 0.0

        self.is_solved = False
        self.has_invalid_move = False

        # Reinitialize conversation
        if self.is_local_context:

            neighboring_obstacles = get_neighboring_obstacles(
                self.current_position,
                self.obstacles
            )

            initial_prompt = (
                f"Starting position: {self.current_position}\n"
                f"Goal position: {self.goal_position}\n"
                f"Neighboring Obstacles: {neighboring_obstacles}"
            )

        else:

            initial_prompt = (
                f"Starting position: {self.current_position}\n"
                f"Goal position: {self.goal_position}\n"
                f"Obstacles: {self.obstacles}"
            )

        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": initial_prompt}
        ]

        self.response_label.config(text="-")
        self.status_label.config(text="Ready")
        self.time_label.config(text="Last inference: -")

        self.next_button.config(state="normal")

        self.draw_grid()
        self.update_status()


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":

    # Example environment
    grid_info = {
        "grid": [
            [".", ".", ".", ".", "."],
            [".", ".", ".", ".", "."],
            [".", ".", ".", ".", "."],
            [".", ".", ".", ".", "."],
            [".", ".", ".", ".", "."]
        ],
        "start": [0, 0],
        "goal": [4, 4],
        "obstacles": [
            [1, 0],
            [1, 2],
            [2, 2],
            [3, 1],
            [3, 3]
        ]
    }

    models = [
        "gemma:7b",
        "llama3.2:3b",
        "qwen3.5:2b"
    ]

    root = tk.Tk()

    app = GridNavigationGUI(
        root=root,
        grid_info=grid_info,
        models=models,
        is_local_context=False,
        max_attempts=25
    )

    root.mainloop()
