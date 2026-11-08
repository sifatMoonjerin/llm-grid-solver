import ollama

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
