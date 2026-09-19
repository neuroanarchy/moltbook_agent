import ollama


messages=[
    {"role": "system", "content": "You are a red team offensive pentester focused on identifying and exploiting vulnerabilities in target systems set by the user."}
]

def ask_model(user_input):
    messages.append({"role": "user", "content": user_input})
    response = ollama.chat(model="qwen3:8b",
                           messages=messages,
                           think=False,
                           )
    assistant_message = response.message.content
    messages.append({"role": "assistant", "content": assistant_message})
    return assistant_message

while True:
    user_input = input("You: ")
    answer = ask_model(user_input)
    print("Agent:", answer)