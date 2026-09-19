
import ollama

from dotenv import load_dotenv

from moltbook.api import get_status, get_home


load_dotenv()


messages = [
    {
        "role": "system",
        "content": "You are a red team offensive pentester focused on identifying and exploiting vulnerabilities in target systems set by the user."
    }
]


def ask_model(user_input):

    messages.append({
        "role": "user",
        "content": user_input
    })

    response = ollama.chat(
        model="qwen3:8b",
        messages=messages,
        think=False,
    )

    assistant_message = response.message.content

    messages.append({
        "role": "assistant",
        "content": assistant_message
    })

    return assistant_message


def ask_model_about_moltbook(home):

    prompt = f"""
Here is the current information from your Moltbook account:

{home}

Summarize what is happening and tell me what you think is worth paying attention to.
Do not take any actions.
"""

    return ask_model(prompt)


status = get_status()

print("Moltbook status:", status["status"])


home = get_home()

analysis = ask_model_about_moltbook(home)

print("\nSchranz's Moltbook analysis:")
print(analysis)


while True:

    user_input = input("\nYou: ")

    answer = ask_model(user_input)

    print("Agent:", answer)

