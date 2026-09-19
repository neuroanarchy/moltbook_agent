import json
from pathlib import Path


# NEW: The file where Schranz's persistent memory will live.
MEMORY_FILE = Path("memory.json")


# NEW: Load all previously stored memories.
def load_memory():

    if not MEMORY_FILE.exists():
        return []

    with open(MEMORY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


# NEW: Save the complete memory list to disk.
def save_memory(memory):

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(
            memory,
            file,
            indent=4,
            ensure_ascii=False
        )


# NEW: Add one new memory to persistent storage.
def add_memory(memory):

    memories = load_memory()

    memories.append(memory)

    save_memory(memories)


# NEW: Return the most recent memories.
def get_recent_memories(limit=5):

    memories = load_memory()

    return memories[-limit:]
