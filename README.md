# Moltbook Agent

Small open-source AI agent for Moltbook using Python + Ollama + Qwen.

## Setup

```bash
git clone <REPO_URL>
cd moltbook_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3:8b
```

Create `.env` in the project root:

```env
MOLTBOOK_API_KEY=your_moltbook_api_key
```

Then run:

```bash
python -m agents.agent
```

Make sure Ollama is installed and running before starting the agent.
