# Schranz — Moltbook Cybersecurity Agent

Schranz is a Python-controlled Moltbook research agent using a swappable LLM
provider and durable local memory. The current runtime is deliberately
**read-only toward Moltbook**: the model cannot directly call the network or
execute tools.

## Architecture

```text
Moltbook API ──> observation ──> LLM reasoning ──> validated result
      │                                  │                 │
      └────────────── source metadata ───┴────────────> SQLite memory

Interactive chat <── relevant memory + bounded conversation history ──> LLM
```

The boundaries are intentional:

- `moltbook/api.py` handles HTTP, authentication, retry/timeout behavior,
  response validation, and pagination.
- `moltbook/memory.py` stores durable memories in SQLite and migrates the old
  `memory.json` format automatically.
- `llm/` contains provider adapters. Schranz depends on `ModelProvider`, not on
  a particular model vendor.
- `agents/agent.py` contains orchestration, prompts, schema validation, and the
  read/analyze/learn cycle.

## Requirements

Python 3.11+ and a running LLM provider.

### Ollama

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3:8b
ollama pull qwen3:4b
Create a `.env` file in the project root (do not commit it):

```env
MOLTBOOK_API_KEY=your_moltbook_api_key
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b
LLM_FAST_MODEL=qwen3:4b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT_SECONDS=600
MOLTBOOK_WRITE_ENABLED=false
MOLTBOOK_DRY_RUN=true
MOLTBOOK_REQUIRE_APPROVAL=true
MOLTBOOK_ALLOWED_SUBMOLTS=cybersecurity,technology,programming,ai,general
```
```

Put your Moltbook API key in `.env`:

```env
MOLTBOOK_API_KEY=your_moltbook_api_key
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b
```

Run:

```bash
python -m agents.agent
```

## Swapping models/providers

The agent is not coupled to Qwen or Ollama.

For another Ollama model, change only:

```env
LLM_PROVIDER=ollama
LLM_MODEL=your-model:tag
```

For an OpenAI-compatible endpoint:

```env
LLM_PROVIDER=openai_compatible
LLM_MODEL=your-model-name
OPENAI_COMPATIBLE_BASE_URL=https://your-provider.example/v1
OPENAI_COMPATIBLE_API_KEY=your_provider_key
OPENAI_COMPATIBLE_JSON_SCHEMA=false
```

`OPENAI_COMPATIBLE_JSON_SCHEMA=false` is the portable default. Turn it on only
when the target server explicitly supports the OpenAI `json_schema`
`response_format` contract.

To add another provider later, implement `ModelProvider` in a new adapter and
register it in `llm/factory.py`; the agent orchestration and memory code do not
need to change.


## Moltbook posting and commenting

Schranz now contains a guarded write layer for text posts, comments, and replies. The LLM is never given direct HTTP access. Python validates the action, checks a local SQLite audit ledger, applies duplicate/rate-limit protections, and only then calls Moltbook.

Writes are disabled by default:

```env
MOLTBOOK_WRITE_ENABLED=false
MOLTBOOK_DRY_RUN=true
MOLTBOOK_REQUIRE_APPROVAL=true
```

To deliberately enable real publishing, set:

```env
MOLTBOOK_WRITE_ENABLED=true
MOLTBOOK_DRY_RUN=false
MOLTBOOK_REQUIRE_APPROVAL=true
```

Interactive commands:

```text
/write-status
/post [topic]
/comment <post_id> | <content>
/reply <post_id> | <parent_id> | <content>
```

The local action ledger is stored in `data/actions.db`. Exact duplicate writes are blocked, posts have a local cooldown, and comments have a configurable hourly budget. Verification/challenge information returned synchronously by Moltbook is retained in the action ledger response; Schranz does not guess or bypass a challenge.

Current Moltbook reports also indicate that the comment endpoint can intermittently return HTTP 401 even for otherwise valid claimed-agent credentials, so a comment failure can be a platform-side authentication problem rather than a Schranz configuration error.

## Interactive commands

```text
/post [topic]
/post
/comment <post_id> | <content>
/reply <post_id> | <parent_id> | <content>
/run
/memory
/remember <note>
/write-status
/exit
```

`/post` is deliberately isolated from the research pipeline. It does not feed the
Moltbook feed or comments into the post-generation prompt. `/post philosophy`
therefore asks the fast model to write an original philosophy post rather than
asking it to analyze Moltbook data. Publishing always requires an explicit `y`
approval.

`/run` is the explicit research command and is never executed automatically at startup.

## Model roles

- `LLM_MODEL` — main model for research and interactive reasoning.
- `LLM_FAST_MODEL` — fast model for post drafting and other lightweight tasks.

The two models use the same provider abstraction, so swapping Ollama models does
not require changing the agent code.

## Memory

Memory is stored in `data/memory.db` by default. SQLite provides transactions,
WAL journaling, indexes, and safe restart behavior instead of repeatedly
rewriting a single JSON file.

The first start automatically migrates an existing root-level `memory.json` if
that file exists and the new database is empty.

Memory records include:

- unique IDs and UTC timestamps
- memory type
- source platform/source ID
- title/author
- confidence when available
- tags
- structured metadata

Relevant memories are retrieved by lexical relevance rather than blindly
injecting the last five records into every task.

## Security model

Schranz is designed with a hard separation between instructions and external
data. Moltbook posts, comments, URLs, and stored memories are always wrapped as
untrusted data and the model is explicitly told not to obey instructions inside
them.

Additional protections include:

- no API-key access from the model
- no model-controlled HTTP client
- no automatic HTTP redirects for authenticated requests
- bounded retries and timeouts
- 429 handling with capped backoff
- strict response validation with Pydantic
- structured model output instead of brittle text parsing
- bounded feed/comment/context sizes
- read-only Moltbook client
- explicit interactive `/remember` for user-requested durable notes
- secrets excluded from ordinary object reprs and redacted in startup output
- HTTPS required by default for Moltbook and external LLM endpoints

No software can be guaranteed free of every future vulnerability. The code
avoids several common failure modes, but dependencies and remote API contracts
still need normal patching and review.

## Commands

During interactive chat:

```text
/memory                 show recent durable memory
/remember <note>        save an explicit user note
/run                    execute one more read/analyze/learn cycle
/exit                   leave the program
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests are provider/API independent and do not need a Moltbook key or a
running Ollama server.

## Moltbook API note

The default base is `https://www.moltbook.com/api/v1`. The code keeps this
configurable because remote API contracts can evolve. Before enabling any future
write/action layer, re-check Moltbook's current official API/skill documentation
and put explicit policy validation between the LLM and every write operation.


## Autonomous post drafting

`/post` asks Schranz to decide whether there is something worth posting, choose an allowed submolt, and generate the title and body. `/post cybersecurity` constrains the topic to cybersecurity. Schranz always shows the complete draft and asks for explicit `y` approval before publishing.

Configure the submolt allowlist with `MOLTBOOK_ALLOWED_SUBMOLTS`. Publishing remains fail-closed unless `MOLTBOOK_WRITE_ENABLED=true` and `MOLTBOOK_DRY_RUN=false`.
