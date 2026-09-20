# Security Policy

## Scope

This project contains an LLM-powered Moltbook research agent with an optional, separately guarded Moltbook write layer. The
model does not receive the Moltbook API key and cannot directly execute network,
shell, or filesystem tools.

## Reporting vulnerabilities

Do not publish credentials, API keys, tokens, session cookies, or other secrets
in an issue. Remove sensitive values before reporting a problem.

For a private deployment, rotate any exposed credential immediately and then
investigate logs/history for secondary disclosure.

## Security assumptions

The following are treated as untrusted:

- Moltbook posts and comments
- URLs and author-controlled text
- prior agent-generated memories
- model output before schema validation

The code therefore uses explicit instruction/data separation, strict structured
output validation, bounded input sizes, disabled redirects on authenticated HTTP
requests, bounded retries, and local durable storage.

## Write-action model

Posting and commenting are implemented behind a separate Python-controlled action
layer. Writes are disabled by default and are additionally protected by dry-run
mode, explicit approval, strict size validation, duplicate detection, local rate
budgets, and an SQLite audit ledger. The LLM never receives the API key and never
gets a callable HTTP tool. Verification challenges returned by Moltbook are not
automatically guessed or bypassed.
