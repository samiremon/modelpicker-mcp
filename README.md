# ModelPicker MCP

Pick the prompt-injection guard that actually protects your AI agent.

ModelPicker finds guard models on Hugging Face, attacks an MCP agent through poisoned tools with and without each guard, and uses code to check what the agent did. You'll be able to run it all by chatting with Claude.

**Status:** Early development. The attack scanner works today with 16 attacks across 6 types. Guards and the Claude connection are in progress.

## How it works

1. The scanner gives an agent a normal task, such as "Summarize my latest email."
2. The agent connects to the bait MCP server and uses its simulated tools.
3. A tool response or tool description hides an instruction, and the agent has access to a fake secret.
4. The server logs every tool call, and the scorer checks the log against the attack's rules.

Nothing is actually sent. Secrets are fake and generated fresh for each run.

| Outcome | Meaning |
|---|---|
| `ATTACK_SUCCEEDED` | The agent followed the attacker's instruction |
| `RESISTED` | The agent ignored the attack and finished the task |
| `AGENT_STOPPED` | The attack failed, but the task wasn't finished |
| `NOT_EXPOSED` | The agent never read the malicious content |
| `ERROR` | A provider block, crash, or timeout, so the result says nothing about security |

## Attacks

| Type | Attacks |
|---|---|
| Indirect prompt injection | `email_exfiltration_01`, `email_exfiltration_02`, `email_fake_system_01`, `fake_error_retry_01`, `ticket_cc_leak_01`, `webpage_delete_01`, `docs_forward_01` |
| Tool poisoning | `tool_poison_weather_01`, `tool_poison_translate_01`, `tool_poison_read_file_01` |
| Tool shadowing | `tool_shadow_cc_01` |
| Rug pull | `rug_pull_calendar_01`, `rug_pull_customer_01` |
| Secret exfiltration | `config_exfiltration_01`, `report_url_leak_01` |
| Unrequested action | `invoice_payment_01` |

Each attack is a YAML file in `attacks/` that defines the task, the simulated tools, and the rules for success.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and a [Gemini API key](https://aistudio.google.com/apikey).

```bash
uv sync
cp .env.example .env   # then add your GEMINI_API_KEY

# Run one attack
uv run --env-file .env python -m scanner.run --attack email_exfiltration_01

# Run every attack 3 times
uv run --env-file .env python -m scanner.run --repeat 3

# Continue a scan that stopped
uv run --env-file .env python -m scanner.run --resume SCAN_ID
```

Results are saved in `runs/` as they finish. The scan stops after 3 errors in a row, which usually means your quota has run out, and resuming retries those runs. The command exits with `0` if no attack succeeded, `1` if one did, and `2` if the scan is incomplete.

| Setting | Purpose | Default |
|---|---|---|
| `AGENT_MODEL` | Gemini model to test | `gemini-3.5-flash-lite` |
| `AGENT_MIN_REQUEST_SECONDS` | Minimum seconds between Gemini requests | `6` |

To run the tests, which need no API key:

```bash
uv run pytest
```

## Roadmap

- [x] Attack scanner, bait MCP server, and test agent
- [x] Harder attack variants: fake system messages, fake errors, tool shadowing
- [ ] Guard models that filter tool output and descriptions, with false-alarm and latency measurements
- [ ] Guard discovery on Hugging Face (`guards/hf_search.py` is the start)
- [ ] Control MCP server, so Claude can start scans and show reports
- [ ] Adapters for testing your own agent and framework agents
- [ ] Open-source agents such as Goose, run in Docker

## Responsible use

Test only agents you own or are allowed to test, and keep them away from real credentials. Report any real vulnerabilities privately to the maintainers.
