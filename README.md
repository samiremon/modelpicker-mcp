# ModelPicker MCP

**Compare text-classification models using measured results on your data, directly from Claude.**

ModelPicker is a Python Model Context Protocol server. It discovers models and datasets on Hugging Face, imports labeled examples, runs compatible classifiers on the server's CPU, and compares quality, speed, errors and memory. Save evaluations, export reports and generate Python integration code without leaving the conversation.

Supported tasks are **single-label text classification**: sentiment, spam, ticket routing and similar tasks with trained classifiers and expected labels. Arbitrary text generation, RAG scoring, images and GPU execution are outside the current scope.

## What does a user need?

| Requirement | Local connection | Remote connection |
|---|---|---|
| Install ModelPicker | On the user's computer | On the server host only |
| Source code | Clone or download this repository | Not required on the client |
| Python/inference libraries | On the user's computer | On the server host |
| Where inference runs | User's CPU | Server host's CPU |
| Connection | Local stdio configuration | Public HTTPS `/mcp` URL + OAuth |
| Tunnel | Not needed | Needed if the host lacks a public HTTPS endpoint |

**A configuration file alone does not install ModelPicker.** Each local user must install the project and generate configuration pointing to their own Python environment. Paths from another computer will not work. This repository provides a source installation, not a packaged `.mcpb` desktop extension or a published standalone installer.

For local evaluation, install:

- **uv**, following its [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
- **Git** to clone the repository, or download and extract its GitHub ZIP.
- **Claude Desktop** or another MCP client supporting local stdio servers.
- **Python 3.11–3.13** in the runtime environment. The recommended uv command downloads Python 3.11 when needed.

Internet access is needed for installation and uncached model/dataset downloads. RAM/disk needs depend on the models. **16 GB RAM and several GB of free disk are practical starting recommendations, not enforced minimums.** Defaults allow 4 GiB observed worker RAM and reserve 1 GiB available system RAM; the OS and client also need memory.

No GPU or paid inference API is required. Public unrestricted Hub resources generally need no token; private/gated resources require access and possibly `HF_TOKEN`. Download, storage, hardware and electricity costs still apply. Linux CPU execution is recorded in [validation](docs/VALIDATION.md); other operating systems need their own validation.

## Install and connect locally

### 1. Get the code

```bash
git clone https://github.com/samiremon/modelpicker-mcp.git
cd modelpicker-mcp
```

Alternatively, extract the GitHub ZIP and open a terminal inside the extracted directory.

### 2. Install the CPU runtime

With uv installed, run the setup helper:

```bash
python3 scripts/setup_local.py
```

On Windows use `python scripts/setup_local.py` or `py scripts/setup_local.py`. The helper installs the locked local runtime, creates private data folders, generates your desktop configuration and prints hardware diagnostics. It does not overwrite Claude's configuration.

To authorize a dedicated dataset folder:

```bash
python3 scripts/setup_local.py --dataset-dir /absolute/path/to/datasets
```

Or perform the installation manually:

```bash
uv sync --frozen --python 3.11 --extra local
uv run --no-sync modelpicker setup --dataset-dir ./examples
uv run --no-sync modelpicker doctor
```

The manual example authorizes this repository's demo data. Without `--dataset-dir`, setup uses a previously saved folder or `~/.modelpicker/datasets`. Only files inside the authorized folder can be imported. Do not authorize your entire home directory.

### 3. Configure Claude Desktop

1. Open the generated `~/.modelpicker/desktop-config.json` (Windows: `%USERPROFILE%\.modelpicker\desktop-config.json`).
2. In Claude Desktop open **Settings → Desktop app → Local MCP servers → Edit config**. Labels can vary by client version.
3. Merge the generated `mcpServers.modelpicker` entry into the existing configuration, preserving other servers/settings. This is a local configuration, not a remote connector URL.
4. Fully quit Claude Desktop, including its tray process, and reopen it.
5. Confirm `modelpicker` is running and enable it in the conversation's tools/connectors menu.

The generated configuration resembles this; use your own generated absolute paths:

```json
{
  "mcpServers": {
    "modelpicker": {
      "command": "/absolute/path/to/modelpicker-mcp/.venv/bin/python",
      "args": ["-m", "modelpicker", "serve"],
      "env": {
        "MODELPICKER_HOME": "/absolute/path/to/.modelpicker",
        "MODELPICKER_DATASET_DIR": "/absolute/path/to/datasets",
        "MODELPICKER_CACHE_DIR": "/absolute/path/to/.modelpicker/models",
        "MODELPICKER_CPU_THREADS": "2"
      }
    }
  }
}
```

Windows setup generates a `.venv\Scripts\python.exe` command. Keep the environment at its configured location; after moving the project, regenerate and update the entry. Claude launches the stdio process itself. **No tunnel or ModelPicker OAuth password is needed locally.**

### 4. Test it in chat

> Use ModelPicker to inspect my computer and show its capabilities.

If you authorized the repository's `examples` folder:

> Import demo.csv using text and label columns. Find compatible English binary sentiment classifiers. Compare all 24 rows and show me the plan before starting. Wait for my approval, then evaluate and explain the results.

The authored 24-row dataset demonstrates the workflow; it is not a production benchmark. For a public dataset:

> Inspect stanfordnlp/sst2's validation split and import 100 rows using sentence as text and label as the expected answer. Compare compatible binary sentiment classifiers.

Use validation, not SST-2's hidden-label test split. Candidates require compatible trained heads and explicit label mappings; model popularity does not establish quality.

### pip alternative

On Linux/Windows, create a Python 3.11 environment and install CPU Torch before the local extra:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[local]'
modelpicker setup --dataset-dir ./examples
modelpicker doctor
```

Windows: use your Python 3.11 executable to create the environment and activate with `.venv\Scripts\Activate.ps1` in PowerShell. macOS: install Torch from the default PyPI index instead. The `local` extra is needed for inference and Hub dataset inspection/import; the base install supplies discovery and MCP only.

## Remote connection: no client-side code installation

Install ModelPicker on the **machine that will run inference**. Stop any stdio process using the same `MODELPICKER_HOME` before starting HTTP mode. One service owns each home; multiple trusted clients can connect to the same HTTP service.

### 1. Create a public HTTPS endpoint

Use a trusted reverse proxy, or install [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) and start a temporary tunnel:

```bash
cloudflared tunnel --url http://127.0.0.1:7860
```

Keep the terminal running. Copy its generated HTTPS origin, such as `https://YOUR-HOST.trycloudflare.com`.

### 2. Start ModelPicker

In another terminal in the repository:

```bash
uv run --no-sync modelpicker setup
uv run --no-sync modelpicker serve --transport streamable-http --public-url https://YOUR-HOST.trycloudflare.com
```

`--public-url` takes the external HTTPS **origin**, without `/mcp`. By default the server binds to `127.0.0.1:7860`. OAuth discovery and hostname checks use the configured external origin.

### 3. Add the connector

In Claude's **Settings → Connectors → Add custom connector**, enter:

```text
Name: ModelPicker
URL:  https://YOUR-HOST.trycloudflare.com/mcp
```

Complete browser authorization using the password in the host's `~/.modelpicker/owner-password.txt`. Enter it **on the ModelPicker consent page**, never in chat. Remote mode requires authentication and supports OAuth discovery, dynamic registration, authorization code + PKCE, rotating refresh tokens and revocation.

The client needs neither Python nor this repository. Evaluation runs on the **server host's CPU**. Keep the host awake and the service/tunnel running. Claude's remote connector connects from Anthropic's infrastructure; it cannot use your computer's `localhost` as its URL. See [Claude's connector documentation](https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp).

Quick-tunnel hostnames change on restart: update the public origin, restart the service and reconnect. OAuth state is in memory, so service restarts require reconnection. Cloudflare Quick Tunnels do not support SSE; use `streamable-http`. Legacy SSE is available for compatible deployments at `/sse`.

**One installation is for one trusted owner, not public multi-tenant hosting.** Connected clients share local datasets, saved evaluations and CPU resources. Independent owners need separate installations or a future design with individual accounts and isolated storage.

## Capabilities and all 23 tools

| Stage | Tools |
|---|---|
| Hardware and limits | `get_capabilities`, `configure_resources` |
| Dataset discovery | `search_datasets`, `inspect_huggingface_dataset` |
| Import and inspect | `import_huggingface_dataset`, `import_dataset_rows`, `import_dataset_file`, `import_dataset_url`, `inspect_dataset` |
| Model discovery | `search_models`, `inspect_model` |
| Plan and execute | `create_evaluation_plan`, `get_plan`, `start_evaluation`, `get_evaluation`, `cancel_evaluation` |
| Review | `get_failures`, `compare_evaluations` |
| Save, export and repeat | `list_saved`, `export_report`, `export_integration`, `schedule_evaluation`, `delete_saved` |

Also includes the `guided_model_selection` MCP prompt.

- **Data:** CSV/JSONL/JSON, pasted examples, owner-approved direct HTTPS URLs and bounded Hugging Face streaming. Chat attachments are not automatically available; supply rows, a supported URL or an authorized local path.
- **Reproducibility:** pinned revisions, dataset/sample hashes, seeded sampling, recorded row IDs and explicit label mappings. Each candidate sees identical sampled rows. Generic label polarity is never guessed.
- **Execution:** sequential CPU subprocess inference, safetensors only, no remote Python code. Background progress, cancellation, timeouts, queue limits, resource controls and restart recovery.
- **Metrics:** accuracy, macro F1, per-label precision/recall/F1, confusion matrix, Wilson accuracy interval, errors, p50/p95 latency, throughput, actual parameter count and observed process RSS.
- **Decisions:** recommendations subject to requested quality, latency and error constraints; no model qualifying is a valid outcome.
- **Reuse:** saved evaluations, comparisons, Markdown/JSON reports, Python integration export and explicitly requested scheduled repeats. Schedules need an awake service; there are no push notifications. Downloadable-file rendering depends on the client.

## Configuration

Setup persists dataset/cache paths in `~/.modelpicker/settings.json`. CLI commands load `.env` from their working directory; desktop launches may use another directory, so provide required settings explicitly in the generated MCP configuration. Keep tokens and passwords out of Git.

| Variable | Default |
|---|---|
| `MODELPICKER_HOME` | `~/.modelpicker` |
| `MODELPICKER_DATASET_DIR` | Saved value, otherwise `$MODELPICKER_HOME/datasets` |
| `MODELPICKER_CACHE_DIR` | Saved value, otherwise `$MODELPICKER_HOME/models` |
| `MODELPICKER_CPU_THREADS` | `2` |
| `MODELPICKER_MAX_ROWS` | `1000` evaluated rows/model |
| `MODELPICKER_MAX_IMPORT_ROWS` | `10000` imported rows |
| `MODELPICKER_MAX_MODEL_DOWNLOAD_MB` | `1500` MiB/model |
| `MODELPICKER_MAX_WORKER_RAM_MB` | `4096` MiB observed worker RSS |
| `MODELPICKER_MIN_AVAILABLE_RAM_MB` | `1024` MiB reserve |
| `MODELPICKER_MAX_JOB_SECONDS` | `3600`, including download/loading |
| `MODELPICKER_DATASET_URL_HOSTS` | `raw.githubusercontent.com` |
| `MODELPICKER_PUBLIC_URL` | Required HTTPS origin for remote mode |
| `MODELPICKER_OWNER_PASSWORD` | Optional override of generated password file |
| `HF_TOKEN` | Optional token for private/gated resources |

Up to 5 candidates per plan and 3 active/queued jobs. Imports are limited to 20 MiB and 20,000 characters/text. Local imports reject traversal and escaping symlinks. URL imports require approved hosts and reject redirects/private addresses. RAM checks are periodic, not an OS-level sandbox.

## Measurement and privacy

**Local inference does not make chat private from the provider.** Pasted text and returned tool results reach Claude or your chosen chat provider. Default results omit raw example text; `get_failures(include_text=true)` discloses it and should be used only when requested. Local storage is unencrypted. Deletion does not erase original files, model caches or forensic traces.

p50/p95 measure sequential tokenization + CPU inference at batch size 1, excluding download, loading and warmup. Throughput is successful calls divided by measured inference time, not a concurrent load test. Weight size is not runtime memory; performance depends on hardware, input length and load.

Balanced sampling changes label proportions. Bounded-buffer Hub streaming is not uniform sampling across the entire dataset. Duplicate/related examples undermine interval assumptions. Public benchmarks may overlap training data and do not establish production performance. Use representative labeled data and check licenses before reuse.

## Troubleshooting

| Problem | Check |
|---|---|
| Claude cannot start the server | Generated absolute Python path exists; fully restart Claude; inspect Local MCP server logs. |
| `No module named modelpicker` | Install in the same environment referenced by the configuration. |
| Missing inference packages | Install `--extra local`, then run `modelpicker doctor`. |
| Dataset import rejected | File must be inside the authorized directory; rerun setup with `--dataset-dir` if needed. |
| Another service owns this home | Stop it, use its HTTP endpoint, or choose a separate `MODELPICKER_HOME`. |
| Authorization failed | Reconnect for a fresh request (expires after 10 minutes), allow the consent cookie, and enter the current password. An environment override takes precedence over the password file. |
| Remote URL stopped working | Keep host/tunnel running, update changed origins and reconnect after restart. |
| Resource limit reached | Reduce row/model size or adjust installation ceilings to fit available hardware. |

## Development and evidence

```bash
uv sync --frozen --python 3.11 --extra local --extra dev
uv run --no-sync pytest -q
uv run --no-sync ruff check modelpicker tests scripts server.py test_setup.py
uv run --no-sync ruff format --check modelpicker tests scripts server.py test_setup.py
```

Tests use controlled fixtures and cover imports, boundaries, labels, metrics, job supervision, exports, stdio MCP and HTTP OAuth. They do not prove every branded client/OS integration. A separate live smoke test downloads a real classifier when missing:

```bash
uv run --no-sync python scripts/live_smoke.py
```

See [architecture](docs/ARCHITECTURE.md), [demo guide](docs/DEMO.md), [recorded validation](docs/VALIDATION.md) and the [four-page portfolio](portfolio/ModelPicker-Portfolio.pdf).
