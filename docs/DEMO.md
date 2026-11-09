# Recording the demo

## Prepare

1. Run `uv sync --frozen --extra local --extra dev`.
2. Run `MODELPICKER_DATASET_DIR="$PWD/examples" uv run modelpicker setup`.
3. Connect the generated stdio configuration to a compatible desktop surface, or run authenticated
   Streamable HTTP behind HTTPS and add the `/mcp` URL to a supported web client.
4. Run `uv run python scripts/live_smoke.py` to pre-download and verify the example model.
5. Keep your owner password, tokens and sensitive file paths out of the recording.

An initial local installation/connector consent is required. Do not claim “no setup.” After connecting,
the routine workflow is chat-only. A chat attachment needs explicit client-specific transfer support;
the reliable demo sources are pasted rows, an authorized local file, or a Hub dataset.

## Suggested conversation

> I want to find an English sentiment classifier for my laptop. Inspect my resources first.

> Use my demo.csv file. The columns are text and label. Use all 24 rows.

> Inspect distilbert/distilbert-base-uncased-finetuned-sst-2-english and other compatible candidates.
> Show output labels and download sizes before choosing.

> Make the evaluation plan. Prioritize quality and require macro F1 of at least 0.8.

> Run it. How is it progressing?

> Show the measured comparison and explain which requirements passed.

> Show the incorrect predictions, including their text.

> Give me Python integration code and a Markdown report.

For 100 public examples, use `stanfordnlp/sst2`, validation split, `sentence` / `label` columns.
For three models, first inspect each model for compatible labels and safetensors. Discovery can return
models that are not runnable by this backend; explain that and select alternatives.

## What the video should show

- The user chooses a dataset source and row count.
- The assistant asks only for missing requirements and label mappings.
- A saved plan identifies actual models/revisions and the same rows for every model.
- A real background evaluation returns progress and observed ETA.
- Results show actual metrics with sample-size limitations, not size-derived performance estimates.
- Follow-up questions inspect failures, export code or test more examples.

Use labeled authored examples for a workflow demonstration, not claims of production accuracy. If you
edit out waiting time, say so. First downloads can dominate runtime. A cached report should be identified
as a saved run. Verify each advertised chat-client integration in its real interface before recording it.

## If something fails

- Runtime missing: install the local extra and CPU torch; check `modelpicker doctor`.
- Missing labels: use a labeled validation split or provide expected labels.
- Generic model labels: inspect the model card and provide a verified mapping; do not guess.
- No safetensors / custom model code: choose another supported classifier.
- Memory limit: close other programs or choose smaller models; adjust limits within installation ceilings.
- Remote connection: check the exact public origin, HTTPS forwarding, owner consent and `/mcp` path.
- Home already in use: close the old service or share its HTTP endpoint between clients.
- Request finished before evaluation: use the returned job ID with `get_evaluation`.

This project does not provide a cloud inference subsidy or always-on hosting. Local jobs require an awake
computer; a chat client is not guaranteed to send unsolicited notifications when a background job ends.
