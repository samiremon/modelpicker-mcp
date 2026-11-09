# Validation record

Validated on 2026-11-09 (Asia/Dhaka), Linux x86_64, Intel Core i7-1255U, approximately 15 GiB RAM.
CPU evaluation used two threads, batch size 1, 256-token truncation, Python 3.11.17,
PyTorch 2.15.0+cpu and Transformers 4.57.6.

## Real inference

100 streamed examples from `stanfordnlp/sst2`, validation split; seed 42 and bounded shuffle buffer 200.
Source revision: `8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb`.
Saved snapshot SHA256: `9338ce72c6f824a4e83ef08af761076a61ddeb89ac1413a8a27f8760031fbd13`.

| Model | Revision | Accuracy | Macro F1 | p50 ms | p95 ms | Errors |
|---|---|---:|---:|---:|---:|---:|
| distilbert/distilbert-base-uncased-finetuned-sst-2-english | 714eb0fa89d2f80546fda750413ed43d93601a13 | 0.93 | 0.9300 | 21.51 | 28.84 | 0 |
| xuancoblab2023/tiny-bert-sst2-distilled | 2069895b5184ce108139265d91f7e2234e0a70bf | 0.50 | 0.3800 | 0.99 | 1.73 | 0 |
| Harsha901/tinybert-imdb-sentiment-analysis-model | 0cd5d1ac6c06eb0f5f81b7022f95691528c981fa | 0.84 | 0.8400 | 3.64 | 5.42 | 0 |

All 300 predictions completed without execution errors. Worker elapsed time was approximately 39 seconds,
including loading/downloading the two smaller models; DistilBERT weights were already cached.
This is one run on one machine, not a speed guarantee. The smallest model's poor score is retained,
not hidden. Public dataset performance is not proof of production quality or independent of training data.

Generated integration code was executed successfully and returned a valid mapped label.
The complete stdio MCP flow also ran three real models against all 24 authored examples (72 predictions),
including plan creation, background polling and report/integration export through MCP tools.
An earlier real evaluation of the authored 24-row CSV also completed without execution errors.

## Automated verification

The 22-test network-independent suite covers dataset normalization, paths/symlinks, sampling, label mappings,
classification metrics, constrained recommendations, persistence, resource settings, cancellation,
timeouts, memory supervision, crashes, partial results, exports, and setup configuration.
It includes a real stdio MCP client/server subprocess and HTTP OAuth authorization/PKCE, code replay
rejection, token rotation, revocation, host protection, and state shared across stateless tool calls.

The suite is also run in a separate environment without Torch/Transformers/datasets to verify the
discovery-only installation and offline test fixtures do not depend on the heavyweight runtime.

The Docker image built successfully from the lockfile. A non-root container (UID 1000) with a fresh named
data volume started successfully: `/health` returned 200, unauthenticated `/mcp` returned 401, OAuth
resource discovery returned 200, and Torch/Transformers/datasets imported successfully. The test container
and temporary volume were removed afterward. The local `modelpicker-mcp:demo` image remains available.

## Limits of this validation

Real branded Claude/Gemini/ChatGPT interfaces, external HTTPS tunnels, macOS, Windows, ARM hardware,
and GPU execution are not established by these tests. Initial account/connector setup is still necessary.
No claim is made that arbitrary Hub models or unlabeled datasets are compatible. See the README's explicit
supported-task and deployment boundaries.
