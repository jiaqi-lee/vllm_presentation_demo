# vllm_presentation_demo

Interactive streaming demo for vllm-ascend product showcases.

The demo provides a terminal REPL where every prompt you type is forwarded
to an already-running vLLM service as a streaming chat-completions request,
tokens are printed to the screen as they arrive, and an
`ais_bench`-style performance table (TTFT / TPOT / ITL / E2EL / RPS / TPS / ...)
is printed once the stream finishes. There is no dataset-initialization wait
in front of the user because the request is sent the moment the prompt is
entered.

## Why this demo

`ais_bench`-style dataset runs are great for repeatable benchmarking but they
hide the model from the audience: the user only sees a final table after a
long dataset-preparation pause. This demo keeps the same performance
formulas used by `ais_bench` but exposes them on a per-prompt basis with
visible streaming output, so the audience can both *feel* and *measure* the
speedup of enabling/disabling vllm-ascend features.

## Requirements

- Python 3.9+
- A running vLLM service exposing `POST /v1/chat/completions` (streaming).
  Defaults assume `http://127.0.0.1:8004` with model name `minimax`, matching
  the curl template in `需求描述.txt`.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Start the REPL against the default endpoint:

```bash
python demo.py
```

Or point it at a different service:

```bash
python demo.py --url http://127.0.0.1:8004 \
               --model minimax \
               --max-tokens 200 \
               --temperature 0
```

Inside the REPL you can type any prompt and press Enter. Streaming output is
printed live, followed by two performance tables.

### REPL commands

| Command                | Purpose                                                       |
|------------------------|---------------------------------------------------------------|
| `/help`                | Show available commands                                       |
| `/config`              | Print the current configuration as JSON                       |
| `/set <key> <value>`   | Update a setting (`base_url`, `model`, `max_tokens`, `temperature`, `enable_thinking`, `timeout`) |
| `/exit`, `/quit`, `/q` | Exit the demo                                                 |

Ctrl+C during streaming cancels just that request and returns you to the
prompt. Ctrl+C at the empty prompt exits.

## What request is actually sent

The forwarded request body matches the curl template from `需求描述.txt` exactly,
with only the user prompt substituted into `messages[0].content`:

```json
{
  "model": "minimax",
  "messages": [{"role": "user", "content": "<your prompt>"}],
  "stream": true,
  "temperature": 0,
  "max_tokens": 200,
  "chat_template_kwargs": {"enable_thinking": false},
  "stream_options": {"include_usage": true}
}
```

`stream_options.include_usage` is added so vLLM emits the final `usage`
block (with `prompt_tokens` / `completion_tokens`) in the closing SSE chunk.
This is the same mechanism `ais_bench`'s `VLLMCustomAPIChat` uses to obtain
token counts, and the demo relies on it for accurate TPOT / TPS reporting.

## Performance metrics

All formulas mirror `ais_bench`:

| Metric                    | Formula                                             |
|---------------------------|-----------------------------------------------------|
| TTFT                      | `time_points[1] - time_points[0]`                   |
| TPOT                      | `(latency - ttft) / (output_tokens - 1)` if `output_tokens > 1` else 0 |
| ITL                       | `np.diff(time_points[1:])`                          |
| E2EL / Benchmark Duration | `time_points[-1] - time_points[0]`                  |
| OutputTokenThroughput     | `output_tokens / latency`                           |
| Request Throughput        | `1 / infer_time`                                    |
| Input Token Throughput    | `input_tokens / infer_time`                         |
| Output Token Throughput   | `output_tokens / infer_time`                        |
| Total Token Throughput    | `(input_tokens + output_tokens) / infer_time`       |
| Prefill Token Throughput  | `input_tokens / ttft`                               |
| Concurrency               | `latency / infer_time`                              |

Time-based metrics are reported in milliseconds and throughput metrics in
`token/s` / `req/s`, matching `ais_bench`'s `_add_units_to_*` helpers.

## How streaming output stays honest

The client uses an asyncio producer/consumer split:

- The **producer** reads SSE chunks from vLLM. The moment a non-empty content
  delta arrives it records `time.perf_counter()` (which becomes the next
  `time_points[i]`) and `put_nowait`s the text into an unbounded queue. It
  never waits on stdout.
- The **consumer** drains the queue and writes to stdout.

Even if the terminal is slow, only display is delayed; TTFT / ITL / TPOT
always reflect the true server-to-client timings.

## Project layout

```
vllm_presentation_demo/
├── README.md
├── requirements.txt
├── demo.py
└── presentation_demo/
    ├── __init__.py
    ├── config.py        # CLI + DemoConfig
    ├── client.py        # async SSE streaming client + producer/consumer
    ├── metrics.py       # ais_bench-compatible per-request metric computation
    ├── table.py         # tabulate fancy_grid rendering of two ais_bench-style tables
    └── repl.py          # REPL main loop and control commands
```
