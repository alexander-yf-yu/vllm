# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

The `@AGENTS.md` import above is the canonical source for the **contribution policy**
(duplicate-work checks, no-busywork rules, accountability) and the **development
workflow** (environment setup, build/install, tests, linters, commit trailers).
Read it first; do not duplicate its commands here. The sections below add what it
does not cover: the runtime architecture and this checkout's local build specifics.

## Non-negotiables (from AGENTS.md)

- **Never** use system `python3` or bare `pip`. Everything goes through `uv` and `.venv/bin/python`.
- Python-only change → `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto` (skips the C++/CUDA recompile).
  C/C++/CUDA/Rust change → `uv pip install -e . --torch-backend=auto` (recompiles).
- Run one test: `.venv/bin/python -m pytest tests/path/to/test_file.py -v` (`source activate` does not persist in non-interactive shells).
- Lint as CI does: `pre-commit run --all-files`; mypy: `pre-commit run mypy-3.12 --all-files --hook-stage manual`. Line length 88.

## Architecture (V1 engine)

V1 (`vllm/v1/`) is the only engine; `vllm/engine/llm_engine.py` is a thin alias
(`LLMEngine = V1LLMEngine`). The system is a frontend → IPC client → core loop →
worker pipeline that spans **three process tiers**, which is the single most
important thing to internalize before debugging "engine core" or multiprocessing errors:

```
Main process            EngineCore process (child)        Worker process(es) (grandchildren)
LLM / AsyncLLM    --ZMQ-->  Scheduler -> Executor   --->   GPUWorker/CPUWorker -> ModelRunner
(EngineCoreClient)          (core.py loop)                  (one torch model per TP/PP/DP rank)
```

### Request lifecycle
- **Entry points** (`vllm/entrypoints/`): offline `LLM` class (`llm.py`) and the
  OpenAI-compatible server (`openai/api_server.py`, launched by `vllm serve` via
  `cli/`). There are also `anthropic/` and `grpc_server.py` surfaces.
- Both converge on **`EngineCoreClient.make_client()`** (`vllm/v1/engine/core_client.py`),
  which picks one of: `InprocClient` (in-process), `SyncMPClient` (for `LLM`),
  `AsyncMPClient` (for `AsyncLLM`). The MP clients talk to the engine over **ZMQ
  sockets + msgpack** (`vllm/v1/serial_utils.py` `MsgpackEncoder/Decoder`); large/
  multimodal tensors go via a separate torch IPC channel, not msgpack.
- **`EngineCore` / `EngineCoreProc`** (`vllm/v1/engine/core.py`) runs the busy loop
  in the child process: `Scheduler` → `Executor` → workers, looping on the ZMQ queue.
  Spawned by `launch_core_engines()` (`vllm/v1/engine/utils.py`).
- Tokenization/detokenization bracket the loop in the main process:
  `InputProcessor` (`vllm/v1/engine/input_processor.py`) and `OutputProcessor`
  (`output_processor.py`, handles logprobs + streaming deltas).

### Scheduling & KV cache (the performance core)
- `vllm/v1/core/sched/scheduler.py` — continuous batching + chunked prefill;
  emits a `SchedulerOutput` batch each step.
- `vllm/v1/core/kv_cache_manager.py` + `block_pool.py` — paged KV block
  allocation; prefix caching is a hashed block lookup in the block pool.
- `vllm/v1/kv_cache_interface.py` — `KVCacheConfig`/layout spec.

### Models, attention, platforms (the hardware-portability seam)
- `vllm/model_executor/models/registry.py` — 290+ architectures mapped to classes.
- **`vllm/v1/attention/selector.py` `get_attn_backend()`** is the key portability
  abstraction: it asks `vllm/platforms/{cuda,rocm,cpu,...}.py`
  (`current_platform.get_attn_backend_cls()`) to pick a backend from
  `vllm/v1/attention/backends/` based on head size, dtype, KV-cache dtype, MLA, etc.
- `vllm/v1/executor/abstract.py` `Executor.get_class()` routes to `MultiprocExecutor`,
  `RayDistributedExecutor`, or `UniProcExecutor`. Workers in `vllm/v1/worker/`
  (`gpu_worker.py`/`gpu_model_runner.py`, `cpu_worker.py`) consume `SchedulerOutput`
  → produce `ModelRunnerOutput`.

### Config
- `VllmConfig` (`vllm/config/vllm.py`) aggregates ModelConfig, CacheConfig,
  ParallelConfig, SchedulerConfig, CompilationConfig, etc., and is threaded through
  via the thread-local `set_current_vllm_config()` context (`vllm/config/__init__.py`).

### Native code
- **`csrc/`** → compiled into `vllm._C` (`.abi3.so`) via `CMakeLists.txt` + `cmake/`.
  `VLLM_TARGET_DEVICE` selects cuda/rocm/cpu (`cmake/cpu_extension.cmake` for CPU/macOS).
- **`rust/`** → `vllm-rs`, an optional Rust serving frontend that talks to EngineCore
  over the same ZMQ+msgpack path; builds the `_rust_tool_parser` extension via
  `setuptools-rust` (toolchain pinned in `rust-toolchain.toml`).
- **torch.compile / inductor** is used at runtime to JIT-compile model graphs during
  warmup — meaning a working C/C++ compiler is required at *run* time, not just build time.

## Local build: this checkout (Apple Silicon / CPU)

This is a from-source **CPU** build (no CUDA; the only option on Apple Silicon).
Env: Apple M5, Python 3.12 venv at `.venv`, torch 2.11.0, Rust 1.95.
Full details and rebuild commands are in **`SETUP_NOTES.md`**. Key gotchas:

- This machine's Command Line Tools libc++ is **incomplete** (missing `<map>` etc.;
  present only in the SDK). Both the build *and* runtime torch.compile need:
  `export CPLUS_INCLUDE_PATH="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1"`
  Permanent fix: reinstall CLT (`sudo rm -rf /Library/Developer/CommandLineTools && xcode-select --install`).
- macOS multiprocessing is `spawn`, so any script that constructs `LLM(...)` at module
  scope **must** guard it under `if __name__ == "__main__":` (the engine-core child re-imports the module).
- Keep `VLLM_CPU_KVCACHE_SPACE` (GB) small on 16 GB RAM; oversizing fails KV-cache allocation.
- Quick check: `.venv/bin/python smoke_test.py` (tiny opt-125m CPU generation).

> `SETUP_NOTES.md` and `smoke_test.py` are local additions, not upstream files.
