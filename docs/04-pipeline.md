# The inter-file pipeline — design, dead ends, and the version that works

**Problem:** Whisper's structure per file is `encode (NPU) → decode (CPU, autoregressive)`.
During CPU decoding the NPU idles (~50 % duty cycle). For one-window files — the
majority of conversational audio — there is nothing *inside* a file to overlap.

**Insight:** overlap *across* files. While the CPU decodes file N, the NPU pre-encodes
the first window of file N+1. The NPU becomes the saturated resource; throughput is
bounded by `max(NPU, CPU)` instead of their sum. Measured: **+35 %** (60 s → 39.3 s
on 30 files), byte-identical outputs.

## Dead end 1: intra-file speculation

Whisper advances its window by the timestamp of the last generated token
(roll-forward), so window N+1's start offset is only known *after* decoding window N.
Speculating `offset + 3000 frames` is almost always wrong (the real delta is
~2900-something) → the speculative forward is discarded → zero gain. Abandoned after
reading `whisper_full_with_state`'s seek loop.

## Dead end 2: naive prefetch (pipeline v1)

Launch the prefetch thread *before* `whisper_full(file N)`:

```
worker: mel(file N+1) → lock NPU mutex → forward(N+1)
main:   whisper_full(file N) → take: join(worker) → …
```

**Deadlock.** The take joins the worker before the main thread ever reaches its
decode phase; the worker waits for a decode-phase signal that never comes.

First fix attempt: drop the signal, keep only the NPU mutex. No deadlock — but the
worker's forward executes *before* the main thread's encode (mutex inversion), so
every file pays forward(N+1) + forward(N) serialized: **83 s, worse than sequential.**

## Working design (pipeline v2): callback at decode-entry

The only party that knows "the CPU is now decoding" is whisper.cpp itself. v2 adds:

1. **`pipeline_prefetch_callback`** in `whisper_full_params` — invoked by
   `whisper_full_with_state` right when the main thread enters its decode loop.
2. The CLI registers a launcher that (once per file) calls
   `whisper_pipeline_prefetch(ctx, next_samples, n)`.
3. The prefetch worker: mel (CPU, private buffer) → **wait on a phase condition
   variable** until the main thread is decoding → NPU mutex → conv+NPU forward into a
   double-buffered `pipe_embd_enc`.
4. Next file's `whisper_full` consumes the result: join → if the sample pointer
   matches, swap mel and `embd_enc` buffers; the first window skips encoding entirely
   (only the cross-KV pass runs).

### Invariants that make it safe

- **Exactly one NPU forward in flight**: all forwards (main and worker) serialize on
  a single mutex — the [NPU rule](07-npu-rules.md) is structurally enforced.
- No concurrent access to shared state: the worker borrows `state->mel` and the conv
  scheduler **under the same mutex** the main thread's encoder uses; the decoder
  touches neither.
- Stale results are dropped by sample-pointer identity + a generation counter
  (resumed speech invalidates an in-flight speculation).
- Anti-deadlock: the take force-signals the phase CV before joining.

### MSVC pitfalls (fork-specific)

- Aggregate initializers in `whisper_full_default_params` are **positional** —
  `/*.field =*/` are just comments. Inserting the two new fields mid-struct required
  moving the initializer to the exact matching position.
- The pipeline code must compile under the lib's C++11 constraint (std::thread,
  condition_variable, atomic — all C++11).

## Where the code lives

All in `patches/0001-whisper-npu-full-patch.patch`: `src/whisper.cpp`
(state fields, `whisper_pipeline_prefetch`, take logic, decode-phase signal),
`include/whisper.h` (API + params), `examples/cli/cli.cpp` (lookahead reader +
launcher), CMake option `WHISPER_PIPELINE`.

Runtime toggle: set `WHISPER_NO_PIPELINE=1` to fall back to sequential (A/B testing).
