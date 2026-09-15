# Benchmarks

Benchmark programs are TypeScript files in this directory. `raw/` contains the final, immutable JSON records cited by the project README. Generated charts, browser profiles, captures, and ad-hoc profiling output are intentionally ignored.

## Real-model evaluation

```sh
npm run bench:model
npm run bench:model:decode
```

The primary run covers 256–4,096 prompt tokens, alternating FP16 and compressed paths with one short warmup and three repetitions per workload. Host wall-clock timing includes GPU execution, waits, and logits readback; it excludes loading, tokenization, pipeline creation, and UI rendering. Prefill computes and reads logits after every 32-token chunk.

The decode supplement measures 24 identical supplied tokens per path and context after prefill. It is teacher-forced timing from one trajectory, not autonomous-generation throughput or repeated-run variability. Missing greedy decode samples caused by immediate EOS are preserved as missing, never treated as zero.

Final records:

- `raw/2026-09-15T10-43-22-413Z-model.json`
- `raw/2026-09-15T10-52-04-928Z-model-decode.json`

## Synthetic attention evaluation

```sh
npm run bench:attention
```

The final optimized record compares FP16 attention, the materialized compressed baseline, and fused compressed attention over the same data and device. It uses five warmups, twenty samples, rotating path order, and GPU timestamps. The result isolates attention-kernel behavior; it is not an end-to-end model benchmark.

Final record: `raw/2026-09-15T09-27-00-112Z-phase4-phase4-bench.json`.

Each raw record includes hardware metadata, measurement boundaries, samples, buffer accounting, source hashes, and correctness-gate hashes. Explicit allocation excludes driver reservations and physical peak VRAM.
