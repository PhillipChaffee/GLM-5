# Regen concurrency ramp: measured throughput-vs-concurrency curve and the token-accounting audit

Issue [#39](https://github.com/PhillipChaffee/GLM-5/issues/39), under [#23](https://github.com/PhillipChaffee/GLM-5/issues/23) (map [issue #1](https://github.com/PhillipChaffee/GLM-5/issues/1)). One warm-weights B200:4 server boot (sglang TP4 target, recipe below), four concurrency arms of 256 identical rows each from the smoke split head (`perfectblend_train.jsonl` rows 1–256, resume off, temp 0.7 / top_p 0.8 / top_k 20, effort low, clear_thinking on), then a CPU retokenization audit of the stored corpora. Ramp run: 26.8 min server time all-in (**≈ $11** on 4×B200 at the recorded $25/node-h — rate source: [#23 session-3 briefing](https://github.com/PhillipChaffee/GLM-5/issues/23#issuecomment-6104880313), consistent with [modal.com/pricing](https://modal.com/pricing) B200 $6.25/GPU-h ×4). Runs (Modal app records, look up by id): ramp `ap-lUqNkMur6ABBepLuFAHUE7`, audit `ap-dNpVNjjyxdPUrC9BDuCiIO`. Raw corpora on volume `glm-drafter-run` at `data/ramp/c{64,128,256,512}.jsonl`.

## The curve (the acceptance)

Completion tokens from the server's own `usage` accounting (now aggregated by the regen client, `experiments/glm-drafter/scripts/data/generate_train_data_glm.py`); window = client wall time from submit to final drain.

| arm | rows success/attempted | completion tok/s aggregate | tok/GPU | rows/min | usage tok/row |
|---|---|---|---|---|---|
| c64 | 255/256 | 724 | 181 | 63.7 | 682 |
| c128 | 255/256 | 1,352 | 338 | 122.7 | 661 |
| c256 | 255/256 | 1,493 | 373 | 126.7 | 707 |
| c512 | 256/256 | 1,436 | 359 | 129.4 | 666 |

**Saturation at c≈128–256: throughput flattens at ~1.4–1.5k tok/s aggregate (~370 tok/s/GPU); c512 gains nothing (−4% vs c256).** Queue never non-zero (`#queue-req: 0`), zero timeouts/retractions, one spec of a single context-overflow error per arm at c64–c256 (the same borderline row sampling differently each time; c512 caught a lucky 256/256).

**Recommended operating point for the full corpus regen: c256.** c128 is within 3% of it; c192 unnecessary; beyond 256 the plateau is measured, not projected.

## Audit verdict — what the smoke's 258 tok/s/node actually was

Four hypotheses from [the serving-bounds research](https://github.com/PhillipChaffee/GLM-5/blob/main/docs/research/glm53-flash-serving-bounds.md), checked against the ramp's own server banner and the fresh accounting:

1. **Token accounting (refuted as the gap's cause).** The regen client previously printed *no* token stats at all — the smoke's 1,033 tok/s (258/GPU) was reconstructed off-line from the stored corpus. The audit stage retokenized the smoke corpus with the target tokenizer: **full tok/row mean 710, p50 162, p90 2,097** — reproducing the recorded scratch analysis exactly. Server usage counts at each arm (661–707 tok/row) sit within 2–5% of the corpus-retokenized counts (629–695; the delta is per-turn template framing tokens that usage counts and the corpus does not). Conclusion: both channels count *full* generated tokens including reasoning, so the smoke's counting is honest as recorded.

Reasoning share, same audit: visible-vs-full split on the stored think spans reads **full/visible = 1.06–1.08** across all five corpora (the split nonzero at ~6–8% — the think tag ids did match, else the ratio would read 1.00): at `reasoning_effort=low` the rationale is a small slice of the generated tokens on this workload, never the hidden-4x the cookbook note hypothesized.
2. **DSA backend fallback (refuted).** Banner: `attention_backend dsa`, `Set DSA backends for fp8_e4m3 KV Cache: prefill=trtllm, decode=trtllm`, `dsa_topk_backend sgl-kernel`, MoE `flashinfer_trtllm` on SM10. The change through the fragile per-arch backends (#42012/#40286/#37105) never engaged: the good trtllm pair ran.
3. **Effective decode batch (partially true, not the binding limit).** `max_running_requests capped to 204 by the mamba state cache` (1,022 slots × 5/state/request at `mamba_full_memory_ratio 0.9`); KV pool 5,929,472 tokens fp8_e4m3 (39.12 GB/rank), in-flight demand ≤ ~100k tokens (<2% of pool). The formal cap truncates the c256/c512 arms to 204 admitted requests — yet the observed server-side decode batch peaked ≈128 mid-window, i.e. **the actual ceiling is workload-shaped, not admission-shaped** (see 4).
4. **Missing CUDA graphs (refuted).** Decode graphs: `backend full`, 30 tiers captured through bs=512 (~90 s, `avail_mem` 24.7 GB after), replay live during the ramp (`cuda graph: True` in every decode line). Prefill graphs are off by engine design for KDA hybrids ("Breakable CUDA graph is incompatible with KDA hybrid linear attention"), matching the published KDA serving recipe.

**Why ~1,500 aggregate still sits far below the published spec-off anchors (2,660 @ c64, 4,828 @ c256; 4×GB300/GB200 cookbook cells):** those anchors measure single-turn, 256-token-OUTPUT, temp-0 decode-bound windows. Regen is multi-turn (~1.34 turns/row measured), ~700-token outputs, sampling on, with a turn-boundary re-prefill of the growing conversation between decode phases, and decode batches oscillating ~29–128 mid-window. sglang's own decode-interim lines hit 2,300+ tok/s in bursts — the engine is healthy; the aggregate is prefill-decode mixing + client per-sample pipeline limited. The 5–10k tok/s/node planning assumption inherited from #11 is refuted on this workload: ~370 tok/s/GPU is the measured regime.

## Cross-check against the recorded smoke run

| quantity (c64) | this ramp | recorded smoke |
|---|---|---|
| tok/s aggregate | 724 | 1,033 (derived: 698 × 710 / 480 s) |
| tok/row (full) | 682 (usage) / 648 (corpus) | 710 (corpus retok) |
| rows/min | 63.7 | 87 |

Token basis aligns (audit proof above); the ~30% rate residual is unattributed engine/config drift between builds — the regen image resolves `pip_install("sglang[all]")` unpinned, so the 2026-10-13 build (torch 2.9.1, transformers 5.10.2) is a different engine from the 2026-10-09 smoke build. Left open for the next session: record `pip freeze` of the regen image at build time and pin the resolved version for the real run, so stage costs stop re-anchoring with each brew.

## Cost re-anchor for the full run (feeds #45)

Measured regen rate at the recommended c256: **126.7 rows/min** (last arm's 129.4 within noise). 1.42M rows → 11,206 min ≈ **187 node-hours ≈ $4,700** on 4×B200 at $25/node-h (rate source: [#23 billing arithmetic](https://github.com/PhillipChaffee/GLM-5/issues/23#issuecomment-6104880313); [modal.com/pricing](https://modal.com/pricing) B200). Replaces the recorded briefing line (~$6–8k from 87 rows/min extrapolated at c64) and the addendum's speculative $0.7–3k (5–10k tok/s/node — refuted above). ±10%: the 256-row windows sample a random-equivalent head of the shuffled split; the resolved throughput plateau is broad (c256–c512 all 1,436–1,493).

Error tail: ~0.4% context-overflow failures at the smoke's 12288-context/4096-token recipe, non-deterministic per arm on borderline rows (c512 served all 256 where c64/128/256 each dropped one); the regen client skips and resumes; no cost impact.

## Levers deliberately not pulled (named for later)

- Pythonside: the client's per-sample sequential-turn pipeline (one thread per sample, turn waits inline) is a candidate for a batched two-phase loop (submit all turn-1 for N samples, then turn-2...) — could cut per-turn serialization but doesn't change the measured plateau shape.
- `--mem-fraction-static` / `--max-mamba-cache-size` levers exist (#37 ticket-adjacent) but admission never bound; pointless here.
- #57's token-budget regen admission (length-sorted streaming) touches the *cache* stage, not this client.
