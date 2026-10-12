# Is a GLM-5.3-Flash target hidden-state cache published anywhere? A wide survey

Research supporting the drafter-evaluation and cache-build work under [Train the GLM-5.3-Flash drafter via DeepSpec on Modal, end to end](https://github.com/PhillipChaffee/GLM-5/issues/23) (map #1). Compiled 2026-10-12 by web research across Hugging Face (name + full-text + org search), GitHub (code/org/issue/release search), ModelScope (including Chinese keywords), Kaggle, Zenodo, Figshare, OSF, and the DSpark/DFlash papers' artifact statements. Motivated by the question "has anyone published the cache so we can consume it instead of building ~62 TB ourselves?"

## Answer in one paragraph

**No. After a venue-by-venue sweep, no published artifact contains a GLM-5.3-Flash target hidden-state cache — not at our tap set, not at any tap set, not at any scale, from any target precision. The build is ours.** DeepSpec releases checkpoints + code and instructs users to build their own ~38 TB cache for Qwen3-4B; TorchSpec streams hidden states without storing them by design; ModelScope (including 蒸馏缓存 / 隐状态 queries) is empty; Kaggle/Zenodo/Figshare/OSF are clean; the papers release code only. An independent searcher (jayleaton's `docs/DRAFTER-SEARCH.md`, 2026-09-29) reached the same negative result two weeks earlier. The ecosystem structurally doesn't share caches: every serious actor either regenerates text and re-runs the target (community caches are fp8 tap subsets in the hundreds of GB) or streams without storing. What the sweep *did* surface: **a trained, MIT-licensed DSpark drafter for our exact target** (`RedHatAI/GLM-5.3-Flash-speculator.dspark-preview`) whose checkpoint is public even though its training hidden states are not — a zero-training evaluation arm and potential warm-start that did not exist in the map's decisions.

## 1. Venue-by-venue results

- **DeepSpec (deepseek-ai)**: README "Released Checkpoints" = 12 drafters (Eagle3/DFlash/DSpark x Qwen3-4B/8B/14B, Gemma4-12B) trained on per-target regenerated open-perfectblend; users build the cache locally (README warns "roughly 38 TB for the default Qwen/Qwen3-4B setting"). GitHub releases API: **0 releases**. HF org datasets: theorem-proving text only. Open issues confirm the pain point is real: #21 "Support online training without precomputing the target cache" (open), #26 "add support for FP8 target cache quantization" (open RFC). <https://github.com/deepseek-ai/DeepSpec>
- **TorchSpec (lightseekorg)**: streams hidden states from engines to trainers via a bounded Mooncake staging store — "without materializing them on disk" by design. Released datasets are text regen corpora, not hidden states. No GLM-5.3-Flash support. <https://github.com/lightseekorg/TorchSpec>
- **ModelScope**: API queries `dspark` → 2 prompt sets; `GLM-5.3` → 2 (a fidelity-study mirror, a docker image); `hidden state` → 0; 蒸馏缓存 → 0; 隐状态 → 0; `GLM` → 110 results, all ChatGLM-era text/benchmarks/calibration mirrors. `incoai/GLM-5.3-Flash-DFlash2` mirrored (the NC-ND drafter, no data). <https://modelscope.cn>
- **GitHub**: one in-flight GLM-5.3-Flash training program (`jayleaton/glm53-tensorfold-spark`, T1/T2 DFlash re-fit program) keeps its tap dumps private by design ("Keep it on the Spark and delete it after training"). Its `docs/DRAFTER-SEARCH.md` (2026-09-29) enumerates every published GLM-5.3-Flash drafter/dataset and found no consumable cache — an independent corroboration of this survey. Code search (~512 hits for GLM-5.3-Flash + hidden states/cache) surfaces only engine implementations, serving kits, and fidelity/KLD eval docs — nobody publishes cached tensors.
- **Kaggle / Zenodo / Figshare / OSF**: clean (API-verified). Kaggle `glm-5.3` → 20 GGUF quant shards; `dflash` → mirrors + runtime payloads; Zenodo/Figshare/OSF → unrelated hits only.
- **Papers**: DSpark (arXiv:2607.05147) artifact statement releases trained checkpoints + DeepSpec, "checkpoints + code only; no cache, no training-data dump." DFlash (arXiv:2602.06036, ICML 2026) releases code (z-lab/dflash); z-lab's HF org has ~60 DFlash drafters for other families (GLM-5.1, not Flash) and eval/text-regen datasets only.

## 2. Adjacent artifacts — found, not consumable

- **`PixelML/glm53-flash-specdec-sliceB`** (35.2 GB, MIT, 2026-09-05): hidden-state capture from an **AWQ W4A16** target (not FP8-as-served), taps [5,14,24,33,42] (incoai DFlash2's, not ours), 455,367 tokens (~0.05% of our budget), int16 bit-views of bf16. Best audit value: independently confirms the tap convention ("hc_post-materialized + stream-mean") our capture must match.
- **`brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits`** (~450 GB-class): 42 routed layers x 1,310,720 BF16 hidden rows + router weights + a `terminal/last_hidden.bf16.bin` — but captured from the **BF16 reference weights** (not FP8-as-served), at MoE block inputs for quantization calibration, over a sealed panel with restricted-use windows. Panel, not cache.
- **`malaiwah/GLM-5.3-Flash-fidelity-suite-v1`** and calibration-activations siblings: final-hidden-only captures over 512 contexts, and all-layer attn_in/mlp_in/router_logits over 92x2048 tokens. Fidelity evidence, not training data.
- **Text regens** (reduce data-prep cost, not the capture): `shanjiaz/glm-5.3-flash-open-perfectblend-regen-25pct` (25% of Red Hat's 1.73M-row set; licence unstated), `Zek-Takai/glm53-flash-harvest` (MIT, 86k responses / 246M tokens), `mgoin/open-perfectblend-glm5.2-regen` (full corpus, but GLM-5.2 teacher). The remaining 75% of Red Hat's regen text and 100% of their hidden-state extraction are unpublished.

## 3. The GLM-5.3-Flash drafter family (all drafters, no data)

| Drafter | Licence | Taps | Notes |
| --- | --- | --- | --- |
| `RedHatAI/GLM-5.3-Flash-speculator.dspark-preview` | **MIT** | 20/28/32/36/40/44 + final 45 | **DSpark**, our algorithm family, our exact target; card discloses 1,731,254 regenerated open-perfectblend rows, "hidden-state extraction ran across two GB300 trays"; publishes checkpoint only |
| `modal-labs/GLM-5.3-Flash-DFlash` | MIT | 23/27/31/35/39/43 | Overlaps our union at 23/42/43; no published acceptance; training data undisclosed |
| `canada-quant/GLM-5.3-Flash-DFlash2-E/F/G` | Apache-2.0 | 9 taps (G) | G trained on 737k self-generated samples |
| `incoai/GLM-5.3-Flash-DFlash2` | CC BY-NC-ND | 5/14/24/33/42 | The #12 baseline (gated) |
| `avlp12/GLM-5.3-Flash-Alis-MTP-Drafter` | MIT | — | MLX extraction of the stock layer-45 MTP head for Apple Silicon, not trained; author has 0 dataset uploads |

Red Hat's model card states the training setup in detail and publishes **only the checkpoint**: "Auxiliary hidden-state extraction infrastructure is required for training only." Our own `phillipchaffee/dspark-proxy-run` (Qwen3-4B, includes a small DeepSpec target cache) remains the only public GLM-effort-adjacent hidden-state artifact — and it is ours, Qwen-target.

## 4. Why the ecosystem doesn't share caches

Every serious actor either (a) regenerates from regen-text corpora and re-runs the target — jayleaton budgets ~1.0 TB / 0.18 TB for ~11M generated tokens with 5/9 fp8 taps; the community norm is fp8 tap subsets in the hundreds of GB, not full bf16 per-token captures — or (b) streams hidden states without storing them (TorchSpec/Mooncake; DeepSpec issue #21 requests exactly this upstream). A 62 TB bf16 per-token cache at 8 taps is an outlier design nobody has published at any scale, for any model. And a cache is **locked to everything** — exact checkpoint revision, tap layers, stream convention (mHC), chat template, tokenizer — so even a found cache would need auditing against our capture spec rather than blind consumption.

## 5. Gaps

- Gated/private HF repos are invisible to unauthenticated search; a cache could sit behind a gate (only HF auth + org membership would confirm).
- Kaggle search indexes titles/metadata only.
- One Zenodo query variant 504'd twice; covered with alternate queries.
- Red Hat's full 1.73M-row regen text could exist under an unexpected name; only the 25% subset surfaced.

## Sources

- <https://github.com/deepseek-ai/DeepSpec> (+ releases API, issues #21, #26)
- <https://arxiv.org/abs/2607.05147> and <https://arxiv.org/html/2607.05147v1> (artifact statement)
- <https://arxiv.org/abs/2602.06036> and <https://github.com/z-lab/dflash>
- <https://github.com/lightseekorg/TorchSpec>
- <https://github.com/jayleaton/glm53-tensorfold-spark/blob/main/docs/DRAFTER-SEARCH.md>
- <https://huggingface.co/datasets/PixelML/glm53-flash-specdec-sliceB>
- <https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits>
- <https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1>
- <https://huggingface.co/datasets/shanjiaz/glm-5.3-flash-open-perfectblend-regen-25pct>
- <https://huggingface.co/datasets/Zek-Takai/glm53-flash-harvest>
- <https://huggingface.co/datasets/mgoin/open-perfectblend-glm5.2-regen>
- <https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark-preview>
- <https://huggingface.co/datasets/phillipchaffee/dspark-proxy-run>
- ModelScope API: /api/v1/datasets?Query={dspark, GLM-5.3, GLM, hidden state, eagle3, speculative, drafter, 蒸馏缓存, 隐状态}
- Kaggle/Zenodo/Figshare/OSF public APIs (queries listed in §1)
