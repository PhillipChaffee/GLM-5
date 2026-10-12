# Which target-model layers should feed the GLM-5.3-Flash DSpark drafter?

Research for the target-model layer-set decision ticket. Author: deep-research agent (opencode-13793), 2026-10-12.
Read-only research; nothing in the repo was modified.

All sglang citations are against the pinned serving SHA `961bcf48` (image `lmsysorg/sglang:nightly-dev-cu13-20261010-961bcf48`); line numbers at current main (`166b4ae`, fetched 2026-10-12) are given where they differ. Local sglang sources were read from an ephemeral shallow clone of main plus per-file fetches at the pinned SHA `961bcf48` (the clone has since been deleted; every citation stands against the pinned SHA or main as dated). Repo rule honored: no GLM chat-format control-token literals appear anywhere in this report; all files read under `experiments/glm-drafter/` were read through the mask filter.

---

## Q1 - Mechanics: the glm5_next capture, and why id 44 cannot fire

### The `[val+1]` input-indexed convention

`Glm5NextForCausalLM.set_dflash_layers_to_capture` (sglang `python/sglang/srt/models/glm5_next.py:1526-1538` @ `961bcf48`; 1547-1559 at main):

```python
self.capture_aux_hidden_states = True
self.model.dflash_capture = True
# Capturing before layer k + 1 gives the completed output of layer k
self.model.layers_to_capture = [val + 1 for val in layer_ids]
```

The comment states the semantics: **the capture machinery taps the input of a layer**, and input of layer `k+1` == completed output of layer `k`. So a requested `target_layer_ids` entry `k` is served by capturing during layer `k+1`'s forward. The same `[val + 1]` mapping is used by the EAGLE3 setter (`glm5_next.py:1514-1524`, which for `layer_ids=None` instead uses the fixed default `[2, num_layers // 2, num_layers - 3]` set directly, no shift) and by every other EAGLE3/DFlash target in the tree (llama.py:922-945 with the comment "we plus 1 here because in sglang, for the ith layer, it takes the output of the (i-1)th layer as aux hidden state"; qwen3.py:740-750 "SGLang captures 'before layer i'. To capture the hidden state after target layer k (HF-style), we capture before layer k + 1").

Inside the text model's forward, the loop is `for i in range(normal_start_layer, normal_end_layer)` — for TP1/last-PP-rank that is `i in 0..44` — and the capture callback is handed to the layer only `if i in self.layers_to_capture` (`glm5_next.py:1158-1187` @ `961bcf48`). The decoder layer forwards the callback into its **attention-boundary prepare** (`glm5_next.py:942-948`: `self.attn_boundary.prepare(hidden_states, forward_batch, capture=capture_output)`), i.e. the capture fires on the raw stream the previous layer handed on, **before** the next layer's input norm. The closure then applies the mHC contraction (`glm5_next.py:1099-1102`):

```python
def _prepare_aux_hidden_state(self, aux_hidden_state):
    if self.dflash_capture and self.config.mhc:
        aux_hidden_state = hc_contract(aux_hidden_state, self.config.hc_mult)
    return aux_hidden_state
```

`hc_contract(x, n) = x.unflatten(-1, (n, -1)).mean(dim=-2)` (kernels `ops/layernorm/mhc.py:1919-1921`) — the unweighted mean over the `hc_mult` mHC streams, bit-identical to the training-side fold (`experiments/glm-drafter/deepspec_glm/deepspec/modeling/dspark/glm5/hidden_states.py:36-43`, regression-tested per the compat report C.1).

### Why the last layer's output is structurally uncapturable there

For requested id 44 (== `num_hidden_layers - 1`), the map emits `layers_to_capture = [..., 45]`. The loop runs `i in 0..44` only, so `45` never matches; the capture point "input of layer 45" does not exist. Result: **6 of the 7 requested features are collected** (`AuxHiddenStateList` at `glm5_next.py:1156`; the 6 loop captures fire in ascending layer order for requested ids 1, 12, 23, 34, 42, 43).

Where the final norm runs relative to capture: **after the entire layer loop**. `glm5_next.py:1199-1210` @ `961bcf48`:

```python
hidden_states = residual_batch.complete_output(hidden_states, forward_batch)
if not forward_batch.forward_mode.is_idle():
    hidden_states = residual_batch.final_norm(hidden_states, forward_batch, self.norm)
...
return hidden_states, aux_hidden_states
```

`self.norm = RMSNorm(config.hidden_size, ...)` (`glm5_next.py:1040`). There is no capture argument on this call, and no code path after it appends to `aux_hidden_states`. So the closure cannot produce anything post-final-norm: the loop's last possible capture (input of layer 44 = output of layer 43) is already inside the loop, and the norm happens after the loop ends.

### What exactly gets stored per capture point

- For requested id `k < 44`: the value captured at input of layer `k+1` is the raw mHC stream state that layer `k`'s FFN exit handed on — the pre-norm residual-stream output of layer `k` — folded by `hc_contract` to `[B, S, D]`. This equals what the training cache stores for id `k` (mean-folded raw block output), which is why ids 1..43 serve bit-identically to training.
- The mHC state machinery writes streams at the FFN exit and **contracts them only after the last layer** (`layers/layer_boundary/residual/mhc.py:222-231`: `if self.state.is_last_layer: hidden_states = hc_contract(hidden_states, self.state.hc_mult)`), so the value entering `self.norm` at `glm5_next.py:1204` is the **folded, pre-final-norm** last-layer state. `self.norm` then produces the post-final-norm state that feeds the LM head.

### Is the post-final-norm state trivially available elsewhere? Yes

It is the **return value of the final-norm call** (`glm5_next.py:1204-1206`) on every non-idle forward — prefill and every verification step — and it is exactly what flows into the logits processor / LM head (`Glm5NextForConditionalGeneration.forward`, glm5_next.py:1619-1665). A capture of id 44 therefore needs ~3 lines after the `final_norm` call: if the final id is requested, `aux_hidden_states.capture(hidden_states)` (the post-norm tensor). Note the important subtlety verified below (Q5a): the existing `capture=` seam on `residual_batch.final_norm` (`layers/layer_boundary/residual/batch.py:59-114`) delivers the norm's **input**, not its output (`layers/layer_boundary/residual/access.py:40-54` — `capture(residual)` receives the updated residual; `RMSNorm.forward` returns `(normed, residual_new)` where `residual_new = x + residual`, `layers/layernorm.py:832-839, 867-870`).

### Where the drafter's fc crashes

The DFlash/DSPARK draft projects the concatenated features in `project_target_hidden` (`models/dflash.py:771-787`): it shape-checks `target_hidden.shape[-1] == fc.in_features` (`num_context_features * hidden_size` = 7 × 4096 = 28672 for our checkpoint; `fc` built at `dflash.py:700-716`) and raises the documented ValueError ("DFLASH target_hidden feature dim mismatch... the target model is capturing a different number of layer features than the draft checkpoint/config expects") on the first prefill, when only 6 × 4096 = 24576 arrive. The drafter-side consumer chain is `write_target_hidden_kv → project_target_hidden` (`models/dspark.py:729-743`).

---

## Q2 - Cross-model survey of sglang capture implementations

Wiring (`model_executor/model_runner_components/attention_backend_setup.py:40-66` @ main; verified): under DFLASH/DSPARK the runner calls `model.set_dspark_layers_to_capture(ids)` if the target has it, else `set_dflash_layers_to_capture(ids)`, else raises. So **the target's setter decides the convention**, and the drafter checkpoint's ids must have been authored against that same convention.

| Target model / setter | Convention | Can it produce the post-final-norm state? | Does the served drafter consume it? |
|---|---|---|---|
| **glm5_next** `set_dflash_layers_to_capture` (glm5_next.py:1526-1538) | input-indexed: ids shifted `[val+1]`, capture at input of layer k+1 = output of layer k; mHC `hc_contract` fold | **No** — id 44 maps to nonexistent "layer 45"; final norm runs after the loop (1204-1206) and is never captured | No shipped drafter can even get 7 features; DFlash2 drafter avoids it ([5,14,24,33,42] of 45, verified from `incoai/GLM-5.3-Flash-DFlash2/config.json`) |
| **kimi_k3** `set_dspark_layers_to_capture` (kimi_k3.py:2989-3001) + `set_dflash_layers_to_capture` (3003-3006, comment: "DFLASH target_layer_ids name layer outputs, which is what the DSPARK taps already capture here, so reuse them without the usual +1 shift") | output-indexed: ids used directly; standard path captures at next layer's input for `i-1 in captures` (2809-2839); last layer captured via `capture=` passed into `residual_batch.final_norm` when `end_layer - 1 in captures` (2865-2875) | **Partially** — it captures the final norm's **input** (pre-norm), not the post-norm output (see Q5a) | **No** — the shipped `RadixArk/Kimi-K3-DSpark` drafter taps `[7, 23, 51, 67, 83]` of **93** layers (config.json verified); the final-capture branch is unused by it |
| **deepseek_v2** (V3-family) eagle3 (deepseek_v2.py:3401-3417) | hybrid heuristic: `if layer_ids[0] == 1: [val+1] else: ids as-is`, with TODO "(Qiaolin-Yu): check if other draft models need similar layer id adjustment" | via the MTP/EAGLE input path, yes (below) | DeepSeek-style MTP (EAGLE worker) consumes the **post-norm last hidden state** as the drafter's input (`deepseek_nextn.py:194-226`: embedding + `hnorm(spec_info.hidden_states)` → `eh_proj`; the worker passes `return_hidden_states_before_norm=False`, eagle_worker_v2.py:977/1116, and the logits processor stores the post-norm `pruned_states` under `CaptureHiddenMode.LAST`, logits_processor.py:828-853) |
| **deepseek_v2** dflash (3418-3427) | input-indexed `[val+1]` | No | DFlash2-style drafters avoid it |
| **deepseek_v4** `set_dspark_layers_to_capture` (4773-4783) | output-indexed, ids direct; forward captures **after** layer i runs (`i in dspark_layers_to_capture`, 4508-4516), mHC mean-fold via `.mean(dim=1)`; final `self.norm` runs after (4537) and is not captured | No | DSV4.1-Flash's bundled DSpark drafter consumes encoder hidden states (paper Fig. 3); no final-layer tap |
| **qwen3 / qwen3_moe / qwen3_next / llama / gpt_oss / lfm2 / laguna / inkling / nemotron_h / gemma4 / nanbeige / mimo_v2 / qwen3_vl / ...** eagle3+dflash setters | input-indexed `[val+1]` for explicit ids; defaults `[2, N//2, N-3]` (no shift) | qwen3.py:747-750 comment shows the convention is deliberate; none capture post-norm | EAGLE3 drafts consume fused mid-layer features (official EAGLE-3 taps outputs of layers [1, N//2-1, N-4], see Q4); **gpt_oss is the exception — see next row** |
| **gpt_oss** eagle3 (gpt_oss.py:1336-1351) + forward (707-734) | boundary-indexed: "boundary 0 is the embedding output, and boundary i + 1 is the output after transformer block i"; setter **preserves ids as-is when `max(layer_ids) == num_layers`** ("Preserve IDs that already include the final hidden-state boundary") | **Yes — the only such path in the tree**: when `end_layer in layers_to_capture`, `capture=` is passed into `residual_batch.final_norm` (726-734) — which yields the final norm's **input** (pre-norm boundary) | **Yes**: `nvidia/gpt-oss-120b-Eagle3-v3` requests boundaries `[24, 30, 36]` of 36 layers and consumes the final pre-norm boundary (PR #32334); other gpt-oss Eagle3 heads consume the "final post-norm hidden state" via the MTP-style path |
| **multi_layer_eagle** worker (multi_layer_eagle_worker_v2.py:654, 845) | requests `return_hidden_states_before_norm=True` | consumes the **pre-norm** last hidden state | variant-specific |

Net: **every multi-layer feature-tap drafter shipped on sglang avoids the final decoder layer**; the only final-state consumers are (i) the EAGLE/MTP family, whose drafter *input* is the post-norm last hidden state (DeepSeek-V3-style MTP, standard EAGLE), and (ii) the gpt-oss EAGLE3-v3 drafter, which consumes the *pre-norm* final boundary as one of three features.

---

## Q3 - Upstream reasoning: convention origin, Kimi PR, prior art on final capture

**Origin of `[val+1]` on glm5_next.** Added by PR **#36708 "[DFLASH] Support GLM-5.3-Flash hidden-state capture"** (jianc99, merged 2026-08-27 into the #36507 support branch; reached main via the #36507 squash merge, i.e. v0.5.20): "implement the standard `set_dflash_layers_to_capture` model hook... keep the existing EAGLE3 capture behavior unchanged... add unit coverage for mHC contraction, EAGLE3 isolation, and target-layer mapping". So the DFLASH setter on glm5_next deliberately inherited the pre-existing EAGLE3 `[val+1]` convention rather than choosing output-indexed semantics. A follow-up, **#36755 "Fix DFLASH aux hidden-state capture on mHC models"**, fixed `hidden_states + residual` crashing with `residual=None` on mHC (the residual is folded into the widened stream), and registered the capture unit test with CI. The convention itself is older: the llama.py comment ("we plus 1 here...") and the qwen3.py comment ("SGLang captures 'before layer i'") predate the 6.7k-commit window I could blame locally; the default `[2, N//2, N-3]` mirrors official EAGLE-3's own taps (Q4).

**The Kimi K3 DSpark capture.** Added with Kimi-K3 model support in PR **#32541 "[Kimi] Support kimi-k3"** (git `--diff-filter=A` on kimi_k3.py), later touched by #40794 ("[Spec] Support DFLASH for Kimi K3") and #38991 (the DCP+DSpark eager-mode fix the compat report cites — a `extend_prefix_lens is None` crash, not a capture-semantics change). The design comment is the load-bearing quote (kimi_k3.py:3003-3006): DFLASH ids "name layer outputs, which is what the DSPARK taps already capture here, so reuse them without the usual +1 shift." I.e. sglang now carries **two live conventions**: DFlash targets shift ids; DSpark-ready targets (kimi_k3, deepseek_v4, kimi_linear, bailing_moe_v3) use ids directly with output-time capture.

**Prior art on final-layer/final-boundary capture — it has happened before, with the same crash.** PR **#32334 "[Speculative Decoding] Fix GPT-OSS EAGLE3 hidden states"** (fixes #32226, by Codex):

> "`nvidia/gpt-oss-120b-Eagle3-v3` requests target hidden-state boundaries `[24, 30, 36]`. SGLang shifted these to `[25, 31, 37]` and did not capture the final boundary after all 36 GPT-OSS blocks, so only two 2,880-wide states were returned. The EAGLE3 draft expected three states and crashed while combining tensors of width 8,640 and 5,760."

Identical failure class to ours: the last requested id is the final boundary, the `[val+1]` shift makes it unreachable, only K-1 features arrive, the drafter's projection crashes on shape. The fix: "Interpret GPT-OSS EAGLE3 layer IDs as hidden-state boundary indices. **Capture a requested final GPT-OSS boundary before the target model's final normalization.**" Note carefully: sglang's established final-boundary capture is **pre-norm**, and the gpt-oss drafter was trained (or at least served) against that. PR #32334 also had to honor the draft's `norm_before_fc: true` (normalize concatenated aux states before the FC), showing draft checkpoints differ in where they expect normalization.

**Is the exclusion of the final layer deliberate or a boundary bug?** Both, in different places:

- On glm5_next dflash it is an **accidental structural hole**: #36708 wanted "output of layer k" semantics (its own test names it "target-layer mapping"), and the `[val+1]` machinery simply cannot express "output of the last layer" because there is no layer 45. Nobody discussed the final layer in #36708; the shipped DFlash2 checkpoint sidesteps it.
- On DeepSpec upstream it is **deliberate and documented**: `deepspec/eval/base_evaluator.py:100-112` (`assert_no_final_target_layer`, called from both the dspark and eagle3 evaluators):

> "target_layer_ids must not include the final target decoder layer {last_layer_id}. Transformers output_hidden_states stores the final normalized hidden state at that position, while target cache generation stores raw decoder-layer outputs. Use an earlier layer and regenerate the target cache and draft checkpoint."

That is a **train/serve consistency guard, not a quality claim** (see Q4). The HF convention it references is real: HF transformers appends hidden states before each layer and appends the **post-final-norm** state as the last entry, so the raw output of the final layer never exists in the tuple (verified in `transformers` v4.46 `modeling_llama.py:924-968`; the current main refactored the API but the convention stands).

**Other upstream breadcrumbs.** deepseek_v2's eagle3 setter carries a live TODO about convention adjustment across draft models ("check if other draft models need similar layer id adjustment", deepseek_v2.py:3405-3406); the open PR #38642 (Qwen4-Exp capture, following the #36708 pattern) documents that glm5_next "gates contraction on a `dflash_capture` flag to preserve its legacy EAGLE3 behavior" — i.e. glm5_next's EAGLE3 capture stays un-contracted by design. No PR or issue discusses capturing *post*-final-norm states for a multi-layer drafter; the only norm-relative discussions are #32334's pre-norm boundary and the multi-layer-eagle `return_hidden_states_before_norm=True` choice.

---

## Q4 - Is the final layer good or bad for a drafter? (literature)

### What each method actually consumes

- **EAGLE-1** (arXiv:2401.15077, ICML 2024): draft input = target token embedding + the target's **top-layer feature**, which the EAGLE-3 paper describes as "the features of the target model (the features before the LM head)" / "the features immediately before the LM head" (arXiv:2503.01840v3, Sections 1 and 2.2). The LM head's input is the **post-final-norm** state; sglang's standard EAGLE/MTP worker indeed passes `return_hidden_states_before_norm=False` (eagle_worker_v2.py:977). So EAGLE-1/MTP-style drafters are trained on the post-norm final state as their primary conditioning signal.
- **Medusa** (arXiv:2401.10774): extra decoding heads on the backbone's final hidden state — the same post-norm feature that feeds the LM head (EAGLE-3 paper §1 groups EAGLE and Medusa as "reuse the top-layer features... immediately before the LM head").
- **Hydra** (arXiv:2402.05109): sequentially-dependent draft heads for Medusa — still consumes the target's final hidden state per position, not a multi-layer spread.
- **EAGLE-3** (arXiv:2503.01840): abandons feature prediction; fuses **low/mid/high** features — "We record the low, middle, and high-level feature sequences from the target model's forward pass" (§3.1), concatenated 3k → FC → k. The official implementation taps **outputs of layers [1, N//2 - 1, N-4]** (SafeAILab/EAGLE `eagle/model/modeling_llama_kv.py`, forward loop: `if idx==len(self.layers)-3 or idx==len(self.layers)//2 or idx==2: all_hidden_states += (hidden_states,)` — appended before each named layer runs, so these are inputs to layers 2, N//2, N-3 = outputs of 1, N//2-1, N-4) — and the post-norm append at the end is gated off (and a duplicate is explicitly commented out with "# !!!"), so **EAGLE-3 does not use the post-norm state at all**. The draft fc is `hidden_size * 3` (cnets.py:532-534). Ablation (paper Table 2, LLaMA-3.1-8B/MT-bench): EAGLE-2 3.16x/τ4.05 → +remove-feature-constraint 3.82x/τ5.37 → **+fused features 4.40x/τ6.13** — multi-layer fusion buys ~+0.58x speedup and +0.76 τ over the single-feature variant, and is what enables the data-scaling law (Fig. 1).
- **DFlash** (arXiv:2602.06036, the method DSpark builds on): context features = "hidden states from a set of target layers {l1,...,lm} concatenated and projected" (Eq. 2 area, quoted in the DSpark paper §2; DFlash paper §3). Implementation policy is explicit (DFlash §5, "Implementation"): "**The target hidden features are extracted from 5 layers uniformly selected between the second layer and the third-to-last layer of the target model.**" Ablations: more target hidden features help (Table 7: 5-H beats 3-H on all three tasks), and 5 draft layers is the speedup optimum (Table 6). The paper's rationale for hidden-state taps: they "capture long-range dependencies and task-specific semantics, and — crucially — implicitly encode information about future token predictions" (§3, citing Samragh et al. 2025).
- **DSpark** (arXiv:2607.05147): inherits the DFlash context-feature design wholesale (§2, Eq. 2-3: `H_ctx = RMSNorm(W_c[H(l1);...;H(lm)])`, KV-injected into every draft layer). **The paper never names concrete layer ids, never ablates layer choice, and never discusses the final layer.** The layer-set dimension is simply not explored in the paper.
- **DeepSeek-V4.1-Flash** (arXiv:2609.19969): "We omit the MTP module during backbone pre-training and use DSpark... We train DSpark separately after the backbone pre-training stage" (§2.1, §2.4.3); the drafter is 3 Transformer blocks consuming **encoder hidden states** (Fig. 3: DSpark fed by "Embedding ... Enc. Hidden States") with a Markov head and confidence head. No layer ids published in the paper; the architecture chapter says nothing about final-layer taps.
- **DeepSeek-V3 MTP** (referenced in both the GLM-5 report and the V4.1-Flash paper as "the MTP module in DeepSeek-V3, which is trained jointly with the backbone"): the MTP module's input is the shared embedding + the main model's output hidden state — i.e. the **post-final-norm** state — and sglang serves exactly that (Q2 table). GLM-5 itself ships an MTP with 3-way parameter sharing (GLM-5 tech report §2.1, lines ~204-215; acceptance length 2.76 vs DeepSeek-V3.2's 2.55).

### Shipped DeepSpec-family configs all avoid the final layer

From `deepseek-ai/DeepSpec` (GitHub, verified `config/dspark|dflash|eagle3/*.py`):

| Config | Target layers | target_layer_ids | max id |
|---|---|---|---|
| dspark/dflash/eagle3 qwen3-4b & qwen3-8b | 36 | [1, 9, 17, 25, 33] | N-3 |
| dspark/dflash/eagle3 qwen3-14b | 40 | [1, 10, 19, 28, 37] | N-3 |
| dspark/dflash/eagle3 gemma4-12b | 48 (HF `num_hidden_layers` verified) | [5, 17, 29, 41, 46] | N-2 |
| Kimi-K3-DSpark (served, sglang) | 93 (draft config `num_target_layers`) | [7, 23, 51, 67, 83] | N-10 |
| DFlash2 GLM-5.3-Flash (served) | 45 | [5, 14, 24, 33, 42] | N-3 |
| **our smoke drafter** | 45 | **[1, 12, 23, 34, 42, 43, 44]** | **N-1 — unique in the ecosystem** |

### Why post-final-norm might help (the case for)

1. **It is the target's current next-token intent, exactly.** The post-norm state is the LM head's input; for a full-rank head, EAGLE-3 §1 notes the top-layer feature "aligns directly with the logits of the next token." A drafter imitating the target's next-token distribution gets the cleanest possible imitation target — this is precisely why DeepSeek's production MTP consumes it as the module's input, and why EAGLE-1/Medusa were built on it.
2. **Production precedent**: DeepSeek-style MTP served through sglang's EAGLE worker consumes the post-norm last hidden state end-to-end (eagle_worker_v2.py:977/1116 → deepseek_nextn.py:194-226), and GLM-5's own shipped draft is an MTP of this family. Post-norm consumption is not exotic; it is the default for autoregressive drafters.
3. **Deeper features measurably help drafters**: DFlash Table 7 (5 taps beat 3), EAGLE-3 Table 2 (fused multi-layer beats single top-layer), and the DFlash §3 argument that hidden states "implicitly encode information about future token predictions." A 44-tap is the deepest possible tap.
4. **Our pipeline made it consistent**: the GLM port resolved the HF-vs-hook convention collision by re-normalizing the final capture in the cache path (`glm_target_hidden_states_from_hooks`, hidden_states.py:80-100 — "The last decoder layer's raw stream output is folded and re-normalized so it matches the evaluator-side `output_hidden_states` entry for the same layer (which is the post-norm `last_hidden_state`)"), so id 44 is trained consistently post-norm on both data paths. The problem upstream guards against does not exist in our training data.

### Why it might hurt (the case against)

1. **Nobody ships it.** Every multi-layer drafter shipped on sglang or released by DeepSpec avoids the final layer (table above); DeepSpec *asserts* it away; DFlash's published policy stops at the third-to-last layer; EAGLE-3's official taps stop at N-4. This is circumstantial but remarkably uniform.
2. **The uniform avoidance has a mundane explanation, though**: upstream DeepSpec's assert message shows the exclusion exists because the **two data paths disagree** at the final position (HF tuple: post-norm; hook cache: raw), not because of measured quality. DFlash2's [5,14,24,33,42] equally may reflect capture-path convenience (its capture cannot deliver 44 on glm5_next) rather than an ablated finding. **No published ablation measures final-tap quality either way.**
3. **Redundancy**: 42, 43, 44 are adjacent; deep adjacent residual streams are highly correlated, so the marginal information of the 3rd consecutive tap is likely small. DFlash Table 7's gains came from spreading taps (3-H → 5-H was spread, not clustered). The new selection paper (below) would predict 42/43/44 to score poorly on its CKA diversity criterion.
4. **Distribution mismatch risk at serve time**: mid taps are pre-norm residual sums (unbounded scale); 44 as trained is post-norm (per-token normalized scale ~1). Heterogeneous input scales are already the training regime (44 post-norm vs 1..43 raw), so the fc/hidden_norm learned around it — but only if serving reproduces the same post-norm feature. A naive port of the Kimi capture seam would deliver the final norm's *input* (pre-norm) — silently mismatching training (no crash; degraded acceptance) — the exact confusion DeepSpec's assert exists to prevent. This is the sharpest practical risk and it is about the *patch*, not the layer (see Q5a).
5. **Weak counter-anecdote**: in PR #32334's own measurements, gpt-oss Eagle3 heads that consume the final post-norm state scored lower acceptance (1.926/1.735) than the boundary-tap v3 (2.720) and long-context (2.289) heads — but these are different draft models, so it is confounded and weak.

### Newer layer-selection literature (2025-2026)

**arXiv:2610.12939 "Surfing the Activation Wave: Analyzing the Information and Geometry of LLM Hidden States for Speculative Decoding"** (Goel et al., submitted 2026-10-09; NeurIPS 2026 workshops Axiom/GDDL/PriGM): a principled framework for exactly our question — multi-layer hidden-state selection for speculative decoding scored by (i) informativeness (matrix/Shannon entropy of hidden states and of their concatenation) and (ii) diversity/redundancy (centered kernel alignment, CKA, across depth), plus token-entropy filtering. Its selected sets "outperform fixed baseline choices, including the default `[2, N//2, N-3]` rule" in EAGLE3-style decoding. Directly applicable: it gives Phillip a **measurable, offline** criterion for whether 42/43/44 are redundant (high pairwise CKA → drop two) and whether 44 adds informativeness — runnable on the already-built cache without any GPU serving test. It also validates that position-based heuristics (which is what every shipped config uses) are beatable, i.e. the field treats layer choice as an open, empirical question.

---

## Q5 - Synthesis for the decision

### (a) Is the ~10-line Kimi-pattern patch the established way, and what is its risk surface?

**The seam is established; the naive Kimi-pattern is the wrong shape for our training convention.** Three verified facts:

1. `residual_batch.final_norm(..., capture=...)` exists and is production-proven (kimi_k3.py:2865-2875; gpt_oss.py:726-734), and its accumulators are CUDA-graph-safe by design (`AuxHiddenStateList.capture` clones borrowed storage; `AuxHiddenStatePacker.for_batch` aliases the decode graph runner's preallocated buffer — `layers/aux_hidden_states.py:20-56`).
2. **But the seam delivers the norm's input, not its output**: `final_norm_pair` calls `capture(residual)` after `norm(hidden_states, residual)` returns `(normed, residual_new)` (`residual_access.py:40-54`; RMSNorm returns the pre-norm sum as `residual_new`, layernorm.py:832-870). Kimi's comment confirms the intent: the captured value is "what the next layer's attention input reads... (the final norm's, after the last layer)" — i.e. the final norm's **read**, pre-norm. gpt_oss's PR words it identically: "Capture a requested final GPT-OSS boundary **before** the target model's final normalization."
3. Our drafter was trained with id 44 = **post-final-norm** (both data paths, hidden_states.py:59-69 and 97-98). Passing `aux_hidden_states.capture` into `final_norm` Kimi-style would therefore serve a pre-norm feature where training used post-norm — a silent acceptance-length degradation, not a crash.

**The correct patch is simpler than the Kimi pattern**: after `residual_batch.final_norm(...)` returns (`glm5_next.py:1204-1206` @ `961bcf48`; 1225-1227 at main), the returned `hidden_states` *is* the post-final-norm state; if the requested max id equals `num_hidden_layers - 1`, append it via `aux_hidden_states.capture(hidden_states)` (~5 lines, plus keeping `set_dflash_layers_to_capture` from emitting the impossible `val+1` for that id). Capture order is preserved iff the final id is the **maximum** of `target_layer_ids` (it is, for the 7-union); a config that put 44 in the middle would need positional insertion.

Risk surface, itemized:

- **CUDA graphs**: the DFlash loop capture already runs inside decode graphs today; the post-norm capture sits in the same captured region after the loop, appending to the same accumulator. gpt_oss ships exactly this shape. Idle forwards skip `final_norm` (`glm5_next.py:1203`) — the guard `if not forward_batch.forward_mode.is_idle()` must wrap the capture (the DFLASH worker does not run idle forwards, but guard anyway).
- **PP**: `final_norm` runs only on the last PP rank, and the setter already early-returns on non-last ranks — consistent.
- **DeferredFinalize**: the `capture=` seam raises on a deferred MoE handoff (residual_batch.py:97-103), but glm5_next does not pass `takes_handoff=True`, so `stream.export` completes the handoff eagerly (residual_stream.py:211-232) — not reachable for this patch's placement (capture after the return, not via the seam).
- **mHC fold placement**: the last layer's streams are contracted before `self.norm` (mhc.py:225-231), so the captured post-norm value is `self.norm(folded_last_stream)` — exactly the training-side `final_norm(tensor)` computation (hidden_states.py:97-98). Numerically the same op; the one remaining question is fused-kernel rounding order (`access.final_norm_pair` deliberately keeps add+norm together, residual_access.py:43-46) — a parity spot-check against the training cache is cheap and decisive.
- **Upstreamability**: mirrors #32334's precedent (a setter that understands the final boundary) and would fix every future DeepSpec-trained glm5_next drafter; the qwen3.py comment ("To capture the hidden state after target layer k (HF-style)...") shows the intent is already half-written in the tree.

### (b) Production precedent of a drafter consuming post-final-norm states via sglang?

- **Yes, but not in the DFlash/DSPARK family**: DeepSeek-V3-style MTP (and standard EAGLE) drafters consume the target's post-final-norm last hidden state as their input — `return_hidden_states_before_norm=False` everywhere in eagle_worker_v2/tp_worker, `CaptureHiddenMode.LAST` storing the post-norm `pruned_states` (logits_processor.py:828-853), and the NextN draft consuming it after `hnorm` + `eh_proj` (deepseek_nextn.py:194-226). GLM-5's own shipped MTP draft is this family.
- **No, for multi-layer feature-tap drafters**: no shipped DSPARK/DFlash drafter consumes the post-norm state; the only drafter that taps the final boundary at all (gpt-oss EAGLE3-v3) consumes it **pre-norm**; and the shipped Kimi-K3 DSpark drafter does not tap the final layer (its ids stop 10 layers early). So "7-union with a post-norm 44" would be the first of its kind in production; "post-norm consumption" itself is production-standard.

### (c) What the evidence implies for the candidate sets

All four arms are cache-compatible (the 7-union cache already stores per-layer features for ids 1,12,23,34,42,43,44 — per the briefing/numbers context, the union exists precisely so subset arms are config-only retrains). Ranking by evidence:

1. **7-union [1,12,23,34,42,43,44] with the post-norm capture patch** — keeps the trained step_20 checkpoint and its smoke-validated config; 44 carries the target's exact next-token intent (the MTP/EAGLE-1 precedent for that feature's value); DFlash Table 7 says more taps help. Conditions: implement the patch as the **post-norm return-value capture** (not the Kimi seam); live-verify prefill + decode + CUDA-graph before the expensive cache stage; order constraint (44 must be the max id). This arm's unique risk is the unproven 42/43/44 redundancy and the un-upstreamed patch.
2. **5-stock [1,12,23,34,42]** (retrain `fc` for 5 features, zero sglang patches) — the only arm that matches every shipped precedent exactly (DeepSpec configs, DFlash2 on this very target, DFlash's published policy: 5 taps ending at N-3). Serves on stock images forever. Cost: retrain (smoke-cheap) and giving up the 44 tap. If the live test of (1) shows any capture/graph awkwardness, this is the safe harbor.
3. **5-spread [1,12,23,34,44] (patched)** — best of both shapes: DFlash-style spread + the single deepest tap; kills the 42/43 redundancy concern; same patch as (1), same cache. Natural follow-up if CKA/entropy analysis of the cached features shows 42/43 add little beyond 44 (runnable offline per arXiv:2610.12939's method).
4. **6 = [1,12,23,34,42,43] (retrain, stock)** — needs retrain like (2)/(3) but keeps adjacent 42/43 without the 44 intent feature; strictly dominated by (2) unless a measurement shows 43 (but not 44) is uniquely valuable, for which there is no evidence anywhere.

Genuinely unknown / measurable (cheap first):

- **Offline, no GPU serving**: run entropy + pairwise-CKA analysis (arXiv:2610.12939) over the cached 7-union features to quantify 42/43/44 redundancy and 44's informativeness; parity-check `self.norm(folded_stream)` against the training cache's stored id-44 feature (catches the fused-rounding question and any pre/post-norm confusion before serving).
- **Live smoke (single GPU-hour class)**: does the patched post-norm capture fire identically on prefill and under decode CUDA-graph replay (compare against the training cache on a fixed prompt)? Does the drafter's acceptance length move at all between 6-feature and 7-feature feeding (the step_20 checkpoint accepts only the 7-feature shape; a 6-feature run needs the retrained fc — or simply verify the 7-feature path first)?
- **Unknown anywhere in the literature**: no published ablation isolates the final layer's marginal value for a parallel/multi-tap drafter. If the full run happens, the 5-spread vs 7-union comparison (config-only retrains on the same cache) would be the first such data point.

---

## Bottom line for Phillip

1. **The blocker is real and structural, and it has happened before**: glm5_next's dflash capture (PR #36708, inheriting the EAGLE3 `[val+1]` convention) maps requested id 44 to a nonexistent capture point; the final norm runs after the loop and is never captured; the drafter's `fc` crashes at first prefill with the documented shape ValueError (dflash.py:771-787). sglang already fixed the identical bug class on gpt-oss (PR #32334) by treating the final id as a boundary and capturing at the final norm.
2. **But copy the intent, not the letter, of the Kimi pattern**: Kimi's and gpt_oss's final captures deliver the final norm's **input** (pre-norm). Our drafter was trained on id 44 = **post-final-norm** (both training data paths re-normalize deliberately). The right patch is ~5 lines capturing the `final_norm` **return value** — which is the LM head's input, already computed every verification step. The shared `capture=` seam is graph-safe and production-proven, so risk is low, but the seam itself would silently mismatch training if used verbatim.
3. **Every shipped multi-layer drafter avoids the final layer — but for consistency reasons, not measured quality reasons.** DeepSpec upstream hard-asserts against it because its two data paths disagree at that position (HF gives post-norm, hooks give raw); our port resolved that collision by standardizing on post-norm, so the upstream objection does not apply to our checkpoint. No paper (DSpark, DFlash, EAGLE-3) ablates layer choice; the uniform avoidance is convention, not evidence. Post-norm final-state consumption *is* production-standard in the EAGLE/MTP family (DeepSeek MTP, GLM-5's own MTP).
4. **Recommendation shape**: proceed with the 7-union + post-norm patch as primary (keeps the trained checkpoint, fixes the class of problem for all future glm5_next drafters, upstreamable per #32334 precedent), with **5-stock [1,12,23,34,42]** as the zero-patch fallback and **5-spread [1,12,23,34,44]** as the redundancy-optimized variant — all three are config-only retrains on the already-planned 7-union cache. Before the expensive cache build: run the offline entropy/CKA analysis on cached features (2610.12939 gives the recipe) and the live parity/acceptance smoke of the patched capture.
5. **The one thing not to do**: implement the Kimi seam verbatim and serve. It would feed a pre-norm feature where training used post-norm — no crash, quietly degraded acceptance, and precisely the mismatch class DeepSpec's `assert_no_final_target_layer` exists to prevent.

---

## Confidence & gaps

- **High confidence**: capture mechanics and the structural impossibility of id 44 on stock glm5_next (read at both `961bcf48` and main); the crash site; the Kimi/gpt_oss final-capture semantics being pre-norm (traced through `final_norm_pair` → `RMSNorm` return convention); DeepSpec's `assert_no_final_target_layer` and released configs; DFlash's published layer policy; EAGLE-3's official taps and fusion ablation; the shipped Kimi-K3-DSpark and DFlash2 GLM configs; PR numbers and titles (#36708, #36755, #32334, #32541, #38991, #42489, #42624, #30261).
- **Medium confidence**: that upstream DeepSpec's cache-vs-evaluator disagreement is the *sole* reason for the final-layer exclusion (their assert message says so; no design doc found saying more); the EAGLE-1 "post-norm" reading (based on EAGLE-3's description "features immediately before the LM head" plus sglang's `return_hidden_states_before_norm=False` behavior — I did not run EAGLE-1's original code).
- **Unverified / open**: whether the gpt-oss EAGLE3-v3 drafter was *trained* on the pre-norm boundary or the convention was chosen at serve time (the PR only proves serving works and acceptance is good); exact DSpark-for-DSV4 drafter layer ids (bundled in the target config, not fetched); whether glm5_next's fused mHC add+norm rounding makes the served post-norm feature bit-identical to the training cache (spot-check needed); pre-#32334 git blame of the llama.py comment (predates my local history window; the comment text and convention are verified regardless).
- **Not obtainable in this session**: a live GPU reproduction of the crash (static analysis only, per the compat report's own unknowns list).

## Sources

Repo (read-only, masked reads):

- `experiments/glm-drafter/deepspec_glm/deepspec/modeling/dspark/glm5/hidden_states.py` (worktree `23-glm-drafter-run`) — training capture convention
- `experiments/glm-drafter/notes/dspark-compat-report.md` — the compatibility map being researched
- `papers/DSpark-2607.05147.txt`, `papers/DFlash-2602.06036.txt`, `papers/DeepSeek-V4.1-Flash-2609.19969.txt`, `papers/GLM-5-technical-report-2602.15763.txt`

sglang sources: per-file raw fetches at `961bcf48` (saved to ephemeral scratch, since deleted) plus a shallow clone of main `166b4ae`; file references below carry line numbers per SHA.

- `python/sglang/srt/models/glm5_next.py` (1040, 1099-1102, 1158-1187, 1199-1210, 1514-1538 @961bcf48; 1225-1227, 1547-1559 @main)
- `python/sglang/srt/models/kimi_k3.py` (2674, 2809-2847, 2865-2919, 2989-3006 @961bcf48)
- `python/sglang/srt/models/gpt_oss.py` @main (707-734, 1336-1363)
- `python/sglang/srt/models/dflash.py` (619-626, 700-716, 771-787)
- `python/sglang/srt/models/dspark.py` (483, 729-743, 759)
- `python/sglang/srt/models/deepseek_v2.py` @main (3401-3427), `deepseek_v4.py` @main (4508-4516, 4773-4783), `deepseek_nextn.py` @main (194-226)
- `python/sglang/srt/models/{qwen3.py:740-750, llama.py:922-945, qwen3_moe.py:1141-1166, qwen3_next.py:1225-1256}`
- `python/sglang/srt/layers/layer_boundary/residual/{batch.py:59-114, access.py:40-54, stream.py:188-232, mhc.py:200-231}`
- `python/sglang/srt/layers/{layernorm.py:821-870, aux_hidden_states.py:20-56, logits_processor.py:828-853}`
- `python/sglang/kernels/ops/layernorm/mhc.py:1919-1921`
- `python/sglang/srt/speculative/{eagle_worker_v2.py:905-985, 1090-1120; multi_layer_eagle_worker_v2.py:640-670}`
- `python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:40-66`

GitHub (sgl-project/sglang PRs, fetched via GitHub MCP search): #36708, #36755, #36507, #32334 (fixes #32226), #32541, #30261, #38991, #42489, #42624, #38642, #32281, #40794.

DeepSpec (github.com/deepseek-ai/DeepSpec @ main `005e03b8`): `deepspec/eval/base_evaluator.py:100-112`, `deepspec/eval/{dspark,eagle3}/evaluator.py`, `scripts/data/prepare_target_cache.py:93-133`, `config/{dspark,dflash,eagle3}/*.py`.

Hugging Face configs: `RadixArk/Kimi-K3-DSpark` (config.json), `incoai/GLM-5.3-Flash-DFlash2` (config.json), `google/gemma-4-12b-it` (num_hidden_layers=48).

Literature: arXiv:2503.01840 (EAGLE-3, incl. Table 2 ablation and §3.1 feature description; official code SafeAILab/EAGLE `eagle/model/modeling_llama_kv.py` + `cnets.py`), arXiv:2401.10774 (Medusa), arXiv:2402.05109 (Hydra, cited in EAGLE-3), arXiv:2607.05147 (DSpark, Eq. 2-3, local sidecar), arXiv:2602.06036 (DFlash §5 + Tables 6-7, local sidecar), arXiv:2609.19969 (DeepSeek-V4.1-Flash §2.1/§2.4.3, local sidecar), arXiv:2602.15763 (GLM-5 tech report MTP, local sidecar), arXiv:2610.12939 (Surfing the Activation Wave, 2026-10-09), HF transformers v4.46 `modeling_llama.py` (output_hidden_states convention).
