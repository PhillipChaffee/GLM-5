# DSpark drafter (GLM-5.3-Flash) ↔ sglang DSPARK loader compatibility map

Scope: our trained `Glm5DsparkModel` drafter checkpoint (transformers `save_pretrained` output on Modal: `step_N/model.safetensors + config.json + train_config.py`) vs sglang's builtin `--speculative-algorithm DSPARK` path, with GLM-5.3-Flash (`glm5_next`, 320B FP8 hybrid: 11 DSA full-attention + 34 KDA layers) as the target.

Evidence base: local sources at `experiments/glm-drafter/deepspec_glm/deepspec/modeling/dspark/glm5/{modeling.py,config.py,hidden_states.py}`; upstream DeepSpec at `deepseek-ai/DeepSpec` (`deepspec/modeling/dspark/{common.py,markov_head.py}`); sglang @ main `1814ece57fc283253e4e65980d4607d2207958c3` (fetched 2026-10-10; the DSPARK parser was previously pinned at `961bcf48` — unchanged in the relevant parts). All sglang paths below are relative to `python/sglang/srt/` unless noted. GitHub MCP was down; everything was fetched via raw.githubusercontent/`gh api`.

---

## A. Config contract

### A.1 What sglang actually reads

Two parsers run in sequence for the **draft** checkpoint's HF config:

1. `speculative/dflash_utils.py` → `parse_dflash_draft_config` (base parser). Reads, from a nested `dflash_config` dict **or** top-level keys (`_cfg_get` accepts dict-or-attribute; text config is consulted via `_get_text_config`):
   - `num_hidden_layers` (from the draft text config; `min_value=1`)
   - `num_target_layers` (top-level or in `dflash_config`; `min_value=1`)
   - `block_size` ("Keep support for current checkpoints where block_size is top-level")
   - `target_layer_ids` (must be a non-empty list of ints; range-checked against the target layer count in `DFlashDraftConfig.resolve_target_layer_ids`)
   - `mask_token` (string, defaults `"<|MASK|>"`) and `mask_token_id` (int ≥ 0)
   - optional family extras we do not use: `conv_kernel_size`/`conv_group_size`, `selector_rank`/`selector_top_k`, `output_multiplier`, `final_logit_softcapping`, `projector_type` (domino), `attention_sink_bias`, `attention_value_scale`
2. `speculative/dspark_components/dspark_config.py` → `parse_dspark_draft_config` (wraps #1). Resolution order per key: `dspark_*`-prefixed top-level keys → nested `dspark_config` dict → plain keys on text/top-level config:
   - `dspark_block_size` | `dspark_config.block_size` | `block_size` → **gamma**
   - `dspark_markov_rank` | `markov_rank` (default 0; must be > 0 — "DSpark requires markov_rank > 0")
   - `dspark_markov_head_type` | `markov_head_type` — must be in `("vanilla", "gated", "rnn")`, required when `markov_rank > 0`
   - `dspark_noise_token_id` | `dspark_config.mask_token_id` | `mask_token_id`
   - `dspark_target_layer_ids` | base parser's `target_layer_ids`
   - `get_dspark_sample_from_anchor(config)`: `sample_from_anchor`, **default True**
   - `checkpoint_bundles_dspark_draft`: a TARGET config carrying `dspark_block_size` / `dspark_markov_rank` / `dspark_noise_token_id` / `dspark_target_layer_ids` marks a bundled draft head (not our layout; we serve a standalone draft checkpoint)

Runtime validation in `resolve_runtime_config` (`dspark_config.py:84-134`):

- gamma: from config `block_size`, or `--speculative-num-draft-tokens` (gamma = num_draft_tokens − 1) with a warning on mismatch
- **`mask_token_id` is required** ("DSpark requires mask_token_id to be set in the draft model config") and **must be `< target vocab size`** (checked against the target's embedding row count, padded vocab included)

Worker consumption (`dspark_worker_v2.py:224-234`): `gamma`, `verify_num_draft_tokens = gamma + 1`, `sample_from_anchor`, `mask_token_id`. With `sample_from_anchor=True`, query rows = gamma (row 0 = the committed/bonus token embedded by the **target** embedding, rows 1..gamma−1 = `mask_token_id`); verify window = gamma+1.

Relevant CLI flags (`arg_groups/fields/spec.py`): `--speculative-algorithm` (DSPARK is builtin, line 33), `--speculative-draft-model-path` (36-42), `--speculative-num-draft-tokens` (66-69), `--speculative-dspark-block-size` (78-81: "verify window is gamma + 1 … Omit to auto-infer gamma from the draft checkpoint block_size"), `--speculative-dspark-confidence-sts-path` (90-97, optional STS calibration; losslessness unaffected), `--speculative-dspark-sps-table-path` (82-89), `--speculative-draft-attention-backend` (137-143), `--speculative-draft-kv-cache-dtype` (151-163), `--speculative-use-block-verification` (120-124).

### A.2 Our config → sglang expectation (diff table)

Our `config.json` is what `build_draft_config` produced (`deepspec_glm/.../dspark/glm5/config.py`): a **deepcopy of the GLM-5.3-Flash text config** plus drafter overrides — so it carries extra GLM fields (`mhc`, `hc_mult`, `layer_types`, `linear_attn_config`, MoE fields, `num_nextn_predict_layers`, …). Values the builder set explicitly: `architectures ["Glm5DsparkModel"]`, `model_type "glm5_dspark"`, `num_target_layers 45`, `num_hidden_layers 5`, `block_size 7`, `tie_word_embeddings false`, `layer_types ["full_attention"]×5`, `mlp_layer_types ["dense"]×5`, `mask_token_id 154821`, `target_layer_ids [1,12,23,34,42,43,44]`, `num_anchors 512`, `enable_confidence_head true`, `confidence_head_with_markov true`, `markov_rank 256`, `markov_head_type "vanilla"`.

| our key (value) | sglang reader | verdict / action |
|---|---|---|
| `model_type: "glm5_dspark"` | transformers `AutoConfig` on the draft path | **EDIT.** Unknown to transformers; `get_config` will fail without `trust_remote_code` + shipped code. The shipped precedent — `incoai/GLM-5.3-Flash-DFlash2/config.json` — uses `model_type: "qwen3"` with sglang keys riding along as extra attrs. Change to `"qwen3"` (or ship remote-code `config.py`). Note `configs/hybrid_arch.py:126-133` excludes draft models from the glm5_next KDA registry, so a qwen3-typed draft won't be misread as hybrid. |
| `architectures: ["Glm5DsparkModel"]` | `models/registry.py:118-150` maps `config.architectures[0]` → `EntryClass.__name__` | **EDIT (with new class).** Keep the string, but register a class with exactly this name (see B). Today's DSPARK EntryClasses are `Qwen3DSparkModel`, `LingDSparkModel`, `DSparkDraftModel`, `Lfm2DSparkDraftModel`, `DFlash2DraftModel` — none matches. |
| `block_size: 7` | → gamma = 7; verify window 8 | OK as-is. `--speculative-dspark-block-size 7` / `--speculative-num-draft-tokens 8` optional (warning if inconsistent). |
| `num_hidden_layers: 5` | draft depth (also `DSparkDraftMixin.num_stages`, which neutralizes the inherited `num_nextn_predict_layers=1` MTP heuristic — `models/dspark.py` comment) | OK. |
| `num_target_layers: 45` | base parser | OK. |
| `target_layer_ids: [1,12,23,34,42,43,44]` | parsed + range-checked | Parses fine; **7 features expected by `fc`. But glm5_next capture only delivers 6 — id 44 never fires (see C).** Blocker, needs sglang patch or id change. |
| `mask_token_id: 154821` | required; `< 154880` target vocab | OK (passes both checks). Semantics: rows 1..6 of each draft block embed this id through the **target** embedding — matches training's `create_noise_embed` (`common.py:264-294`). |
| `markov_rank: 256` | must be > 0 | OK. |
| `markov_head_type: "vanilla"` | ∈ {vanilla, gated, rnn} | OK. |
| `enable_confidence_head: true`, `confidence_head_with_markov: true` | `build_confidence_head` (`models/dspark.py`): `with_markov` defaults to `markov_rank > 0`; input dim = hidden + rank | OK. Checkpoint **must** carry `confidence_head.*` weights or load errors out (explicit ValueError). Note the flag name is `enable_confidence_head`, not `confidence_head_alpha` — alpha is a train-time arg only (`enable = alpha > 0`); it never lands in config.json. |
| `sample_from_anchor` (absent) | defaults **True** | OK — matches training (block row 0 = anchor/committed token). Set it explicitly to `true` for legibility. |
| `num_anchors: 512` | **not read** by sglang (serving runs one block per step; no anchor sampling) | Harmless leftover. |
| `layer_types: ["full_attention"]×5` | `get_dflash_layer_types` → each layer `(window=-1, ENCODER_ONLY)` since `is_causal` is absent (`models/dflash.py:155-189`) | OK — ENCODER_ONLY = bidirectional, matching training's mask. **Keep `is_causal` absent** (GLM HF text config has none). |
| `head_dim` — inherited from the GLM text config = **0** | `DFlashAttention`: `head_dim = int(getattr(config, "head_dim", hidden//heads))` — `0` is not None → **head_dim 0** → crash | **EDIT: set `head_dim: 64`.** (Training's `Glm5DsparkAttention` fell back to `hidden//heads = 64` because `0` is falsy — `modeling.py:90-92`; sglang has no such fallback.) |
| `intermediate_size` — inherited from GLM text config = 12288 | `DFlashMLP` builds from it | **VERIFY** against the trained MLP weight shapes (builder deepcopied the target value; the class default was 10240 — only the checkpoint's `gate_proj.shape[0]` is ground truth). Whatever it is, config must match the tensors. |
| `vocab_size 154880`, `hidden_size 4096`, `num_attention_heads 64`, `num_key_value_heads 64`, `rms_norm_eps 1e-5`, `hidden_act "silu"`, `attention_bias false` | backbone dims | OK (64/64 heads → dense MHA, TP-divisible). |
| `rope_theta` / `rope_parameters` (inherited; transformers fills a default when absent) | `DFlashAttention` builds neox RoPE **unconditionally** (`models/dflash.py:244-259`) | Landmine only if the stock attention class is reused (it applies RoPE the drafter never saw — we trained NoPE). Irrelevant once the NoPE subclass (B) is used. |
| `tie_word_embeddings: false` | — | OK. |
| `mhc`, `hc_mult`, `linear_attn_config`, MoE fields, `num_nextn_predict_layers` (inherited) | not read on the draft side | Harmless except `num_nextn_predict_layers` (neutralized via `num_stages`). |
| *(missing)* `use_cache`, `dtype` | — | Recommended hygiene (DFlash2 sets both): `use_cache: false`, `dtype: "bfloat16"`. |

### A.3 Exact config.json edits for our checkpoint

```jsonc
{
  "model_type": "qwen3",                       // was "glm5_dspark" (DFlash2 precedent)
  "architectures": ["Glm5DsparkModel"],        // keep; requires the new sglang class (B)
  "head_dim": 64,                              // was 0 (inherited) — hard crash without this
  "intermediate_size": 12288,                  // VERIFY vs trained gate_proj shape[0]
  "use_cache": false,
  "dtype": "bfloat16",
  "sample_from_anchor": true                   // explicit; default already true
  // everything else unchanged: block_size 7, num_hidden_layers 5, num_target_layers 45,
  // target_layer_ids [1,12,23,34,42,43,44], mask_token_id 154821, markov_rank 256,
  // markov_head_type "vanilla", enable_confidence_head true, confidence_head_with_markov true,
  // layer_types, mlp_layer_types, tie_word_embeddings false, vocab 154880, hidden 4096,
  // num_attention_heads 64, num_key_value_heads 64, rms_norm_eps 1e-5
}
```

---

## B. Weight-name contract

### B.1 sglang `models/dspark.py` expected names vs our parameter names

The serving class is `DSparkDraftModel(DSparkDraftMixin, DFlashDraftModel)` — the backbone is `DFlashDraftModel` (`models/dflash.py`). Its docstring states the checkpoint contract: "transformer weights for `layers.*`, `fc.weight`, `hidden_norm.weight` for projecting target context features, `norm.weight` for final normalization" (`dflash.py:619-626`). Embedding/lm_head are **shared from the target** (`DSparkDraftMixin.attach_shared_modules`); the loader skips `lm_head.`, `rotary_emb.`, and (non-Nemotron) `embed_tokens.` weights (`_DSPARK_SKIPPED_WEIGHT_PREFIXES` + skip in `DSparkDraftMixin.load_weights`).

| our checkpoint param (`Glm5DsparkModel`, `save_pretrained`) | sglang param | resolution |
|---|---|---|
| `embed_tokens.weight` | *(skipped)* target's `embed_tokens` attached | ✅ by design (we trained it as a frozen copy — matches `initialize_embeddings_and_head(freeze=True)`) |
| `lm_head.weight` | *(skipped)* target's `lm_head` attached; logits via `compute_base_logits` | ✅ |
| `layers.N.self_attn.q_proj.weight` | `layers.N.self_attn.qkv_proj.weight` (shard `q`) | ✅ auto-remapped — `dflash.py:828-835` `stacked_params_mapping = [("qkv_proj","q_proj","q"), ("qkv_proj","k_proj","k"), ("qkv_proj","v_proj","v"), ("gate_up_proj","gate_proj",0), ("gate_up_proj","up_proj",1)]` |
| `layers.N.self_attn.k_proj.weight` | `qkv_proj` shard `k` | ✅ auto |
| `layers.N.self_attn.v_proj.weight` | `qkv_proj` shard `v` | ✅ auto |
| `layers.N.self_attn.o_proj.weight` | `o_proj` | ✅ direct |
| `layers.N.self_attn.q_norm.weight`, `k_norm.weight` | `q_norm.weight`, `k_norm.weight` (head-dim `RMSNorm`) | ✅ direct (ours is `Glm5NextRMSNorm(head_dim)` = weight-only RMSNorm, same as sglang's) |
| `layers.N.mlp.gate_proj.weight` | `mlp.gate_up_proj` shard 0 | ✅ auto |
| `layers.N.mlp.up_proj.weight` | `mlp.gate_up_proj` shard 1 | ✅ auto |
| `layers.N.mlp.down_proj.weight` | `mlp.down_proj` | ✅ direct |
| `layers.N.input_layernorm.weight`, `post_attention_layernorm.weight` | same names | ✅ direct |
| `norm.weight` | `norm.weight` | ✅ direct |
| `fc.weight` | `fc.weight` (shape-checked: `dflash.py:889-915` — `num_context_features(=7) × 4096 → 4096`) | ✅ direct, same name |
| `hidden_norm.weight` | `hidden_norm.weight` | ✅ direct |
| `markov_head.markov_w1.weight` | `VanillaMarkov.markov_w1` = `nn.Embedding(vocab, rank)` (`models/dspark.py`; identical to DeepSpec `markov_head.py:17`) | ✅ exact match |
| `markov_head.markov_w2.weight` | `VanillaMarkov.markov_w2` = `nn.Linear(rank, vocab, bias=False)` (`markov_head.py:18` ↔ `dspark.py`) | ✅ exact match |
| `confidence_head.proj.weight` / `.bias` | `DSparkConfidenceHead.proj` = `nn.Linear(hidden(+rank), 1, bias=True, fp32)`; DeepSpec `AcceptRatePredictor.proj = nn.Linear(input_dim, 1)` (`common.py:43-49`) | ✅ exact match, incl. bias=True; `with_markov=True` gives input dim 4096+256 on both sides |

Name resolution is prefix-robust: `resolve_param_name` (`dflash.py:846-860`) tries the raw name, `"model."+name`, and stripped-`model.` forms; HF keys without a `model.` prefix (ours, since `Glm5DsparkModel` owns the submodules directly) resolve directly.

**Zero weight remaps are needed.** The loader fuses q/k/v → qkv and gate/up → gate_up by itself.

### B.2 The real mismatch: attention FORM

- **Ours (training):** NoPE — `_forward_backbone` deletes `position_ids` entirely (`modeling.py:371-383`); attention is bidirectional over context **strictly before the anchor** plus bidirectional within the noise block, via a flex BlockMask (`modeling.py:146-165`; DeepSpec `common.py:86-96`: `mask_context = kv_idx < seq_len & kv_idx < anchor_pos`, `mask_draft = same block`). No positional encoding anywhere.
- **sglang stock:** `DFlashAttention.__init__` **always** builds `rotary_emb = get_rope(...)` (`dflash.py:244-259`); forward applies neox RoPE (fused table kernel or per-tensor, `dflash.py:354-372`); the ctx-KV injection path applies `k_norm` **and RoPE** to the target-hidden-derived context keys (`write_target_hidden_kv` → `apply_k_rope`, `models/dspark.py`). Layer type defaults to ENCODER_ONLY (bidirectional) for `full_attention` layers when `is_causal` is absent — that part matches us — but the RoPE application is a functional divergence from training.

### B.3 Verdict

**Do not reuse `Qwen3DSparkModel`/`DSparkDraftModel` unchanged; do write a new thin per-arch class** — exactly the `lfm2_dspark.py` pattern extended with an attention override (`models/lfm2_dspark.py` is 9 lines: `class Lfm2DSparkDraftModel(DSparkDraftModel): pass`; the extensible seams already exist: `DFlashDraftModel.decoder_layer_cls` and `DFlashDecoderLayer.attention_cls` class attributes, used the same way by `DFlashLagunaDecoderLayer`/`DFlashLagunaAttention`).

New file `python/sglang/srt/models/glm5_dspark.py`, sketch:

```python
from sglang.srt.models.dflash import DFlashAttention, DFlashDecoderLayer
from sglang.srt.models.dspark import DSparkDraftModel


class Glm5DsparkAttention(DFlashAttention):
    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__(config, layer_id, quant_config, prefix)
        self.rotary_emb = None  # NoPE: keep the attribute so the fused-KV
        # bundle builder exits gracefully (returns None
        # -> per-layer ctx path), and forward never ropes

    def forward(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)  # no rope
        attn_output = self.attn(q, k, v, forward_batch)  # ENCODER_ONLY == bidirectional
        return self.o_proj(attn_output)[0] if isinstance(...) else self.o_proj(attn_output)

    def apply_k_rope(self, positions, k):
        return k  # ctx-injection path: identity instead of rope


class Glm5DsparkDecoderLayer(DFlashDecoderLayer):
    attention_cls = Glm5DsparkAttention


class Glm5DsparkModel(DSparkDraftModel):
    decoder_layer_cls = Glm5DsparkDecoderLayer


EntryClass = [Glm5DsparkModel]
```

Notes on the seams this relies on:

- `DSparkDraftMixin.__init__` → `DFlashDraftModel.__init__` builds `self.layers` from `decoder_layer_cls` (`dflash.py:669-685`), so the class attribute is enough.
- Fused ctx-KV write: `_build_fused_kv_write_bundle` reads `attn0.rotary_emb` and returns `None` unless it is a full `RotaryEmbedding` (`models/dspark.py`) → with `rotary_emb = None` it falls back to the per-layer path (`kv_proj_only` → `apply_k_norm` → `apply_k_rope` → `set_kv_buffer`), which our `apply_k_rope` override makes rope-free. Keep `SGLANG_DSPARK_STACKED_CTX_KV` off (default) — `_project_ctx_kv_stacked` would touch `rotary_emb`; a NoPE stacked-KV variant is an easy later optimization (stacking needs no rope at all).
- Sampling: vanilla head is fully supported, including the fused greedy kernel (`VanillaMarkov.sample_block_greedy_fused` via `MarkovGreedyStep`) and the folded-proposal graph path (`SGLANG_DSPARK_FOLDED_PROPOSAL`).
- Everything else (markov head, confidence head, fc/hidden_norm/norm, TP sharding of qkv/gate_up) flows through the base classes untouched.

Registration: drop the file into `sglang.srt.models` (it is auto-discovered via `EntryClass`, `models/registry.py:118-150`) — either patch the image or use `SGLANG_EXTERNAL_MODEL_PACKAGE`.

---

## C. Target side: glm5_next + DSPARK

### C.1 What already exists (shipped)

- **Target support:** `models/glm5_next.py` (`EntryClass = [Glm5NextForConditionalGeneration]`, line 1961), `configs/glm5_next.py` (45 layers; `layer_types` with 11 `deepseek_sparse_attention` + 34 `linear_attention` KDA; `mhc=true, hc_mult=4`; `qk_rope_head_dim=0`), `models/glm5_next_nextn.py` (native MTP). Cookbook (`docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx`): GLM-5.3-Flash supported since **v0.5.20**; MTP served via EAGLE 5/1/6; **DFlash2** already ships with a trained block-draft on this exact target (`incoai/GLM-5.3-Flash-DFlash2`), using the hidden-state capture from PR #36708 — i.e., **"glm5_next + dflash-family draft worker" is a supported, exercised combination**.
- **DSPARK worker:** `SpeculativeAlgorithm.DSPARK` → `DSparkWorkerV2` (`speculative/spec_info.py:323-328`), sharing the DFLASH-family verify machinery (`is_dflash_family`); DSPARK adds ragged verify, the confidence head + STS + SPS-table scheduling, mask-token noise blocks, and target-hidden KV injection.
- **Target hidden-state capture wiring:** `model_executor/model_runner_components/attention_backend_setup.py:40-66` — under DFLASH/DSPARK the runner calls `set_dspark_layers_to_capture(ids)` if the target has it, else `set_dflash_layers_to_capture(ids)`, else **raises**. glm5_next has the latter (`models/glm5_next.py:1526-1538`), which maps `layers_to_capture = [val + 1 for val in layer_ids]` (capture before layer k = output of layer k−1 = our after-layer convention) and sets `dflash_capture=True`, which triggers the mHC contract `hc_contract(aux, hc_mult)` (`models/glm5_next.py:1099-1102`). That contract is mean-fold over the `hc_mult` streams — **bit-identical to our training fold** (`deepspec_glm/.../glm5/hidden_states.py:16-23` `[B,S,N,D].mean(dim=2)`; regression-tested in `test/registered/unit/layer_boundary/test_glm5_next_dflash_capture.py:31-38`).
- **KDA verify path:** `arg_groups/glm5_next_hook.py:18-36` — with any speculative algorithm on GLM-5-Next (SM100), `--linear-attn-verify-backend` is auto-pinned to `nv_cutedsl` ("the fused Kimi-K3/DSPARK CuTeDSL kernel" per `arg_groups/kimi_k3_hook.py`). The Kimi hook additionally defaults the **draft** backend to `trtllm_mha` for DSPARK on SM100 (dense MQA draft); glm5_next's hook does not, so `--speculative-draft-attention-backend` falls back to the process prefill backend (`speculative/draft_worker_common.py:31-62`) — set it explicitly.
- **KDA/mamba state commit after verify:** the worker detects the target as KDA-hybrid (`mambaish_config` → `hybrid_kda_config` → `glm5_next_config`, `configs/hybrid_arch.py:126-137`) and calls `update_mamba_state_after_mtp_verify` with chain-layout step indices (`dspark_worker_v2.py:474-479, 1095-1164`). This is the piece that makes spec-decode safe on recurrent KDA layers.
- Both hooks are dispatched by model type from `arg_groups/model_hook.py:246-259`.

### C.2 The one genuine blocker: the 7th capture point (layer 44)

Our `fc` consumes **7** concatenated features (`len(target_layer_ids) = 7` → `fc.in_features = 28672`), and per our training convention the id `44 == num_hidden_layers − 1` resolves to the **post-final-norm** last hidden state (`hidden_states.py:26-49`: `layer_id == len(hidden_states) − 2 → final_hidden_state`).

glm5_next's dflash capture cannot produce it: `layers_to_capture = [val+1]` puts 45 in the list, and the capture closure only fires `if i in self.layers_to_capture` for layers `0..44` (`models/glm5_next.py:1158-1187`) → **only 6 features reach `project_target_hidden`** → `fc` shape mismatch ValueError at the first prefill (`dflash.py:771-787`).

Kimi K3 solves the identical problem for its DSpark draft: its `set_dspark_layers_to_capture` uses layer ids directly (output-capture, `models/kimi_k3.py:2989-3006` — note the comment "DFLASH target_layer_ids name layer outputs, which is what the DSPARK [capture wants]"), and its forward captures the **post-final-norm** output when the last layer index is in the capture list (`models/kimi_k3.py:2865-2875`, `capture=aux_hidden_states.capture` passed into `final_norm`).

Fix options, cheapest first:

1. **Small sglang patch on glm5_next** mirroring Kimi's pattern: in `set_dflash_layers_to_capture`, special-case `val == num_hidden_layers − 1` → capture the post-`final_norm` hidden (pass `capture` into `residual_batch.final_norm` like Kimi does) instead of emitting `val+1`. A few lines in `models/glm5_next.py`; also covers any future glm5_next drafter trained the DeepSpec way.
2. Re-pick capture ids ≤ 42 and retrain `fc` (the shipped DFlash2 draft does exactly this: `target_layer_ids [5,14,24,33,42]` — strong evidence the shipped team avoided the last layer on purpose).

### C.3 Other recorded constraints (not blockers for a single-node TP run)

- Speculative decoding (either family) does not run with DP-Attention (cookbook "Neither algorithm runs with DP-Attention"; DSPARK dense-draft compact verify + attn-DP explicitly rejects cuda graph, `dspark_worker_v2.py:295-307`).
- PD disaggregation + spec does not start at the current cut (draft-graph capture width assert on the prefill role; cookbook known-limitations).
- `--speculative-draft-kv-cache-dtype fp8_e4m3` is available to halve the 5-layer draft pool (`fields_spec.py:151-163`).
- glm5_next does not implement the pre-gather `set_dspark_target_hidden_projector` protocol (Kimi does) → projection runs post-gather on the draft side: works, slightly less optimal.

**Net: between "glm5_next served by sglang" and "glm5_next + DSPARK draft worker" stand exactly three things — the new drafter class (B), the last-layer capture patch (C.2), and the config edits (A.3). The KDA verify path, mHC capture folding, and mamba-state commit are already wired for this target.**

---

## D. Version pin recommendation

- **v0.5.21** (`lmsysorg/sglang:v0.5.21`, `-cu130` variant; released 2026-10-02) contains DSPARK (LFM2 #40651, merged 09-22), glm5_next + DFlash2 support (capture #36708), and the Kimi-K3 DSpark fix #38991 (merged 10-01). It does **not** contain: #42489 (10-06, GLM-5.3-Flash DFlash decoding perf on B300), #42624 (10-08, "Mask uncommitted slots before commit-inject cache lookups" — correctness-adjacent for the ctx-inject path), #43012 (10-08, block verification for the DFlash family), #42987 (10-09, folded draft sampling), and the unified-memory DSpark admission #40492 (10-08). Since we are bringing up a new draft on the target that #42489/#42624 directly touch, v0.5.21 is the wrong cut.
- **main @ `961bcf48`** is exactly what the nightly image `lmsysorg/sglang:nightly-dev-cu13-20261010-961bcf48` was built from (Docker Hub tag embeds the SHA), and it is the same SHA our earlier `dspark_config.py` reference was taken at — i.e., that image provably contains the DSPARK parser/loader state we analyzed, plus every DSpark fix through 10-10. Current main HEAD is `1814ece57fc283253e4e65980d4607d2207958c3` (only marginally ahead).
- **Recommendation:** pin `lmsysorg/sglang:nightly-dev-cu13-20261010-961bcf48` on the existing 4×B300 CUDA-13 baseline (the #12 baseline ran `v0.5.21` = the cu130 family, so the CUDA/driver story is unchanged; B300/Blackwell needs the cu13 image family — cu130 release tags exist since `v0.5.20-cu130`, and `nightly-cu134-*` images are the CUDA-13.4 alternative). Nightlies rotate: if the tag is gone, rebuild the repo at source SHA `961bcf48` with the CUDA-13 Dockerfile, or fall back to the rolling `dev-cu13` tag. sglang ships its own torch inside the image; there is no host torch/CUDA pin to manage, which is why the image pin (not a pip pin) is the right control point.

---

## E. BOTTOM LINE

Minimal change list to serve our checkpoint:

1. **Config edits on the checkpoint (no retraining):** `model_type → "qwen3"`, keep `architectures ["Glm5DsparkModel"]`, `head_dim 0 → 64`, confirm `intermediate_size` against the trained MLP shapes, add `use_cache: false` / `dtype: "bfloat16"` / explicit `sample_from_anchor: true`. Everything else already parses and validates (gamma=7 from `block_size`; `mask_token_id 154821 < 154880`; vanilla markov; confidence head with markov).
2. **No weight remaps.** sglang's loader already fuses q/k/v → `qkv_proj` and gate/up → `gate_up_proj`, skips the checkpoint's own `embed_tokens`/`lm_head` in favor of the target's, and loads `fc`/`hidden_norm`/`norm`/`markov_head.{markov_w1,markov_w2}`/`confidence_head.proj` verbatim.
3. **New sglang file** `python/sglang/srt/models/glm5_dspark.py` (~40 lines): `Glm5DsparkModel(DSparkDraftModel)` + NoPE `Glm5DsparkAttention(DFlashAttention)` (no rotary; `apply_k_rope` = identity; `rotary_emb = None`) + `Glm5DsparkDecoderLayer(DFlashDecoderLayer)` with `attention_cls` override; `EntryClass = [Glm5DsparkModel]`. Pattern proven by `lfm2_dspark.py` + the Laguna `attention_cls` override.
4. **Small sglang patch to glm5_next capture** (or a verified no-op if a live test shows otherwise): handle `target_layer_id == num_hidden_layers − 1` by capturing the post-final-norm hidden, mirroring Kimi K3's DSpark capture. Without it, the 7-feature `fc` gets 6 features and dies at first prefill.
5. **Server flags** (target flags from the GLM-5.3-Flash recipe: `--attention-backend dsa --dsa-*-backend trtllm/tilelang --linear-attn-backend triton --kv-cache-dtype fp8_e4m3 --quantization fp8 …`):

   ```bash
   --speculative-algorithm DSPARK
   --speculative-draft-model-path <ckpt>
   --speculative-dspark-block-size 7        # or --speculative-num-draft-tokens 8
   --speculative-draft-attention-backend <trtllm_mha|fa4|…>   # pick explicitly on B300
   # optional: --speculative-dspark-confidence-sts-path <fit with sglang.benchmark.dspark_sts_fit>
   # optional: --speculative-dspark-sps-table-path (ragged/cap-accept scheduling)
   ```

6. **Image pin:** `lmsysorg/sglang:nightly-dev-cu13-20261010-961bcf48` (main @ `961bcf48`, CUDA 13), not v0.5.21.

**UNKNOWNS that need a live GPU test:**

- Whether any code path already supplies the 7th (post-final-norm) capture on glm5_next — static read says no; the failure mode is a `fc.weight`-shape ValueError naming `num_context_features` at the first prefill, which is unambiguous either way.
- Which draft attention backend is correct/fast for a NoPE ENCODER_ONLY dense draft on B300 (`trtllm_mha` is Kimi's DSPARK default; DFlash2 on this target runs `fa4`; flashinfer/triton are the generic fallbacks).
- Trained `intermediate_size` (12288 inherited from the target vs 10240 class default) — read `gate_proj.weight.shape` off the checkpoint before finalizing config.
- Serving-time context semantics vs training: at serve time every draft block attends bidirectionally over the **entire** committed prefix (context = all positions before the anchor), while training sampled anchors over a bounded window — accept-length sanity check with the DSpark observability counters (`dspark_observability.py`) and `SGLANG_SIMULATE_ACC_LEN` before trusting throughput numbers.
- Whether the inherited GLM rope fields in the draft config trip anything outside our subclass (they shouldn't — the NoPE class never reads them).
- CUDA-graph capture with the new class end-to-end (folded proposal requires the vanilla fused kernel — our head is vanilla, so it should engage; verify `SGLANG_DSPARK_FOLDED_PROPOSAL` logs).
- Confirm `mask_token_id 154821` is the exact noise id training embedded (config says so; spot-check one `create_noise_embed` equivalence in eval).
- KDA-state correctness after partial accept under real sampling (`update_mamba_state_after_mtp_verify` path) — exercised by Kimi-K3 DSpark upstream, but our target/backends combination (dsa + nv_cutedsl) should be smoke-tested for state-restore correctness.

---

## Addendum (2026-10-13, issue #47 bring-up): live corrections and the working recipe

The bring-up smoke served the step_20 checkpoint end to end through DSPARK on the pinned nightly (app `experiments/glm-drafter/dspark_bringup.py`); three static conclusions above need corrections, and the working recipe is now measured.

1. **`SGLANG_DSPARK_STACKED_CTX_KV` defaults to True at this SHA** — the "keep it off (default)" note in §B.3 is stale. The stacked ctx-KV path is where a NoPE drafter must intervene: the mixin's `_project_ctx_kv_stacked` makes a single `rotary_emb` call, and everything else in the stacked path (weights, per-layer k-norms) is arch-agnostic. `glm5_dspark.py` therefore carries the NoPE stacked override (the parent's projection minus that one rope call) — the "easy NoPE variant" §B.3 predicted.
2. **§A.3's config edit list is not sufficient.** Editing keys in place leaves the GLM hybrid fields (`linear_attn_config`, `index_*` DSA keys, `mhc`/`hc_*`, MoE fields, `num_nextn_predict_layers`) on the draft config, and sglang then treats the dense 5-layer drafter as a hybrid: the draft KV pool builds with unsharded full-width (64-head) rows while the draft attention is TP-sharded, and decode-graph capture dies in the KV store (`store_cache` size mismatch, batch 336 vs 84). The serve config must be a **clean dense whitelist** — only what `parse_dflash_draft_config`/`parse_dspark_draft_config` plus `DFlashAttention`/`DFlashMLP` read, i.e. the shape of the shipped DFlash2 config (DFlash2 boots green on this same nightly with its clean config). Also keep `max_position_embeddings` at the target's value: sglang derives the server context from the draft config too, and a smaller draft value hard-errors at boot.
3. **Draft attention backend: fa4 works; trtllm_mha breaks this host.** fa4 (the DFlash2 precedent) forces the draft KV dtype to bf16 (it cannot read the target's fp8 KV), which declines the fused-draft region and builds a standalone draft pool — correct once the config is clean. `trtllm_mha` keeps the fp8 fused path, whose page sizing forces `page_size=64`; this target's DSA index pool runs `index_kpool=4` (inherited from the GLM config) and requires 256-token pages, so the target pool asserts at boot. Recorded as a glm5_next+DSPARK incompatibility at this SHA, not a backend defect.
4. **Working recipe, proven on 4×B300 at B200 rates** (image `lmsysorg/sglang:nightly-dev-cu13-20261010-961bcf48`): stock image + two in-place patches — the glm5_next capture (post-final-norm return-value capture per #33's resolution: setter special-cases the final id, forward appends `final_norm`'s return) and `glm5_dspark.py` (NoPE attention + decoder-layer class + stacked override) — plus the clean-dense serve config (`prepare_drafter` stage builds it from the checkpoint config + safetensors header) and flags `--speculative-algorithm DSPARK --speculative-draft-model-path <serve copy> --speculative-dspark-block-size 7 --speculative-draft-attention-backend fa4` over the #12 target recipe. Boot ≈ 20 min on warm volume weights (load 156 s; graphs: prefill 367 s, target_verify 82 s, draft_decode 42 s), ≈ $25–40/attempt. Spec counters populate (`spec_accept_length`, `spec_accept_rate`, `spec_verify_calls_total`, `spec_num_draft_tokens` 8 = γ+1); the 20-step step_20 smoke checkpoint measures counter-τ ≈ 1.0 — liveness, not quality; the undertrained drafter's proposals are ~random relative to the target.
