# GLM-5.3-Flash Speculative Decoding

Vocabulary for the effort to assess and, if feasible, deploy a DSpark-style parallel drafter for GLM-5.3-Flash. Terms come from the papers in `papers/`.

## Speculative decoding

**Speculative decoding**:
A small drafter proposes a block of candidate tokens; the target verifies the whole block in one forward pass and accepts the longest consistent prefix, appending one bonus token. Output distribution is identical to the target's.
_Avoid_: lossless decoding, assisted generation

**Target**:
The large model being served whose token distribution is authoritative (here, GLM-5.3-Flash).

**Drafter**:
The small model proposing candidate tokens for the target to verify.

**Accepted length (τ)**:
Tokens accepted per decoding round, including the bonus token.

**Anchor token**:
The target-generated token from the previous round that seeds the next drafting round.

**Bonus token**:
The token the target generates itself after accepting a draft prefix.

## DSpark

**DSpark**:
DeepSeek-AI's speculative decoding framework: a semi-autoregressive drafter plus confidence-scheduled verification, deployed in DeepSeek-V4 serving (arXiv:2607.05147).
_Avoid_: DeSpark

**DeepSpec**:
DeepSeek-AI's open-sourced training repository for speculative decoding algorithms (Eagle3, DFlash, DSpark), with released drafter checkpoints.

**Parallel drafter**:
A drafter producing all γ candidate tokens in a single forward pass; drafting latency is independent of block size, but positions predict independently.

**Semi-autoregressive drafter**:
DSpark's drafter: a parallel backbone plus a lightweight sequential head, keeping single-pass drafting while conditioning each position on the previously sampled token in the block.

**Suffix decay**:
Falling acceptance probability at later block positions because each parallel position marginalizes over all possible predecessors instead of the one actually sampled.
_Avoid_: acceptance decay, multi-modal collision (the underlying cause)

**Markov head**:
DSpark's default sequential module: a low-rank bigram transition bias added to the parallel backbone's logits.

**Confidence head**:
Predicts, per draft position, the probability that the token survives verification given all preceding tokens in the block were accepted. Calibrated post-hoc.

**Sequential Temperature Scaling (STS)**:
Post-hoc left-to-right calibration of cumulative survival probabilities against empirical acceptance rates.

**Hardware-aware prefix scheduler**:
Chooses each request's verification length to maximize expected system-wide throughput given the engine's profiled capacity curve and current load; preserves losslessness.

**KV injection**:
Feeding target-model hidden states into the drafter's attention as extra context, so the drafter conditions on the target's computation.

## GLM-side baseline

**MTP**:
Multi-Token Prediction: native speculative-decoding layers shipped with GLM models. GLM-5.3-Flash includes an MTP layer.
_Avoid_: MTP head, draft head

**MTP-1**:
The single-token-per-step native speculative baseline that DSpark replaced in DeepSeek-V4 serving; the natural baseline for this effort.
