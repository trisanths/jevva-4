# jevva-4

Fast one-pass multiple-choice decisions ("Jev") on Gemma 4 vision-language models, measured on a single NVIDIA L40S at batch 1.

A Jev decision is one forward pass. The model reads a frame plus a question with lettered options, and the answer is the softmax over the logits of the offered letters at the answer position. There is no decoding, so latency is one prefill. This repo holds the code, launch scripts and result files for four lines of work done between 2026-09-28 and 2026-10-07:

1. Jev-tuning and a fast path for Gemma 4 E2B and E4B: an E2B decision in 7.6 ms end to end at 70% image accuracy.
2. A fast path for Gemma 4 26B-A4B: 49.4 ms end to end at 80.8% image accuracy.
3. A negative result: turning the dense Gemma 4 12B into a mixture of experts (MoEfication) does not pay.
4. A negative result: letting E2B prefill and mapping its states into the 26B's KV cache scores below E2B alone.

"End to end" always means uint8 frame plus question text in, option probabilities out, including GPU image preprocessing and tokenization, timed on the GPU host. Network time is not included.

## Results

### Evaluation set

4,400 multiple-choice items: ScienceQA-IMG test (800), A-OKVQA val (800), AI2D test (800), MMStar (600), MMLU test (800), ARC-Challenge test (600). "Image" accuracy is the first four (3,000 items), "text" the last two (1,400). Frames are resized to 640x360 or 640x480 before the processor. Image budgets of 70, 140 and 280 are Gemma 4's soft-token counts per image.

### Jev-tuning (E2B, E4B)

LoRA r16 / alpha 32 on the language layers plus a trainable final norm. The loss is cross-entropy over the offered letters only, with options reshuffled per example and a random image budget per batch. Training used about 30K train-split items: A-OKVQA train, ScienceQA train, MMLU auxiliary train, ARC, ARC-Easy and CommonsenseQA. AdamW lr 1e-4, betas 0.9/0.95, 100 warmup steps, cosine, effective batch 16. AI2D and MMStar never appear in training.

| model | budget | image acc base | image acc Jev-tuned | text acc base | text acc Jev-tuned | image ECE base | image ECE tuned |
|---|---|---|---|---|---|---|---|
| E2B | 70 | 53.7 | 70.1 | 61.2 | 71.2 | 0.304 | 0.041 |
| E2B | 280 | 56.0 | 72.6 | 61.2 | 71.2 | 0.287 | 0.043 |
| E4B | 70 | 65.5 | 76.3 | 75.9 | 79.4 | 0.236 | 0.047 |
| E4B | 280 | 70.8 | 79.3 | 75.9 | 79.4 | 0.204 | 0.041 |

Gains hold on the two held-out sources (E2B at 70: AI2D 48.9 to 66.1, MMStar 37.3 to 48.5). Full metrics are in `results/jev/summary.json`.

### E2B / E4B fast path (L40S, batch 1, budget 70)

| model | precision | compile | end to end p50 (ms) |
|---|---|---|---|
| E2B | bf16 | default | 10.93 |
| E2B | FP8 per-tensor | default | 8.83 |
| E2B | FP8 + int4 tail | max-autotune-no-cudagraphs | 7.64 |
| E4B | FP8 per-tensor | default | 13.26 |
| E4B | FP8 + int4 tail | max-autotune-no-cudagraphs | 11.57 |

Stock Hugging Face single forward pass of E2B was 76 ms. The steps that got it down: static shapes and one CUDA graph (18 ms), torch.compile (10.2 ms), FP8 (8.1 ms on the GPU, 8.8 ms end to end), FP8 plus int4 on the tail (7.6 ms). Accuracy on the deployed path: E2B-jev 69.6 to 70.0 across bf16, FP8 and FP8+int4; E4B-jev 75.4 bf16 and 74.9 FP8+int4.

Two cuts are specific to one-pass decisions. Gemma 4 E2B and E4B share K/V across their last 20 (E2B) or 18 (E4B) layers, so those layers only need to run on the answer token. And only the offered letters' rows of the tied embedding are needed for the readout. Together they cut 23% (E2B) and 16% (E4B) of latency, measured by ablation (`results/jev/ablation.jsonl`). Stage breakdown is in `results/jev/stages.jsonl`. A10G runs are in `results/jev-a10g/`.

### 26B-A4B fast path (L40S, batch 1, untuned weights)

`src/jev_fast26.py` runs the Hugging Face modules with vLLM 0.30's fused MoE kernels. Experts are quantized at load from the bf16 checkpoint, either to 4-bit (symmetric round-to-nearest, groups of 32, repacked for the Marlin MoE kernel) or to FP8 per output channel. The vision tower is compiled and captured in one CUDA graph, optionally FP8.

| experts | vision tower | budget | end to end p50 (ms) | image acc |
|---|---|---|---|---|
| Marlin 4-bit | bf16 | 280 | 57.0 | 80.6 |
| Marlin 4-bit | FP8 | 280 | 49.4 (p99 50.1) | 80.8 |
| Marlin 4-bit | FP8 | 140 | 32.6 | 76.7 |
| Marlin 4-bit | FP8 | 70 | 27.9 | 73.8 |

FP8 experts score 80.6 at 280. vLLM 0.30 used as a server for the same one-pass call took 99 to 116 ms on this card. The H200 serving baseline we compared against reported 79.7 ms p50 on its server timer, with bf16 weights, a TensorRT vision tower, 280 image tokens and 8 generated tokens. Those setups differ from ours, so treat the comparison as rough.

The 26B language model alone, against sequence length (Marlin experts, compiled graph, real token windows; `results/jev/lm_short.jsonl`):

| tokens | 1 | 4 | 8 | 16 | 32 | 64 | 144 |
|---|---|---|---|---|---|---|---|
| ms | 8.1 | 9.9 | 10.9 | 12.8 | 14.3 | 18.0 | 22.0 |
| experts touched per layer (of 128) | 8 | 16 | 23 | 37 | 48 | 71 | 85 |

At batch 1 the model is bound by the weights it reads. The 8.1 ms floor for one token is mostly the attention and dense MLP weights, about 1.9B parameters, which this path leaves in bf16. Quantizing them is the obvious next cut.

### Negative result: MoEfying the 12B

Gemma 4 12B (dense, encoder-free) split into experts with oracle routing at 50% of the FFN, all-items accuracy (dense 81.2): 8x1920 experts 36.8, best subset 38.5, shared plus 3 of 7 50.0, 32x480 55.3, 128x120 67.8, random 27.2. Even unstructured per-token neuron selection reaches only 79.9 at 50%, 76.0 at 25% and 54.0 at 12.5%, and loses to simply cutting image tokens at matched FLOPs. The 26B-A4B is cheaper and more accurate than the 12B at every image budget (280: 81.2 at 5.64 TFLOP vs 79.2 at 7.34). The 12B also decides late: a logit lens is at chance until layer 35 of 48, so early exit buys little. Files: `results/moefy/`.

### Negative result: small-model prefill handed to the 26B

E2B-jev reads the frame and question. A mapper turns its residual stream (layers 4, 9 and 14) into every 26B layer's keys and values, and the 26B runs only the 9 chat-template tokens before reading out the letter. The header tokens' KV comes from the 26B itself. Budget 70, same 4,400 items:

| setup | image | text | end to end (ms) |
|---|---|---|---|
| E2B-jev alone | 68.8 | 69.6 | 7.6 to 8.8 |
| 26B direct | 73.7 | 88.6 | 27.9 |
| hybrid, closed-form ridge mapping (Heo et al., arXiv 2608.03893) | 55.5 | 53.3 | 33.7 |
| hybrid, trained mapper (57.5M) + 26B LoRA (11.5M), CE + KL to the 26B, 1 epoch | 66.7 | 66.4 | 33.9 |
| control: same E2B states, same loss and parameter count, no 26B | 65.6 | 62.4 | n/a |

The hybrid stays below the small model it reads from, and in this build it was also slower than the 26B on its own. Given its own KV through the same code path, the 26B reproduces its own answers (10 of 10), so the harness is not the cause. `results/handoff/research_notes.txt` has a source-checked literature and workload review behind this verdict, including Kimi K3 and GLM-5.3.

## Layout

`src/`, by experiment:

- shared: `common.py` (loading, batching, FP8 and int4 quantization, scoring), `data.py` (evaluation and training loaders, prompt format)
- MoEfication: `moe.py`, `partition.py`, `stage1.py`, `stage1_26b.py`, `stage2.py`, `analysis.py`, `plot.py`, `timing.py`
- 12B early exit: `jevnet.py`, `exit_probe.py`, `jev_grid.py`, `jev_fast.py`
- Jev-tuning and the E2B/E4B fast path: `train_jev.py`, `jev_base.py`, `jev_fast_e.py`, `fast_bench_e.py`, `fast_eval_e.py`, `e2e_bench.py`, `prof_stages.py`, `ablate_e.py`, `plot_jev.py`
- 26B: `jev_vllm.py`, `tower26.py`, `jev_fast26.py`, `test_marlin.py`, `lm_short.py`
- KV handoff: `hx_core.py`, `hx_prep.py`, `hx_ridge.py`, `hx_train.py`, `hx_c1.py`, `hx_eval.py`, `hx_bench.py`, `hx_test.py`, `hx_test2.py`
- debugging the harness: `check_harness.py`, `check_mask.py`, `debug_*.py`, `sdpa_check.py`, `smoke.py`

`infra/` holds the AWS scripts. `gpu.sh` brings a box up and down (`up | stop | start | down | status | ssh-config | allow-ip | sync`). It needs `MOEFY_BUCKET` (an S3 bucket you own) and `MOEFY_KEY_NAME` (an EC2 key pair). Optional settings are `MOEFY_REGION` (default `us-east-1`), `MOEFY_HF_SECRET` (a Secrets Manager id holding a Hugging Face token, default `huggingface`), `MOEFY_ROLE`, `MOEFY_KEY_FILE`, `MOEFY_BOX_NAME`, `MOEFY_SSH_ALIAS` and `MOEFY_SPOT`. The first-boot script writes `~/.moefy_env` on the box, which the run scripts source. `hunt_l40s.sh` retries g6e capacity every 5 minutes. The `chain_*.sh` and `hx_*.sh` files are the exact run sequences used on the boxes.

`results/` holds the summaries and per-run records; per-item probability files and the trained adapters (`e2b-jev` 92 MB, `e4b-jev` 133 MB) are not included.

## Environment

Everything ran on AWS (g6e L40S, g5 A10G), nothing locally. Two virtualenvs:

- Hugging Face stack: torch 2.11.0+cu128, transformers 5.17.0, torchao 0.18.0, peft.
- vLLM stack (26B and handoff work): vllm 0.30.0, torch 2.13.0+cu130, transformers 5.17.0, torchao 0.18, accelerate.

Model snapshots: gemma-4-E2B-it `3e22461f`, gemma-4-E4B-it `ee0ef602`, gemma-4-26B-A4B-it `4d7ae498`.

## Pitfalls we hit

- transformers 5.17 needs `processor_kwargs=dict(padding=True)` for padded batches; passing `padding=True` directly silently changes results.
- The plain Hugging Face forward applies the image bidirectional mask on global layers too. `common.py` patches it to the vision-model mask.
- SDPA on a single unpadded sequence disagreed with eager attention on the 12B. Reference runs use eager.
- Rowwise FP8 (torchao) hangs under CUDA graph replay on sm89. Per-tensor FP8 works.
- Uncompiled torchao FP8 is slower than bf16 at batch 1.
- Merge LoRA adapters in bf16 before quantizing.
- vLLM's Triton int4 MoE kernel has no tuned config for the L40S and runs 2x slower than FP8. Marlin fixes it.

## Caveats

Single runs, no seeds swept. The Jev-tuned models trained on the A-OKVQA and ScienceQA train splits, so those two eval sources are in-distribution; AI2D and MMStar are the held-out check. Latency numbers are GPU-host end to end at batch 1 and exclude network round trips.
