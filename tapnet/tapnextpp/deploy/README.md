# TAPNext++ deployment: ONNX / TensorRT export and benchmarking

Tools to run TAPNext++ (and TAPNext) frame-by-frame with low latency on a
local NVIDIA GPU.

| File | Purpose |
|------|---------|
| `step_model.py` | `TAPNextStep`: one online step as a pure tensor function (same weights, same math). `PointTokenizer`: builds query tokens without the full model. |
| `export_onnx.py` | Exports `TAPNextStep` to ONNX (fp16 or fp32) and verifies it against the reference `TAPNext.forward` with onnxruntime. |
| `trt_runner.py` | Minimal TensorRT 10/11 runtime wrapper with on-GPU ping-pong state, and `build_engine` (Python API, no `trtexec` needed). |
| `benchmark.py` | GPU latency/FPS of the reference model, the step model (eager, CUDA graph, `torch.compile`) and a TensorRT engine, plus deviation from fp32. |
| `eval_davis.py` | TAP-Vid DAVIS accuracy of any variant: PyTorch fp32/fp16, other input resolutions, simulated FP8, padded query slots, an ONNX graph or a TensorRT engine. |
| `quantize_fp8.py` | FP8 post-training quantization with NVIDIA ModelOpt, calibrated on real frames plus the recurrent state the model has at that point. |
| `optimize_local.py` | One command for the target machine: export, build, benchmark and evaluate a grid of resolutions and query counts, then write `report.md`. |

## Why a separate step model?

`tapnext_torch.TAPNext.forward` is written for research use. It branches in
Python on whether a cache exists. It carries a dataclass of 24 state tensors
and uses `einsum` and a custom autograd `sqrt`. It also keeps a 200 MB sin-cos
lookup buffer that is only needed on query frames. `TAPNextStep` keeps the
network and weights but exposes a fixed-signature graph:

```
inputs : frame [1,H,W,3] f32 in [-1,1] | point_tokens [1,Q,768] f32 | reset [1] f32
         rg_state [12,N,768] f32 | conv_state [12,N,3,768] f16
outputs: tracks [1,Q,2] (y,x in 256x256 model space) | visible_logits [1,Q,1]
         rg_state_out | conv_state_out                       (N = (H/8)*(W/8) + Q)
```

In fp16 mode, the matmuls and attention run in fp16. LayerNorm/RMSNorm, the
residual stream and the RG-LRU recurrence stay in fp32, as under
`torch.autocast`. The resulting ONNX file is explicitly typed, which is what
TensorRT expects with `--stronglyTyped`, the only mode in TensorRT 11.

Point tokens only change on frames where a query starts, so cache them. Unused
query slots can stay on the "unknown" token, which is the same token a query
sees before its start frame, and be activated later.

## Verified (CPU, fp32 reference vs. export)

| Check | Result |
|-------|--------|
| `TAPNextStep` fp32 vs `TAPNext.forward`, 6 frames, incl. late queries | max 6e-5 px |
| ONNX fp16 (onnxruntime) vs fp32 reference, 256 px, 256 queries | max 0.002 px, visibility 100% equal |
| ONNX fp16 vs fp32 reference, 512 px checkpoint, 64 queries | max 0.0012 px |
| `PointTokenizer` vs `TAPNext.embed_queries` | bit-exact |

The fp16 ONNX graph contains only standard ops (Gemm, MatMul,
LayerNormalization, Softmax, ArgMax, Erf, Tanh, Conv, ...). Its weights are
394 MB.

`benchmark.py` and `trt_runner.py` need a CUDA GPU and TensorRT. They were not
run on hardware when this was written.

## Run it on the target machine

The quickest path is the sweep, which downloads the checkpoint and DAVIS,
builds one TensorRT engine per configuration and writes `report.md` with fps
and DAVIS accuracy side by side:

```bash
pip install -e ".[torch]" onnx onnxscript onnxruntime tensorrt mediapy scipy absl-py
python -m tapnet.tapnextpp.deploy.optimize_local --workdir tapnextpp_opt \
    --resolutions 256,224,192 --queries 64,256
```

The individual steps:

```bash
pip install -e ".[torch]" onnx onnxscript onnxruntime   # from a checkout of this repo
pip install tensorrt   # or the TensorRT tarball; needs a Blackwell-capable release
wget https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt   # 256x256 model

# 1. Export + verify (any machine, CPU is fine)
python -m tapnet.tapnextpp.deploy.export_onnx --checkpoint tapnextpp_ckpt.pt \
    --resolution 256 --num_queries 256 --precision fp16 \
    --output tapnextpp_256_q256_fp16.onnx

# 2. Build the engine ON THE TARGET GPU (engines are GPU/driver specific).
#    The extra flags are no-ops (defaults) in TensorRT 11 and needed in 10.x.
trtexec --onnx=tapnextpp_256_q256_fp16.onnx --saveEngine=tapnextpp_256_q256_fp16.plan \
    --stronglyTyped --useCudaGraph --noDataTransfers --useSpinWait
#    trtexec prints the GPU compute time per inference: that is the
#    best-case per-frame latency.

# 3. End-to-end comparison incl. accuracy vs fp32
python -m tapnet.tapnextpp.deploy.benchmark --checkpoint tapnextpp_ckpt.pt \
    --resolution 256 --num_queries 256 --engine tapnextpp_256_q256_fp16.plan
python -m tapnet.tapnextpp.deploy.benchmark --checkpoint tapnextpp_ckpt.pt \
    --resolution 256 --num_queries 256 --modes step_graph --fp16_accumulation
```

On Windows, `torch.compile` needs Triton. The `*_compile*` modes may fail
there; `step_graph` and `trt` do not use it.

## Accuracy of the speed levers (TAP-Vid DAVIS, measured on CPU)

Query-first, all points of a video tracked jointly online, `tapnextpp_ckpt.pt`,
all 30 videos (`eval_davis.py`):

| Variant | AJ | delta_avg | OA | AJ change |
|---------|---:|----------:|---:|----------:|
| 256x256 fp32 (reference, matches the paper's 66.6 / 79.9 / 92.1) | 66.59 | 79.94 | 92.12 | - |
| 256x256, linear layers in FP8 E4M3 (per-tensor, simulated) | 66.31 | 79.74 | 91.86 | -0.28 |
| 256x256, padded to 64 fixed query slots ("unknown" tokens) | 66.60 | 79.91 | 92.11 | +0.01 |
| 224x224 input (25 videos; reference on the same 25: 67.05) | 43.01 | 58.32 | 86.40 | -24.0 |

* FP8 is close to free in accuracy and roughly doubles tensor-core peak, so it
  is the main lever towards 60 fps. `quantize_fp8.py` produces the ModelOpt
  FP8 graph; verify the built engine with `eval_davis.py --engine`.
* Fixed-size engines are fine: unused slots on the "unknown" token do not
  change the results.
* Lower input resolution is not usable without fine-tuning. The model was
  only trained at 256 (and 512 for the 512 checkpoint).

## What to expect: compute budget

TAPNext-B processes every 8x8 patch as a token through 12 ViT + 12 SSM
blocks. That costs about 385 MFLOPs per token plus attention, mostly in
image tokens:

| Input | Queries | GFLOPs / frame | Effective TFLOPS needed for 60 fps |
|-------|--------:|---------------:|-----------------------------------:|
| 256x256 | 16 | 440 | 26 |
| 256x256 | 64 | 462 | 28 |
| 256x256 | 256 | 553 | 33 |
| 256x256 | 1024 | 942 | 57 |
| 512x512 | 64 | 2238 | 134 |
| 512x512 | 256 | 2372 | 142 |

Published PyTorch + `torch.compile` numbers from the TAPNext/TAPNext++
papers:

* H100: 189-197 fps, latency-bound at about 5 ms.
* V100: 70 fps at 256 queries and 42 fps at 1024 queries. Both correspond to
  about 39 effective TFLOPS, roughly 31% of the V100's 125 TFLOPS fp16 peak.

An RTX 5070 Laptop GPU has 36 SMs at up to about 2.35 GHz. GeForce Blackwell
runs fp16 tensor math with fp32 accumulation at half rate, which gives a dense
fp16 peak of about 43 TFLOPS at full boost, less in a power-limited chassis.
fp16 accumulation, FP8 or INT8 roughly double that peak. So, at 256x256:

* PyTorch at V100-like efficiency: about 25 fps.
* TensorRT fp16 at about 45-60% efficiency: about 35-50 fps.
* 60 fps at 256 queries needs about 77% of the fp16 peak. That is unlikely,
  so plan on one of these:
  * FP8 / INT8 quantization of the linear layers, for example NVIDIA
    TensorRT Model Optimizer PTQ, then rebuild the engine. Validate accuracy
    on TAP-Vid DAVIS first.
  * fp16 accumulation, via `--fp16_accumulation` in PyTorch. Check the
    deviation column.
  * Fewer image tokens. A non-square grid such as 256x192 for 4:3 input is
    supported by the position-embedding interpolation but was not trained.
    Accuracy must be checked.
* 512x512 is out of reach at 60 fps on this GPU: about 4x the compute of
  256x256.

All numbers above are estimates; use `benchmark.py` / `trtexec` to measure.
