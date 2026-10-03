# TAPNext++ online tracking on an RTX 500 Ada Generation Laptop GPU

Measured 2026-10-03 with `tapnextpp_ckpt.pt` at 256x256. **60 fps is not
reachable on this GPU.** The best measured configuration is a TensorRT FP8
engine at **38.9 fps** (25.7 ms mean, 26.7 ms p95) with DAVIS AJ 66.42, or
**35.0 fps** at AJ 66.71 if the 0.3 AJ matters more than 4 fps. The gap to
60 fps is explained in [Why not 60 fps](#why-not-60-fps).

## The machine

| | |
|---|---|
| GPU | NVIDIA RTX 500 Ada Generation Laptop GPU, sm_89, 16 SMs (2048 CUDA cores) |
| VRAM | 4093 MiB, 16 MiB L2 |
| Driver | 596.71, CUDA 13.2, WDDM |
| Power limit | 30 W (= default = min = max in this scope; the part's ceiling is 35 W) |
| Clocks | 3105 MHz max; 2025 MHz in short bursts; **1395-1560 MHz sustained** at 27-32 W, 58-65 C |
| Power mode | on AC, Windows "Best performance" overlay (`ded574b5-...`), already active |
| Host | Windows 11 Pro 26200, 31.5 GiB RAM |

`nvidia-smi -pl 35` is refused ("Changing power management limit is not
supported in current scope"), and this Windows build exposes only the Balanced
scheme, whose performance overlay was already set to maximum. So 30 W is the
hard envelope, and it is what holds the SM clock at less than half its 3105 MHz
maximum under load. Every number below is with that envelope.

### Measured GEMM ceiling

4096^3 GEMM, 60 iterations, `torch.matmul` / `torch._scaled_mm`:

| precision | ms | TFLOPS |
|---|---:|---:|
| fp32 (tf32 tensor cores) | 19.20 | 7.16 |
| fp16, fp32 accumulate | 8.14 | 16.89 |
| fp16, fp16 accumulate | 7.57 | 18.16 |
| fp8 e4m3 | 2.37 | 57.95 |

At 462 GFLOPs per frame, 60 fps needs 27.7 effective TFLOPS. Note that fp16
with fp32 accumulation runs at full rate here: the half-rate behaviour the
README expected applies to consumer Blackwell, not to Ada.

### Software

* Python 3.12.13, `torch 2.11.0+cu128`, `tensorrt 11.3.0.99` (cu13 wheels),
  `onnx`, `onnxscript`, `onnxruntime 1.30`.
* Second venv for FP8: `nvidia-modelopt 0.47.0`, `onnxruntime-gpu 1.22`,
  `torch 2.11.0+cu128`. The CUDA EP only loads with torch's DLL directory on
  `PATH` (`.venv-mo/Lib/site-packages/torch/lib`, which carries
  `cublasLt64_12.dll` and `cudnn64_9.dll`); without it ModelOpt silently
  calibrates on the CPU.
* `trtexec` and Nsight Systems are not available — the pip TensorRT wheels ship
  neither — so profiling used `tensorrt.IProfiler` (see
  [Where the time goes](#where-the-time-goes)).
* `torch.compile` is unusable: `TritonMissing` on Windows. The `ref_compile`
  and `step_compile_graph` modes fail, which is why the comparison below is
  against eager PyTorch and CUDA graphs only. The published 189-197 fps (H100)
  and 70 fps (V100) figures do use `torch.compile`.

## Results

`fps` and `p95 ms` are per-frame, including reading the tracks back to the host
(`benchmark.py`). `dev px` is the deviation from the fp32 reference model over
24 frames of the synthetic panning video. DAVIS is query-first with all points
of a video tracked jointly, metrics in 256x256 space.

### Sustained (`--warmup 200 --frames 400`)

This is the number an application sees: 200 warm-up frames are enough for the
clock to settle into the 30 W envelope.

| variant | mean ms | p95 ms | fps | eff. TFLOPS | max dev px | mean dev px |
|---|---:|---:|---:|---:|---:|---:|
| **TensorRT FP8, Gemm only, q64** | **25.61** | **26.56** | **39.0** | 18.0 | 4.83 | 0.077 |
| TensorRT FP8, Gemm only, q64, builder level 3 | 25.73 | 26.72 | 38.9 | 18.0 | 4.83 | 0.077 |
| TensorRT FP8, `lin_out` + SSM `down` kept fp16, q64 | 28.57 | 30.16 | 35.0 | 16.2 | 3.26 | 0.064 |
| TensorRT fp16, q64 | 37.58 | 38.83 | 26.6 | 12.3 | 0.108 | 0.0016 |
| PyTorch `step_graph` fp16, q64 | 53.53 | 54.32 | 18.7 | 8.6 | 0.115 | 0.0013 |

The FP8 `max dev px` of 4.8 is one query slot whose soft-argmax lands in the
neighbouring bin; `mean dev px` of 0.077 is the honest figure, and the DAVIS
numbers below confirm it.

### Default methodology (`--warmup 30`, 100-200 frames)

Lower than the sustained figures for the TensorRT engines because the first
seconds of the measurement window still contain the clock ramp.

| variant | mean ms | p50 ms | p95 ms | fps | max dev px |
|---|---:|---:|---:|---:|---:|
| `ref` (`TAPNext.forward`, autocast fp16), q64 | 82.18 | 81.55 | 86.40 | 12.2 | 0.670 |
| `step` (eager fp16), q64 | 60.31 | 60.84 | 64.26 | 16.6 | 0.115 |
| `step_graph` (CUDA graph), q64 | 56.08 | 55.09 | 62.67 | 17.8 | 0.115 |
| `step_graph --fp16_accumulation`, q64 | 55.56 | 54.59 | 61.24 | 18.0 | 0.342 |
| TensorRT fp16, q64 | 39.65 | 39.04 | 43.93 | 25.2 | 0.108 |
| `step_graph`, q256 | 65.54 | 64.38 | 73.71 | 15.3 | 3.863 |
| TensorRT fp16, q256 | 44.06 | 43.68 | 48.93 | 22.7 | 3.872 |
| TensorRT FP8, Gemm only, q64 | 28.93 | 29.16 | 30.96 | 34.6 | 4.826 |
| `ref_compile`, `step_compile_graph` | - | - | - | - | `TritonMissing` |

### DAVIS accuracy

| variant | videos | AJ | delta_avg | OA |
|---|---:|---:|---:|---:|
| fp32 CPU reference (prior session) | 30 | 66.59 | 79.94 | 92.12 |
| TensorRT fp16, q64 | 30 | **66.60** | 79.92 | 92.12 |
| TensorRT fp16, q256 | 30 | **66.60** | 79.70 | 92.20 |

Videos 0-3 (`goat`, `car-roundabout`, `motocross-jump`, `breakdance`) are the
FP8 calibration set, so the FP8 engines are scored on videos 4-29 and the fp16
engines are re-averaged over the same 26 for comparison:

| variant (26 videos, 4-29) | AJ | delta_avg | OA | AJ vs fp16 |
|---|---:|---:|---:|---:|
| TensorRT fp16, q64 | 66.86 | 80.10 | 91.83 | - |
| TensorRT fp16, q256 | 66.77 | 79.81 | 91.86 | -0.09 |
| TensorRT FP8, `lin_out` + SSM `down` kept fp16 | 66.71 | 80.09 | 91.68 | **-0.15** |
| TensorRT FP8, all 108 Gemms | 66.42 | 79.89 | 91.51 | **-0.44** |

All 30 DAVIS videos have at most 64 query points, so the 64-slot engine covers
the benchmark; the 256-slot engine costs 4 fps for nothing here. A 256-slot FP8
engine was not built: it needs its own 2.4 GB calibration set and the query
count is not the bottleneck.

The FP8 accuracy cost lands where `eval_davis.py --fp8_sim` predicted on the
CPU (-0.28 AJ). The worst-conditioned FP8 activations are the gated-MLP down
projections (`node_linear_{9i+4}`, per-tensor amax up to 1.2e4) and the RG-LRU
output projections (`node_linear_{9i+2}`, up to 5.8e3). Keeping those 24 of 108
Gemms in fp16 recovers 0.29 AJ for 3.9 fps:

```
python -m tapnet.tapnextpp.deploy.quantize_fp8 ... \
    --nodes_to_exclude "^node_linear_2$ ^node_linear_4$ ^node_linear_11$ ..."
```

## Recommended configuration

```bash
# 1. fp16 step graph (any machine)
python -m tapnet.tapnextpp.deploy.export_onnx --checkpoint tapnextpp_ckpt.pt \
    --resolution 256 --num_queries 64 --precision fp16 \
    --output r256_q64_fp16.onnx --verify_frames 0

# 2. FP8 post-training quantization (modelopt venv, torch/lib on PATH)
python -m tapnet.tapnextpp.deploy.quantize_fp8 \
    --checkpoint tapnextpp_ckpt.pt --davis_pkl tapvid_davis/tapvid_davis.pkl \
    --onnx r256_q64_fp16.onnx --output r256_q64_fp8.onnx \
    --calibration_eps "cuda:0 cpu"

# 3. Engine on the target GPU (3 GiB workspace fits the 4 GB card)
python -c "from tapnet.tapnextpp.deploy import trt_runner; \
    trt_runner.build_engine('r256_q64_fp8.onnx', 'r256_q64_fp8.plan', 3, 3.0)"
```

38.9 fps, 26.7 ms p95, AJ 66.42 (-0.44 vs fp16), 197 MB engine (the fp16 one
is 380 MB). Builder
optimization level 5 is worth 0.5% over level 3 for 3x the build time (99 s vs
28 s) — not worth it. Peak VRAM during `eval_davis.py --engine` is about
2.4 GB, because the fp32 PyTorch model is still loaded for the point tokenizer;
on this card that leaves no room to run two variants in one process.

## Why not 60 fps

60 fps is 16.7 ms per frame. The best engine is 25.6 ms, so the shortfall is
1.54x. It is not a tuning problem:

1. **The fp16 engine already runs at 73% of this GPU's fp16 GEMM ceiling**
   (12.3 of 16.9 effective TFLOPS). There is no headroom left in fp16.
2. **FP8 only accelerates the GEMMs, and they are 61% of the frame.** Of the
   25.6 ms FP8 frame, about 12.4 ms is GEMM, 2.9 ms attention and 10.4 ms
   norms, RG-LRU recurrence, state concat/slice and elementwise work that stays
   in fp16/fp32. Even with the GEMMs at the measured 58 TFLOPS FP8 roofline
   (7.2 ms) the frame is 20.4 ms = 49 fps, and with *free* GEMMs it is 13.2 ms
   = 76 fps. 60 fps sits inside that narrow band and below what the memory
   traffic of the state allows.
3. **The state is 95 MiB per frame.** The RG-LRU state is 12 x 1088 x 768 fp32
   (38 MiB) and the conv state 12 x 1088 x 3 x 768 fp16 (57 MiB); each frame
   reads and writes both. That is roughly 1-2 ms on this GPU's memory bus and
   it does not shrink with quantization.
4. **The chassis is the real limit.** 30 W holds the SM clock at 1395-1560 MHz
   against a 3105 MHz maximum. The same silicon at full clock would be about
   1.5x faster, which still would not reach 60 fps, and the power limit cannot
   be raised from software here.

The levers that remain all cost accuracy or need training: 224x224 input drops
AJ to about 43 (already measured), and fewer image tokens or a smaller backbone
would need fine-tuning. Reaching 60 fps on an unmodified TAPNext-B at 256x256
needs roughly 3x this GPU's sustained tensor throughput and about 2x its
memory bandwidth.

## Where the time goes

`tensorrt.IProfiler` over 50-60 frames, grouped by fused-kernel family (the
profiler synchronizes per kernel, so the sum exceeds the wall time by about 26%
and over-weights small kernels; shares are relative within one column).

| group | fp16 engine | FP8 engine |
|---|---:|---:|
| linear layers (`node_linear*`, `__mye*` GEMM kernels) | 27.9 ms, 60.7% | 15.7 ms, 48.5% |
| attention (`_gemm_mha_v2`, 12 kernels, fp16 in both) | 3.59 ms, 7.8% | 3.62 ms, 11.2% |
| state concat / slice / transpose | 5.17 ms, 11.2% | 5.22 ms, 16.1% |
| LayerNorm + RMSNorm | 2.66 ms, 5.8% | 2.27 ms, 7.0% |
| RG-LRU recurrence | 2.13 ms, 4.6% | 1.94 ms, 6.0% |
| other elementwise (GELU, gating, temporal conv) | 3.03 ms, 6.6% | 2.22 ms, 6.9% |
| heads, patch conv, tokenizer-side matmuls | 1.47 ms, 3.2% | 1.43 ms, 4.4% |
| sum of profiled kernels | 45.95 ms | 32.40 ms |
| real wall time | 37.58 ms | 25.61 ms |

FP8 cut the GEMM time by 1.78x and left everything else untouched, which is
exactly the 1.46x end-to-end speedup observed. The attention matmuls and the
RG-LRU gate matmuls were deliberately left in fp16 (see below); quantizing them
would address at most another 8% of the frame.

## Script fixes

Each is a separate commit on `ccr-e3e60b5e-ir9pr0`.

1. **`export_onnx.py`: survive a non-UTF-8 console.** `torch.onnx` prints
   status glyphs, which raises `UnicodeEncodeError` on a cp1252 Windows console
   after the export has written nothing. `PYTHONUTF8=1` also works as a
   workaround.
2. **`quantize_fp8.py`: `torch.cuda.empty_cache()` before the ModelOpt child.**
   `calibration_data()` leaves about 2.4 GB in the caching allocator while
   ModelOpt calibrates on the same 4 GB GPU.
3. **`quantize_fp8.py`: make the FP8 graph usable at all.** Three independent
   defects, each enough on its own to produce an engine whose tracks are 240 px
   off from the first frame:
   * `--use_external_data_format` corrupts initializers on write. With ModelOpt
     0.47.0, three of the 438 tensors of the 256 px / 64 query graph came back
     wrong (`l2_up_b`, `l4_qkv_b`, `l5_lin_x_b`, errors up to 5e4) even with
     quantization reduced to a single node. The 394 MB step graph fits in one
     ONNX file, so the flag is now only passed above the 2 GiB protobuf limit,
     and `_check_weights_survived()` compares the pass-through initializers and
     raises rather than handing over a broken graph.
   * `--high_precision_dtype fp16` casts down the part of the step graph that
     is fp32 on purpose. The RG-LRU input (`xc`) reaches 7e4 per tensor, past
     the fp16 range, so the recurrence overflows. The default is now fp32 and
     the value is a flag.
   * Quantizing every op puts FP8 Q/DQ on the residual stream (amax up to
     6.1e4, feeding LayerNorm and RMSNorm) and on the RG-LRU gate matmuls
     (amax 7.0e4). Only `Gemm` is quantized now, which is exactly what
     `eval_davis.py --fp8_sim` simulates; `--op_types_to_quantize` and
     `--nodes_to_exclude` expose the rest.

   `--disable_mha_qdq` was not needed: ModelOpt reports "Found 0 MHA (QK_AV)
   Patterns" for this graph and never quantizes the attention matmuls.

Not changed, but measured: enqueueing the engine on a dedicated CUDA stream
instead of the default one (TensorRT warns about the extra
`cudaStreamSynchronize` calls) makes no difference once the clocks are warm —
25.49 ms default vs 25.31 ms dedicated, median of 5 interleaved reps of 120
frames. It only shortens the cold-start phase. `trt_runner.py` was left alone.

Everything else in `benchmark.py`, `trt_runner.py`, `optimize_local.py` and
`eval_davis.py --engine` ran correctly on first contact with the GPU, including
`build_engine` against TensorRT 11, where `BuilderFlag.FP16` and
`platform_has_fast_fp16` no longer exist and strongly typed networks are the
only mode.
