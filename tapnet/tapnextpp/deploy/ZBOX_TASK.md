# Task for a Claude Code session on the ZBox

Start Claude Code in this repo on the ZBox (Claude Desktop app, or
`claude remote-control` in the repo folder) and say:
"Do the task in tapnet/tapnextpp/deploy/ZBOX_TASK.md".

## Goal
Get TAPNext++ online point tracking to >= 60 fps on this machine's NVIDIA GPU
while keeping TAP-Vid DAVIS accuracy, and report measured numbers. The GPU is
reportedly an RTX 5070 Laptop/Mobile; confirm with `nvidia-smi`.

## Setup
1. `git fetch origin ccr-e3e60b5e-ir9pr0 && git checkout ccr-e3e60b5e-ir9pr0`.
   Read `tapnet/tapnextpp/deploy/README.md` first.
2. Python env with CUDA PyTorch that supports Blackwell (sm_120), then
   `pip install -e ".[torch]" onnx onnxscript onnxruntime tensorrt mediapy scipy absl-py einops`.
   For FP8, also `pip install "nvidia-modelopt[onnx]"`, ideally in a separate
   venv because it pulls onnxruntime-gpu.
3. Record `nvidia-smi`: GPU model, power limit, clocks. Run on AC power at
   maximum performance. A desktop RTX 5070 (48 SMs, 12 GB) instead of the
   Laptop part (36 SMs, 8 GB) changes expectations; note which it is.

## Known (measured on CPU in a cloud session)
- `TAPNextStep` matches `TAPNext.forward` to 6e-5 px. The fp16 ONNX export
  matches fp32 to 0.002 px.
- DAVIS (query-first, 30 videos, `tapnextpp_ckpt.pt`, 256x256):
  - fp32: AJ 66.59 / delta_avg 79.94 / OA 92.12.
  - Simulated FP8 linear layers: AJ 66.31.
  - Padding to 64 query slots: AJ 66.60.
  - 224x224 input: AJ about 43. Lower resolution is NOT an option, and the
    512 checkpoint costs about 4x the compute.
- Compute: about 462 GFLOPs/frame (256 px, 64 queries) or 553 (256 queries).
  On a 5070 Laptop (about 43 TFLOPS fp16 peak with fp32 accumulation), fp16
  TensorRT is expected at roughly 35-50 fps. FP8 roughly doubles the peak.
- Never run on a GPU yet: `benchmark.py`, `trt_runner.py` (incl.
  `build_engine`), `optimize_local.py`, `eval_davis.py --engine`. Fix small
  bugs minimally and commit them.
- The ModelOpt FP8 ONNX graph does not run in onnxruntime. Validate it
  through a TensorRT engine only.

## Steps
1. fp16 sweep:
   `python -m tapnet.tapnextpp.deploy.optimize_local --workdir tapnextpp_opt --resolutions 256 --queries 64,256`
   This writes `tapnextpp_opt/report.md`. Engine DAVIS AJ should be within
   about 0.1 of 66.6.
2. If below 60 fps, try FP8:
   - Quantize:
     `python -m tapnet.tapnextpp.deploy.quantize_fp8 --checkpoint tapnextpp_opt/tapnextpp_ckpt.pt --davis_pkl tapnextpp_opt/tapvid_davis/tapvid_davis.pkl --onnx tapnextpp_opt/r256_q64_fp16.onnx --output tapnextpp_opt/r256_q64_fp8.onnx --calibration_eps "cuda:0 cpu"`
     Calibration data is about 60 MB per sample.
   - Build the engine with `trt_runner.build_engine`, or with
     `trtexec --stronglyTyped`.
   - Measure fps with `benchmark.py --engine` and accuracy with
     `eval_davis.py --engine --videos 4,5,...,29`. Videos 0-3 were used for
     calibration.
   - If AJ drops by more than about 0.5, retry with `--disable_mha_qdq` or
     exclude the RG-LRU gate MatMuls and Add/LayerNorm inputs from
     quantization.
3. Also try `benchmark.py --modes step_graph --fp16_accumulation` and note
   fps and the deviation column.
4. If still below 60 fps, profile the engine
   (`trtexec --loadEngine=... --dumpProfile --separateProfileRun`) and report
   where the time goes.

## Done means
Commit `tapnet/tapnextpp/deploy/RESULTS_<gpu>.md` containing:
- the GPU details;
- fps, p95 latency and DAVIS AJ for every variant tried;
- the recommended >= 60 fps configuration, or the best achievable with an
  explanation;
- any bug fixes.

Push to `ccr-e3e60b5e-ir9pr0`.
