# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Benchmarks online (frame-by-frame) TAPNext++ inference on a CUDA GPU.

  python -m tapnet.tapnextpp.deploy.benchmark --checkpoint tapnextpp_ckpt.pt \
      --resolution 256 --num_queries 256 \
      [--engine tapnextpp_256_q256_fp16.plan] [--fp16_accumulation]

Modes (--modes, comma separated; failures are reported and skipped):
  ref                 TAPNext.forward, eager, autocast fp16 (votsp2026 wrapper).
  ref_compile         TAPNext.forward + torch.compile (the papers' setting).
  step                TAPNextStep fp16, eager.
  step_graph          TAPNextStep fp16 replayed as one CUDA graph per frame.
  step_compile_graph  torch.compile(max-autotune) TAPNextStep in a CUDA graph.
  trt                 TensorRT engine via trt_runner (added when --engine set).

Each frame is timed until its tracks are on the host, i.e. the latency an
online application sees. Accuracy is reported as the deviation from the fp32
reference model over the first --check_frames frames of a synthetic panning
video.
"""

import argparse
import json
import subprocess
import time

import numpy as np
from tapnet.tapnextpp.deploy import export_onnx
from tapnet.tapnextpp.deploy import step_model
import torch

ALL_MODES = ('ref', 'ref_compile', 'step', 'step_graph', 'step_compile_graph')


def gpu_info():
  props = torch.cuda.get_device_properties(0)
  info = (
      f'{props.name}: {props.multi_processor_count} SMs, '
      f'{props.total_memory / 2**30:.1f} GiB, torch {torch.__version__}'
  )
  try:
    query = 'clocks.sm,clocks.max.sm,power.draw,power.limit,pstate'
    out = subprocess.run(
        ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader'],
        capture_output=True, text=True, timeout=10, check=True,
    ).stdout.strip()
    info += f'\n  nvidia-smi ({query}): {out}'
  except (OSError, subprocess.SubprocessError):
    pass
  return info


class Bench:
  """Holds the model, video, queries and the fp32 reference tracks."""

  def __init__(self, args):
    self.args = args
    self.size = (args.resolution, args.resolution)
    self.model = step_model.load_tapnext(args.checkpoint, 'cuda')
    for p in self.model.parameters():
      p.requires_grad_(False)
    self.tokenizer = step_model.PointTokenizer.from_model(self.model).cuda()
    self.video = export_onnx.synthetic_video(
        args.video_frames, args.resolution).cuda()
    self.queries = export_onnx.grid_queries(args.num_queries).cuda()
    self.tokens = [self.tokenizer(self.queries, 0), self.tokenizer(self.queries, 1)]
    self.reference = self._fp32_reference()

  def frame(self, t):
    return self.video[:, t % self.video.shape[1]]

  def tokens_for(self, t):
    return self.tokens[0 if t == 0 else 1]

  @torch.no_grad()
  def _fp32_reference(self):
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    run = self.make_ref(autocast=False)
    tracks = [run(t)[0].float().cpu() for t in range(self.args.check_frames)]
    torch.backends.cuda.matmul.allow_tf32 = tf32
    return tracks

  def make_ref(self, autocast=True, compile_model=False):
    fwd = torch.compile(self.model) if compile_model else self.model
    state = None

    def run(t):
      nonlocal state
      if t == 0:
        state = None
      frame = self.frame(t)[:, None]
      with torch.no_grad(), torch.autocast(
          'cuda', dtype=torch.float16, enabled=autocast):
        if state is None:
          tracks, _, vis, state = fwd(video=frame, query_points=self.queries)
        else:
          tracks, _, vis, state = fwd(video=frame, state=state)
      return tracks[:, 0], vis[:, 0]

    return run

  def make_step(self):
    step = step_model.TAPNextStep(
        self.model, self.size, self.args.num_queries, torch.float16).cuda()
    state = None

    def run(t):
      nonlocal state
      if t == 0:
        state = step.init_state('cuda')
      reset = torch.full((1,), float(t == 0), device='cuda')
      with torch.no_grad():
        tracks, vis, *state = step(
            self.frame(t), self.tokens_for(t), reset, *state)
      return tracks, vis

    return run

  def make_step_graph(self, compile_model=False):
    step = step_model.TAPNextStep(
        self.model, self.size, self.args.num_queries, torch.float16).cuda()
    fn = step
    if compile_model:
      fn = torch.compile(step, mode='max-autotune-no-cudagraphs')
    frame = self.frame(0).clone()
    tokens = self.tokens[1].clone()
    reset = torch.zeros(1, device='cuda')
    rg, conv = step.init_state('cuda')
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.no_grad():
      for _ in range(3):  # Warm up (and compile) outside of graph capture.
        fn(frame, tokens, reset, rg, conv)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
      tracks, vis, new_rg, new_conv = fn(frame, tokens, reset, rg, conv)
      rg.copy_(new_rg)
      conv.copy_(new_conv)

    def run(t):
      if t == 0:
        rg.zero_()
        conv.zero_()
      frame.copy_(self.frame(t))
      tokens.copy_(self.tokens_for(t))
      reset.fill_(float(t == 0))
      graph.replay()
      return tracks, vis

    return run

  def make_trt(self):
    from tapnet.tapnextpp.deploy import trt_runner  # pylint: disable=g-import-not-at-top

    runner = trt_runner.TRTStepRunner(self.args.engine)
    if runner.num_queries != self.args.num_queries:
      raise ValueError(
          f'Engine has {runner.num_queries} queries, '
          f'--num_queries is {self.args.num_queries}.')

    def run(t):
      if t == 0:
        runner.reset()
      return runner(self.frame(t), self.tokens_for(t))

    return run

  def measure(self, run):
    args = self.args
    times, deviations = [], []
    for t in range(args.warmup + args.frames):
      start = time.perf_counter()
      tracks, vis = run(t)
      tracks_host = tracks.float().cpu()  # Synchronizes, like a real app.
      vis.cpu()
      elapsed = time.perf_counter() - start
      if t >= args.warmup:
        times.append(elapsed)
      if t < args.check_frames:
        deviations.append((tracks_host - self.reference[t]).abs())
    times_ms = 1000 * np.array(times)
    dev = torch.stack(deviations)
    return {
        'mean_ms': times_ms.mean(),
        'p50_ms': np.percentile(times_ms, 50),
        'p95_ms': np.percentile(times_ms, 95),
        'fps': 1000 / times_ms.mean(),
        'max_dev_px': dev.max().item(),
        'mean_dev_px': dev.mean().item(),
    }


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--resolution', type=int, default=256)
  parser.add_argument('--num_queries', type=int, default=256)
  parser.add_argument('--modes', default=','.join(ALL_MODES))
  parser.add_argument('--engine', default=None, help='TensorRT .plan file.')
  parser.add_argument('--frames', type=int, default=300)
  parser.add_argument('--warmup', type=int, default=30)
  parser.add_argument('--check_frames', type=int, default=24)
  parser.add_argument('--video_frames', type=int, default=48)
  parser.add_argument(
      '--fp16_accumulation', action='store_true',
      help='Allow fp16 accumulation in cuBLAS (2x tensor-core peak on '
      'GeForce, lower precision). Affects the PyTorch modes only.')
  parser.add_argument('--json_out', default=None,
                      help='Also write {mode: results} to this JSON file.')
  args = parser.parse_args()
  if args.check_frames > args.warmup + args.frames:
    parser.error('--check_frames must not exceed --warmup + --frames.')

  if args.fp16_accumulation:
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True

  print(gpu_info())
  gflops = step_model.flops_per_frame(
      (args.resolution // 8) ** 2, args.num_queries) / 1e9
  print(f'{args.resolution}x{args.resolution}, {args.num_queries} queries, '
        f'~{gflops:.0f} GFLOPs/frame (60 fps needs ~{gflops * 60 / 1e3:.1f} '
        'effective TFLOPS)')
  bench = Bench(args)

  modes = [m for m in args.modes.split(',') if m]
  if args.engine and 'trt' not in modes:
    modes.append('trt')
  makers = {
      'ref': bench.make_ref,
      'ref_compile': lambda: bench.make_ref(compile_model=True),
      'step': bench.make_step,
      'step_graph': bench.make_step_graph,
      'step_compile_graph': lambda: bench.make_step_graph(compile_model=True),
      'trt': bench.make_trt,
  }
  print(f'\n{"mode":<20}{"mean ms":>9}{"p50 ms":>9}{"p95 ms":>9}{"FPS":>8}'
        f'{"max dev px":>12}{"mean dev px":>13}')
  results = {}
  for mode in modes:
    try:
      result = bench.measure(makers[mode]())
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'{mode:<20} FAILED: {type(e).__name__}: {str(e)[:200]}')
      results[mode] = {'error': f'{type(e).__name__}: {str(e)[:500]}'}
      continue
    results[mode] = {k: float(v) for k, v in result.items()}
    print(f'{mode:<20}{result["mean_ms"]:>9.2f}{result["p50_ms"]:>9.2f}'
          f'{result["p95_ms"]:>9.2f}{result["fps"]:>8.1f}'
          f'{result["max_dev_px"]:>12.3f}{result["mean_dev_px"]:>13.4f}')
    torch.cuda.empty_cache()
  if args.json_out:
    with open(args.json_out, 'w') as f:
      json.dump({'gpu': gpu_info(), 'gflops_per_frame': gflops,
                 'modes': results}, f, indent=1)


if __name__ == '__main__':
  main()
