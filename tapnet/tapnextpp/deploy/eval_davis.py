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

"""Evaluates deployment variants of TAPNext++ on TAP-Vid DAVIS (query first).

  python -m tapnet.tapnextpp.deploy.eval_davis --checkpoint tapnextpp_ckpt.pt \
      --davis_pkl tapvid_davis/tapvid_davis.pkl --resolution 256 \
      [--precision fp16] [--fp8_sim] [--pad_queries 256] \
      [--engine tapnextpp_256_q256_fp16.plan] [--device cuda] [--out res.json]

Tracking is online and joint over all queries of a video, as in
colabs/torch_tapnextpp_demo.ipynb (visibility = logit > 0). Metrics are
computed in 256x256 coordinates whatever the input --resolution.

Variants:
  --resolution     Model input size (frames are resized to it). Fewer image
                   tokens is the biggest speed lever; only 256 (and 512 for the
                   512 checkpoint) were trained.
  --fp8_sim        Fake-quantizes every linear layer's input and weight to FP8
                   E4M3 with per-tensor scales (what a TensorRT FP8 engine does
                   with dynamic-range scales); attention and norms unchanged.
  --pad_queries    Pads each video to a fixed number of query slots with
                   "unknown" tokens, as a fixed-shape engine requires.
  --engine         Runs a TensorRT engine instead of PyTorch (implies padding
                   to the engine's query count; resolution must match).
  --onnx           Same for an ONNX graph in onnxruntime. The FP8 graph from
                   quantize_fp8.py only runs in TensorRT (use --engine).
"""

import argparse
import importlib
import json
import os
import sys
import time
import types
import typing

import numpy as np
from tapnet.tapnextpp.deploy import step_model
import torch
from torch.nn import functional as F

_UNUSED_QUERY_T = 1e9  # Query slot that never starts -> "unknown" token.


class _StubModule(types.ModuleType):

  def __getattr__(self, name):
    return typing.Any


def _evaluation_datasets():
  """Imports tapvid.evaluation_datasets without TensorFlow / chex installed.

  The functions used here (resize, query sampling, metrics) are numpy only.

  Returns:
    The tapnet.tapvid.evaluation_datasets module.
  """
  for name in ('tensorflow', 'tensorflow_datasets', 'chex'):
    try:
      importlib.import_module(name)
    except ImportError:
      sys.modules[name] = _StubModule(name)
  return importlib.import_module('tapnet.tapvid.evaluation_datasets')


def fp8_qdq(x: torch.Tensor) -> torch.Tensor:
  """Per-tensor FP8 E4M3 quantize-dequantize."""
  amax = x.detach().abs().amax().float().clamp(min=1e-12)
  scale = amax / 448.0
  q = (x.float() / scale).to(torch.float8_e4m3fn).float() * scale
  return q.to(x.dtype)


class FP8SimStep(step_model.TAPNextStep):
  """TAPNextStep whose linear layers see FP8-rounded inputs and weights."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._qweights = {}

  def _linear(self, x, w, b):
    key = w.data_ptr()
    if key not in self._qweights:
      self._qweights[key] = fp8_qdq(w)
    return F.linear(fp8_qdq(x.to(self.compute_dtype)), self._qweights[key], b)


def load_davis(path, eval_size=256):
  """Yields (name, frames_uint8, sample) with sample coords at eval_size."""
  ed = _evaluation_datasets()
  import pickle  # pylint: disable=g-import-not-at-top

  with open(path, 'rb') as f:
    data = pickle.load(f)
  for name, item in data.items():
    points = item['points'] * np.array([eval_size, eval_size])
    sample = ed.sample_queries_first(
        item['occluded'], points, np.zeros((points.shape[1], 1, 1, 3)))
    yield name, item['video'], sample


def make_runner(args, model, device):
  """Returns run(frame [1,H,W,3], tokens [1,Q,C], t) -> (tracks, vis_logits)."""
  size = (args.resolution, args.resolution)
  if args.onnx:
    import onnxruntime as ort  # pylint: disable=g-import-not-at-top

    sess = ort.InferenceSession(
        args.onnx, providers=ort.get_available_providers())
    inputs = {i.name: i for i in sess.get_inputs()}
    conv_dtype = (np.float16 if 'float16' in inputs['conv_state'].type
                  else np.float32)
    state = []

    def run_ort(frame, tokens, t):
      if t == 0:
        state[:] = [np.zeros(inputs['rg_state'].shape, np.float32),
                    np.zeros(inputs['conv_state'].shape, conv_dtype)]
      tracks, vis, *new_state = sess.run(None, {
          'frame': frame.cpu().numpy(),
          'point_tokens': tokens.cpu().numpy(),
          'reset': np.array([float(t == 0)], np.float32),
          'rg_state': state[0],
          'conv_state': state[1],
      })
      state[:] = new_state
      return torch.from_numpy(tracks), torch.from_numpy(vis)

    return run_ort, inputs['point_tokens'].shape[1]

  if args.engine:
    from tapnet.tapnextpp.deploy import trt_runner  # pylint: disable=g-import-not-at-top

    runner = trt_runner.TRTStepRunner(args.engine)

    def run_trt(frame, tokens, t):
      if t == 0:
        runner.reset()
      tracks, vis = runner(frame, tokens)
      return tracks.clone(), vis.clone()

    return run_trt, runner.num_queries

  dtype = torch.float16 if args.precision == 'fp16' else torch.float32
  cls = FP8SimStep if args.fp8_sim else step_model.TAPNextStep
  step = cls(model, size, args.pad_queries or 1, dtype).to(device).eval()
  state = []

  def run_torch(frame, tokens, t):
    if t == 0:
      state[:] = step.init_state(device, num_queries=tokens.shape[1])
    reset = torch.full((1,), float(t == 0), device=device)
    tracks, vis, *new_state = step(frame, tokens, reset, *state)
    state[:] = new_state
    return tracks, vis

  return run_torch, args.pad_queries


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--davis_pkl', required=True)
  parser.add_argument('--resolution', type=int, default=256)
  parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
  parser.add_argument('--fp8_sim', action='store_true')
  parser.add_argument('--pad_queries', type=int, default=0)
  parser.add_argument('--engine', default=None)
  parser.add_argument('--onnx', default=None,
                      help='Run an exported graph in onnxruntime instead.')
  parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                      else 'cpu')
  parser.add_argument('--videos', default='',
                      help='Comma-separated video indices (default: all).')
  parser.add_argument('--out', default=None,
                      help='JSON file; per-video results are appended and '
                      'videos already in it are skipped.')
  args = parser.parse_args()

  ed = _evaluation_datasets()
  device = torch.device(args.device)
  model = step_model.load_tapnext(args.checkpoint, device)
  for p in model.parameters():
    p.requires_grad_(False)
  tokenizer = step_model.PointTokenizer.from_model(model).to(device)
  run, pad_to = make_runner(args, model, device)

  done = {}
  if args.out and os.path.exists(args.out):
    with open(args.out) as f:
      done = json.load(f)
  wanted = {int(v) for v in args.videos.split(',') if v}

  for index, (name, video, sample) in enumerate(load_davis(args.davis_pkl)):
    if (wanted and index not in wanted) or name in done:
      continue
    start = time.time()
    frames = ed.resize_video(video, (args.resolution, args.resolution))
    frames = torch.from_numpy(frames).to(device).float() / 127.5 - 1.0
    queries = torch.from_numpy(sample['query_points']).float().to(device)
    num_real = queries.shape[1]
    if pad_to:
      if pad_to < num_real:
        raise ValueError(f'{name}: {num_real} queries > {pad_to} slots.')
      pad = torch.zeros(1, pad_to - num_real, 3, device=device)
      pad[..., 0] = _UNUSED_QUERY_T
      queries = torch.cat([queries, pad], dim=1)

    tracks, visible = [], []
    with torch.inference_mode():
      for t in range(frames.shape[0]):
        tr, vis = run(frames[t : t + 1], tokenizer(queries, t), t)
        tracks.append(tr[0, :num_real].float().cpu())
        visible.append(vis[0, :num_real, 0].float().cpu() > 0)
    tracks = torch.stack(tracks, dim=1).numpy()[None]  # [1, Q, T, 2] (y, x)
    occluded = ~torch.stack(visible, dim=1).numpy()[None]
    scalars = ed.compute_tapvid_metrics(
        sample['query_points'],
        sample['occluded'],
        sample['target_points'],
        occluded,
        tracks[..., ::-1],
        query_mode='first',
    )
    done[name] = {k: float(np.mean(v)) for k, v in scalars.items()}
    secs = time.time() - start
    print(f'[{index:2d}] {name:<20} AJ {done[name]["average_jaccard"]:.3f} '
          f'pts {done[name]["average_pts_within_thresh"]:.3f} '
          f'OA {done[name]["occlusion_accuracy"]:.3f} '
          f'({frames.shape[0]} frames, {secs:.0f}s)', flush=True)
    if args.out:
      with open(args.out, 'w') as f:
        json.dump(done, f, indent=1)

  keys = ('average_jaccard', 'average_pts_within_thresh', 'occlusion_accuracy')
  means = {k: np.mean([v[k] for v in done.values()]) for k in keys}
  print(f'{len(done)} videos: AJ {100 * means[keys[0]]:.2f}  '
        f'delta_avg {100 * means[keys[1]]:.2f}  OA {100 * means[keys[2]]:.2f}')


if __name__ == '__main__':
  main()
