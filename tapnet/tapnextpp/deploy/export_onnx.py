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

"""Exports one online TAPNext++ step to ONNX and verifies it.

Example (fp16 with fp32 recurrence, ready for TensorRT --stronglyTyped):

  python -m tapnet.tapnextpp.deploy.export_onnx \
      --checkpoint tapnextpp_ckpt.pt --resolution 256 --num_queries 256 \
      --precision fp16 --output tapnextpp_256_q256_fp16.onnx

Writes `<output>` (graph), `<output>.data` (weights) and
`<output minus .onnx>.tokens.npz` (the learned point tokens for
`step_model.PointTokenizer`). With --verify_frames > 0 the exported graph is
run in onnxruntime and compared against the reference
`tapnext_torch.TAPNext.forward` (fp32) on a synthetic moving video.

Requires: torch>=2.6, onnx, onnxscript; onnxruntime for verification.
"""

import argparse
import time

import numpy as np
from tapnet.tapnextpp.deploy import step_model
import torch
from torch.nn import functional as F

INPUT_NAMES = ['frame', 'point_tokens', 'reset', 'rg_state', 'conv_state']
OUTPUT_NAMES = ['tracks', 'visible_logits', 'rg_state_out', 'conv_state_out']


def synthetic_video(num_frames, size, shift=3, seed=0):
  """[1, T, H, W, 3] in [-1, 1]: a smooth texture panning `shift` px/frame."""
  gen = torch.Generator().manual_seed(seed)
  pad = 16 + shift * num_frames
  base = torch.rand(1, 3, (size + pad) // 8, (size + pad) // 8, generator=gen)
  base = F.interpolate(
      base, size=(size + pad, size + pad), mode='bicubic', align_corners=False
  ).clamp(0, 1)
  frames = [
      base[0, :, 8 : 8 + size, 8 + shift * t : 8 + shift * t + size]
      for t in range(num_frames)
  ]
  return (torch.stack(frames).permute(0, 2, 3, 1)[None] * 2 - 1).contiguous()


def grid_queries(num_queries, start_frame=0):
  """[1, Q, 3] (t, y, x) queries on a regular grid in 256x256 model space."""
  side = int(np.ceil(np.sqrt(num_queries)))
  coords = torch.linspace(24, 232, side)
  ys, xs = torch.meshgrid(coords, coords, indexing='ij')
  yx = torch.stack([ys.flatten(), xs.flatten()], -1)[:num_queries]
  t = torch.full((num_queries, 1), float(start_frame))
  return torch.cat([t, yx], -1)[None]


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--resolution', type=int, default=256)
  parser.add_argument('--num_queries', type=int, default=256)
  parser.add_argument('--precision', choices=['fp16', 'fp32'], default='fp16')
  parser.add_argument('--output', required=True)
  parser.add_argument('--opset', type=int, default=18)
  parser.add_argument('--verify_frames', type=int, default=3)
  args = parser.parse_args()

  dtype = torch.float16 if args.precision == 'fp16' else torch.float32
  model = step_model.load_tapnext(args.checkpoint)
  for p in model.parameters():
    p.requires_grad_(False)
  size = (args.resolution, args.resolution)
  step = step_model.TAPNextStep(model, size, args.num_queries, dtype).eval()
  print('Exporting:', step_model.describe(step))

  tokenizer = step_model.PointTokenizer.from_model(model)
  tokens_path = args.output.removesuffix('.onnx') + '.tokens.npz'
  tokenizer.save(tokens_path)

  queries = grid_queries(args.num_queries)
  example = (
      synthetic_video(1, args.resolution)[:, 0],
      tokenizer(queries, 0),
      torch.ones(1),
      *step.init_state(),
  )
  start = time.time()
  with torch.no_grad():
    torch.onnx.export(
        step,
        example,
        args.output,
        dynamo=True,
        opset_version=args.opset,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        external_data=True,
    )
  print(f'Wrote {args.output} (+ .data) and {tokens_path} '
        f'in {time.time() - start:.0f}s')

  if args.verify_frames > 0:
    verify(model, tokenizer, args)


def verify(model, tokenizer, args):
  """Runs the ONNX graph in onnxruntime against the reference model."""
  import onnxruntime as ort  # pylint: disable=g-import-not-at-top

  sess = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
  shapes = {i.name: i.shape for i in sess.get_inputs()}
  state_types = {i.name: i.type for i in sess.get_inputs()}
  conv_dtype = np.float16 if 'float16' in state_types['conv_state'] else np.float32
  rg = np.zeros(shapes['rg_state'], np.float32)
  conv = np.zeros(shapes['conv_state'], conv_dtype)

  video = synthetic_video(args.verify_frames, args.resolution)
  queries = grid_queries(args.num_queries)
  # Query/track coordinates live in 256x256 model space at any resolution.
  state = None
  with torch.no_grad():
    for t in range(args.verify_frames):
      frame = video[:, t : t + 1]
      if state is None:
        ref_tracks, _, ref_vis, state = model(video=frame, query_points=queries)
      else:
        ref_tracks, _, ref_vis, state = model(video=frame, state=state)
      ref_tokens = step_model.make_point_tokens(model, queries, t)
      tokens = tokenizer(queries, t)
      tracks, vis, rg, conv = sess.run(None, {
          'frame': frame[:, 0].numpy(),
          'point_tokens': tokens.numpy(),
          'reset': np.array([1.0 if t == 0 else 0.0], np.float32),
          'rg_state': rg,
          'conv_state': conv,
      })
      track_err = np.abs(tracks - ref_tracks[:, 0].numpy())
      vis_agree = ((vis > 0) == (ref_vis[:, 0].numpy() > 0)).mean()
      print(
          f'  frame {t}: |onnx - reference| tracks max {track_err.max():.4f}'
          f' px, mean {track_err.mean():.5f} px; visibility agreement'
          f' {100 * vis_agree:.1f}%; tokenizer max err'
          f' {(tokens - ref_tokens).abs().max():.2e}'
      )


if __name__ == '__main__':
  main()
