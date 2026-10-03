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

"""FP8 post-training quantization of an exported TAPNext++ step graph.

  python -m tapnet.tapnextpp.deploy.quantize_fp8 \
      --checkpoint tapnextpp_ckpt.pt --davis_pkl tapvid_davis/tapvid_davis.pkl \
      --onnx r256_q64_fp16.onnx --output r256_q64_fp8.onnx

Calibration inputs are real video frames together with the recurrent state the
model actually has at that point (from running the PyTorch step model), so the
state inputs see realistic ranges. By default the first --calib_videos DAVIS
videos are used; evaluate on the others to avoid calibrating on test data.

Quantization itself is done by NVIDIA TensorRT Model Optimizer
(`pip install "nvidia-modelopt[onnx]"`), which inserts FP8 Q/DQ nodes that
TensorRT turns into FP8 GEMMs. Build the result like any other step graph
(trt_runner.build_engine / trtexec --stronglyTyped) and check it with
`eval_davis.py --engine`; onnxruntime cannot run it (ModelOpt places some
FP8 tensors on Transpose nodes, which only TensorRT accepts).
`eval_davis.py --fp8_sim` estimates the accuracy cost without a GPU.

Calibration data is large (about 60 MB per sample at 256 px / 64 queries,
because the recurrent state is part of the input).
"""

import argparse
import subprocess
import sys

import numpy as np
import onnx
from tapnet.tapnextpp.deploy import eval_davis
from tapnet.tapnextpp.deploy import step_model
import torch


def calibration_data(args, resolution, num_queries):
  """Returns {input name: array}, samples concatenated along axis 0."""
  ed = eval_davis._evaluation_datasets()  # pylint: disable=protected-access
  device = torch.device(args.device)
  model = step_model.load_tapnext(args.checkpoint, device)
  tokenizer = step_model.PointTokenizer.from_model(model).to(device)
  step = step_model.TAPNextStep(
      model, (resolution, resolution), num_queries, torch.float32
  ).to(device).eval()
  frame_ids = sorted(int(f) for f in args.calib_frames.split(','))
  samples = {k: [] for k in ('frame', 'point_tokens', 'reset', 'rg_state',
                             'conv_state')}
  for index, (name, video, sample) in enumerate(
      eval_davis.load_davis(args.davis_pkl)):
    if index >= args.calib_videos:
      break
    frames = ed.resize_video(video, (resolution, resolution))
    frames = torch.from_numpy(frames).to(device).float() / 127.5 - 1.0
    queries = torch.from_numpy(sample['query_points']).float().to(device)
    pad = torch.zeros(1, num_queries - queries.shape[1], 3, device=device)
    pad[..., 0] = eval_davis._UNUSED_QUERY_T  # pylint: disable=protected-access
    queries = torch.cat([queries, pad], dim=1)
    rg, conv = step.init_state(device)
    last = min(max(frame_ids), frames.shape[0] - 1)
    with torch.inference_mode():
      for t in range(last + 1):
        tokens = tokenizer(queries, t)
        reset = torch.full((1,), float(t == 0), device=device)
        if t in frame_ids or t == last:
          samples['frame'].append(frames[t : t + 1].cpu().numpy())
          samples['point_tokens'].append(tokens.cpu().numpy())
          samples['reset'].append(reset.cpu().numpy())
          samples['rg_state'].append(rg.cpu().numpy())
          samples['conv_state'].append(conv.cpu().numpy())
        _, _, rg, conv = step(frames[t : t + 1], tokens, reset, rg, conv)
    print(f'  calibration video {index} ({name}): frames <= {last}', flush=True)
  return {k: np.concatenate(v, axis=0) for k, v in samples.items()}


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--davis_pkl', required=True)
  parser.add_argument('--onnx', required=True, help='fp16 graph to quantize.')
  parser.add_argument('--output', required=True)
  parser.add_argument('--calib_videos', type=int, default=4)
  parser.add_argument('--calib_frames', default='0,1,4,12,24,40')
  parser.add_argument('--calibration_method', default='max',
                      choices=['max', 'entropy'])
  parser.add_argument('--calibration_eps', default='cpu',
                      help="e.g. 'cuda:0 cpu' or 'trt cuda:0 cpu'.")
  parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                      else 'cpu')
  args = parser.parse_args()

  graph = onnx.load(args.onnx, load_external_data=False)
  shapes = {
      i.name: [d.dim_value for d in i.type.tensor_type.shape.dim]
      for i in graph.graph.input
  }
  resolution = shapes['frame'][1]
  num_queries = shapes['point_tokens'][1]
  conv_is_fp16 = (graph.graph.input[4].type.tensor_type.elem_type ==
                  onnx.TensorProto.FLOAT16)
  print(f'{args.onnx}: {resolution}px, {num_queries} query slots')

  data = calibration_data(args, resolution, num_queries)
  if conv_is_fp16:
    data['conv_state'] = data['conv_state'].astype(np.float16)
  calib_path = args.output.removesuffix('.onnx') + '.calib.npz'
  np.savez(calib_path, **data)
  print(f'Wrote {calib_path} ({data["frame"].shape[0]} samples)')
  # ModelOpt calibrates in a child process on the same GPU; hand the VRAM that
  # the caching allocator still holds from calibration_data() back to it.
  if torch.cuda.is_available():
    torch.cuda.empty_cache()

  cmd = [
      sys.executable, '-m', 'modelopt.onnx.quantization',
      '--onnx_path', args.onnx,
      '--quantize_mode', 'fp8',
      '--calibration_method', args.calibration_method,
      '--calibration_data_path', calib_path,
      '--calibration_eps', *args.calibration_eps.split(),
      '--high_precision_dtype', 'fp16',
      '--use_external_data_format',
      '--output_path', args.output,
  ]
  print('$', ' '.join(cmd), flush=True)
  subprocess.run(cmd, check=True)


if __name__ == '__main__':
  main()
