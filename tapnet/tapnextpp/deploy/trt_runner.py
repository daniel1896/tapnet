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

"""Runs a TensorRT engine built from `export_onnx.py` output.

Build the engine first, e.g. (TensorRT 10.x and 11.x):

  trtexec --onnx=tapnextpp_256_q256_fp16.onnx \
      --saveEngine=tapnextpp_256_q256_fp16.plan --stronglyTyped

Usage:

  runner = TRTStepRunner('tapnextpp_256_q256_fp16.plan')
  tokenizer = PointTokenizer.load('tapnextpp_256_q256_fp16.tokens.npz').cuda()
  runner.reset()
  for t, frame in enumerate(frames):  # [1, H, W, 3] float32 in [-1, 1], cuda
    tracks, visible_logits = runner(frame, tokenizer(queries, t))

The recurrent state never leaves the GPU: two state buffers are swapped
(ping-pong) every frame, so no state copies are needed.
"""

import tensorrt as trt
import torch

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
}


def build_engine(
    onnx_path: str,
    plan_path: str,
    optimization_level: int = 3,
    workspace_gib: float = 4.0,
) -> None:
  """Builds a strongly typed engine (same as `trtexec --stronglyTyped`).

  Needs only the `tensorrt` Python package (pip wheels do not ship trtexec).
  Build on the GPU that will run the engine.

  Args:
    onnx_path: ONNX file from export_onnx.py (external weights next to it).
    plan_path: Where to write the serialized engine.
    optimization_level: TensorRT builder optimization level (0-5); higher
      searches more kernels and builds slower.
    workspace_gib: Builder workspace limit.
  """
  logger = trt.Logger(trt.Logger.WARNING)
  builder = trt.Builder(logger)
  flags = 0
  strongly_typed = getattr(
      trt.NetworkDefinitionCreationFlag, 'STRONGLY_TYPED', None)
  if strongly_typed is not None:  # Default (and possibly removed) in TRT 11.
    flags |= 1 << int(strongly_typed)
  network = builder.create_network(flags)
  parser = trt.OnnxParser(network, logger)
  if not parser.parse_from_file(onnx_path):
    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    raise RuntimeError(f'ONNX parse failed: {errors}')
  config = builder.create_builder_config()
  config.set_memory_pool_limit(
      trt.MemoryPoolType.WORKSPACE, int(workspace_gib * 2**30))
  config.builder_optimization_level = optimization_level
  engine = builder.build_serialized_network(network, config)
  if engine is None:
    raise RuntimeError('TensorRT engine build failed (see log above).')
  with open(plan_path, 'wb') as f:
    f.write(engine)


class TRTStepRunner:
  """Thin wrapper around a TAPNextStep TensorRT engine."""

  def __init__(self, engine_path: str, device: str = 'cuda'):
    self._logger = trt.Logger(trt.Logger.WARNING)
    self._runtime = trt.Runtime(self._logger)
    with open(engine_path, 'rb') as f:
      self.engine = self._runtime.deserialize_cuda_engine(f.read())
    if self.engine is None:
      raise RuntimeError(f'Could not deserialize {engine_path}.')
    self.context = self.engine.create_execution_context()
    self.device = torch.device(device)

    def alloc(name):
      shape = tuple(self.engine.get_tensor_shape(name))
      dtype = _TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]
      return torch.zeros(shape, dtype=dtype, device=self.device)

    self.frame = alloc('frame')
    self.point_tokens = alloc('point_tokens')
    self.reset_flag = alloc('reset')
    self.tracks = alloc('tracks')
    self.visible_logits = alloc('visible_logits')
    self.rg = [alloc('rg_state'), alloc('rg_state')]
    self.conv = [alloc('conv_state'), alloc('conv_state')]
    for name, tensor in (
        ('frame', self.frame),
        ('point_tokens', self.point_tokens),
        ('reset', self.reset_flag),
        ('tracks', self.tracks),
        ('visible_logits', self.visible_logits),
    ):
      self.context.set_tensor_address(name, tensor.data_ptr())
    self.reset()

  @property
  def num_queries(self) -> int:
    return self.point_tokens.shape[1]

  def reset(self) -> None:
    """Starts a new sequence: zero state and reset flag on the next frame."""
    for buf in self.rg + self.conv:
      buf.zero_()
    self.reset_flag.fill_(1.0)
    self._src = 0

  @torch.no_grad()
  def __call__(self, frame: torch.Tensor, point_tokens: torch.Tensor):
    """Advances one frame.

    Args:
      frame: [1, H, W, 3] float32 in [-1, 1] on the GPU.
      point_tokens: [1, Q, C] float32 from `PointTokenizer`.

    Returns:
      tracks [1, Q, 2] (y, x, 256x256 model space) and visible_logits
      [1, Q, 1]. Both are views of internal buffers that the next call
      overwrites; clone them if you need to keep them.
    """
    self.frame.copy_(frame)
    self.point_tokens.copy_(point_tokens)
    src, dst = self._src, 1 - self._src
    ctx = self.context
    ctx.set_tensor_address('rg_state', self.rg[src].data_ptr())
    ctx.set_tensor_address('conv_state', self.conv[src].data_ptr())
    ctx.set_tensor_address('rg_state_out', self.rg[dst].data_ptr())
    ctx.set_tensor_address('conv_state_out', self.conv[dst].data_ptr())
    if not ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
      raise RuntimeError('TensorRT execution failed.')
    self.reset_flag.zero_()  # Stream-ordered after the launch above.
    self._src = dst
    return self.tracks, self.visible_logits
