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

"""Export-friendly single-frame step of TAPNext / TAPNext++.

`tapnext_torch.TAPNext.forward` is written for training and research use: it
branches in Python on whether a recurrent cache exists, keeps the state in a
dataclass of 24 tensors, uses `einsum`, a custom autograd `sqrt` and a 200 MB
sin-cos lookup buffer for query embedding. None of that is needed to advance
an online tracker by one frame, and several of those constructs are awkward for
ONNX / TensorRT.

`TAPNextStep` re-expresses exactly one online step (t == 1) of the same network
with the same weights as a pure tensor function:

  (frame, point_tokens, reset, rg_state, conv_state)
      -> (tracks, visible_logits, new_rg_state, new_conv_state)

* `point_tokens` are computed outside the graph with `make_point_tokens`, which
  calls the original `TAPNext.embed_queries`. They only change on frames where a
  query starts, so callers can cache them.
* `reset` is 1.0 on the first frame and 0.0 afterwards. It replaces the
  `cache is None` branch of the RG-LRU input normalisation.
* The recurrent state of all 12 layers is stacked into two tensors. The RG-LRU
  state is always float32 (as in the original); the temporal-conv state uses the
  compute dtype (as under `torch.autocast`).

With `compute_dtype=torch.float16` the matmuls / attention run in fp16 while
norms, the residual stream and the linear recurrence stay in fp32. That mirrors
what `torch.autocast` does for the reference model and gives TensorRT an
explicitly typed graph (build it with `--stronglyTyped`).
"""

from __future__ import annotations

import math

import numpy as np
from tapnet.tapnext import tapnext_torch
import torch
from torch import nn
from torch.nn import functional as F


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
  return F.gelu(x, approximate='tanh')


class TAPNextStep(nn.Module):
  """One online TAPNext step as a stateless tensor function."""

  def __init__(
      self,
      model: tapnext_torch.TAPNext,
      image_size: tuple[int, int] = (256, 256),
      num_queries: int = 256,
      compute_dtype: torch.dtype = torch.float32,
  ):
    super().__init__()
    if image_size[0] % model.patch_size[0] or image_size[1] % model.patch_size[1]:
      raise ValueError(
          f'image_size {image_size} must be a multiple of the patch size'
          f' {model.patch_size}.'
      )
    self.image_size = tuple(image_size)
    self.num_queries = num_queries
    self.width = model.width
    self.num_layers = len(model.blocks)
    self.num_heads = model.blocks[0].vit_block.num_heads
    self.grid_h = image_size[0] // model.patch_size[0]
    self.grid_w = image_size[1] // model.patch_size[1]
    self.num_image_tokens = self.grid_h * self.grid_w
    self.num_tokens = self.num_image_tokens + num_queries
    first_ssm = model.blocks[0].ssm_block
    self.conv_width = first_ssm.recurrent_block.conv_1d.temporal_width
    self.lru_heads = first_ssm.recurrent_block.rg_lru.num_heads
    self.rms_eps = first_ssm.temporal_pre_norm.eps
    self.compute_dtype = compute_dtype

    cd = compute_dtype
    f32 = torch.float32

    def buf(name, tensor, dtype):
      self.register_buffer(name, tensor.detach().clone().to(dtype))

    buf('patch_w', model.lin_proj.weight, cd)
    buf('patch_b', model.lin_proj.bias, f32)
    with torch.no_grad():
      pos_emb = model._video_pos_emb(self.grid_h, self.grid_w)  # pylint: disable=protected-access
    buf('image_pos_emb', pos_emb.reshape(self.num_image_tokens, self.width), f32)
    buf('encoder_norm_w', model.encoder_norm.weight, f32)
    buf('encoder_norm_b', model.encoder_norm.bias, f32)
    self.encoder_norm_eps = model.encoder_norm.eps

    for i, blk in enumerate(model.blocks):
      ssm = blk.ssm_block
      rec = ssm.recurrent_block
      lru = rec.rg_lru
      p = f'l{i}_'
      buf(p + 'tnorm', ssm.temporal_pre_norm.scale, f32)
      buf(p + 'cnorm', ssm.channel_pre_norm.scale, f32)
      buf(p + 'lin_y_w', rec.linear_y.weight, cd)
      buf(p + 'lin_y_b', rec.linear_y.bias, cd)
      buf(p + 'lin_x_w', rec.linear_x.weight, cd)
      buf(p + 'lin_x_b', rec.linear_x.bias, cd)
      buf(p + 'lin_out_w', rec.linear_out.weight, cd)
      buf(p + 'lin_out_b', rec.linear_out.bias, cd)
      buf(p + 'conv_w', rec.conv_1d.w, f32)
      buf(p + 'conv_b', rec.conv_1d.b, f32)
      # -8 * softplus(a_param) is a constant at inference time.
      buf(p + 'neg8_softplus_a', -8.0 * F.softplus(lru.a_param), f32)
      buf(p + 'in_gate_w', lru.input_gate.w, f32)
      buf(p + 'in_gate_b', lru.input_gate.b, f32)
      buf(p + 'a_gate_w', lru.a_gate.w, f32)
      buf(p + 'a_gate_b', lru.a_gate.b, f32)
      # MLPBlock: ffw_up.w is [2, C, 4C]; fuse both up projections into one
      # [C, 8C] matmul (same math as the einsum).
      up_w = ssm.mlp_block.ffw_up.w
      up_b = ssm.mlp_block.ffw_up.b.reshape(2, -1)
      buf(p + 'up_w', torch.cat([up_w[0], up_w[1]], dim=1).t(), cd)
      buf(p + 'up_b', torch.cat([up_b[0], up_b[1]], dim=0), cd)
      buf(p + 'down_w', ssm.mlp_block.ffw_down.weight, cd)
      buf(p + 'down_b', ssm.mlp_block.ffw_down.bias, cd)

      vit = blk.vit_block
      attn = vit.self_attention
      buf(p + 'ln1_w', vit.ln_1.weight, f32)
      buf(p + 'ln1_b', vit.ln_1.bias, f32)
      buf(p + 'qkv_w', attn.in_proj_weight, cd)
      buf(p + 'qkv_b', attn.in_proj_bias, cd)
      buf(p + 'proj_w', attn.out_proj.weight, cd)
      buf(p + 'proj_b', attn.out_proj.bias, cd)
      buf(p + 'ln2_w', vit.ln_2.weight, f32)
      buf(p + 'ln2_b', vit.ln_2.bias, f32)
      buf(p + 'mlp1_w', vit.mlp[0].weight, cd)
      buf(p + 'mlp1_b', vit.mlp[0].bias, cd)
      buf(p + 'mlp2_w', vit.mlp[3].weight, cd)
      buf(p + 'mlp2_b', vit.mlp[3].bias, cd)
      self.vit_ln_eps = vit.ln_1.eps

    self.coordinate_head = _clone_head(model.coordinate_head)
    self.visible_head = _clone_head(model.visible_head)

  def _p(self, layer: int, name: str) -> torch.Tensor:
    return getattr(self, f'l{layer}_{name}')

  def _linear(self, x, w, b):
    return F.linear(x.to(self.compute_dtype), w, b)

  def _rms_norm(self, x, scale):
    var = torch.mean(torch.square(x), dim=-1, keepdim=True)
    return x * torch.rsqrt(var + self.rms_eps) * (scale + 1)

  def _block_diag(self, x, w, b):
    # x: [N, H*D], w: [H, D, D] -> [N, H*D]; same as the einsum in
    # tapnext_lru_modules.BlockDiagonalLinear, written as a batched matmul.
    n = x.shape[0]
    xh = x.reshape(n, self.lru_heads, -1).transpose(0, 1)  # [H, N, D]
    y = torch.matmul(xh, w) + b[:, None, :]  # [H, N, D]
    return y.transpose(0, 1).reshape(n, -1)

  def init_state(
      self, device: torch.device | str | None = None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    rg_state = torch.zeros(
        self.num_layers, self.num_tokens, self.width,
        dtype=torch.float32, device=device,
    )
    conv_state = torch.zeros(
        self.num_layers, self.num_tokens, self.conv_width - 1, self.width,
        dtype=self.compute_dtype, device=device,
    )
    return rg_state, conv_state

  def forward(
      self,
      frame: torch.Tensor,
      point_tokens: torch.Tensor,
      reset: torch.Tensor,
      rg_state: torch.Tensor,
      conv_state: torch.Tensor,
  ):
    """Advances the tracker by one frame.

    Args:
      frame: [1, H, W, 3] float32 RGB frame in [-1, 1] at `image_size`.
      point_tokens: [1, Q, C] float32 tokens from `make_point_tokens`.
      reset: [1] float32, 1.0 on the very first frame and 0.0 afterwards.
      rg_state: [L, N, C] float32 RG-LRU state (zeros on the first frame).
      conv_state: [L, N, K-1, C] temporal-conv state (zeros on first frame).

    Returns:
      tracks: [1, Q, 2] float32 (y, x) positions in 256x256 model space.
      visible_logits: [1, Q, 1] float32; visible if > 0.
      new_rg_state, new_conv_state: states to pass to the next call.
    """
    cd = self.compute_dtype
    c = self.width

    # Patch embedding (Conv2d with stride == kernel) + learned position emb.
    img = frame.permute(0, 3, 1, 2).to(cd)  # [1, 3, H, W]
    tok = F.conv2d(img, self.patch_w, stride=self.patch_w.shape[-2:])
    tok = tok.flatten(2).transpose(1, 2)[0].float() + self.patch_b
    tok = tok + self.image_pos_emb  # [h*w, C]
    x = torch.cat([tok, point_tokens[0].float()], dim=0)  # [N, C] float32

    reset = reset.reshape(1, 1).float()
    new_rg, new_conv = [], []
    for i in range(self.num_layers):
      p = lambda name, i=i: self._p(i, name)

      # ---- SSM residual block (Griffin / Hawk) -------------------------------
      raw = x
      h = self._rms_norm(raw, p('tnorm'))
      y = _gelu_tanh(self._linear(h, p('lin_y_w'), p('lin_y_b')))
      xb = self._linear(h, p('lin_x_w'), p('lin_x_b'))  # [N, C]

      # Causal temporal conv over (state, current frame).
      window = torch.cat([conv_state[i], xb[:, None, :]], dim=1)  # [N, K, C]
      new_conv.append(window[:, 1:])
      xc = (window.float() * p('conv_w')[None]).sum(1) + p('conv_b')

      # RG-LRU, one step, always in float32.
      gate_x = torch.sigmoid(
          self._block_diag(xc, p('in_gate_w'), p('in_gate_b')))
      gate_a = torch.sigmoid(
          self._block_diag(xc, p('a_gate_w'), p('a_gate_b')))
      log_a = gate_a * p('neg8_softplus_a')
      a = torch.exp(log_a)
      multiplier = torch.sqrt(1.0 - torch.exp(2.0 * log_a))
      multiplier = reset + (1.0 - reset) * multiplier
      h_t = a * rg_state[i] + xc * gate_x * multiplier
      new_rg.append(h_t)

      out = self._linear(h_t.to(cd) * y, p('lin_out_w'), p('lin_out_b'))
      residual = out.float() + raw
      h = self._rms_norm(residual, p('cnorm'))
      up = self._linear(h, p('up_w'), p('up_b'))
      gate, val = up.chunk(2, dim=-1)
      mlp = self._linear(_gelu_tanh(gate) * val, p('down_w'), p('down_b'))
      x = mlp.float() + residual

      # ---- Spatial ViT block -------------------------------------------------
      h = F.layer_norm(x, (c,), p('ln1_w'), p('ln1_b'), self.vit_ln_eps)
      qkv = self._linear(h, p('qkv_w'), p('qkv_b'))  # [N, 3C]
      qkv = qkv.reshape(-1, 3, self.num_heads, c // self.num_heads)
      q, k, v = qkv.permute(1, 2, 0, 3).unbind(0)  # each [heads, N, D]
      o = F.scaled_dot_product_attention(q[None], k[None], v[None])[0]
      o = o.transpose(0, 1).reshape(-1, c)
      x = self._linear(o, p('proj_w'), p('proj_b')).float() + x
      h = F.layer_norm(x, (c,), p('ln2_w'), p('ln2_b'), self.vit_ln_eps)
      h = F.gelu(self._linear(h, p('mlp1_w'), p('mlp1_b')))
      x = self._linear(h, p('mlp2_w'), p('mlp2_b')).float() + x

    x = F.layer_norm(
        x, (c,), self.encoder_norm_w, self.encoder_norm_b,
        self.encoder_norm_eps,
    )
    point_x = x[self.num_image_tokens:][None]  # [1, Q, C]
    tracks, visible_logits = self._heads(point_x)
    return (
        tracks,
        visible_logits,
        torch.stack(new_rg, dim=0),
        torch.stack(new_conv, dim=0).to(cd),
    )

  def _heads(self, x):
    """Same as TAPNext.prediction_heads (without returning the logits)."""
    soft_argmax_threshold = 20
    softmax_temperature = 0.5
    track_logits = self.coordinate_head(x)
    position_x, position_y = track_logits.chunk(2, dim=-1)
    num_bins = position_x.shape[-1]
    index = torch.arange(num_bins, device=x.device, dtype=torch.float32)
    tracks = []
    for logits in (position_x, position_y):
      argmax = logits.argmax(dim=-1, keepdim=True).float()
      mask = (torch.abs(argmax - index) <= soft_argmax_threshold).float()
      probs = F.softmax(logits * softmax_temperature, dim=-1) * mask
      probs = probs / probs.sum(dim=-1, keepdim=True)
      tracks.append(torch.sum(probs * index, dim=-1, keepdim=True))
    tracks = torch.cat(tracks, dim=-1) + 0.5
    return tracks, self.visible_head(x)


def _clone_head(head: nn.Sequential) -> nn.Sequential:
  clone = nn.Sequential(*[
      type(m)(m.in_features, m.out_features) if isinstance(m, nn.Linear)
      else type(m)(m.normalized_shape, eps=m.eps) if isinstance(m, nn.LayerNorm)
      else type(m)()
      for m in head
  ])
  clone.load_state_dict(head.state_dict())
  return clone.float()


@torch.no_grad()
def make_point_tokens(
    model: tapnext_torch.TAPNext,
    query_points: torch.Tensor,
    frame_index: int,
) -> torch.Tensor:
  """Point tokens for one online frame, exactly as TAPNext.forward builds them.

  Args:
    model: The reference TAPNext model (its `embed_queries` is reused).
    query_points: [1, Q, 3] (t, y, x) queries in 256x256 model space, where t is
      the absolute frame index at which each query starts.
    frame_index: Absolute index of the frame about to be processed.

  Returns:
    [1, Q, C] float32 tokens: the query token on a query's start frame, the
    "unknown" token before it and the "mask" token after it.
  """
  shifted = torch.cat(
      [query_points[..., :1] - frame_index, query_points[..., 1:]], dim=-1
  )
  return model.embed_queries(1, shifted.float())[:, 0].float()


class PointTokenizer(nn.Module):
  """Standalone `make_point_tokens` that does not need the full model.

  Only the three learned tokens (3 x C floats) are model specific; the sin-cos
  query embedding is deterministic and rebuilt here. This lets a TensorRT
  deployment build point tokens without loading the ~1 GB PyTorch checkpoint.
  """

  def __init__(
      self,
      mask_token: torch.Tensor,
      unknown_token: torch.Tensor,
      point_query_token: torch.Tensor,
      image_size: tuple[int, int] = (256, 256),
  ):
    super().__init__()
    width = mask_token.shape[-1]
    self.image_size = tuple(image_size)
    self.register_buffer('mask_token', mask_token.reshape(1, 1, 1, width))
    self.register_buffer('unknown_token', unknown_token.reshape(1, 1, 1, width))
    self.register_buffer(
        'point_query_token', point_query_token.reshape(1, 1, 1, width)
    )
    pos = tapnext_torch.posemb_sincos_2d(image_size[0], image_size[1], width)
    # Same layout trick as TAPNext.embed_queries: [1, C, W, H].
    pos = torch.from_numpy(pos).view(1, image_size[0], image_size[1], width)
    self.register_buffer('query_pos_embed', pos.permute(0, 3, 2, 1).contiguous())

  @classmethod
  def from_model(cls, model: tapnext_torch.TAPNext) -> 'PointTokenizer':
    return cls(
        model.mask_token.detach().float().cpu(),
        model.unknown_token.detach().float().cpu(),
        model.point_query_token.detach().float().cpu(),
        model.image_size,
    )

  def save(self, path: str) -> None:
    np.savez(
        path,
        mask_token=self.mask_token.cpu().numpy(),
        unknown_token=self.unknown_token.cpu().numpy(),
        point_query_token=self.point_query_token.cpu().numpy(),
        image_size=np.array(self.image_size),
    )

  @classmethod
  def load(cls, path: str) -> 'PointTokenizer':
    data = np.load(path)
    return cls(
        torch.from_numpy(data['mask_token']),
        torch.from_numpy(data['unknown_token']),
        torch.from_numpy(data['point_query_token']),
        tuple(int(v) for v in data['image_size']),
    )

  @torch.no_grad()
  def forward(self, query_points: torch.Tensor, frame_index: int):
    """See `make_point_tokens`; returns [1, Q, C] float32."""
    query_points = query_points.float()
    rel_t = (query_points[..., :1] - frame_index).unsqueeze(1)  # [1, 1, Q, 1]
    size = torch.tensor(self.image_size, device=query_points.device)
    grid = (query_points[..., 1:].unsqueeze(1) / size) * 2 - 1
    pos = F.grid_sample(self.query_pos_embed, grid, align_corners=False)
    query_tokens = self.point_query_token + pos.permute(0, 2, 3, 1)
    tokens = torch.where(rel_t == 0, query_tokens, self.mask_token)
    tokens = torch.where(rel_t > 0, self.unknown_token, tokens)
    return tokens[:, 0]


def load_tapnext(
    checkpoint_path: str, device: torch.device | str = 'cpu'
) -> tapnext_torch.TAPNext:
  """Loads a TAPNext / TAPNext++ PyTorch (Lightning) checkpoint."""
  model = tapnext_torch.TAPNext(image_size=(256, 256))
  ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
  state_dict = ckpt.get('state_dict', ckpt)
  state_dict = {k.removeprefix('tapnext.'): v for k, v in state_dict.items()}
  model.load_state_dict(state_dict)
  return model.to(device).eval()


def flops_per_frame(
    num_image_tokens: int, num_queries: int, width: int = 768, depth: int = 12
) -> float:
  """Approximate FLOPs of one TAPNext-B online step (matmuls + attention)."""
  n = num_image_tokens + num_queries
  macs_per_token = (
      3 * width * width  # linear_x / linear_y / linear_out
      + 3 * width * 4 * width  # gated MLP (2 up + 1 down)
      + 2 * width * 64  # block-diagonal RG-LRU gates (12 heads x 64)
      + 4 * width * width  # qkv + out projection
      + 2 * width * 4 * width  # ViT MLP
  )
  attention_macs = 2 * n * n * width
  return 2.0 * depth * (n * macs_per_token + attention_macs)


def describe(step: TAPNextStep) -> str:
  gflops = flops_per_frame(step.num_image_tokens, step.num_queries) / 1e9
  return (
      f'{step.image_size[0]}x{step.image_size[1]} input, '
      f'{step.num_image_tokens} image + {step.num_queries} point tokens, '
      f'~{gflops:.0f} GFLOPs/frame, '
      f'{math.prod((step.num_layers, step.num_tokens, step.width)) * 4 / 2**20:.0f}'
      ' MiB RG-LRU state'
  )
