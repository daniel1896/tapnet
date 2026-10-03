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

"""One-command speed/accuracy sweep of TAPNext++ on the local GPU.

For every (resolution, num_queries) in the grid this script:
  1. exports the fp16 step graph to ONNX (export_onnx.py),
  2. builds a TensorRT engine on this GPU (trt_runner.build_engine),
  3. measures per-frame latency/FPS of PyTorch (CUDA graph) and TensorRT
     (benchmark.py),
  4. measures TAP-Vid DAVIS accuracy of the TensorRT engine (eval_davis.py),
and writes a markdown report. Steps whose outputs exist are skipped, so the
script can be re-run after a failure.

  python -m tapnet.tapnextpp.deploy.optimize_local --workdir tapnextpp_opt \
      [--resolutions 256,224,192] [--queries 64,256] [--skip_eval]

Downloads the 256x256 TAPNext++ checkpoint and TAP-Vid DAVIS (1.7 GB) into
--workdir if they are missing. Needs: CUDA PyTorch, onnx, onnxscript,
tensorrt (pip install tensorrt), mediapy, scipy, absl-py.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.request
import zipfile

CHECKPOINT_URL = (
    'https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt')
DAVIS_URL = 'https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip'


def _download(url, dest):
  if dest.exists():
    return
  print(f'Downloading {url} ...', flush=True)
  tmp = dest.with_suffix(dest.suffix + '.part')
  urllib.request.urlretrieve(url, tmp)
  tmp.rename(dest)


def _run(cmd, log_path):
  """Runs a module of this package, teeing output to log_path."""
  cmd = [sys.executable, '-m'] + cmd
  print('  $', ' '.join(cmd), flush=True)
  start = time.time()
  with open(log_path, 'w') as log:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, check=False)
    log.write(proc.stdout)
  tail = '\n'.join(proc.stdout.strip().splitlines()[-12:])
  print('    ' + tail.replace('\n', '\n    '), flush=True)
  print(f'    ({time.time() - start:.0f}s, exit {proc.returncode})', flush=True)
  return proc.returncode == 0


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--workdir', default='tapnextpp_opt')
  parser.add_argument('--checkpoint', default=None,
                      help='Defaults to <workdir>/tapnextpp_ckpt.pt.')
  parser.add_argument('--davis_pkl', default=None,
                      help='Defaults to <workdir>/tapvid_davis/tapvid_davis.pkl')
  parser.add_argument('--resolutions', default='256,224,192')
  parser.add_argument('--queries', default='64,256')
  parser.add_argument('--optimization_level', type=int, default=3)
  parser.add_argument('--skip_eval', action='store_true',
                      help='Skip the DAVIS accuracy evaluation.')
  args = parser.parse_args()

  work = pathlib.Path(args.workdir)
  work.mkdir(parents=True, exist_ok=True)
  ckpt = pathlib.Path(args.checkpoint or work / 'tapnextpp_ckpt.pt')
  _download(CHECKPOINT_URL, ckpt)
  davis = pathlib.Path(
      args.davis_pkl or work / 'tapvid_davis' / 'tapvid_davis.pkl')
  if not args.skip_eval and not davis.exists():
    archive = work / 'tapvid_davis.zip'
    _download(DAVIS_URL, archive)
    with zipfile.ZipFile(archive) as z:
      z.extractall(davis.parent.parent)

  from tapnet.tapnextpp.deploy import trt_runner  # pylint: disable=g-import-not-at-top

  rows = []
  for res in [int(r) for r in args.resolutions.split(',')]:
    for q in [int(n) for n in args.queries.split(',')]:
      tag = f'r{res}_q{q}_fp16'
      print(f'\n=== {tag}', flush=True)
      onnx_path = work / f'{tag}.onnx'
      plan_path = work / f'{tag}.plan'
      bench_json = work / f'{tag}.bench.json'
      eval_json = work / f'{tag}.davis.json'
      row = {'tag': tag, 'resolution': res, 'queries': q}
      rows.append(row)

      if not onnx_path.exists() and not _run(
          ['tapnet.tapnextpp.deploy.export_onnx', '--checkpoint', str(ckpt),
           '--resolution', str(res), '--num_queries', str(q),
           '--precision', 'fp16', '--output', str(onnx_path),
           '--verify_frames', '0'], work / f'{tag}.export.log'):
        row['error'] = 'export failed'
        continue
      if not plan_path.exists():
        print(f'  building {plan_path.name} ...', flush=True)
        start = time.time()
        try:
          trt_runner.build_engine(
              str(onnx_path), str(plan_path), args.optimization_level)
        except Exception as e:  # pylint: disable=broad-exception-caught
          row['error'] = f'engine build failed: {e}'
          print('   ', row['error'], flush=True)
          continue
        print(f'    ({time.time() - start:.0f}s)', flush=True)
      if not bench_json.exists():
        _run(['tapnet.tapnextpp.deploy.benchmark', '--checkpoint', str(ckpt),
              '--resolution', str(res), '--num_queries', str(q),
              '--modes', 'step_graph', '--engine', str(plan_path),
              '--json_out', str(bench_json)], work / f'{tag}.bench.log')
      if bench_json.exists():
        bench = json.loads(bench_json.read_text())
        row['gpu'] = bench['gpu'].splitlines()[0]
        row['gflops'] = bench['gflops_per_frame']
        for mode, result in bench['modes'].items():
          row[mode] = result
      if not args.skip_eval:
        if not eval_json.exists() or len(json.loads(eval_json.read_text())) < 30:
          _run(['tapnet.tapnextpp.deploy.eval_davis', '--checkpoint', str(ckpt),
                '--davis_pkl', str(davis), '--resolution', str(res),
                '--engine', str(plan_path), '--out', str(eval_json)],
               work / f'{tag}.davis.log')
        if eval_json.exists():
          per_video = json.loads(eval_json.read_text()).values()
          for key, name in (('average_jaccard', 'aj'),
                            ('average_pts_within_thresh', 'delta_avg'),
                            ('occlusion_accuracy', 'oa')):
            row[name] = 100 * sum(v[key] for v in per_video) / len(per_video)
          row['videos'] = len(per_video)

  report = _report(rows)
  (work / 'report.md').write_text(report)
  (work / 'report.json').write_text(json.dumps(rows, indent=1))
  print('\n' + report)
  print(f'Wrote {work / "report.md"} and {work / "report.json"}')


def _report(rows):
  """Markdown table of the sweep."""

  def fmt(value, spec):
    return '-' if value is None else format(value, spec)

  gpu = next((r['gpu'] for r in rows if 'gpu' in r), 'unknown GPU')
  lines = [
      f'# TAPNext++ local sweep on {gpu}', '',
      '| config | GFLOPs | TensorRT fps | TensorRT p95 ms | PyTorch graph fps'
      ' | TRT max dev px | DAVIS AJ | delta_avg | OA | note |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
  ]
  for r in rows:
    trt, torch_graph = r.get('trt', {}), r.get('step_graph', {})
    cells = [
        r['tag'], fmt(r.get('gflops'), '.0f'),
        fmt(trt.get('fps'), '.1f'), fmt(trt.get('p95_ms'), '.2f'),
        fmt(torch_graph.get('fps'), '.1f'), fmt(trt.get('max_dev_px'), '.3f'),
        fmt(r.get('aj'), '.1f'), fmt(r.get('delta_avg'), '.1f'),
        fmt(r.get('oa'), '.1f'),
        r.get('error') or trt.get('error') or torch_graph.get('error') or '',
    ]
    lines.append('| ' + ' | '.join(cells) + ' |')
  lines += ['', 'fps = online per-frame rate incl. reading tracks back to the '
            'host. DAVIS: query-first, all points tracked jointly, '
            '256x256 metric space.', '']
  return '\n'.join(lines)


if __name__ == '__main__':
  main()
