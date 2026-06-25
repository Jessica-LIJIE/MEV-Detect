#!/usr/bin/env bash
# E2 云 GPU 一键实验：单卡规模曲线 + 多池 PSO 1/2/4 卡
#
# 用法（在项目根目录）:
#   chmod +x scripts/run_e2_all.sh
#   ./scripts/run_e2_all.sh
#
# 仅 2 卡机器:
#   GPUS=1,2 ./scripts/run_e2_all.sh

set -euo pipefail

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPUS="${GPUS:-1,2,4}"
RECORD_ID="${RECORD_ID:-mock_mp_001}"
PARTICLES="${PARTICLES:-8000}"
MAX_ITER="${MAX_ITER:-80}"
REPEATS="${REPEATS:-3}"
TOP_K="${TOP_K:-32}"

if [[ ! -d .venv ]]; then
  echo "ERROR: .venv not found. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p data/E2-data data/figures logs

echo "========== 环境 =========="
echo "ROOT=$ROOT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| gpus', torch.cuda.device_count())"
python -c "import torch; x=torch.rand(4,device='cuda'); torch.cuda.synchronize(); print('cuda probe: ok')" \
  || { echo "WARN: CUDA probe failed; E2-A may fall back to CPU only"; }

echo ""
echo "========== E2-A: 单卡规模曲线 (CPU vs CUDA) =========="
python scripts/benchmark_multipool_scale.py \
  --record-id "$RECORD_ID" \
  2>&1 | tee logs/e2a_scale.log

python - << 'PY'
import json
from pathlib import Path
p = Path("data/E2-data/multipool_scale_curve.json")
if p.is_file():
    d = json.loads(p.read_text(encoding="utf-8"))
    print("crossover:", d.get("crossover_note"))
    print("100k×K=32 ms:", d.get("single_gpu_100k_k32_ms"))
PY

echo ""
echo "========== E2-B: 多池 PSO (${GPUS} 卡) =========="
echo "particles=$PARTICLES max_iter=$MAX_ITER repeats=$REPEATS top_k=$TOP_K"

python scripts/benchmark_multipool_multi_gpu.py \
  --gpus "$GPUS" \
  --record-id "$RECORD_ID" \
  --top-k "$TOP_K" \
  --particles "$PARTICLES" \
  --max-iter "$MAX_ITER" \
  --repeats "$REPEATS" \
  --seed 42 \
  2>&1 | tee logs/e2b_multigpu.log

python - << 'PY'
import json
from pathlib import Path
p = Path("data/E2-data/multipool_multi_gpu_benchmark.json")
if p.is_file():
    rows = json.loads(p.read_text(encoding="utf-8"))["results"]
    print("\n--- multipool multi-GPU summary ---")
    for r in rows:
        print(
            f"{r['world_size']} GPU | {r['elapsed_ms_mean']:.1f} ms | "
            f"speedup {r.get('speedup', 1):.2f}x | "
            f"eff {r.get('parallel_efficiency_pct', 100):.1f}% | "
            f"fitness {r['best_fitness_mean']:.2f}"
        )
PY

echo ""
echo "========== 打包结果 =========="
TS="$(date +%Y%m%d_%H%M%S)"
ZIP="data/E2-data/e2_cloud_${TS}.zip"
zip -r "$ZIP" \
  data/E2-data/multipool_scale_curve.json \
  data/E2-data/multipool_multi_gpu_benchmark.json \
  data/E2-data/ddp_multipool_*gpu_r*.json \
  data/figures/e2_*.png \
  logs/e2a_scale.log logs/e2b_multigpu.log \
  2>/dev/null || zip -r "$ZIP" \
  data/E2-data/multipool_scale_curve.json \
  data/E2-data/multipool_multi_gpu_benchmark.json \
  data/figures/e2_*.png \
  logs/e2a_scale.log logs/e2b_multigpu.log

echo ""
echo "Done."
echo "  JSON: data/E2-data/multipool_scale_curve.json"
echo "  JSON: data/E2-data/multipool_multi_gpu_benchmark.json"
echo "  ZIP:  $ZIP"
echo "Download the ZIP from the cloud panel, then shutdown the instance."
