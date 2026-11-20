#!/bin/bash
# RQ1 exploration: serve Qwen3-8B with vLLM on one GPU, run explore_rq1.py against it.
# Submit from Trust_Laundering/:
#   sbatch runner/slurm/explore_rq1.sbatch                       # smoke test: 4 pairs
#   LIMIT=0 sbatch --time=03:00:00 runner/slurm/explore_rq1.sbatch  # all 144 banking pairs
#SBATCH --job-name=tl_explore_rq1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-highvram
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40&bf16
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail

LIMIT="${LIMIT:-4}"
ATTACKS="${ATTACKS:-A1 A2}"
WORKERS="${WORKERS:-8}"
MODEL_ID="Qwen/Qwen3-8B"
SERVED_NAME="qwen3-8b"
WS=/scratch3/workspace/udhanuka_umass_edu-cs684
OUT_DIR="$WS/trust_laundering/results"
VLLM_ENV=/work/pi_sniekum_umass_edu/udhanuka_umass_edu/.conda/envs/vllm-gptoss
PORT=$((20000 + SLURM_JOB_ID % 10000))

echo "host=$(hostname) date=$(date) job=$SLURM_JOB_ID port=$PORT limit=$LIMIT attacks=$ATTACKS"
nvidia-smi --query-gpu=name,memory.total --format=csv
module load conda/latest
# Compute nodes have no /usr/local/cuda; FlashInfer JIT-compiles kernels and needs nvcc.
module load cuda/12.8
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}"
export VLLM_USE_FLASHINFER_SAMPLER=0   # torch sampler; avoids a JIT build at warmup
module list 2>&1

# Read-only shared weights; keep every cache off /home (85% full) and off /project.
export HF_HUB_CACHE=/datasets/ai/qwen3/hub
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME="$WS/cache"
export VLLM_CACHE_ROOT="$WS/cache/vllm"
export TRITON_CACHE_DIR="$WS/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$WS/cache/inductor"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$OUT_DIR" "$XDG_CACHE_HOME" logs

echo "vllm: $("$VLLM_ENV/bin/python" -c 'import vllm; print(vllm.__version__)')"
"$VLLM_ENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo "waiting for vLLM (pid $VLLM_PID)..."
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then echo "vLLM up after $((i * 5))s"; break; fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "vLLM died; tail of log:"; tail -50 "logs/vllm-$SLURM_JOB_ID.log"; exit 1; fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "vLLM not up after 15 min"; exit 1; }

conda activate trustlaundering
which python && python --version
python runner/explore_rq1.py \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$SERVED_NAME" \
  --suites banking --attacks $ATTACKS --limit "$LIMIT" --workers "$WORKERS" \
  --out-dir "$OUT_DIR"
echo "done $(date)"
