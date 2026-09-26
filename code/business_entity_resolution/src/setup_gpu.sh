#!/bin/bash
set -euo pipefail
# ──────────────────────────────────────────────────────────────────────
# GPU instance bootstrap for Amazon ML Challenge compound pipeline
#
# Recommended instance: g5.4xlarge (us-east-1)
#   A10G 24GB VRAM, 16 vCPU, 64GB RAM
#   ~$1.62/hr on-demand → $70 budget covers ~43 hours
#
# Prerequisites:
#   1. Launch g5.4xlarge with Deep Learning AMI (Ubuntu 22.04)
#   2. SCP dataset:
#        scp -i key.pem -r /home/ubuntu/student_resource/ ubuntu@NEW_IP:/home/ubuntu/
#   3. SCP code:
#        scp -i key.pem -r code/ ubuntu@NEW_IP:/home/ubuntu/ml-challenge/
#   4. SSH in and run: bash src/setup_gpu.sh
#
# After setup:
#   cd /home/ubuntu/ml-challenge/code/business_entity_resolution
#   python -X utf8 src/pipeline.py all 2>&1 | tee pipeline.log
# ──────────────────────────────────────────────────────────────────────

echo "=== GPU Pipeline Setup — $(date) ==="

sudo apt-get update -qq
sudo apt-get install -y -qq libgomp1 tmux htop

pip install --upgrade pip -q
pip install torch --index-url https://download.pytorch.org/whl/cu121 -q

# faiss-gpu: pip package is just 'faiss-gpu' (faiss-gpu-cu12 does not exist on PyPI)
# conda is the officially supported path; pip fallback works on Deep Learning AMI
pip install faiss-gpu -q || conda install -c pytorch -c nvidia faiss-gpu=1.15.1 -y -q

# sentence-transformers >= 6.0.0 REQUIRED — v6.0.0 (Aug 2025) fixed a silent
# fp16 scoring bug in cross-encoder inference. Earlier versions produce wrong
# CE scores in half-precision without any error or warning.
pip install "sentence-transformers>=6.0.0" -q

pip install lightgbm -q
pip install rapidfuzz jellyfish scikit-learn datasets -q
pip install "transformers>=4.40.0" accelerate -q
pip install "optimum[onnxruntime-gpu]" -q
pip install numpy scipy -q

echo ""
echo "=== Verification ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "
import torch, faiss, sentence_transformers, lightgbm, rapidfuzz
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')
print(f'torch={torch.__version__}')
print(f'faiss-gpu OK (ngpus={faiss.get_num_gpus()})')
print(f'sentence-transformers={sentence_transformers.__version__}')
print(f'lightgbm={lightgbm.__version__}')
print(f'rapidfuzz={rapidfuzz.__version__}')
"
echo "=== Setup complete — $(date) ==="
