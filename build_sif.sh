#!/bin/bash
# Rebuild a portable Vina-GPU-2.1 .sif containing the CURRENT source (all fixes).
# Kernels compile per-machine at first run (make source + VINA_GPU_HOME), so the image
# works on any NVIDIA GPU — give it to other computers as-is.
# Usage: bash build_sif.sh [output.sif]
set -e
cd "$(dirname "$0")"
OUT="${1:-autodock-vina-gpu-2.1.sif}"
IMG=autodock-vina-gpu:latest
echo "[1/2] docker build -> $IMG  (compiles boost + vina from src/; ~15-20 min first time)"
docker build -t "$IMG" .
echo "[2/2] singularity build $OUT  <- docker image"
singularity build "$OUT" "docker-daemon://$IMG"
echo "DONE: $OUT   (run: singularity exec --nv $OUT autodock-vina-gpu --help)"
