#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
make
inputs=("$@")
if [ ${#inputs[@]} -eq 0 ]; then
  inputs=(inputs/*.mp4)
fi
for input in "${inputs[@]}"; do
  build/teleop-camera-latency-analysis --input "$input" --output "outputs/$(basename "$input")"
done
