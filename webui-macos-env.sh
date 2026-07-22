#!/bin/bash
####################################################################
#                    macOS (Apple Silicon) defaults                 #
# Please modify webui-user.sh to change these instead of this file #
####################################################################

export install_dir="$HOME"
export COMMANDLINE_ARGS="--skip-torch-cuda-test --upcast-sampling --no-half-vae --use-cpu interrogate --opt-sub-quad-attention"
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Memory guardrail: limit PyTorch MPS to ~70% of RAM (leaves headroom for macOS + apps)
# On 16GB: caps at ~11GB. On 8GB: ~5.6GB. On 24GB+: ~16.8GB.
# Set to 0.0 to disable (not recommended on 8/16GB machines)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7

# Apple Silicon only — install latest PyTorch with MPS support
export TORCH_COMMAND="pip install torch torchvision torchaudio"

####################################################################
