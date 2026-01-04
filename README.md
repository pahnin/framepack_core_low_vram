# framepack_core_low_vram
Core of Framepack next-frame (next-frame-section) prediction neural network structure with some optimizations


framepack_core

Stripped-down core of FramePack Studio for HunyuanVideo generation, optimized for 8GB VRAM consumer GPUs. Removes Gradio UI, focuses on headless inference with VAE tiling, memory optimizations, and code cleanup.

​
Features

    Low VRAM Operation: Runs full video generation (320x320@8fps) on 8GB GPUs via dynamic swapping, CPU offload, Sage attention, and MAG cache.

​

FramePack Anti-Drifting: Progressive next-frame prediction with inverted sampling for long videos without quality degradation.

​

HunyuanVideo DiT Backbone: Packed Transformer3D model with 3D VAE, full attention, and dual-stream text-video fusion.

    ​

    VAE Tiling Enhancements: Memory-efficient decoding for high-res outputs.

    Queue System: Job queue + worker for batch processing (used by upstream SDXL-Hunyuan pipelines).

    LoRA Support: Standard diffusers LoRA loading.

    Timestamped Prompts: Sectioned prompt blending for complex scenes.

Note: Some dead code remains (e.g., unused Gradio helpers). Queue runner and job system support external integrations like sdxl_hunyuan_video_pipeline.

​
Quick Start

bash
# Clone and setup
git clone <your-framepack-core-repo>
cd framepack_core
# Install PyTorch (CUDA 12.1+ for Sage attention)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt  # Or manual: diffusers transformers accelerate xformers sageattention
# Install Sage Attention
Use instructions from Sage Attention github
# Run sample (downloads ~13B HunyuanVideo to ./hf_download)
python -m src.framepack_core.sample_usage

Sample Output: 4s video (8 frames @ 2fps) in outputs/hunyuan_only/. Adjust VRAM settings auto-detects >60GB for high mode.

​
Usage

Core entry: HunyuanVideoGenerator in src/framepack_core/generator.py.

python
from src.framepack_core.generator import HunyuanVideoGenerator
from src.framepack_core.settings import Settings

settings = Settings()
settings.set("model_path", "hunyuanvideo-community/HunyuanVideo")
settings.set("low_vram_mode", True)  # 8GB mode
settings.set("width", 320)
settings.set("num_frames", 8)
settings.set("gpu_memory_preservation", 4)  # GB reserved

generator = HunyuanVideoGenerator(settings=settings, high_vram=False)
generator.load_models("hunyuanvideo-community/HunyuanVideo")
video = generator.generate_video(prompt="your prompt", seed=42, ...)

See src/framepack_core/sample_usage.py for full example.

​
Architecture

text
src/framepack_core/
├── generator.py          # Main HunyuanVideoGenerator
├── diffusers_helper/     # Model wrappers, MAG cache, k-diffusion sampler
│   ├── models/           # Packed HunyuanVideoTransformer3D
│   ├── hunyuan.py        # VAE tiling, CLIP vision
│   └── pipelines/        # Worker, LoRA manager
├── pipelines/            # Queue runner, prompt blender (for upstream)
├── settings.py           # Config (width, steps, dtype=float16)
└── sample_usage.py       # Standalone demo

VRAM Optimizations
Feature	Benefit	8GB Impact
Sage/Flash Attention	Reduces KV cache	-2-3GB peak
Dynamic CPU Offload	Model swapping	Fits 13B model
MAG Cache (0.15 thresh)	Skips redundant frames	20-30% faster
VAE Tiling Decode	High-res without OOM	+50% res possible
float16 + low_vram_mode	Half precision	Baseline 6-8GB

Tested on RTX 3050 8gb/ RTX 5060 with vram limit set to 4 gb
​
Queue System (Upstream Usage)

For batch jobs (e.g., in sdxl-hunyuan-pipeline):


# Run worker
python -m src.framepack_core.queue_runner

​
Requirements

    Python 3.10+

    PyTorch 2.4+ (CUDA 12.1+)

    diffusers, transformers, accelerate, xformers, sageattention

    psutil, einops, opencv-python (utils)

No Gradio/ComfyUI deps. Offline HF cache: HF_HOME=./hf_download.

​
Development

    Sync as Git submodule: git submodule add  framepack_core_low_vram src/framepack_core

    Run tests: python -m src.framepack_core.sample_usage

    Dead code: gradio/ folder safe to ignore/remove.

License

Apache-2.0 (upstream FramePack Studio).
​
