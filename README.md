# framepack_core_low_vram
### Lightweight HunyuanVideo core from FramePack Studio, aggressively optimized for 8GB VRAM GPUs.
 Strips UI code, adds tiling/memory optimizations, and restructures for 
headless use in pipelines like SDXlHunyuan pipeline , 3D reconstruction 
workflows.*

🙏 With utmost respect and gratitude to the original FramePack by @lllyasviel and FramePack Studio contributors. This is a stripped-down derivative focused on low-VRAM headless inference.

Why This Repo Exists
This fork exists to solve real-world deployment constraints:

Aggressive VAE tiling for 8GB VRAM (RTX 3050 8GB) to eliminate OOM errors

Removed 90% of Gradio UI - no previews, job streams, or web interfaces

Restructured worker code for maintainability and module reusability in other projects

Pure headless operation - runs standalone or as pipeline backend

Quick Start
```bash
git clone https://github.com/pahnin/framepack_core_low_vram
cd framepack_core_low_vram

# PyTorch CUDA 12.1+ (Sage attention)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Run sample - downloads ~13GB HunyuanVideo to ./hf_download
python -m src.framepack_core.sample_usage
```

Output: 4s video (8 frames) in outputs/hunyuan_only/. Auto-detects VRAM mode.
Usage
```
pythonfrom src.framepack_core.generator import HunyuanVideoGenerator
from src.framepack_core.settings import Settings

settings = Settings()
settings.set("model_path", "hunyuanvideo-community/HunyuanVideo")
settings.set("low_vram_mode", True)      # 8GB mode
settings.set("gpu_memory_preservation", 4)  # Reserve 4GB
settings.set("width", 320)
settings.set("num_frames", 8)

generator = HunyuanVideoGenerator(settings=settings, high_vram=False)
generator.load_models("hunyuanvideo-community/HunyuanVideo")
video = generator.generate_video(prompt="futuristic drone city flythrough", seed=42)
```
Full example: src/framepack_core/sample_usage.py

Architecture
```
src/framepack_core/
├── generator.py          # 🎯 Main HunyuanVideoGenerator
├── diffusers_helper/     # Model wrappers + optimizations
│   ├── models/           # Packed HunyuanVideoTransformer3D
│   ├── hunyuan.py        # VAE tiling + CLIP vision
│   └── k_diffusion/      # UniPC sampler
├── pipelines/            # Queue runner + prompt blender
├── settings.py           # Centralized config
└── sample_usage.py       # Standalone demo
```
VRAM
Tested: RTX 3050 8GB, RTX 4060 with 4GB reservation.
Requirements
texttorch&gt;=2.4 (CUDA 12.1+)
diffusers transformers accelerate
xformers sageattention
psutil einops opencv-python

No Gradio/ComfyUI. Offline: HF_HOME=./hf_download
License
Apache-2.0 (inherits from FramePack/FramePack Studio)
