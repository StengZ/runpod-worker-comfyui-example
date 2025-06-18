# Use Nvidia CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 

# According to the basic package and tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libgl1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up excess image packages
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# comfy-cli
RUN pip install comfy-cli==1.2.7

# ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install --cuda-version 11.8 --nvidia

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install ComfyUI dependencies
RUN pip install --no-cache-dir \
    accelerate>=1.1.1 \
    numpy==1.25.0 \
    aiohttp \
    addict \
    albumentations==1.4.3 \
    bitsandbytes>=0.41.1 \
    blind-watermark \
    blend_modes \
    cmake \
    gguf \
    color-matcher \
    colour-science \
    dill \
    diffusers>=0.31.0 \
    clip_interrogator>=0.6.0 \
    einops \
    facexlib \
    fastapi \
    filelock \
    ftfy \
    fvcore \
    GitPython \
    google-generativeai \
    hydra-core \
    imageio \
    importlib_metadata \
    iopath \
    insightface \
    joblib \
    kornia>=0.7.1 \
    lark \
    librosa \
    loguru \
    matplotlib \
    matrix-client==0.4.0 \
    mediapipe \
    numba \
    numexpr \
    omegaconf \
    onnx \
    onnxruntime \
    open-clip-torch>=2.24.0 \
    opencv-contrib-python \
    opencv-python \
    opencv-python-headless>=4.0.1.24 \
    openai \
    pandas \
    peft>=0.12.0 \
    piexif \
    pilgram \
    Pillow \
    protobuf \
    psd-tools \
    psutil \
    pygithub \
    pymatting \
    pytorch-lightning>=2.2.1 \
    pyyaml \
    pyzbar \
    qrcode \
    rembg \
    reportlab \
    requests \
    rich \
    scikit-image \
    scikit-learn \
    scipy \
    safetensors>=0.4.2 \
    segment-anything \
    sentencepiece \
    soundfile \
    spandrel \
    svgwrite \
    svglib \
    timm \
    tokenizers>=0.13.3 \
    toml \
    torch==2.5.1 \
    torchvision \
    huggingface-hub==0.26.2 \
    torchaudio \
    torchsde \
    transparent-background \
    transformers>=4.45.0 \
    trimesh[easy] \
    tqdm \
    typer \
    typer_config \
    typing-extensions \
    ultralytics \
    vtracer \
    webcolors \
    websocket-client==1.6.3 \
    wget \
    yacs \
    yapf \
    zhipuai

# Install runpod
RUN pip install runpod requests uuid pathlib fastapi[standard]==0.115.4

ENV NODES_DIR=/comfyui/custom_nodes

# Clone ComfyUI repository

RUN git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /comfyui/custom_nodes/ComfyUI-Custom-Scripts
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git /comfyui/custom_nodes/ComfyUI-KJNodes
RUN git clone https://github.com/yolain/ComfyUI-Easy-Use.git /comfyui/custom_nodes/ComfyUI-Easy-Use
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git /comfyui/custom_nodes/ComfyUI-Impact-Pack
RUN git clone https://github.com/rgthree/rgthree-comfy.git /comfyui/custom_nodes/rgthree-comfy
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git /comfyui/custom_nodes/ComfyUI-Impact-Subpack
RUN git clone https://github.com/city96/ComfyUI-GGUF.git /comfyui/custom_nodes/ComfyUI-GGUF
RUN git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git /comfyui/custom_nodes/ComfyUI_Comfyroll_CustomNodes
RUN git clone https://github.com/BadCafeCode/masquerade-nodes-comfyui.git /comfyui/custom_nodes/masquerade-nodes-comfyui
RUN git clone https://github.com/chflame163/ComfyUI_LayerStyle_Advance.git /comfyui/custom_nodes/ComfyUI_LayerStyle_Advance
RUN git clone https://github.com/Ryuukeisyou/comfyui_face_parsing.git /comfyui/custom_nodes/comfyui_face_parsing


# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py Skin_Enhancer_Standard_api.json ./
RUN chmod +x /start.sh

# Start the container
CMD ["/start.sh"]
