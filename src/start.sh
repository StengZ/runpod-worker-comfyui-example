#!/usr/bin/env bash

echo "Downloading models..."

wget -O /comfyui/models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/sTzing/t5xxl/resolve/main/t5xxl_fp8_e4m3fn.safetensors
wget -O /comfyui/models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors

wget -O /comfyui/models/vae/flux_fill_ae.safetensors https://huggingface.co/zw2013/fill_vae/resolve/main/flux_fill_ae.safetensors

wget -O /comfyui/models/checkpoints/illustriousRealismBy_v10VAE.safetensors https://huggingface.co/ExistentialC/illustrious-realism/resolve/main/illustriousRealismBy_v10VAE.safetensors

wget -O /comfyui/models/loras/ILXL_Realism_Slider_V.1.safetensors https://huggingface.co/casque/ILXL_Realism_Slider_V.1/resolve/main/ILXL_Realism_Slider_V.1.safetensors
wget -O /comfyui/models/loras/RealSkin_xxXL_v1.safetensors https://huggingface.co/casque/RealSkin_xxXL_v1/resolve/main/RealSkin_xxXL_v1.safetensors

wget -O /comfyui/models/unet/flux1-dev-Q5_0.gguf https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q5_0.gguf
wget -O /comfyui/models/ultralytics/segm/face_yolov8m.pt https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt

wget -O /comfyui/models/florence2/large-PromptGen-v2.0/added_tokens.json https://huggingface.co/chflame163/ComfyUI_LayerStyle/resolve/main/ComfyUI/models/florence2/large-PromptGen-v2.0/added_tokens.json
wget -O /comfyui/models/florence2/large-PromptGen-v2.0/config.json https://huggingface.co/chflame163/ComfyUI_LayerStyle/resolve/main/ComfyUI/models/florence2/large-PromptGen-v2.0/config.json
wget -O /comfyui/models/florence2/large-PromptGen-v2.0/configuration_florence2.py https://huggingface.co/chflame163/ComfyUI_LayerStyle/resolve/main/ComfyUI/models/florence2/large-PromptGen-v2.0/configuration_florence2.py
wget -O /comfyui/models/florence2/large-PromptGen-v2.0/generation_config.json https://huggingface.co/chflame163/ComfyUI_LayerStyle/resolve/main/ComfyUI/models/florence2/large-PromptGen-v2.0/generation_config.json
wget -O /comfyui/models/florence2/large-PromptGen-v2.0/model.safetensors https://huggingface.co/chflame163/ComfyUI_LayerStyle/resolve/main/ComfyUI/models/florence2/large-PromptGen-v2.0/model.safetensors
wget -O /comfyui/models/florence2/large-PromptGen-v2.0/modeling_florence2.py https://huggingface.co/chflame163/ComfyUI_LayerStyle/resolve/main/ComfyUI/models/florence2/large-PromptGen-v2.0/modeling_florence2.py
wget -O /comfyui/models/florence2/large-PromptGen-v2.0/preprocessor_config.json https://huggingface.co/chflame163/ComfyUI_LayerStyle/resolve/main/ComfyUI/models/florence2/large-PromptGen-v2.0/preprocessor_config.json
wget -O /comfyui/models/florence2/large-PromptGen-v2.0/processing_florence2.py https://huggingface.co/chflame163/ComfyUI_LayerStyle/resolve/main/ComfyUI/models/florence2/large-PromptGen-v2.0/processing_florence2.py
wget -O /comfyui/models/florence2/large-PromptGen-v2.0/special_tokens_map.json https://huggingface.co/chflame163/ComfyUI_LayerStyle/resolve/main/ComfyUI/models/florence2/large-PromptGen-v2.0/special_tokens_map.json
wget -O /comfyui/models/florence2/large-PromptGen-v2.0/tokenizer.json https://huggingface.co/chflame163/ComfyUI_LayerStyle/resolve/main/ComfyUI/models/florence2/large-PromptGen-v2.0/tokenizer.json
wget -O /comfyui/models/florence2/large-PromptGen-v2.0/tokenizer_config.json https://huggingface.co/chflame163/ComfyUI_LayerStyle/resolve/main/ComfyUI/models/florence2/large-PromptGen-v2.0/tokenizer_config.json
wget -O /comfyui/models/florence2/large-PromptGen-v2.0/vocab.json https://huggingface.co/chflame163/ComfyUI_LayerStyle/resolve/main/ComfyUI/models/florence2/large-PromptGen-v2.0/vocab.json

echo "runpod-worker-comfy: Starting ComfyUI"
python3 /comfyui/main.py --disable-auto-launch --listen 0.0.0.0 --port 8188 &

echo "runpod-worker-comfy: Starting RunPod Handler"
python3 -u /rp_handler.py