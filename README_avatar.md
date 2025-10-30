# Talent Factory - Avatar Training for Persona Foundry

This module extends the Talent Factory to produce avatar-ready model artifacts for the Persona Foundry, starting with image LoRAs (SD 1.5).

## Overview

The Avatar training system creates **Talent Kits** that can be discovered, loaded, and mixed by Persona Foundry with zero manual glue. Each kit contains:

- `talent.json` - Manifest with metadata
- `weights.safetensors` - LoRA weights
- `base.txt` - Base model information
- `prompts.json` - Prompt templates
- `examples/` - Sample generated images
- `controls/` - Optional control files
- `LICENSE.txt` - License information

## Quick Start

### 1. Install Dependencies

**For CUDA (RTX 3060+):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.30.0 transformers accelerate xformers bitsandbytes Pillow opencv-python pyyaml pydantic jsonschema
```

**For Apple Silicon (M1/M2/M3/M4):**
```bash
pip install torch torchvision torchaudio
pip install diffusers==0.30.0 transformers accelerate bitsandbytes Pillow opencv-python pyyaml pydantic jsonschema
export OMP_NUM_THREADS=4
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
```

### 2. Prepare Training Data

**Identity Training:**
```
data/avatar/identity/dot_kairo/images/
├── image1.png
├── image1.txt  # Caption: "dot_kairo, high quality character"
├── image2.jpg
├── image2.txt  # Caption: "dot_kairo, detailed portrait"
└── ...
```

**Style Training:**
```
data/avatar/style/anime_mecha/images/
├── image1.png
├── image1.txt  # Caption: "anime mecha style, detailed"
├── image2.jpg
├── image2.txt  # Caption: "mecha design, high quality"
└── ...
```

### 3. Create Configuration

Copy and modify example configs:
```bash
cp configs/avatar/dot_kairo.yaml configs/avatar/my_character.yaml
# Edit the config file
```

### 4. Train

```bash
python -m talent_factory.image.train_avatar_lora --config configs/avatar/my_character.yaml
```

### 5. Validate and Index

```bash
# Validate all kits
python -m talent_factory.image.validator

# Index valid kits
python -m talent_factory.image.index_avatar_talents > var/avatar_talents.json
```

## Configuration

### YAML Config Structure

```yaml
# Required fields
id: avatar.identity.my_character
kind: lora
sdx_version: "1.5"
base_model: runwayml/stable-diffusion-v1-5
train_data_dir: /data/avatar/identity/my_character/images
resolution: 640
lora_rank: 16
max_train_steps: 4000
learning_rate: 1e-4
batch_size: 1

# Optional fields
token: my_character  # Required for identity talents
negatives: "photo, hyperrealistic, text, watermark"
description: "My custom character"
tags: ["identity", "character"]
author: "Your Name"
license: "Private"
```

### Talent Types

**Identity Talents** (`avatar.identity.*`):
- Unique character LoRAs
- Require `token` field
- Bind silhouette/markings to unique token

**Style Talents** (`avatar.style.*`):
- Reusable style LoRAs
- No token required
- Apply artistic styles (anime, mecha, abstract, etc.)

## Platform Support

### CUDA (RTX 3060+)
- Uses xFormers for memory efficiency
- Gradient checkpointing enabled
- AdamW 8-bit optimizer
- Batch size: 1 (12GB-safe)

### Apple Silicon (M1/M2/M3/M4)
- MPS backend with thermal management
- Gradient accumulation: 8
- BF16/FP16 precision
- Quiet defaults to reduce fan noise

## File Structure

```
talent-factory/
├── schemas/
│   └── avatar_talent.json          # Manifest schema
├── image/
│   ├── talent_kit.py               # Kit management
│   ├── config_loader.py            # Config validation
│   ├── train_avatar_lora.py        # Training wrapper
│   ├── validator.py                # Validation CLI
│   ├── indexer.py                  # Indexing CLI
│   └── index_avatar_talents.py     # Entry point
├── configs/
│   └── avatar/
│       ├── dot_kairo.yaml          # Identity example
│       └── anime_mecha.yaml        # Style example
├── data/
│   └── avatar/
│       ├── identity/
│       │   └── <token>/
│       │       └── images/
│       └── style/
│           └── <style_id>/
│               └── images/
└── talents/
    └── avatar/
        └── <talent_id>/
            ├── talent.json
            ├── weights.safetensors
            ├── base.txt
            ├── prompts.json
            ├── examples/
            ├── controls/
            └── LICENSE.txt
```

## Commands Reference

### Training
```bash
# Train identity LoRA
python -m talent_factory.image.train_avatar_lora --config configs/avatar/dot_kairo.yaml

# Train style LoRA
python -m talent_factory.image.train_avatar_lora --config configs/avatar/anime_mecha.yaml
```

### Validation
```bash
# Validate all kits
python -m talent_factory.image.validator

# Validate specific kit
python -m talent_factory.image.validator --talent-id avatar.identity.dot_kairo

# Verbose output
python -m talent_factory.image.validator --verbose
```

### Indexing
```bash
# Index to stdout
python -m talent_factory.image.index_avatar_talents

# Index to file
python -m talent_factory.image.index_avatar_talents --output var/avatar_talents.json

# Include invalid kits
python -m talent_factory.image.index_avatar_talents --include-invalid
```

## Troubleshooting

### Common Issues

**Out of Memory (CUDA):**
- Reduce `batch_size` to 1
- Enable gradient checkpointing
- Use xFormers
- Reduce `resolution` to 512

**Apple Silicon Thermal Throttling:**
- Set `OMP_NUM_THREADS=4`
- Use MPS watermarks
- Enable gradient accumulation
- Monitor system temperature

**Training Data Issues:**
- Ensure images are 512x512 or larger
- Check caption files exist and are properly formatted
- Verify `train_data_dir` path is correct

### Validation Errors

**Schema Validation Failed:**
- Check `talent.json` format
- Verify required fields are present
- Ensure field types match schema

**Weights Not Found:**
- Check `weights.safetensors` exists
- Verify file permissions
- Ensure training completed successfully

## Integration with Persona Foundry

The Talent Factory produces kits that Persona Foundry can consume directly:

1. **Discovery**: Scan `/talents/avatar/` directory
2. **Loading**: Read `talent.json` manifest
3. **Mixing**: Combine identity + style LoRAs
4. **Rendering**: Use prompt templates from `prompts.json`

## Future Extensions

- **SDXL Support**: `sdx_version: "xl"` with 768px resolution
- **VLM Integration**: Vision-language model support
- **Embeddings**: CLIP embeddings for semantic search
- **Cloud Training**: Optional cloud backend
- **Dataset Tools**: Automated data collection

## License

This module is part of the Talent Factory system. See the main LICENSE file for details.
