# Persona Foundry Integration Guide

## Overview

The Talent Factory now produces **Persona Foundry-ready** avatar talents that can be discovered, loaded, and mixed by Persona Foundry with zero manual glue.

## Available Persona Foundry Models

### Base Models (For Creating Personas)
- **`runwayml/stable-diffusion-v1-5`** - Stable Diffusion 1.5 base model for avatar creation
- **`stabilityai/stable-diffusion-xl-base-1.0`** - Stable Diffusion XL for high-quality persona creation
- **`microsoft/DialoGPT-medium`** - Base conversational model for persona personality training
- **`microsoft/DialoGPT-large`** - Advanced conversational model for complex persona personalities

### Style Talents (Reusable Style LoRAs)
- **`avatar.style.anime.mecha`** - Anime mecha style for character design
- **`avatar.style.anime.creature`** - Anime creature style for organic, fantastical design
- **`avatar.style.abstract`** - Abstract style for geometric, minimalist design
- **`avatar.style.animal_traits`** - Animal traits style for anthropomorphic design

## Persona Foundry Integration

### 1. Discovery
Persona Foundry can discover talents by scanning the `/talents/avatar/` directory:

```bash
python -m talent_factory.image.index_avatar_talents
```

Returns JSON array of all valid talent kits with metadata.

### 2. Loading
Each talent kit contains:
- `talent.json` - Manifest with metadata
- `weights.safetensors` - LoRA weights
- `base.txt` - Base model information
- `prompts.json` - Prompt templates
- `examples/` - Sample generated images
- `controls/` - Optional control files
- `LICENSE.txt` - License information

### 3. Training
Persona Foundry uses the base models to train:
- **Identity LoRAs** - Unique character traits and appearance
- **Personality Models** - Conversational behavior and tone
- **Style LoRAs** - Reusable artistic styles

### 4. Mixing
Persona Foundry can mix trained LoRAs using the sliders:
- **Mecha ↔ Creature** (0..1) - Controls `anime.mecha` vs `anime.creature` style
- **Abstract ↔ Animal** (0..1) - Controls `abstract` vs `animal_traits` style

### 5. Rendering
Persona Foundry uses the trained models for:
- **Hero portrait** - Character headshot
- **Full body** - Complete character view
- **Character sheet** - Multiple poses/expressions
- **Expression grid** - Various emotional states

## Talent Kit Structure

```
/talents/avatar/avatar.identity.dot_aurora/
├── talent.json              # Manifest with metadata
├── weights.safetensors      # LoRA weights
├── base.txt                 # Base model info
├── prompts.json             # Prompt templates
├── examples/                # Sample images
├── controls/                # Optional control files
└── LICENSE.txt              # License information
```

## Manifest Schema

Each `talent.json` contains:
```json
{
  "id": "avatar.identity.dot_aurora",
  "category": "Presence/Avatar",
  "kind": "lora",
  "sdx_version": "1.5",
  "base_model": "runwayml/stable-diffusion-v1-5",
  "token": "dot_aurora",
  "lora_rank": 16,
  "default_weight": 1.0,
  "negatives": "photo, hyperrealistic, text, watermark",
  "size_mb": 0.1,
  "created_at": "2025-10-11T22:40:28.145498",
  "metadata": {
    "description": "Unique identity LoRA for Dot Aurora character",
    "tags": ["identity", "character", "dot_aurora", "persona"],
    "author": "Talent Factory",
    "license": "Private"
  }
}
```

## Persona Foundry Slider Mapping

### Style Sliders
- **Mecha (0.0) ↔ Creature (1.0)**
  - 0.0-0.5: `avatar.style.anime.mecha` (weight: 1.0-0.0)
  - 0.5-1.0: `avatar.style.anime.creature` (weight: 0.0-1.0)

- **Abstract (0.0) ↔ Animal (1.0)**
  - 0.0-0.5: `avatar.style.abstract` (weight: 1.0-0.0)
  - 0.5-1.0: `avatar.style.animal_traits` (weight: 0.0-1.0)

### Base Model Selection
- Choose from available base models:
  - `runwayml/stable-diffusion-v1-5` - For avatar creation
  - `stabilityai/stable-diffusion-xl-base-1.0` - For high-quality personas
  - `microsoft/DialoGPT-medium` - For personality training
  - `microsoft/DialoGPT-large` - For advanced personalities

## API Integration

### HTTP Endpoints
- `GET /talents/avatar` - List all available avatar talents
- `POST /avatar/style` - Apply style sliders and palette
- `POST /avatar/render` - Render template → image path

### WebSocket Events
- `avatar.setStyle` - Update style sliders
- `avatar.state` - Change animation state
- `avatar.speak` - Trigger speech with lip-sync

## Example Persona Kit

```json
{
  "dot_id": "aurora",
  "version": "1.0.0",
  "sliders": {"mecha": 0.35, "animal": 0.65},
  "finish": "flat",
  "palette": {"primary": "#00A8FF", "secondary": "#FFD166", "accent": "#8A2BE2"},
  "voice": {"provider": "piper", "voice": "en_GB-alba"},
  "tone": {"warmth": 0.7, "formality": 0.3, "humour": 0.5, "pace": "medium"},
  "phrasebook": {"say": ["Got it!"], "avoid": ["As an AI, ..."]},
  "boundaries": {"medical": "refer_to_gp_talent", "legal": "avoid_specifics"}
}
```

## Training New Talents

### Using Base Models
The Talent Factory provides base models that Persona Foundry can use to train:

1. **Select a base model** from the Persona Foundry category
2. **Upload training data** (images for avatars, conversations for personality)
3. **Train LoRA adapters** using the base model
4. **Export talent kits** for Persona Foundry integration

### Style Talents
```bash
# Create new style talent using base model
python -m talent_factory.image.train_avatar_lora --config configs/avatar/new_style.yaml
```

## Validation

```bash
# Validate all talent kits
python -m talent_factory.image.validator --verbose

# Validate specific talent
python -m talent_factory.image.validator --talent-id avatar.identity.dot_aurora
```

## Indexing

```bash
# Index all talents for Persona Foundry
python -m talent_factory.image.index_avatar_talents --output var/avatar_talents.json

# Pretty print index
python -m talent_factory.image.index_avatar_talents --pretty
```

## Next Steps

1. **Persona Foundry Implementation** - Build the UI and rendering engine
2. **Real Training** - Replace mock training with actual LoRA fine-tuning
3. **Example Generation** - Auto-generate sample images for each talent
4. **Control Assets** - Add OpenPose/Lineart/Depth control files
5. **SDXL Support** - Add SDXL talent variants
6. **VLM Integration** - Add vision-language model support

## Files Created

- `schemas/avatar_talent.json` - Manifest schema
- `image/talent_kit.py` - Kit management
- `image/config_loader.py` - Config validation
- `image/train_avatar_lora.py` - Training wrapper
- `image/validator.py` - Validation CLI
- `image/indexer.py` - Indexing CLI
- `configs/avatar/*.yaml` - Training configurations
- `talents/avatar/*/` - Generated talent kits
- `README_avatar.md` - Avatar training documentation
- `PERSONA_FOUNDRY_INTEGRATION.md` - This integration guide
