# Talent Factory

**Your local AI workshop – where Dots learn new skills.**

Talent Factory is a local AI workshop for creating, evaluating, and publishing fine-tuned models (*Talents*) that Dots can later use. It runs as a LAN-accessible service with a full visual UI, exposing an MCP Talent Catalogue for integration with Dot Home.

## Features

- **Modern Web UI**: Next.js + Tailwind + ShadCN interface with real-time updates
- **Visual Fine-Tuning**: Complete web UI with no CLI required
- **Hardware Auto-Detection**: Automatically detects GPU/CPU capabilities and filters compatible models
- **Data Preparation**: Upload, clean, mask PII, and validate datasets inline
- **Real-Time Training**: Live progress monitoring streamed over WebSocket
- **Automatic Model Downloads**: Hugging Face integration with local caching and status tracking
- **Evaluation Dashboard**: Base vs tuned model comparison with detailed metrics
- **Safety Evaluation**: Built-in safety checks and rubric assessment
- **MCP Integration**: Exposes Talent Catalogue for Dot Home discovery
- **Local-First Security**: All data stays local by default
- **Audit Trail**: Complete logging for compliance and debugging
- **Optional Authentication**: JWT-based auth system for secure access
- **Feedback System**: Usage metrics and feedback collection for Dot Home
- **Package Management**: .deb and .rpm installers with systemd integration
- **Network Security**: Firewall configuration and mDNS advertising

## Quick Start

### Super Simple Setup (Just Run the Script!)

```bash
cd tools/talent-factory
./start-talent-factory.sh
```

This will:
1. Create a Python virtual environment
2. Install all dependencies
3. Start the backend service on port 8084
4. Start the UI server on port 3004
5. Open Talent Factory in your browser

### Manual Setup

1. **Create Virtual Environment:**
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start Backend Service (FastAPI + WebSocket):**
   ```bash
   python3 main.py
   ```

   The backend listens on `http://localhost:8084` and exposes the WebSocket endpoint at `ws://localhost:8084/ws`.

3. **Install UI Dependencies (first run only):**
   ```bash
   cd ../ui
   npm install
   ```

4. **Start UI Server (Next.js dev server):**
   ```bash
   npm run dev -- --port 3004
   ```
   (Use a different port, e.g. `--port 3200`, if 3004 is already in use.)

5. **Open in Browser:**
   - Talent Factory: http://localhost:3004
   - Backend API: http://localhost:8084
   - MCP Catalogue: http://localhost:8084/mcp/talents

### Model Downloads & Hugging Face Setup

Talent Factory now downloads base models on demand before training. To enable this:

1. **Authenticate with Hugging Face** (only required for gated models):
   ```bash
   huggingface-cli login
   ```
   or set `HF_TOKEN` in your shell. See `HUGGINGFACE_SETUP.md` for the step-by-step guide.

2. **Pick a model in the UI** – each MLX model shows its download status.  
   - Click **Download Model** to cache it locally under `tools/talent-factory/models/`.  
   - Progress is streamed via WebSocket and persisted in the UI.

3. **Train only after the model is cached** – the UI will prevent training until the base model reports `✓ Model ready for training`.

Downloaded models are reused across runs and can be managed via the API (`GET /models/cached`, `DELETE /model/{id}`).

## How It Works

### 1. Dashboard
- View system status and hardware compatibility
- Monitor active training runs
- Browse existing talents
- Check environment profile (GPU, VRAM, RAM)

### 2. New Talent Wizard
- **Step 1: Choose Model** - Select a compatible base model and (if needed) download it from Hugging Face
- **Step 2: Prepare Data** - Upload and clean training dataset
- **Step 3: Train & Evaluate** - Start fine-tuning with live progress
- **Step 4: Publish** - Add to Talent Catalogue for Dot Home

### 3. Talent Catalogue
- Browse published talents
- Test talents with sample inputs
- Export talents for external use
- View talent metadata and metrics

### 4. Settings
- Configure network access and security
- Manage storage locations
- Enable/disable security features
- View audit logs

## API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `GET /env/check` - Hardware environment check
- `GET /models/list` - List all registered models (legacy endpoint)
- `GET /models/available` - List models filtered for the current hardware/backend
- `POST /model/download` - Start Hugging Face model download with progress events
- `GET /model/status/{model_id}` - Check download/cache status for a model
- `GET /models/cached` - List locally cached base models
- `DELETE /model/{model_id}` - Remove a cached base model
- `POST /dataset/ingest` - Upload dataset
- `POST /dataset/clean` - Clean PII from dataset
- `POST /train/start` - Start model training
- `GET /train/status/{id}` - Get training progress
- `POST /train/cleanup` - Manually clear stale training runs
- `POST /evaluate/run` - Run model evaluation
- `POST /talents/publish` - Publish talent
- `GET /dashboard` - Get dashboard data

### MCP Endpoints (for Dot Home)
- `GET /mcp/talents` - List all talents
- `GET /mcp/talents/{id}` - Get talent details
- `GET /mcp/talents/{id}/test` - Test talent
- `GET /mcp/discovery` - Service discovery
- `GET /mcp/health` - MCP health check

## Hardware Requirements

### Minimum Requirements
- **CPU**: 4 cores, 8GB RAM
- **Storage**: 50GB free space (models + datasets)
- **Network**: Local network access

### Recommended Requirements
- **Apple Silicon**: M2/M3 with 16GB+ unified memory (MLX backend)
- **NVIDIA GPU**: 8GB+ VRAM for CUDA/LoRA workflows
- **CPU**: 8+ cores, 16GB+ RAM
- **Storage**: 100GB+ free space for multiple cached models
- **Network**: Gigabit Ethernet

### Supported MLX Models (Apple Silicon)
- **Gemma 2B Instruct (MLX)** – fastest for testing (`mlx-community/gemma-2b-it-4bit`)
- **Phi-2 (MLX)** – efficient small model (`mlx-community/phi-2-4bit`)
- **Mistral 7B Instruct (MLX)** – high quality instruction following (`mlx-community/Mistral-7B-Instruct-v0.1-4bit`)
- **Llama 2 7B Chat (MLX)** – conversational assistant (`mlx-community/Llama-2-7b-chat-hf-4bit`)

CUDA/CPU model support is still available through the legacy training engine, but Apple Silicon + MLX offers the best experience today.

## Security Features

### Local-First Architecture
- All data stays on your machine by default
- No cloud dependencies required
- LAN-only access by default
- Configurable network restrictions

### PII Detection & Masking
- Automatic detection of sensitive information
- Real-time masking of PII in datasets
- Safety scoring for datasets
- Compliance-ready data handling

### Audit Logging
- Complete action logging
- Timestamped audit trail
- Database and file-based logging
- Configurable retention periods

### Network Security
- IP address filtering
- Referer header validation
- Suspicious user agent detection
- Configurable access controls

## File Structure

```
talent-factory/
├── backend/                    # FastAPI backend
│   ├── main.py                # API + WebSocket entrypoint
│   ├── mlx_engine.py          # MLX training implementation
│   ├── model_downloader.py    # Hugging Face download manager
│   ├── websocket_handler.py   # WebSocket connection manager
│   ├── mcp_catalogue.py       # MCP API endpoints
│   └── requirements.txt       # Python dependencies
├── ui/                         # Next.js 13+ app
│   ├── package.json
│   ├── next.config.ts
│   └── src/app/page.tsx       # Main Talent Factory UI
├── models/                     # Cached base models
├── datasets/                   # Uploaded training datasets
├── logs/                       # Backend + training logs
├── certs/                      # TLS certificates (optional)
├── avahi/                      # mDNS service definition
├── start-talent-factory.sh     # One-command startup script
├── HUGGINGFACE_SETUP.md        # Auth/setup guide for model downloads
└── WEBSOCKET_FIX_SUMMARY.md    # Notes on realtime progress updates
```

## Configuration

### Security Configuration
Located in `backend/security_config.json`:

```json
{
  "local_only": true,
  "allowed_networks": ["192.168.0.0/16", "10.0.0.0/8"],
  "block_external": true,
  "require_auth": false,
  "audit_enabled": true,
  "pii_detection": true,
  "data_retention_days": 30
}
```

### Training Configuration
Training parameters are automatically adjusted based on outcome preference:

- **Speed**: Fast training, good results
- **Balanced**: Good speed and quality (default)
- **Quality**: Best results, longer training

Additional hyperparameters (epochs, learning rate, gradient accumulation) are surfaced in the UI and ultimately passed to `backend/mlx_engine.py`. Advanced users can POST to `/train/start` with custom values if different behaviour is required.

## Dot Home Integration

Talent Factory exposes an MCP (Model Context Protocol) API that Dot Home can discover and use:

1. **Discovery**: Dot Home queries `/mcp/discovery` to find Talent Factory
2. **Talent Listing**: Dot Home gets available talents from `/mcp/talents`
3. **Talent Testing**: Dot Home can test talents via `/mcp/talents/{id}/test`
4. **Activation**: Users can enable talents for specific personas in Dot Home

### MCP Service Discovery
Talent Factory advertises itself via mDNS as `talentfactory.local` with:
- HTTP service on port 80/443
- MCP service on port 80
- Automatic discovery on local network

## Troubleshooting

### Common Issues

**Backend Service Not Starting**
- Check if port 8084 is available
- Ensure Python dependencies are installed
- Check the console for error messages
- Verify virtual environment is activated

**UI Not Loading**
- Verify UI server is running on port 3004
- Check browser console for errors
- Ensure all files are in correct locations
- Try refreshing the page

**Training Failures**
- Check GPU availability and VRAM
- Verify dataset format and size
- Check available disk space
- Review training logs in `logs/` directory

**Model Download Stuck or Failing**
- Confirm you're authenticated with Hugging Face (`huggingface-cli whoami`)
- Inspect `/tmp/talent-factory-backend.log` for HTTP errors (401 = auth, 404 = bad model ID)
- Retry from the UI after deleting the partial cache (`DELETE /model/{id}` or remove folder under `models/`)
- Review `HUGGINGFACE_SETUP.md` for full troubleshooting steps

**MCP Integration Issues**
- Verify MCP endpoints are accessible
- Check network connectivity
- Ensure Talent Factory is running
- Review MCP service discovery

### Logs and Debugging

**Audit Logs**: `logs/audit_YYYY-MM-DD.log`
**Training Logs**: `logs/training_*.log`
**Application Logs**: Console output from backend service

**Database**: `registry.db` (SQLite database with all talent metadata)

## Deployment Options

### Package Installation

**Debian/Ubuntu (.deb package):**
```bash
# Build the package
./build-deb.sh

# Install
sudo dpkg -i talent-factory-1.0.0.deb
```

**Red Hat/CentOS (.rpm package):**
```bash
# Build the package
./build-rpm.sh

# Install
sudo rpm -i ~/rpmbuild/RPMS/noarch/talent-factory-1.0.0-1.*.rpm
```

### Docker Deployment

```bash
# Build Docker image
docker build -t talent-factory .

# Run with GPU support
docker run -d --name talentfactory \
  --gpus all \
  -p 80:80 \
  -p 443:443 \
  -v ~/talentfactory:/opt/talent-factory \
  talent-factory:latest
```

### System Service

After package installation, Talent Factory runs as a systemd service:

```bash
# Check status
systemctl status talent-factory

# View logs
journalctl -u talent-factory -f

# Restart service
sudo systemctl restart talent-factory
```

### Network Configuration

**Configure hostname and mDNS:**
```bash
sudo ./scripts/configure-hostname.sh
```

**Configure firewall:**
```bash
sudo ./scripts/configure-firewall.sh
```

**Generate SSL certificates:**
```bash
./scripts/generate-certs.sh
```

## Development

### Adding New Models
1. Update the MLX catalogue in `backend/mlx_engine.py` (`_get_model_info` / `list_available_models`)
2. Register metadata in `backend/main.py` (`get_compatible_models`) so the UI picks it up
3. (Optional) Extend `backend/model_downloader.py` if the model needs custom handling
4. Test model download, caching, and training end-to-end

### Adding New PII Patterns
1. Update `patterns` dictionary in `security.py`
2. Add corresponding replacement in `replacements` dictionary
3. Test PII detection and masking

### Customizing UI
1. Modify `ui/src/app/page.tsx` for layout changes
2. Update Tailwind classes for styling
3. Add new React components for functionality

## Future Enhancements

- **Cloud Sync**: Optional "Dot Cloud" sync for cross-device talents
- **Scheduled Training**: Automated re-training and dataset refresh
- **Distributed Training**: Multi-GPU training support
- **Multi-User**: Role-based access control
- **Studio Integration**: Integration with Futurematic Studio
- **Advanced Evaluation**: More sophisticated safety and performance metrics
- **Model Compression**: Quantization and pruning support
- **Dataset Augmentation**: Automatic data augmentation techniques

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the GarethAPI ecosystem. See the main repository for license information.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs for error messages
3. Create an issue in the repository
4. Contact the development team

---

**Talent Factory** - Where Dots learn new skills, locally and securely.
