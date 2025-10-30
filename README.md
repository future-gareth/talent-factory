# Talent Factory

**Your local AI workshop – where Dots learn new skills.**

Talent Factory is a local AI workshop for creating, evaluating, and publishing fine-tuned models (*Talents*) that Dots can later use. It runs as a LAN-accessible service with a full visual UI, exposing an MCP Talent Catalogue for integration with Dot Home.

## Features

- **Modern Web UI**: Next.js + Tailwind + ShadCN interface with real-time updates
- **Visual Fine-Tuning**: Complete web UI with no CLI required
- **Hardware Auto-Detection**: Automatically detects GPU/CPU capabilities and filters compatible models
- **Data Preparation**: Upload, clean, mask PII, and validate datasets inline
- **Real-Time Training**: Live progress monitoring with WebSocket updates
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

2. **Start Backend Service:**
   ```bash
   python3 main.py
   ```

3. **Start UI Server:**
   ```bash
   cd ui
   python3 -m http.server 3004
   ```

4. **Open in Browser:**
   - Talent Factory: http://localhost:3004
   - Backend API: http://localhost:8084
   - MCP Catalogue: http://localhost:8084/mcp/talents

## How It Works

### 1. Dashboard
- View system status and hardware compatibility
- Monitor active training runs
- Browse existing talents
- Check environment profile (GPU, VRAM, RAM)

### 2. New Talent Wizard
- **Step 1: Choose Model** - Select compatible base model
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
- `GET /models/list` - List compatible models
- `POST /dataset/ingest` - Upload dataset
- `POST /dataset/clean` - Clean PII from dataset
- `POST /train/start` - Start model training
- `GET /train/status/{id}` - Get training progress
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
- **Storage**: 50GB free space
- **Network**: Local network access

### Recommended Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CPU**: 8+ cores, 16GB+ RAM
- **Storage**: 100GB+ free space
- **Network**: Gigabit Ethernet

### Supported Models
- **Llama 2 7B** - 8GB VRAM minimum
- **Llama 2 13B** - 16GB VRAM minimum
- **Mistral 7B** - 8GB VRAM minimum
- **CodeLlama 7B** - 8GB VRAM minimum

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
├── backend/                 # FastAPI backend service
│   ├── main.py             # Main application
│   ├── training_engine.py  # LoRA/PEFT training engine
│   ├── mcp_catalogue.py    # MCP API endpoints
│   ├── security.py         # Security and audit features
│   ├── requirements.txt    # Python dependencies
│   └── venv/               # Virtual environment
├── ui/                      # Web UI
│   └── index.html          # Main UI interface
├── models/                  # Trained models storage
├── datasets/                # Training datasets
├── logs/                    # Audit and training logs
├── certs/                   # SSL certificates (optional)
├── avahi/                   # mDNS service definition
├── start-talent-factory.sh  # Startup script
└── README.md               # This file
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
1. Update `get_model_info()` in `training_engine.py`
2. Add model configuration to `get_compatible_models()` in `main.py`
3. Test model loading and training

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

## Troubleshooting

### Backend Service Not Starting
- Check if port 8084 is available
- Ensure Python dependencies are installed (`pip install -r requirements.txt`)
- Check the console for error messages

### UI Not Loading
- Verify the UI server is running on port 3004
- Check browser console for errors
- Ensure all files are in the correct locations

### mDNS Not Working
- Ensure avahi-daemon is running: `systemctl status avahi-daemon`
- Check firewall settings allow mDNS traffic
- Verify hostname configuration: `hostname` should return `talentfactory`

### Authentication Issues
- Check auth configuration: `curl http://localhost:8084/auth/status`
- Enable/disable auth: `curl -X POST http://localhost:8084/auth/enable -d '{"username":"admin","password":"secret"}'`

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs for error messages
3. Create an issue in the repository
4. Contact the development team

---

**Talent Factory** - Where Dots learn new skills, locally and securely.
