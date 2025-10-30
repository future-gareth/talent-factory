# Talent Factory MCP Server Guide

## Overview

The Talent Factory exposes an MCP (Model Context Protocol) server that allows Dot Home and other clients to discover, browse, test, and use trained AI talents. The MCP server provides a standardized interface for talent management and inference.

## Service Discovery

### Endpoint: `GET /mcp/discovery`

Discover available MCP endpoints and capabilities.

**Request:**
```bash
curl http://localhost:8084/mcp/discovery
```

**Response:**
```json
{
  "service_name": "Talent Factory",
  "service_type": "talent_catalogue",
  "version": "1.0.0",
  "endpoints": [
    "/mcp/talents",
    "/mcp/talents/{id}",
    "/mcp/talents/{id}/test",
    "/mcp/talents/{id}/infer",
    "/mcp/discovery"
  ],
  "capabilities": [
    "talent_discovery",
    "talent_testing",
    "model_inference"
  ]
}
```

## Talent Discovery

### Endpoint: `GET /mcp/talents`

List all available active talents in the catalogue.

**Request:**
```bash
curl http://localhost:8084/mcp/talents
```

**Response:**
```json
{
  "talents": [
    {
      "id": "talent_1761573190224",
      "name": "test1",
      "category": "general",
      "version": "1.0.0",
      "metrics": {
        "accuracy": 0.85,
        "f1_score": 0.82
      },
      "status": "active",
      "created_at": "2025-10-27 13:53:10"
    }
  ]
}
```

### Endpoint: `GET /mcp/talents/{talent_id}`

Get detailed information about a specific talent.

**Request:**
```bash
curl http://localhost:8084/mcp/talents/talent_1761573190224
```

**Response:**
```json
{
  "id": "talent_1761573190224",
  "name": "test1",
  "category": "general",
  "version": "1.0.0",
  "description": "A general-purpose AI assistant",
  "metrics": {
    "accuracy": 0.85,
    "f1_score": 0.82
  },
  "status": "active",
  "created_at": "2025-10-27 13:53:10",
  "updated_at": "2025-10-27 13:53:10",
  "model_path": "/path/to/model",
  "base_model": "llama-2-7b",
  "safety_score": 0.92,
  "rubric_passed": true
}
```

### Endpoint: `GET /mcp/talents/{talent_id}/metadata`

Get comprehensive metadata for a talent, including capabilities and usage limits.

**Request:**
```bash
curl http://localhost:8084/mcp/talents/talent_1761573190224/metadata
```

**Response:**
```json
{
  "talent": {
    "id": "talent_1761573190224",
    "name": "test1",
    ...
  },
  "metadata": {
    "training_data": "...",
    "training_params": "..."
  },
  "capabilities": [
    "text_generation",
    "conversation",
    "task_completion"
  ],
  "usage_limits": {
    "max_tokens": 2048,
    "rate_limit": "100 requests/hour"
  }
}
```

## Talent Testing

### Endpoint: `GET /mcp/talents/{talent_id}/test`

Test a talent with a demo prompt to see sample output.

**Request:**
```bash
curl http://localhost:8084/mcp/talents/talent_1761573190224/test?test_input=Hello
```

**Optional Query Parameters:**
- `test_input` - Custom test prompt (default: "Hello, how are you today?")

**Response:**
```json
{
  "talent_id": "talent_1761573190224",
  "name": "test1",
  "category": "general",
  "test_input": "Hello",
  "test_output": "I'm test1, your AI assistant. I'm here to help you with various tasks and questions. How can I assist you today?",
  "confidence": 0.92,
  "response_time_ms": 150,
  "timestamp": "2025-10-27T14:01:41.022499"
}
```

## Running Inference

### Endpoint: `POST /mcp/talents/{talent_id}/infer`

Run actual inference on a talent with a custom prompt.

**Request:**
```bash
curl -X POST http://localhost:8084/mcp/talents/talent_1761573190224/infer \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How do I create a REST API?",
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

**Request Body:**
```json
{
  "prompt": "Your question or prompt here",
  "max_tokens": 512,        // Optional, default: 512
  "temperature": 0.7,        // Optional, default: 0.7
  "top_p": 0.9              // Optional, default: 0.9
}
```

**Parameters:**
- `prompt` (required) - The input text to send to the talent
- `max_tokens` (optional) - Maximum number of tokens to generate (default: 512)
- `temperature` (optional) - Sampling temperature, 0.0-1.0 (default: 0.7)
- `top_p` (optional) - Nucleus sampling parameter (default: 0.9)

**Response:**
```json
{
  "talent_id": "talent_1761573190224",
  "prompt": "How do I create a REST API?",
  "response": "I'm test1, your AI assistant. I understand you're asking about: How do I create a REST API?. Let me provide a helpful response... I'd be happy to answer your question in detail.",
  "confidence": 0.85,
  "response_time_ms": 0,
  "tokens_generated": 32,
  "timestamp": "2025-10-27T14:01:41.022499"
}
```

## Health and Status

### Endpoint: `GET /mcp/health`

Check the health status of the MCP server.

**Request:**
```bash
curl http://localhost:8084/mcp/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "MCP Talent Catalogue",
  "version": "1.0.0",
  "talents_count": 4
}
```

## Real-time Events

### Endpoint: `GET /mcp/talents/events`

Get recent talent events (publications, updates).

**Request:**
```bash
curl http://localhost:8084/mcp/talents/events
```

**Response:**
```json
{
  "events": [
    {
      "action": "talent_published",
      "talent_id": "talent_1761573190224",
      "timestamp": "2025-10-27T13:53:10",
      "user": "admin"
    }
  ],
  "timestamp": "2025-10-27T14:01:34.631098"
}
```

### Endpoint: `GET /mcp/talents/subscribe` (WebSocket)

Subscribe to real-time talent events via WebSocket.

**Connection:**
```bash
wscat -c ws://localhost:8084/mcp/talents/subscribe
```

**Subscribe Message:**
```json
{
  "type": "subscribe"
}
```

**Event Messages:**
The server will send events when talents are published or updated:
```json
{
  "type": "talent_event",
  "events": [
    {
      "action": "talent_published",
      "talent_id": "talent_123",
      "timestamp": "2025-10-27T14:00:00"
    }
  ],
  "timestamp": "2025-10-27T14:00:00"
}
```

## Usage Examples

### Complete Workflow

1. **Discover available talents:**
```bash
# Get list of all talents
curl http://localhost:8084/mcp/talents
```

2. **Get talent details:**
```bash
# Get specific talent information
curl http://localhost:8084/mcp/talents/talent_1761573190224
```

3. **Test the talent:**
```bash
# Test with default prompt
curl http://localhost:8084/mcp/talents/talent_1761573190224/test

# Test with custom prompt
curl "http://localhost:8084/mcp/talents/talent_1761573190224/test?test_input=Explain%20machine%20learning"
```

4. **Run inference:**
```bash
# Run inference with custom prompt
curl -X POST http://localhost:8084/mcp/talents/talent_1761573190224/infer \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain the concept of fine-tuning in machine learning",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

## Integration with Dot Home

### Initial Setup

1. **Discover Talent Factory:**
```bash
curl http://localhost:8084/mcp/discovery
```

2. **Cache available talents:**
```bash
# Fetch and cache available talents
curl http://localhost:8084/mcp/talents > talents.json
```

### Using a Talent

1. **User selects a talent in Dot Home UI**
2. **Dot Home sends inference request:**
```bash
curl -X POST http://localhost:8084/mcp/talents/{talent_id}/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "User question here"}'
```

3. **Display the response to the user**

## Best Practices

### Temperature Settings

- **Low (0.1-0.3)**: More deterministic, conservative responses
- **Medium (0.5-0.7)**: Balanced creativity and coherence (recommended)
- **High (0.8-1.0)**: More creative, diverse responses

### Token Limits

- **Short responses (64-128 tokens)**: Quick answers, confirmations
- **Medium responses (256-512 tokens)**: Standard explanations
- **Long responses (1024+ tokens)**: Detailed explanations, stories

### Error Handling

All endpoints may return errors. Common status codes:

- `200 OK` - Success
- `404 Not Found` - Talent not found
- `500 Internal Server Error` - Server error

Example error response:
```json
{
  "detail": "Talent not found"
}
```

## Rate Limiting

The MCP server implements rate limiting to prevent abuse:

- **Default**: 100 requests/hour per IP
- **Burst**: 10 requests/minute
- **WebSocket**: Unlimited but connection-based

## Network Configuration

### Local Network Access

The Talent Factory MCP server is accessible on:

- **Host**: `localhost` (local) or `talentfactory.local` (mDNS)
- **Port**: `8084`
- **Protocol**: HTTP/HTTPS

### For Different Environments

**Development:**
```
http://localhost:8084
```

**Production (mDNS):**
```
http://talentfactory.local:8084
```

**Direct IP:**
```
http://192.168.1.100:8084
```

## Troubleshooting

### Talent Not Found

If you get a 404 error:
1. Verify the talent ID is correct
2. Check if the talent status is "active"
3. List all talents: `GET /mcp/talents`

### Slow Inference

If inference is slow:
1. Check system resources (GPU/CPU)
2. Reduce `max_tokens` in the request
3. Verify the talent model is properly loaded

### Connection Issues

If connection fails:
1. Verify Talent Factory is running
2. Check the port (default: 8084)
3. Verify firewall settings

## Advanced Features

### WebSocket Events

Subscribe to real-time events for new talent publications:

```javascript
const ws = new WebSocket('ws://localhost:8084/mcp/talents/subscribe');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('New talent event:', data);
};

ws.send(JSON.stringify({ type: 'subscribe' }));
```

### Batch Operations

Test multiple talents in parallel:

```bash
# Test multiple talents
for talent_id in talent_123 talent_456 talent_789; do
  curl "http://localhost:8084/mcp/talents/${talent_id}/test"
done
```

## Security Considerations

- All MCP endpoints are read-only (for talent discovery)
- Inference endpoints respect rate limits
- Local-first: All data stays on your machine
- Authentication can be enabled via settings

## Support

For issues or questions:
1. Check logs: `tail -f /tmp/talent-factory.log`
2. View API docs: `http://localhost:8084/docs`
3. Review this guide
4. Contact the development team

---

**Talent Factory MCP Server** - Enabling local AI talents for Dot Home
