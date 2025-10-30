#!/bin/bash

# Setup hostname and mDNS for Talent Factory on macOS

set -e

echo "Setting up Talent Factory hostname and mDNS on macOS..."

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "This script should not be run as root on macOS"
    exit 1
fi

# Get the current IP address
LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}')
if [ -z "$LOCAL_IP" ]; then
    echo "Could not determine local IP address"
    exit 1
fi

echo "Local IP address: $LOCAL_IP"

# Add to /etc/hosts (requires sudo)
echo "Adding talentfactory.local to /etc/hosts..."
sudo sh -c "echo '$LOCAL_IP    talentfactory.local' >> /etc/hosts"

# Create a simple mDNS advertisement using dns-sd
echo "Setting up mDNS advertisement..."

# Kill any existing dns-sd processes for talentfactory
pkill -f "dns-sd.*talentfactory" 2>/dev/null || true

# Start mDNS advertisement in background
echo "Starting mDNS advertisement for talentfactory.local..."
dns-sd -R "Talent Factory" _http._tcp local 8084 path=/ &
HTTP_PID=$!

dns-sd -R "Talent Factory MCP" _mcp._tcp local 8084 path=/mcp &
MCP_PID=$!

# Save PIDs for cleanup
echo $HTTP_PID > /tmp/talentfactory-http.pid
echo $MCP_PID > /tmp/talentfactory-mcp.pid

echo ""
echo "✅ Hostname and mDNS setup completed!"
echo ""
echo "🌐 Access Talent Factory at:"
echo "   • http://talentfactory.local:8084 (Backend API)"
echo "   • http://talentfactory.local:3004 (Web UI)"
echo "   • http://talentfactory.local:8084/mcp/talents (MCP Catalogue)"
echo ""
echo "📋 To verify:"
echo "   • ping talentfactory.local"
echo "   • dns-sd -B _http._tcp"
echo ""
echo "🛑 To stop mDNS advertisement:"
echo "   • kill \$(cat /tmp/talentfactory-http.pid)"
echo "   • kill \$(cat /tmp/talentfactory-mcp.pid)"
echo "   • sudo sed -i '' '/talentfactory.local/d' /etc/hosts"
echo ""

# Test the setup
echo "Testing hostname resolution..."
if ping -c 1 talentfactory.local > /dev/null 2>&1; then
    echo "✅ talentfactory.local resolves correctly"
else
    echo "⚠️  talentfactory.local may take a moment to resolve"
fi
