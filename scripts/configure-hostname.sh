#!/bin/bash

# Configure hostname and mDNS advertising for Talent Factory

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

echo "Configuring hostname and mDNS advertising for Talent Factory..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root (use sudo)"
    exit 1
fi

# Configure hostname
HOSTNAME="talentfactory"
echo "Setting hostname to $HOSTNAME..."

# Set hostname temporarily
hostname "$HOSTNAME"

# Set hostname permanently
echo "$HOSTNAME" > /etc/hostname

# Update /etc/hosts
if ! grep -q "talentfactory.local" /etc/hosts; then
    echo "127.0.0.1    talentfactory.local" >> /etc/hosts
    echo "::1          talentfactory.local" >> /etc/hosts
fi

# Install and configure Avahi for mDNS
echo "Installing and configuring Avahi for mDNS..."

# Install Avahi if not present
if ! command -v avahi-daemon &> /dev/null; then
    if command -v apt-get &> /dev/null; then
        apt-get update
        apt-get install -y avahi-daemon avahi-utils
    elif command -v yum &> /dev/null; then
        yum install -y avahi avahi-tools
    elif command -v dnf &> /dev/null; then
        dnf install -y avahi avahi-tools
    else
        echo "Package manager not supported. Please install avahi-daemon manually."
        exit 1
    fi
fi

# Create Avahi service definition
AVAHI_SERVICE="/etc/avahi/services/talentfactory.service"
cat > "$AVAHI_SERVICE" << 'EOF'
<?xml version="1.0" standalone="no"?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
  <name replace-wildcards="yes">Talent Factory on %h</name>
  <service>
    <type>_http._tcp</type>
    <port>80</port>
    <txt-record>path=/</txt-record>
    <txt-record>version=1.0.0</txt-record>
    <txt-record>service=talent-factory</txt-record>
    <txt-record>description=Talent Factory - Your Local AI Workshop</txt-record>
  </service>
  <service>
    <type>_mcp._tcp</type>
    <port>80</port>
    <txt-record>path=/mcp</txt-record>
    <txt-record>version=1.0.0</txt-record>
    <txt-record>service=talent-catalogue</txt-record>
    <txt-record>description=MCP Talent Catalogue for Dot Home</txt-record>
  </service>
  <service>
    <type>_talentfactory._tcp</type>
    <port>8084</port>
    <txt-record>path=/api</txt-record>
    <txt-record>version=1.0.0</txt-record>
    <txt-record>service=talent-factory-api</txt-record>
    <txt-record>description=Talent Factory API</txt-record>
  </service>
</service-group>
EOF

# Configure Avahi daemon
AVAHI_CONFIG="/etc/avahi/avahi-daemon.conf"
if [ -f "$AVAHI_CONFIG" ]; then
    # Backup original config
    cp "$AVAHI_CONFIG" "$AVAHI_CONFIG.backup"
    
    # Update configuration
    sed -i 's/#host-name=.*/host-name=talentfactory/' "$AVAHI_CONFIG"
    sed -i 's/#domain-name=.*/domain-name=local/' "$AVAHI_CONFIG"
    sed -i 's/#enable-dbus=.*/enable-dbus=yes/' "$AVAHI_CONFIG"
    sed -i 's/#disallow-other-stacks=.*/disallow-other-stacks=no/' "$AVAHI_CONFIG"
    sed -i 's/#allow-interfaces=.*/allow-interfaces=eth0,wlan0/' "$AVAHI_CONFIG"
fi

# Start and enable Avahi
systemctl enable avahi-daemon
systemctl start avahi-daemon

# Configure nginx for hostname
NGINX_CONFIG="/etc/nginx/sites-available/talentfactory"
if [ -f "$NGINX_CONFIG" ]; then
    # Update server_name to include talentfactory.local
    sed -i 's/server_name .*/server_name talentfactory.local localhost;/' "$NGINX_CONFIG"
fi

# Test mDNS resolution
echo "Testing mDNS resolution..."
sleep 2

if command -v avahi-resolve &> /dev/null; then
    echo "Available services:"
    avahi-browse -a -t
fi

# Test hostname resolution
echo "Testing hostname resolution..."
if ping -c 1 talentfactory.local &> /dev/null; then
    echo "✓ talentfactory.local resolves correctly"
else
    echo "⚠ talentfactory.local may not resolve yet (this is normal for mDNS)"
fi

# Create a simple test page
TEST_PAGE="/var/www/html/talentfactory-test.html"
cat > "$TEST_PAGE" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Talent Factory - Hostname Test</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .success { color: green; }
        .info { color: blue; }
    </style>
</head>
<body>
    <h1>Talent Factory Hostname Test</h1>
    <p class="success">✓ Hostname configured successfully!</p>
    <p class="info">Hostname: <strong>talentfactory.local</strong></p>
    <p class="info">Access Talent Factory at: <a href="http://talentfactory.local">http://talentfactory.local</a></p>
    <p class="info">MCP Catalogue: <a href="http://talentfactory.local/mcp/talents">http://talentfactory.local/mcp/talents</a></p>
    <hr>
    <p><small>This page confirms that the hostname and mDNS advertising are working correctly.</small></p>
</body>
</html>
EOF

echo ""
echo "Hostname and mDNS configuration completed!"
echo ""
echo "Configuration summary:"
echo "  ✓ Hostname set to: talentfactory"
echo "  ✓ mDNS service: talentfactory.local"
echo "  ✓ HTTP service advertised on port 80"
echo "  ✓ MCP service advertised on port 80"
echo "  ✓ API service advertised on port 8084"
echo ""
echo "Access points:"
echo "  • Web UI: http://talentfactory.local"
echo "  • MCP API: http://talentfactory.local/mcp/talents"
echo "  • Backend API: http://talentfactory.local:8084"
echo ""
echo "Test page: http://talentfactory.local/talentfactory-test.html"
echo ""
echo "To verify mDNS advertising:"
echo "  • From another device on the network: ping talentfactory.local"
echo "  • From macOS: dns-sd -B _http._tcp"
echo "  • From Linux: avahi-browse -a -t"
echo ""
echo "Note: It may take a few minutes for mDNS to propagate across the network."
