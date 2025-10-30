#!/bin/bash

# Configure firewall for Talent Factory - Local subnet access only

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

echo "Configuring firewall for Talent Factory..."

# Detect firewall system
if command -v ufw &> /dev/null; then
    FIREWALL_SYSTEM="ufw"
elif command -v firewall-cmd &> /dev/null; then
    FIREWALL_SYSTEM="firewalld"
elif command -v iptables &> /dev/null; then
    FIREWALL_SYSTEM="iptables"
else
    echo "No supported firewall system detected (ufw, firewalld, or iptables)"
    echo "Please configure your firewall manually to allow local subnet access only"
    exit 1
fi

echo "Detected firewall system: $FIREWALL_SYSTEM"

# Configure based on detected firewall system
case $FIREWALL_SYSTEM in
    "ufw")
        echo "Configuring UFW firewall..."
        
        # Reset UFW to defaults
        ufw --force reset
        
        # Set default policies
        ufw default deny incoming
        ufw default allow outgoing
        
        # Allow SSH (important!)
        ufw allow ssh
        
        # Allow local subnet access to Talent Factory ports
        ufw allow from 192.168.0.0/16 to any port 80
        ufw allow from 192.168.0.0/16 to any port 443
        ufw allow from 192.168.0.0/16 to any port 8084
        ufw allow from 192.168.0.0/16 to any port 3004
        
        # Allow 10.x.x.x subnet
        ufw allow from 10.0.0.0/8 to any port 80
        ufw allow from 10.0.0.0/8 to any port 443
        ufw allow from 10.0.0.0/8 to any port 8084
        ufw allow from 10.0.0.0/8 to any port 3004
        
        # Allow 172.16.x.x subnet
        ufw allow from 172.16.0.0/12 to any port 80
        ufw allow from 172.16.0.0/12 to any port 443
        ufw allow from 172.16.0.0/12 to any port 8084
        ufw allow from 172.16.0.0/12 to any port 3004
        
        # Allow localhost
        ufw allow from 127.0.0.1 to any port 80
        ufw allow from 127.0.0.1 to any port 443
        ufw allow from 127.0.0.1 to any port 8084
        ufw allow from 127.0.0.1 to any port 3004
        
        # Enable UFW
        ufw --force enable
        
        echo "UFW firewall configured successfully"
        ;;
        
    "firewalld")
        echo "Configuring FirewallD..."
        
        # Start and enable firewalld
        systemctl start firewalld
        systemctl enable firewalld
        
        # Create Talent Factory zone
        firewall-cmd --permanent --new-zone=talentfactory
        
        # Set zone target
        firewall-cmd --permanent --zone=talentfactory --set-target=ACCEPT
        
        # Add services
        firewall-cmd --permanent --zone=talentfactory --add-service=http
        firewall-cmd --permanent --zone=talentfactory --add-service=https
        
        # Add ports
        firewall-cmd --permanent --zone=talentfactory --add-port=8084/tcp
        firewall-cmd --permanent --zone=talentfactory --add-port=3004/tcp
        
        # Add sources (local subnets)
        firewall-cmd --permanent --zone=talentfactory --add-source=192.168.0.0/16
        firewall-cmd --permanent --zone=talentfactory --add-source=10.0.0.0/8
        firewall-cmd --permanent --zone=talentfactory --add-source=172.16.0.0/12
        firewall-cmd --permanent --zone=talentfactory --add-source=127.0.0.1
        
        # Reload firewall
        firewall-cmd --reload
        
        echo "FirewallD configured successfully"
        ;;
        
    "iptables")
        echo "Configuring iptables..."
        
        # Flush existing rules
        iptables -F
        iptables -X
        iptables -t nat -F
        iptables -t nat -X
        iptables -t mangle -F
        iptables -t mangle -X
        
        # Set default policies
        iptables -P INPUT DROP
        iptables -P FORWARD DROP
        iptables -P OUTPUT ACCEPT
        
        # Allow loopback
        iptables -A INPUT -i lo -j ACCEPT
        
        # Allow established connections
        iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
        
        # Allow SSH
        iptables -A INPUT -p tcp --dport 22 -j ACCEPT
        
        # Allow local subnets to access Talent Factory ports
        iptables -A INPUT -s 192.168.0.0/16 -p tcp --dport 80 -j ACCEPT
        iptables -A INPUT -s 192.168.0.0/16 -p tcp --dport 443 -j ACCEPT
        iptables -A INPUT -s 192.168.0.0/16 -p tcp --dport 8084 -j ACCEPT
        iptables -A INPUT -s 192.168.0.0/16 -p tcp --dport 3004 -j ACCEPT
        
        iptables -A INPUT -s 10.0.0.0/8 -p tcp --dport 80 -j ACCEPT
        iptables -A INPUT -s 10.0.0.0/8 -p tcp --dport 443 -j ACCEPT
        iptables -A INPUT -s 10.0.0.0/8 -p tcp --dport 8084 -j ACCEPT
        iptables -A INPUT -s 10.0.0.0/8 -p tcp --dport 3004 -j ACCEPT
        
        iptables -A INPUT -s 172.16.0.0/12 -p tcp --dport 80 -j ACCEPT
        iptables -A INPUT -s 172.16.0.0/12 -p tcp --dport 443 -j ACCEPT
        iptables -A INPUT -s 172.16.0.0/12 -p tcp --dport 8084 -j ACCEPT
        iptables -A INPUT -s 172.16.0.0/12 -p tcp --dport 3004 -j ACCEPT
        
        # Allow localhost
        iptables -A INPUT -s 127.0.0.1 -p tcp --dport 80 -j ACCEPT
        iptables -A INPUT -s 127.0.0.1 -p tcp --dport 443 -j ACCEPT
        iptables -A INPUT -s 127.0.0.1 -p tcp --dport 8084 -j ACCEPT
        iptables -A INPUT -s 127.0.0.1 -p tcp --dport 3004 -j ACCEPT
        
        # Save iptables rules
        if command -v iptables-save &> /dev/null; then
            iptables-save > /etc/iptables/rules.v4
        fi
        
        echo "iptables configured successfully"
        ;;
esac

echo ""
echo "Firewall configuration completed!"
echo ""
echo "Allowed access:"
echo "  - Local subnets: 192.168.x.x, 10.x.x.x, 172.16.x.x"
echo "  - Localhost: 127.0.0.1"
echo "  - Ports: 80 (HTTP), 443 (HTTPS), 8084 (API), 3004 (UI)"
echo ""
echo "External access is blocked for security."
echo ""
echo "To check firewall status:"
case $FIREWALL_SYSTEM in
    "ufw")
        echo "  ufw status verbose"
        ;;
    "firewalld")
        echo "  firewall-cmd --list-all-zones"
        ;;
    "iptables")
        echo "  iptables -L -n"
        ;;
esac
