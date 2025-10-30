#!/bin/bash

# Generate self-signed SSL certificates for Talent Factory

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CERTS_DIR="$BASE_DIR/certs"

# Create certs directory if it doesn't exist
mkdir -p "$CERTS_DIR"

# Check if certificates already exist
if [ -f "$CERTS_DIR/talentfactory.local.crt" ] && [ -f "$CERTS_DIR/talentfactory.local.key" ]; then
    echo "SSL certificates already exist at $CERTS_DIR"
    echo "To regenerate, delete the existing certificates and run this script again."
    exit 0
fi

echo "Generating self-signed SSL certificates for talentfactory.local..."

# Generate private key
openssl genrsa -out "$CERTS_DIR/talentfactory.local.key" 4096

# Generate certificate signing request
openssl req -new -key "$CERTS_DIR/talentfactory.local.key" -out "$CERTS_DIR/talentfactory.local.csr" \
    -subj "/C=US/ST=State/L=City/O=Talent Factory/OU=IT Department/CN=talentfactory.local"

# Generate self-signed certificate
openssl x509 -req -days 365 -in "$CERTS_DIR/talentfactory.local.csr" \
    -signkey "$CERTS_DIR/talentfactory.local.key" \
    -out "$CERTS_DIR/talentfactory.local.crt" \
    -extensions v3_req \
    -extfile <(cat <<EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = State
L = City
O = Talent Factory
OU = IT Department
CN = talentfactory.local

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = talentfactory.local
DNS.2 = localhost
DNS.3 = *.local
IP.1 = 127.0.0.1
IP.2 = ::1
EOF
)

# Set permissions
chmod 600 "$CERTS_DIR/talentfactory.local.key"
chmod 644 "$CERTS_DIR/talentfactory.local.crt"

# Clean up CSR file
rm "$CERTS_DIR/talentfactory.local.csr"

echo "SSL certificates generated successfully!"
echo "Certificate: $CERTS_DIR/talentfactory.local.crt"
echo "Private Key: $CERTS_DIR/talentfactory.local.key"
echo ""
echo "Certificate details:"
openssl x509 -in "$CERTS_DIR/talentfactory.local.crt" -text -noout | grep -E "(Subject:|Issuer:|Not Before|Not After|DNS:|IP:)"

echo ""
echo "To use HTTPS, configure your web server to use these certificates."
echo "For nginx, add SSL configuration to your server block:"
echo ""
echo "server {"
echo "    listen 443 ssl;"
echo "    server_name talentfactory.local;"
echo "    ssl_certificate $CERTS_DIR/talentfactory.local.crt;"
echo "    ssl_certificate_key $CERTS_DIR/talentfactory.local.key;"
echo "    # ... rest of your configuration"
echo "}"
