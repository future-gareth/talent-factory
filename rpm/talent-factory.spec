Name:           talent-factory
Version:        1.0.0
Release:        1%{?dist}
Summary:        Talent Factory - Your Local AI Workshop

License:        MIT
URL:            https://github.com/garethapi/talent-factory
Source0:        %{name}-%{version}.tar.gz

BuildArch:      noarch
Requires:       python3 >= 3.8, python3-pip, nginx, avahi, openssl, curl, wget

%description
Talent Factory is a local AI workshop for creating, evaluating, and publishing
fine-tuned models (Talents) that Dots can later use. It runs as a LAN-accessible
service with a full visual UI, exposing an MCP Talent Catalogue for integration
with Dot Home.

Features:
- Visual fine-tuning with no CLI required
- Hardware auto-detection and model filtering
- Data preparation with PII masking
- Real-time training progress monitoring
- Safety evaluation and rubric assessment
- MCP integration for Dot Home discovery
- Local-first security with audit trail

%prep
%setup -q

%build
# No build step required for Python application

%install
rm -rf $RPM_BUILD_ROOT

# Create directories
mkdir -p $RPM_BUILD_ROOT/opt/talent-factory
mkdir -p $RPM_BUILD_ROOT/etc/systemd/system
mkdir -p $RPM_BUILD_ROOT/etc/nginx/sites-available
mkdir -p $RPM_BUILD_ROOT/etc/avahi/services

# Copy application files
cp -r backend $RPM_BUILD_ROOT/opt/talent-factory/
cp -r ui $RPM_BUILD_ROOT/opt/talent-factory/
cp config.yml $RPM_BUILD_ROOT/opt/talent-factory/
cp talent-factory.service $RPM_BUILD_ROOT/etc/systemd/system/
cp -r avahi $RPM_BUILD_ROOT/opt/talent-factory/
cp start-talent-factory.sh $RPM_BUILD_ROOT/opt/talent-factory/
chmod +x $RPM_BUILD_ROOT/opt/talent-factory/start-talent-factory.sh

# Create application directories
mkdir -p $RPM_BUILD_ROOT/opt/talent-factory/{models,datasets,logs,certs}

%post
# Create talent-factory user
if ! id "talentfactory" &>/dev/null; then
    useradd -r -s /bin/false -d /opt/talent-factory talentfactory
fi

# Set up directories and permissions
mkdir -p /opt/talent-factory/{models,datasets,logs,certs}
chown -R talentfactory:talentfactory /opt/talent-factory

# Install Python dependencies
cd /opt/talent-factory/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate self-signed SSL certificates
if [ ! -f /opt/talent-factory/certs/talentfactory.local.crt ]; then
    openssl req -x509 -newkey rsa:4096 -keyout /opt/talent-factory/certs/talentfactory.local.key \
        -out /opt/talent-factory/certs/talentfactory.local.crt -days 365 -nodes \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=talentfactory.local"
    chown talentfactory:talentfactory /opt/talent-factory/certs/*
fi

# Configure nginx
cat > /etc/nginx/sites-available/talentfactory << 'EOF'
server {
    listen 80;
    server_name talentfactory.local;
    
    location / {
        proxy_pass http://localhost:3004;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    location /api {
        proxy_pass http://localhost:8084;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    location /ws {
        proxy_pass http://localhost:8084;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
EOF

# Enable nginx site
ln -sf /etc/nginx/sites-available/talentfactory /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Configure avahi for mDNS
cat > /etc/avahi/services/talentfactory.service << 'EOF'
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
  </service>
  <service>
    <type>_mcp._tcp</type>
    <port>80</port>
    <txt-record>path=/mcp</txt-record>
    <txt-record>version=1.0.0</txt-record>
    <txt-record>service=talent-catalogue</txt-record>
  </service>
</service-group>
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable talent-factory
systemctl start talent-factory

# Restart nginx and avahi
systemctl restart nginx
systemctl restart avahi-daemon

echo ""
echo "Talent Factory is running."
echo "Access at: http://talentfactory.local"
echo ""

%preun
# Stop the service
systemctl stop talent-factory || true
systemctl disable talent-factory || true

# Remove nginx configuration
rm -f /etc/nginx/sites-enabled/talentfactory
rm -f /etc/nginx/sites-available/talentfactory

# Remove avahi service
rm -f /etc/avahi/services/talentfactory.service

# Restart nginx and avahi
systemctl restart nginx || true
systemctl restart avahi-daemon || true

%files
%defattr(-,root,root,-)
/opt/talent-factory
/etc/systemd/system/talent-factory.service

%changelog
* $(date +'%a %b %d %Y') Talent Factory Team <team@garethapi.com> - 1.0.0-1
- Initial release of Talent Factory
- Local AI workshop for fine-tuning models
- MCP integration for Dot Home
- Real-time training progress monitoring
- Safety evaluation and audit trail
