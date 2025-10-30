# Talent Factory Docker Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TF_BASE_DIR=/opt/talent-factory
ENV PORT=8084

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    nginx \
    avahi-daemon \
    avahi-utils \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Create talent-factory user
RUN useradd -r -s /bin/false talentfactory

# Create directories
RUN mkdir -p /opt/talent-factory/{backend,ui,models,datasets,logs,certs,avahi} \
    && chown -R talentfactory:talentfactory /opt/talent-factory

# Set working directory
WORKDIR /opt/talent-factory

# Copy backend requirements
COPY backend/requirements.txt /opt/talent-factory/backend/

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code
COPY backend/ /opt/talent-factory/backend/

# Copy UI code
COPY ui/ /opt/talent-factory/ui/

# Copy configuration files
COPY config.yml /opt/talent-factory/
COPY talent-factory.service /opt/talent-factory/
COPY avahi/talentfactory.service /opt/talent-factory/avahi/

# Copy startup script
COPY start-talent-factory.sh /opt/talent-factory/
RUN chmod +x /opt/talent-factory/start-talent-factory.sh

# Generate self-signed SSL certificates
RUN openssl req -x509 -newkey rsa:4096 -keyout /opt/talent-factory/certs/talentfactory.local.key \
    -out /opt/talent-factory/certs/talentfactory.local.crt -days 365 -nodes \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=talentfactory.local"

# Configure nginx
RUN echo 'server { \
    listen 80; \
    server_name talentfactory.local; \
    location / { \
        proxy_pass http://localhost:3004; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
    } \
    location /api { \
        proxy_pass http://localhost:8084; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
    } \
    location /ws { \
        proxy_pass http://localhost:8084; \
        proxy_http_version 1.1; \
        proxy_set_header Upgrade $http_upgrade; \
        proxy_set_header Connection "upgrade"; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
    } \
}' > /etc/nginx/sites-available/talentfactory

RUN ln -s /etc/nginx/sites-available/talentfactory /etc/nginx/sites-enabled/ \
    && rm /etc/nginx/sites-enabled/default

# Configure avahi for mDNS
RUN echo '<?xml version="1.0" standalone="no"?> \
<!DOCTYPE service-group SYSTEM "avahi-service.dtd"> \
<service-group> \
  <name replace-wildcards="yes">Talent Factory on %h</name> \
  <service> \
    <type>_http._tcp</type> \
    <port>80</port> \
    <txt-record>path=/</txt-record> \
    <txt-record>version=1.0.0</txt-record> \
    <txt-record>service=talent-factory</txt-record> \
  </service> \
  <service> \
    <type>_mcp._tcp</type> \
    <port>80</port> \
    <txt-record>path=/mcp</txt-record> \
    <txt-record>version=1.0.0</txt-record> \
    <txt-record>service=talent-catalogue</txt-record> \
  </service> \
</service-group>' > /etc/avahi/services/talentfactory.service

# Create startup script
RUN echo '#!/bin/bash \
set -e \
echo "Starting Talent Factory..." \
cd /opt/talent-factory \
chown -R talentfactory:talentfactory /opt/talent-factory \
su -s /bin/bash talentfactory -c "cd /opt/talent-factory && ./start-talent-factory.sh" & \
nginx -g "daemon off;" & \
avahi-daemon -D & \
wait' > /usr/local/bin/start-talent-factory-docker.sh

RUN chmod +x /usr/local/bin/start-talent-factory-docker.sh

# Expose ports
EXPOSE 80 443 8084 3004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8084/health || exit 1

# Switch to talent-factory user
USER talentfactory

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/start-talent-factory-docker.sh"]
