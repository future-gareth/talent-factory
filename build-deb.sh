#!/bin/bash

# Build Debian package for Talent Factory

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Create package directory structure
PACKAGE_DIR="talent-factory-1.0.0"
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/DEBIAN"
mkdir -p "$PACKAGE_DIR/opt/talent-factory"

# Copy control file
cp debian/control "$PACKAGE_DIR/DEBIAN/"

# Copy postinst and prerm scripts
cp debian/postinst "$PACKAGE_DIR/DEBIAN/"
cp debian/prerm "$PACKAGE_DIR/DEBIAN/"

# Copy application files
cp -r backend "$PACKAGE_DIR/opt/talent-factory/"
cp -r ui "$PACKAGE_DIR/opt/talent-factory/"
cp config.yml "$PACKAGE_DIR/opt/talent-factory/"
cp talent-factory.service "$PACKAGE_DIR/opt/talent-factory/"
cp -r avahi "$PACKAGE_DIR/opt/talent-factory/"
cp start-talent-factory.sh "$PACKAGE_DIR/opt/talent-factory/"
chmod +x "$PACKAGE_DIR/opt/talent-factory/start-talent-factory.sh"

# Create directories
mkdir -p "$PACKAGE_DIR/opt/talent-factory/{models,datasets,logs,certs}"

# Set permissions
chown -R root:root "$PACKAGE_DIR"

# Build the package
dpkg-deb --build "$PACKAGE_DIR"

echo "Debian package built: ${PACKAGE_DIR}.deb"
echo "Install with: sudo dpkg -i ${PACKAGE_DIR}.deb"
