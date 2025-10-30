#!/bin/bash

# Build RPM package for Talent Factory

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Create source tarball
PACKAGE_NAME="talent-factory"
VERSION="1.0.0"
SOURCE_DIR="${PACKAGE_NAME}-${VERSION}"

rm -rf "$SOURCE_DIR"
mkdir -p "$SOURCE_DIR"

# Copy application files
cp -r backend "$SOURCE_DIR/"
cp -r ui "$SOURCE_DIR/"
cp config.yml "$SOURCE_DIR/"
cp talent-factory.service "$SOURCE_DIR/"
cp -r avahi "$SOURCE_DIR/"
cp start-talent-factory.sh "$SOURCE_DIR/"
chmod +x "$SOURCE_DIR/start-talent-factory.sh"

# Create tarball
tar -czf "${SOURCE_DIR}.tar.gz" "$SOURCE_DIR"

# Move to RPM build directory
mkdir -p ~/rpmbuild/SOURCES
mv "${SOURCE_DIR}.tar.gz" ~/rpmbuild/SOURCES/

# Copy spec file
cp rpm/talent-factory.spec ~/rpmbuild/SPECS/

# Build RPM
rpmbuild -ba ~/rpmbuild/SPECS/talent-factory.spec

echo "RPM package built: ~/rpmbuild/RPMS/noarch/talent-factory-${VERSION}-1.*.rpm"
echo "Install with: sudo rpm -i ~/rpmbuild/RPMS/noarch/talent-factory-${VERSION}-1.*.rpm"
