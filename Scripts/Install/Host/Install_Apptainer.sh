#!/bin/bash
sudo apt update
sudo apt install -y \
    build-essential \
    uuid-dev \
    libgpgme-dev \
    squashfs-tools \
    libseccomp-dev \
    wget \
    pkg-config \
    git

# Install Go
# Go release history: https://go.dev/doc/devel/release
export VERSION=1.18.1 OS=linux ARCH=amd64 # Replace the values as needed
wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz # Downloads the required Go package
sudo tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz # Extracts the archive
rm go$VERSION.$OS-$ARCH.tar.gz # Deletes the ``tar`` file

# Export paths
echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
export PATH=/usr/local/go/bin:$PATH

# Install Apptainer
# Apptainer release history: https://github.com/apptainer/apptainer/releases
export VERSION=1.0.2 # adjust this as necessary
wget https://github.com/apptainer/apptainer/releases/download/v${VERSION}/apptainer-${VERSION}.tar.gz
tar -xzf apptainer-${VERSION}.tar.gz
rm apptainer-${VERSION}.tar.gz # Deletes the ``tar`` file

# Compile Apptainer 
cd apptainer-${VERSION}
./mconfig
make -C ./builddir
sudo make -C ./builddir install
