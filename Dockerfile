FROM base

# Build script amended from https://github.com/tianyikillua/code_aster_on_docker

# Get Ubuntu updates and basic packages
USER root
RUN apt-get update && \
    apt-get upgrade -y --with-new-pkgs -o Dpkg::Options::="--force-confold" && \
    apt-get install -y \
    unzip \
    libglu1 \
    nano \ 
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER aster
WORKDIR /tmp

# Download and install VirtualLab and its requirements
RUN wget -O Install_VirtualLab.sh https://gitlab.com/ibsim/virtuallab/-/raw/master/Scripts/Install/Install_VirtualLab.sh?inline=false && \
    chmod 755 Install_VirtualLab.sh && \
    sudo ~/./Install_VirtualLab.sh -P c -S y -E y -y && \
    source ~/.bashrc

USER root
