FROM ibsim/base:latest

# Build script amended from https://github.com/tianyikillua/code_aster_on_docker

# Set locale environment
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    LANGUAGE=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/home/ibsim/VirtualLab/bin:/opt/SalomeMeca/appli_V2019.0.3_universal:/opt/ERMES/ERMES-CPlas-v12.5:${PATH}" \
    BASH_ENV=/home/ibsim/patch.sh

# Get Ubuntu updates and basic packages
USER root
RUN apt-get update && \
    apt-get upgrade -y --with-new-pkgs -o Dpkg::Options::="--force-confold" && \
    apt-get install -y \
    ubuntu-drivers-common \
    tzdata \
    unzip \
    libglu1 \
    nano \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#COPY hello.sh /home/ibsim/hello.sh
COPY patch.sh /home/ibsim/patch.sh
#RUN chmod +x /home/ibsim/hello.sh
#RUN chmod +x /home/ibsim/patch.sh

USER ibsim
WORKDIR /tmp

# Download and install VirtualLab and its requirements
RUN sudo chmod 755 /home/ibsim/patch.sh && \
    wget -O Install_VirtualLab.sh https://gitlab.com/ibsim/virtuallab/-/raw/master/Scripts/Install/Install_VirtualLab.sh?inline=false && \
    chmod 755 Install_VirtualLab.sh && \
    sudo ./Install_VirtualLab.sh -P c -S y -E y -y && \
    sudo rm /home/ibsim/salome_meca-2019.0.3-1-universal.run && \
    sudo rm /home/ibsim/salome_meca-2019.0.3-1-universal.tgz && \
    sudo rm /home/ibsim/Anaconda3-2020.02-Linux-x86_64.sh && \
    sudo rm /home/ibsim/VirtualLab/Scripts/Install/ERMES-CPlas-v12.5.zip

USER root
