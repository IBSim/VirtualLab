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
################################################################################################
### Stuff to setup the Nvidia CUDA driver taken from offical the Nvidia base 18.04 docker file:
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.4.0/ubuntu1804/base/Dockerfile 
################################################################################################
ENV NVARCH x86_64
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.4 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=450,driver<451"
ENV NV_CUDA_CUDART_VERSION 11.4.43-1
ENV NV_CUDA_COMPAT_PACKAGE cuda-compat-11-4
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/${NVARCH}/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 11.4.0

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-4=${NV_CUDA_CUDART_VERSION} \
    ${NV_CUDA_COMPAT_PACKAGE} \
    && ln -s cuda-11.4 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

#COPY NGC-DL-CONTAINER-LICENSE /

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

################################################################################################
##### END of CUDA Stuff
################################################################################################

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
ENV DEBIAN_FRONTEND=noninteractive
# Install miniconda instead of full anaconda to space space in final image.
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
bash Miniconda3-latest-Linux-x86_64.sh -b && \
sudo rm Miniconda3-latest-Linux-x86_64.sh
COPY salome_meca-2019.0.3-1-universal.tgz /home/ibsim/salome_meca-2019.0.3-1-universal.tgz
# Download and install VirtualLab and its requirements
RUN sudo chmod 755 /home/ibsim/patch.sh && \
    wget -O Install_VirtualLab.sh https://gitlab.com/ibsim/virtuallab/-/raw/BT-Cad2vox/Scripts/Install/Install_VirtualLab.sh?inline=false && \
    chmod 755 Install_VirtualLab.sh && \
    sudo ./Install_VirtualLab.sh -S y -E y -P c -y && \
    sudo rm /home/ibsim/salome_meca-2019.0.3-1-universal.run && \
    sudo rm /home/ibsim/salome_meca-2019.0.3-1-universal.tgz && \
    sudo rm /home/ibsim/VirtualLab/Scripts/Install/ERMES-CPlas-v12.5.zip

RUN sudo ./Install_VirtualLab.sh -C y -y
