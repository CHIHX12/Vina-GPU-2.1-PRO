# Vina-GPU 2.1 — AutoDock-Vina-GPU 2.1
# Base: CUDA 12.2 + Ubuntu 22.04 (GLIBC 2.35, satisfies 2.34 requirement)
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# ── system packages ───────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        wget \
        tar \
        ocl-icd-libopencl1 \
        ocl-icd-opencl-dev \
        clinfo \
    && rm -rf /var/lib/apt/lists/*

# ── Boost 1.84.0 (header-only build + program_options / filesystem / thread) ─
WORKDIR /opt
RUN wget -q https://archives.boost.io/release/1.84.0/source/boost_1_84_0.tar.gz \
    && tar xzf boost_1_84_0.tar.gz \
    && rm boost_1_84_0.tar.gz \
    && cd boost_1_84_0 \
    && ./bootstrap.sh --with-libraries=program_options,filesystem,system,thread \
    && ./b2 -j$(nproc) install \
    && ldconfig

# ── copy Vina-GPU 2.1 source ──────────────────────────────────────────────────
WORKDIR /opt/vina-gpu
COPY src/AutoDock-Vina-GPU-2.1/ ./AutoDock-Vina-GPU-2.1/

# ── patch Makefile and compile ────────────────────────────────────────────────
WORKDIR /opt/vina-gpu/AutoDock-Vina-GPU-2.1
RUN sed -i \
        -e 's|^WORK_DIR=.*|WORK_DIR=/opt/vina-gpu/AutoDock-Vina-GPU-2.1|' \
        -e 's|^BOOST_LIB_PATH=.*|BOOST_LIB_PATH=/opt/boost_1_84_0|' \
        -e 's|^OPENCL_LIB_PATH=.*|OPENCL_LIB_PATH=/usr/local/cuda|' \
        -e 's|^OPENCL_VERSION=.*|OPENCL_VERSION=-DOPENCL_3_0|' \
        -e 's|^GPU_PLATFORM=.*|GPU_PLATFORM=-DNVIDIA_PLATFORM|' \
        Makefile

# build with kernel compiled from source (first-time setup)
RUN make clean && make source -j$(nproc)

# ── runtime wrapper ───────────────────────────────────────────────────────────
RUN printf '#!/bin/bash\nulimit -s 8192\nOCL_CACHE="${HOME}/.cache/vina-gpu"\nmkdir -p "$OCL_CACHE"\nexport OPENCL_BINARY_PATH="${OPENCL_BINARY_PATH:-$OCL_CACHE}"\nexec /opt/vina-gpu/AutoDock-Vina-GPU-2.1/AutoDock-Vina-GPU-2-1 "$@"\n' \
    > /usr/local/bin/autodock-vina-gpu && \
    chmod +x /usr/local/bin/autodock-vina-gpu

# NVIDIA OpenCL ICD for runtime GPU access
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# VINA_GPU_HOME: directory containing OpenCL/ source (used at runtime to compile kernels)
ENV VINA_GPU_HOME=/opt/vina-gpu/AutoDock-Vina-GPU-2.1

WORKDIR /data

ENTRYPOINT ["autodock-vina-gpu"]
CMD ["--help"]
