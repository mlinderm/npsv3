# syntax=docker/dockerfile:1

# Multi-stage build for npsv3:
#   bwa-mem3  Builds bwa-mem3 from source with clang (bwa-mem3 requires clang >= 19 or gcc >= 15)
#   build     Development image with the full toolchain; builds npsv3 (non-editable) into /opt/venv
#   runtime   Slim image with just the virtual environment and the external tools npsv3 invokes (default target)
#
# The runtime image is intended for GPU training via Apptainer on HPC systems, e.g.
#   podman build -t npsv3:latest .
#   podman save --format oci-archive -o npsv3.tar npsv3:latest
#   apptainer build npsv3.sif oci-archive://npsv3.tar
#   apptainer exec --nv npsv3.sif npsv3 ...
# CUDA libraries are provided by the PyTorch wheels, only the host NVIDIA driver is required (bound by `--nv`).

ARG BASE_IMAGE=ubuntu:24.04
ARG PYTHON_VERSION=3.12
ARG BWA_MEM3_VERSION=0.16.0
ARG LLVM_VERSION=19
ARG TARGETARCH

# Define the base image and specific configuration for amd64 (x86_64)
FROM ${BASE_IMAGE} AS base-amd64
ENV NPSV3_ARCH=x86_64

# Define the base image and specific configuration for arm64 (aarch64)
FROM ${BASE_IMAGE} AS base-arm64
ENV NPSV3_ARCH=aarch64

# Dynamically point to resolved base stage based on TARGETARCH
FROM base-${TARGETARCH} AS base

ENV DEBIAN_FRONTEND=noninteractive \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8

# ---------------------------------------------------------------------------------------------------------------------
FROM base AS bwa-mem3

ARG BWA_MEM3_VERSION
ARG LLVM_VERSION

RUN apt-get -qq update && apt-get install --no-install-recommends -yq \
  ca-certificates \
  curl \
  make \
  cmake \
  pkg-config \
  autoconf \
  automake \
  autoconf-archive \
  libtool \
  clang-${LLVM_VERSION} \
  lld-${LLVM_VERSION} \
  libomp-${LLVM_VERSION}-dev \
  zlib1g-dev \
  libdeflate-dev \
  libbz2-dev \
  liblzma-dev \
  libcurl4-openssl-dev \
  libssl-dev \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

# The release tarball includes the vendored submodules (but not .git, so the version is set explicitly). The default
# x86_64 build selects the best SIMD kernels at runtime so the binary is portable across HPC systems.
RUN mkdir -p /opt/bwa-mem3 \
  && curl -fsSL "https://github.com/fg-labs/bwa-mem3/releases/download/v${BWA_MEM3_VERSION}/Source_code_including_submodules.tar.gz" \
    | tar -xzC /opt/bwa-mem3 --strip-components=1 \
  && PATH="/usr/lib/llvm-${LLVM_VERSION}/bin:${PATH}" make -C /opt/bwa-mem3 -j $(nproc) \
    CXX=clang++ CC=clang VERSION_STRING="${BWA_MEM3_VERSION}" LTO=full \
  && install -m 0755 /opt/bwa-mem3/bwa-mem3 /usr/local/bin/ \
  && /usr/local/bin/bwa-mem3 version

# ---------------------------------------------------------------------------------------------------------------------
FROM base AS build

ARG TARGETARCH
ARG PYTHON_VERSION

RUN apt-get -qq update && apt-get install --no-install-recommends -yq \
  ca-certificates \
  build-essential \
  cmake \
  gdb \
  curl \
  gawk \
  git \
  libbz2-dev \
  libcurl4-openssl-dev \
  libdeflate-dev \
  libjansson-dev \
  libjemalloc-dev \
  liblzma-dev \
  libssl-dev \
  pkg-config \
  protobuf-compiler \
  python3 \
  python3-dev \
  zlib1g-dev \
  art-nextgen-simulation-tools \
  bcftools \
  bedtools \
  bwa \
  samtools \
  tabix \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

# Download Boost to speed downstream build (just downloading dependencies)
RUN git clone https://github.com/boostorg/boost.git -b boost-1.89.0 /opt/boost --depth 1 \
  && cd /opt/boost \
  && git submodule update --depth 1 -q --init tools/boostdep \
  && git submodule update --depth 1 -q --init libs/flyweight && python3 tools/boostdep/depinst/depinst.py -X test -g "--depth 1" flyweight \
  && git submodule update --depth 1 -q --init libs/hash2 && python3 tools/boostdep/depinst/depinst.py -X test -g "--depth 1" hash2 \
  && git submodule update --depth 1 -q --init libs/scope && python3 tools/boostdep/depinst/depinst.py -X test -g "--depth 1" scope \
  && git submodule update --depth 1 -q --init libs/dynamic_bitset && python3 tools/boostdep/depinst/depinst.py -X test -g "--depth 1" dynamic_bitset \
  && git submodule update --depth 1 -q --init libs/serialization && python3 tools/boostdep/depinst/depinst.py -X test -g "--depth 1" serialization \
  && ./bootstrap.sh --prefix=/usr/local \
  && ./b2 -d0 link=static cxxflags="-fPIC" cflags="-fPIC" --with-flyweight --with-hash2 --with-scope --with-serialization \
  && ./b2 install \
  && cd / && rm -rf /opt/boost

# Install samblaster and goleft executables
RUN mkdir -p /opt/samblaster && \
  curl -fsSL https://github.com/GregoryFaust/samblaster/releases/download/v.0.1.26/samblaster-v.0.1.26.tar.gz \
    | tar -xzC /opt/samblaster --strip-components=1 && \
  make -C /opt/samblaster -j $(nproc) && \
  cp /opt/samblaster/samblaster /usr/local/bin/. && \
  rm -rf /opt/samblaster

RUN case ${TARGETARCH} in \
    "arm64") GOLEFT_DOWNLOAD=goleft_linux_aarch64 ;; \
    *) GOLEFT_DOWNLOAD=goleft_linux64 ;; \
  esac && \
  curl -fsSL "https://github.com/brentp/goleft/releases/download/v0.2.6/${GOLEFT_DOWNLOAD}" \
    -o /usr/local/bin/goleft && \
  chmod +x /usr/local/bin/goleft

# Install odgi (the static library and headers are needed to build npsv3). odgi compiles with -march=native, which is
# replaced with a fixed baseline (AVX2 on x86_64, the same as required by bwa-mem3) so that the image is portable.
RUN case ${TARGETARCH} in \
    "arm64") ODGI_MARCH=armv8-a ;; \
    *) ODGI_MARCH=x86-64-v3 ;; \
  esac \
  && git clone --depth 1 --recursive --branch node_del https://github.com/mlinderm/odgi.git /opt/odgi \
  && sed -i "s/-march=native/-march=${ODGI_MARCH}/" /opt/odgi/CMakeLists.txt \
  && cmake -S /opt/odgi -B /opt/odgi/build -DBUILD_STATIC=1 \
  && cmake --build /opt/odgi/build -- -j $(nproc) \
  && cp /opt/odgi/bin/odgi /usr/local/bin

COPY --from=bwa-mem3 /usr/local/bin/bwa-mem3 /usr/local/bin/

# Key environment variables for building npsv3 and running tests
# * Force "Unix Makefiles" instead of Ninja due to issues with ExternalProject_Add download and build
# * Install uv-managed Python and the project virtual environment in fixed locations so they can be copied into
#   the runtime image (and are accessible to non-root users, e.g., under Apptainer)
# * Specify the uv copy mode to silence expected warning about hard links in the container filesystem
# * Disable Ray's automatic `uv run` rebuild feature which is not compatible with our build requirements
ENV CMAKE_GENERATOR="Unix Makefiles" \
  CMAKE_INCLUDE_PATH="/opt/odgi/src" \
  CMAKE_LIBRARY_PATH="/opt/odgi/build/handlegraph-prefix/lib:/opt/odgi/lib" \
  UV_PYTHON_INSTALL_DIR=/opt/python \
  UV_PYTHON_PREFERENCE=only-managed \
  UV_PROJECT_ENVIRONMENT=/opt/venv \
  UV_LINK_MODE=copy \
  UV_COMPILE_BYTECODE=1 \
  PATH="/opt/venv/bin:${PATH}" \
  RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
  RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

RUN echo "/lib/${NPSV3_ARCH}-linux-gnu/libjemalloc.so" > /etc/ld.so.preload && \
  PYTHON_VERSION_NODOT=$(echo "${PYTHON_VERSION}" | tr -d '.') && \
  echo "export EXT_BUILD_DIR=build/cp${PYTHON_VERSION_NODOT}-abi3-linux_${NPSV3_ARCH}" > /etc/profile.d/npsv3.sh

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN --mount=type=cache,target=/root/.cache/uv \
  uv python install "${PYTHON_VERSION}" \
  && uv venv --python "${PYTHON_VERSION}" /opt/venv

WORKDIR /opt/npsv3

# Install the locked dependencies first (in a separate layer from the source so that it is cached across source
# changes), including all optional dependencies (extras). npsv3 is built without isolation, so the "build" group must be present before npsv3 itself is built.
RUN --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,source=uv.lock,target=uv.lock \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  uv sync --frozen --no-dev --all-extras --no-install-project

ADD . /opt/npsv3

# Build and install npsv3 itself (non-editable so the runtime image doesn't need the source tree), without host-specific
# code generation in its vendored dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
  SKBUILD_CMAKE_DEFINE="NPSV3_PORTABLE=ON" uv sync --frozen --no-dev --all-extras --no-editable \
  && python -c "import npsv3._native_graph, npsv3._native_realign"

# ---------------------------------------------------------------------------------------------------------------------
FROM base AS runtime

ARG LLVM_VERSION

# Runtime libraries for the native extensions, bwa-mem3 (LLVM OpenMP) and odgi, plus the external tools npsv3 invokes
RUN apt-get -qq update && apt-get install --no-install-recommends -yq \
  ca-certificates \
  gawk \
  libatomic1 \
  libbz2-1.0 \
  libcurl4t64 \
  libdeflate0 \
  libgomp1 \
  libjemalloc2 \
  liblzma5 \
  libomp5-${LLVM_VERSION} \
  libssl3t64 \
  procps \
  zlib1g \
  art-nextgen-simulation-tools \
  bcftools \
  bedtools \
  bwa \
  samtools \
  tabix \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/bin/bwa-mem3 /usr/local/bin/samblaster /usr/local/bin/goleft /usr/local/bin/odgi /usr/local/bin/
COPY --from=build /opt/python /opt/python
COPY --from=build /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
  RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
  RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

RUN echo "/lib/${NPSV3_ARCH}-linux-gnu/libjemalloc.so.2" > /etc/ld.so.preload \
  && python -c "import npsv3._native_graph, npsv3._native_realign, torch" \
  && npsv3 --help > /dev/null \
  && for tool in art_illumina bwa-mem3 bwa samtools samblaster bcftools bedtools tabix bgzip goleft kmc synthBAM; do \
    command -v $tool > /dev/null || { echo "Missing $tool"; exit 1; }; \
  done

WORKDIR /work

CMD ["/bin/bash"]
