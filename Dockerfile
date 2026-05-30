# syntax=docker/dockerfile-upstream:master

# Enable syntax for advanced substitutions (https://docs.docker.com/reference/dockerfile/#environment-replacement)

ARG BASE_IMAGE=ubuntu:24.04 PYTHON_VERSION=3.12

# Define the base image and specific configuration for amd64 (x86_64)
FROM ${BASE_IMAGE} AS base-amd64

# Define the base image and specific configuration for arm64 (aarch64)
FROM ${BASE_IMAGE}  AS base-arm64

ARG TARGETARCH

# Dyanamically point to resolved base stage based on TARGETARCH
FROM base-${TARGETARCH} AS build

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
  art-nextgen-simulation-tools \
  bcftools \
  bedtools \
  bwa \
  sambamba \
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
  && ./bootstrap.sh --prefix=/usr/local \
  && ./b2 \
  && ./b2 install

# Install vg, samblaster and goleft executables
RUN case ${TARGETARCH} in \
    "arm64") VG_DOWNLOAD=vg-arm64 ;; \
    *) VG_DOWNLOAD=vg ;; \
  esac && \
  curl -SL "https://github.com/vgteam/vg/releases/download/v1.67.0/${VG_DOWNLOAD}" \
    -o /usr/local/bin/vg && \
  chmod +x /usr/local/bin/vg

RUN mkdir -p /opt/samblaster && \
  curl -SL https://github.com/GregoryFaust/samblaster/releases/download/v.0.1.26/samblaster-v.0.1.26.tar.gz \
    | tar -xzC /opt/samblaster --strip-components=1 && \
  make -C /opt/samblaster -j $(nproc) && \
  cp /opt/samblaster/samblaster /usr/local/bin/.

RUN case ${TARGETARCH} in \
    "arm64") GOLEFT_DOWNLOAD=goleft_linux_aarch64 ;; \
    *) GOLEFT_DOWNLOAD=goleft_linux64 ;; \
  esac && \
  curl -SL "https://github.com/brentp/goleft/releases/download/v0.2.6/${GOLEFT_DOWNLOAD}" \
    -o /usr/local/bin/goleft && \
  chmod +x /usr/local/bin/goleft

# Install odgi (and libraries)
RUN git clone --depth 1 --recursive --branch node_del https://github.com/mlinderm/odgi.git /opt/odgi \
  && cmake -S /opt/odgi -B /opt/odgi/build -DBUILD_STATIC=1 \
  && cmake --build /opt/odgi/build -- -j $(nproc) \
  && cp /opt/odgi/bin/odgi /usr/local/bin \
  && cp /opt/odgi/lib/odgi.*.so /usr/local/lib  

# Force "Unix Makefiles" instead of Ninja due to issues with ExternalProject_Add download and build
ENV CMAKE_GENERATOR="Unix Makefiles" \
  CMAKE_INCLUDE_PATH="/opt/odgi/src:${CMAKE_INCLUDE_PATH}" \
  CMAKE_LIBRARY_PATH="/opt/odgi/fork/build/handlegraph-prefix/lib:/opt/odgi/lib:${CMAKE_LIBRARY_PATH}" \
  PYTHONPATH="/usr/local/lib:${PYTHON_PATH}" \
  PATH="/root/.local/bin:$PATH" \
  PIP_INDEX_URL=https://pytorch.org \
  UV_INDEX_URL=https://pytorch.org \
  PIP_EXTRA_INDEX_URL=https://pypi.org/simple/ \
  UV_EXTRA_INDEX_URL=https://pypi.org/simple/ \
  HATCH_TEST_ENV=hatch-test.py${PYTHON_VERSION}

RUN case ${TARGETARCH} in \
    "arm64") ARCH="aarch64" ;; \
    *) ARCH="x86_64"  ;; \
  esac && \
  echo "/lib/${ARCH}-linux-gnu/libjemalloc.so" > /etc/ld.so.preload && \
  PYTHON_VERSION_NODOT=$(echo "${PYTHON_VERSION}" | tr -d '.') && \
  echo "export EXT_BUILD_DIR=build/cp${PYTHON_VERSION_NODOT}-abi3-linux_${ARCH}" > /etc/profile.d/npsv3.sh

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install hatch for development (seems to be a regression in 1.16.3)
RUN --mount=type=cache,target=/root/.cache/uv \
  uv python install "${PYTHON_VERSION}" \
  && uv tool install hatch>1.16.3

ADD . /opt/npsv3

WORKDIR /opt/npsv3

# Pre-generate the hatch testing environment
# RUN --mount=type=cache,target=/root/.cache/uv \
#   hatch -e ${HATCH_TEST_ENV} run uv pip install nanobind scikit-build-core[pyproject] \
#   && hatch -e ${HATCH_TEST_ENV} run uv pip install --no-build-isolation -ve .

# TODO: Multistage build to reduce image size and avoid installing build dependencies in the final image
# Adapted from: https://pythonspeed.com/articles/multi-stage-docker-python/

# Create the wheel to prepare for a multistage build
# RUN --mount=type=cache,target=/root/.cache/uv hatch build -t wheel