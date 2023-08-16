FROM tensorflow/tensorflow:2.13.0-gpu

RUN apt-get -qq update && apt-get install --no-install-recommends -yq \
  art-nextgen-simulation-tools \
  bcftools \
  bedtools \
  build-essential \
  bwa \
  cmake \
  curl \
  gawk \
  git \
  libbz2-dev \
  libjemalloc-dev \
  liblzma-dev \
  protobuf-compiler \
  python3-dev \
  python3-distutils \
  samtools \
  tabix \
  && \
  apt-get clean -y && \
  rm -rf /var/lib/apt/lists/*

# Install vg
RUN curl -SL https://github.com/vgteam/vg/releases/download/v1.49.0/vg \
    -o /usr/local/bin/vg \
    && chmod +x /usr/local/bin/vg

# Install odgi (and libraries)
# RUN mkdir -p /opt/odgi \
#   && curl -SL https://github.com/pangenome/odgi/releases/download/v0.8.3/odgi-v0.8.3.tar.gz \
#   | tar -xzC /opt/odgi --strip-components=1 \
#   && cmake -S /opt/odgi -B /opt/odgi/build \
#   && cmake --build /opt/odgi/build -- -j 2

# During development we are mounting our local fork of adgi into the containter

# Needed for odgi Python bindings
ENV LD_PRELOAD="/lib/x86_64-linux-gnu/libjemalloc.so:${LD_PRELOAD}" \
  PYTHONPATH="/opt/odgi/lib:${PYTHON_PATH}" \
  PATH="/opt/odgi/bin:${PATH}"

RUN mkdir -p /opt/samblaster \
    && curl -SL https://github.com/GregoryFaust/samblaster/releases/download/v.0.1.26/samblaster-v.0.1.26.tar.gz \
    | tar -xzC /opt/samblaster --strip-components=1 \
    && make -C /opt/samblaster \
    && cp /opt/samblaster/samblaster /usr/local/bin/.

RUN curl -SL https://github.com/biod/sambamba/releases/download/v0.8.2/sambamba-0.8.2-linux-amd64-static.gz \
    | gzip -dc > /usr/local/bin/sambamba \
    && chmod +x /usr/local/bin/sambamba

RUN curl -SL https://github.com/brentp/goleft/releases/download/v0.2.4/goleft_linux64 \
    -o /usr/local/bin/goleft \
    && chmod +x /usr/local/bin/goleft

ADD . /opt/npsv3

# TODO: Install npsv3

# Install hatch for development
RUN python3 -m pip install --no-cache-dir hatch

WORKDIR /opt/npsv3