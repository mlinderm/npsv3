# npsv3

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

When cloning `npsv3`, make sure to recursively clone all of the submodules, i.e. `git clone --recursive git@github.com:mlinderm/npsv3.git`.

`npsv3` requires Python 3.8+ and a suite of command-line genomics tools. For convenience, a Docker file is provided that installs all of the dependencies. To build that image:

```
docker build -t npsv3 .
```

### Manual installation

To manually install and run NPSV-deep from the source, you will need the following dependencies:

* ART (NGS simulator)
* bwa
* bedtools
* bcftools
* goleft
* htslib (i.e., tabix and bgzip)
* jellyfish (with Python bindings)
* ODGI
* samblaster
* sambamba
* samtools

along with standard command-line utilities, CMake and a C++14 compiler.

We have installed `npsv3` with conda, e.g.,

```
conda create -n npsv3 python=3.11
conda activate npsv3
conda install mkl==2024.0
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Running

Given the multi-step workflow, the typical approach when using the Docker image is to run `npsv3` from a shell. The following command will start a Bash session in the Docker container (replace `/path/to/reference/`directory with the path to directory containing the reference genome and associated BWA indices). `npsv3` is most efficient when the BWA indices are loaded into shared memory. To load BWA indices into shared memory you will need to configure the Docker container with at least 12G of memory and set the shared memory size to 8G or more.

```
docker run --rm --entrypoint /bin/bash \
    --shm-size=8g \
    -v /path/to/reference/directory:/data \
    -w /opt/npsv3 \
    -it \
    npsv3
```

During development we are manually building a fork of ODGI and manually installing the `npsv3` package. When launching the container, mount the directory containing odgi into the container, e.g.,

```
docker run --rm --entrypoint /bin/bash \
    --shm-size=8g \
    -v ~/Research/Data:/data \
    -v `pwd`/../odgi:/opt/odgi \
    -v `pwd`:/opt/npsv3 \
    -w /opt/npsv3 \
    -it \
    npsv3
```

And then from within the container, build odgi
```
cmake -S /opt/odgi -B /opt/odgi/build \
    && cmake --build /opt/odgi/build -- -j 2
```
and finally the npsv3 package
```
python -m pip install -e .
```

You can run the unit tests with `hatch run test`.

## License

`npsv3` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
