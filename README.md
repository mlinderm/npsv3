# npsv3

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

When cloning `npsv3`, make sure to recursively clone all of the submodules, i.e. `git clone --recursive git@github.com:mlinderm/npsv3.git`.

`npsv3` requires Python 3.11+ and a suite of command-line genomics tools. For convenience, a Docker file is provided that installs all of the dependencies. To build that image:

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
python -m pip install -e .
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
    -v ~/Research/data:/data \
    -v `pwd`/../odgi:/opt/odgi \
    -v `pwd`:/opt/npsv3 \
    -w /opt/npsv3 \
    -it \
    npsv3
```

And then from within the container, build odgi (optionally with the `--fresh` option for `cmake` to force reconfiguration):
```
cmake -S /opt/odgi -B /opt/odgi/build --fresh \
    && cmake --build /opt/odgi/build -- -j 2
```
and finally the npsv3 package (which may require the `--break-system-packages` to `pip` depending on the underlying distribution):
```
python3 -m pip install -e .
```

You can run the unit tests with `hatch test`.

## Development

### Building and testing the native extension

The C++ extension build is implemented with scikit-build-score. For easily rebuilding when making changes to the C++ extension
```
hatch -e hatch-test.py3.12 shell
pip install nanobind scikit-build-core[pyproject]
pip install --no-build-isolation -ve .
```

at which point you can run the tests with `pytest <pytest args...>`, e.g., `pytest tests` to run all tests. There are a separate set of C++ units, implemented with GoogleTest that can be built and run with the following (assuming you are using Python 3.11. If not use the correct build directory). Re-building just the C++ tests can be faster than re-building the entire Python package.

```
cmake --build build/cp311-cp311-linux_x86_64 -t graph_test
ctest --test-dir build/cp311-cp311-linux_x86_64
```

To use GDB with pytest, build with debug symbols, then run `python3` under GDB. The `--dist no` disables the distributed test plugin.
```
pip install --no-build-isolation -ve . --config-settings=cmake.build-type="Debug"`
gdb -args python3 -m pytest --dist no tests
```
To use valgrind, similarly build with debug symbols, then run python3 under valgrind. `-p no:warnings` prevents warnings related to NumPy from blocking the tests from running.
```
valgrind --tool=memcheck --track-origins=yes --log-file=valgrind-report.txt python3 -m pytest -p no:warnings tests
```

To force a CMAKE to perform a fresh build, prepend the build command with `CMAKE_ARGS="--fresh"`.

### arm64


```
docker build -f Dockerfile.arm64 --target build -t npsv3-build .
```

    -v `pwd`/../odgi:/opt/odgi \

```
docker run --rm --entrypoint /bin/bash \
    --shm-size=8g \
    -v ~/Research/data:/data \
    -v `pwd`:/opt/npsv3 \
    -w /opt/npsv3 \
    -it npsv3-build
```

For easily rebuilding when making changes to the C++ extension
```
hatch -e hatch-test.py3.12 shell
pip install nanobind scikit-build-core[pyproject]
pip install --no-build-isolation -ve .
pytest tests
```
ctest --test-dir build/cp312-abi3-linux_aarch64 -R AdjacentInsertion -V -N

## License

`npsv3` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
