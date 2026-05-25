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

## Development

### Building and testing the native extension

The C++ extension build is implemented with scikit-build-score. Run the following to rebuild when making changes to the C++ extension. The name of the hatch environment and corresponding build directory are determined by the Python version in use: 
```
hatch -e hatch-test.py3.11 shell
pip install nanobind scikit-build-core[pyproject]
pip install --no-build-isolation -ve .
```
at which point you can run the tests with `pytest <pytest args...>`, e.g., `pytest tests` to run all tests.

There are a separate set of C++ units, implemented with GoogleTest that can be built and run with the following (assuming you are using Python 3.11, if not point to the relevant build directory). Re-building just the C++ tests can be faster than re-building the entire Python package.
```
cmake --build build/cp311-cp311-linux_x86_64 -t graph_test
ctest --test-dir build/cp311-cp311-linux_x86_64
```

To use GDB with pytest, build with debug symbols, then run `python3` under GDB. The `--dist no` disables the distributed test plugin.
```
pip install --no-build-isolation -ve . --config-settings=cmake.build-type="Debug"
gdb -args python3 -m pytest --dist no tests
```

To use valgrind, similarly build with debug symbols, then run python3 under valgrind. `-p no:warnings` prevents warnings related to NumPy from blocking the tests from running.
```
valgrind --tool=memcheck --track-origins=yes --log-file=valgrind-report.txt python3 -m pytest -p no:warnings tests
```

To use Address Sanitizer (ASan), set the CMAKE ENABLE_ASAN option to ON during build
```
pip install --no-build-isolation -ve . --config-settings=cmake.build-type="Debug" --config-settings=cmake.define.ENABLE_ASAN:BOOL=ON
```
then run the tests ensuring libasan is preloaded before execution:
```
LD_PRELOAD="$(gcc -print-file-name=libasan.so):$LD_PRELOAD" python3 -m pytest --dist no tests
```

To run the native tests with GDB, run the tests with `-V` to report the specific test command that failed, e.g., `build/cp311-cp311-linux_x86_64/graph_test "--gtest_filter=GraphConstructionTest.LinksBetweenAltAllelesInSameVariant" "--gtest_also_run_disabled_tests"`, then run that command under GDB, e.g.,
```
gdb -args build/cp311-cp311-linux_x86_64/graph_test "--gtest_filter=GraphConstructionTest.LinksBetweenAltAllelesInSameVariant" "--gtest_also_run_disabled_tests"
```

To force a CMAKE to perform a fresh build, prepend the build command with `CMAKE_ARGS="--fresh"`.

### Developing on arm64 with Docker

Build the container using the provided Docker file:
```
docker build -f Dockerfile.arm64 --target build -t npsv3-build .
```

The following launches a restartable container in the background (for use with the VScode devcontainer extension). We set the shared memory to support loading the BWA indices into shared memory and mount a local directory at `/data` containing the reference genomes, etc.
```
docker run --entrypoint /bin/bash \
    --shm-size=8g \
    -v ~/Research/data:/data \
    -v `pwd`:/opt/npsv3 \
    -w /opt/npsv3 \
    -dt npsv3-build
```
Alternately you can launch directly into an interactive shell:
```
docker run --rm --entrypoint /bin/bash \
    --shm-size=8g \
    -v ~/Research/data:/data \
    -v `pwd`:/opt/npsv3 \
    -w /opt/npsv3 \
    -it npsv3-build
```

As described above, for easily rebuilding when making changes to the C++ extension run the following. The name of the hatch environment and corresponding build directory are determined by the Python version in the container:

```
hatch -e $HATCH_TEST_ENV shell
uv pip install nanobind scikit-build-core[pyproject]
uv pip install --no-build-isolation -ve .
pytest tests
```

Having done the above at least once, you can just build and run the native test suite with:

```
cmake --build $EXT_BUILD_DIR -t graph_test
ctest --test-dir $EXT_BUILD_DIR
```

## License

`npsv3` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
