# npsv3

-----

**Table of Contents**

- [Installation](#installation)
- [Running](#running)
- [Development](#development)
- [License](#license)

## Installation

When cloning NPSV3, make sure to recursively clone all of the submodules, i.e. `git clone --recursive git@github.com:mlinderm/npsv3.git`.

NPSV3 requires Python 3.12+ and a suite of command-line genomics tools. For convenience, a Docker file is provided that installs all of the dependencies. To build that image:

```
docker build --target build -t npsv3 .
```

### Manual installation

To manually install and run NPSV3 from the source, you will need the following dependencies:

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

along with standard command-line utilities, CMake and a C++17 compiler.

We have installed NPSV3 with conda as shown below (note that we set `RAY_ENABLE_UV_RUN_RUNTIME_ENV` to [prevent the Ray library from attempting to reinstall the package](https://github.com/ray-project/ray/issues/54344#issuecomment-3058801560) when using `uv run`.)

```
conda create -n npsv3 python=3.12
conda env config vars set -n npsv3 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
conda activate npsv3
uv pip install -e .
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

NPSV3 is managed with [uv](https://docs.astral.sh/uv/), using [scikit-build-core](https://github.com/scikit-build/scikit-build-core) as the build backend for its native extension.

### Running the Python tests

Run the PyTest test suite with
```
uv run pytest <pytest args...>
```

### Using PDB with ray remote functions

Run the tests with the "legacy" Ray debugger, e.g.
```
RAY_DEBUG=legacy uv run pytest
```

Then from terminal session use `uv run ray debug` to connect.

### Building and testing the native extension

The C++ extension build is implemented with scikit-build-core. npsv3 is configured (see `[tool.uv]` in `pyproject.toml`) to build without isolation, reusing the same environment and CMake build directory (`build/{wheel_tag}`) for fast incremental rebuilds when making changes to the C++ extension.

If the Python tests depend on changes in the C++ extension, reinstall the package with (optionally adding `--config-settings=cmake.build-type="Debug"` for a debug build)
```
uv pip install -e .
```

There are a separate set of C++ units, implemented with GoogleTest that can be built and run with the following (assuming you are using Python 3.12, if not point to the relevant build directory). Re-building just the C++ tests can be faster than re-building the entire Python package.
```
uv run --no-sync cmake --build build/cp312-abi3-linux_x86_64 -t graph_test
uv run --no-sync ctest --test-dir build/cp312-abi3-linux_x86_64
```
For convenience the `build/{wheel_tag}` is embedded in the container as the `$EXT_BUILD_DIR` environment variable, e.g., `uv run --no-sync cmake --build $EXT_BUILD_DIR -t graph_test && uv run --no-sync ctest --test-dir $EXT_BUILD_DIR`.

To use GDB with pytest, build with debug symbols, then run `python3` under GDB. The `--no-sync` argument to `uv run` prevents uv from attempting to rebuild the package without debug symbols. The `--dist no` disables the distributed test plugin.
```
uv pip install -e . --config-settings=cmake.build-type="Debug"
uv run --no-sync gdb --args python -m pytest --dist no <pytest args...>
```

To use valgrind, similarly build with debug symbols, then run python3 under valgrind. `-p no:warnings` prevents warnings related to NumPy from blocking the tests from running.
```
uv run --no-sync valgrind --tool=memcheck --track-origins=yes --log-file=valgrind-report.txt python -m pytest -p no:warnings <pytest args...>
```

To use Address Sanitizer (ASan), set the CMAKE ENABLE_ASAN option to ON during build
```
uv pip install -e . --config-settings=cmake.build-type="Debug" --config-settings=cmake.define.ENABLE_ASAN:BOOL=ON
```
then run the tests ensuring libasan is preloaded before execution:
```
LD_PRELOAD="$(gcc -print-file-name=libasan.so):$LD_PRELOAD" uv run --no-sync python3 -m pytest --dist no <pytest args...>
```

To run the native tests with GDB, run the tests with `-V` to report the specific test command that failed, e.g., `build/cp312-abi3-linux_x86_64/graph_test "--gtest_filter=GraphConstructionTest.LinksBetweenAltAllelesInSameVariant" "--gtest_also_run_disabled_tests"`, then run that command under GDB, e.g.,
```
uv run --no-sync gdb -args build/cp312-abi3-linux_x86_64/graph_test "--gtest_filter=GraphConstructionTest.LinksBetweenAltAllelesInSameVariant" "--gtest_also_run_disabled_tests"
```

To force a CMAKE to perform a fresh build, prepend the build command with `CMAKE_ARGS="--fresh"`.

### Developing on arm64 with Docker

Build the container using the provided Docker file:
```
docker build --target build -t npsv3 .
```

The following launches a restartable container in the background (for use with the VScode devcontainer extension). We set the shared memory to support loading the BWA indices into shared memory and mount a local directory at `/data` containing the reference genomes, etc.
```
docker run --entrypoint /bin/bash \
    --shm-size=8g \
    -v ~/Research/data:/data \
    -v `pwd`:/opt/npsv3 \
    -w /opt/npsv3 \
    -dt npsv3
docker exec -it <container name> bash -l  
```

Alternately you can launch directly into an interactive login shell:
```
docker run --rm \
    --shm-size=8g \
    -v ~/Research/data:/data \
    -v `pwd`:/opt/npsv3 \
    -w /opt/npsv3 \
    -it npsv3 bash -l
```

The development process is the same as described above. The name of the build directory is determined by the Python version in the container; for convenience it is embedded as the `EXT_BUILD_DIR` environment variable.

## License

NPSV3 is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
