# npsv3

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

docker build -t npsv3 .

docker run --rm --entrypoint /bin/bash --shm-size=8g -v ~/Research/Data:/data -v `pwd`:/opt/npsv3 -v `pwd`/../odgi:/opt/odgi -w /opt/npsv3 -it npsv3


```
docker run --rm --entrypoint /bin/bash \
    --shm-size=8g \
    -v /path/to/reference/directory:/data \
    -v `pwd`:/opt/npsv3 \
    -w /opt/npsv3 \
    -it \
    npsv3
```

https://www.tensorflow.org/install/pip


We need to make sure that conda base environment is does not impact compilation or use of hatch environments. To that end, make sure the base environment is not automatically activated,
```
conda config --set auto_activate_base false
```
and clear any vestige of the base environment
```
conda deactivate
unset LDFLAGS
```

Create a conda environment to build npsv3 on Middlebury infrastructure. The update to `xz`` resolves this [issue](https://stackoverflow.com/questions/47633870/rpm-lib64-liblzma-so-5-version-xz-5-1-2alpha-not-found-required-by-lib-li).

```plaintext
module load npsv3

conda create -n npsv3 python=3.8
conda install -c conda-forge cudatoolkit=11.8.0
conda install -c conda-forge 'xz>=5.2.7'

python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.13.*

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

export CC=gcc
export CXX=g++

cmake3 -S $ODGI_HOME -B $ODGI_HOME/build
cmake3 --build $ODGI_HOME/build -- -j 4

python3 -m pip install hatch hydra-submitit-launcher
```

You can then test with `hatch run test` or create a shell environment with `npsv3` available via `hatch shell`.

## License

`npsv3` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
