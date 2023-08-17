# npsv3

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

docker build -t npsv3 .

docker run --rm --entrypoint /bin/bash --shm-size=8g -v ~/Research/Data:/data -v `pwd`:/opt/npsv3 -v `pwd`/../odgi:/opt/odgi -w /opt/npsv3 -it npsv3

```console
pip install npsv3
```

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


conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.13.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

export CC=gcc
export CXX=g++

cmake -S /path/to/odgi -B /path/to/odgi/build
cmake --build /path/to/odgi/build -- -j 4

python3 -m pip install hatch hydra-submitit-launcher

## License

`npsv3` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
