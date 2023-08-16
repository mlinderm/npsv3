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

## License

`npsv3` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
