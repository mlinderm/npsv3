import hydra
import pytest
import ray

from npsv3.util.sample import Sample

@pytest.fixture(scope="session")
def hydra_setup():
    hydra.initialize(config_path="../src/npsv3/conf", version_base=None)
    yield
    hydra.core.global_hydra.GlobalHydra.instance().clear()


@pytest.fixture(scope="session")
def ray_setup():
    ray.init(num_cpus=1, include_dashboard=False)
    yield
    ray.shutdown()


@pytest.fixture(scope="function")
def cfg(request, hydra_setup): #noqa: ARG001
    marker = request.node.get_closest_marker("cfg_overrides")
    _cfg = hydra.compose(
        config_name="config",
        overrides=marker.args if marker else [],
    )
    return _cfg


@pytest.fixture(scope="function")
def hg002_sample():
    return Sample(
        "HG002",
        mean_coverage=25.46,
        mean_insert_size=573.1,
        std_insert_size=164.2,
        sequencer="HS25",
        read_length=148,
    )

@pytest.fixture(scope="function")
def syndip_sample():
    return Sample(
        "CHM1_CHM13",
        mean_coverage=44.76,
        mean_insert_size=344.72,
        std_insert_size=105.32,
        sequencer="HSXn",
        read_length=151,
    )