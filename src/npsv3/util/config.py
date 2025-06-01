import os

from omegaconf import OmegaConf


def setup_resolvers():
    OmegaConf.register_new_resolver("strip_ext", lambda path: os.path.splitext(path)[0])
    OmegaConf.register_new_resolver("len", lambda arg: len(arg))
