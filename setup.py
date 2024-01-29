import os
import platform
import re
import subprocess
import sys
from distutils.spawn import find_executable
from distutils.version import LooseVersion

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Project structure and CMake build steps adapted from
# https://www.benjack.io/2018/02/02/python-cpp-revisited.html
# https://github.com/pybind/cmake_example/blob/master/setup.py

cmake = find_executable(os.environ.get("CMAKE", "cmake"))


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output([cmake, "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            ) from None

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
            if cmake_version < "3.1.0":
                msg = "CMake >= 3.1.0 is required on Windows"
                raise RuntimeError(msg)

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get("CXXFLAGS", ""), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call([cmake, ext.sourcedir, *cmake_args], cwd=self.build_temp, env=env)
        subprocess.check_call([cmake, "--build", ".", *build_args], cwd=self.build_temp)
        print()  # Add an empty line for cleaner output


class SeqLibCMakeBuild(CMakeBuild):
    def run(self):
        root_path = os.path.dirname(os.path.realpath(__file__))
        seqlib_path = os.path.join(root_path, "lib", "seqlib")

        # Apply patches to enable fix compilation issues
        patched = subprocess.run(
            [
                "/usr/bin/patch",
                "--strip=1",
                "--forward",
                "--dry-run",
                "--reverse",
                "-i",
                os.path.join(root_path, "seqlib.patch"),
            ],
            cwd=seqlib_path, check=False,
        )
        if patched.returncode != 0:
            subprocess.check_call(
                ["/usr/bin/patch", "--strip=1", "--forward", "-i", os.path.join(root_path, "seqlib.patch")],
                cwd=seqlib_path,
            )

        super().run()


setup(
    scripts=["scripts/synthBAM"],
    ext_modules=[CMakeExtension("npsv3/npsv3ext")],
    cmdclass={"build_ext": SeqLibCMakeBuild},
    zip_safe=False,
)
