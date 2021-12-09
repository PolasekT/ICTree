import sys

from skbuild import setup


setup(
    name="treeio",
    version="0.0.1",
    description="TreeIO C++ binding",
    author="Tomas Polasek",
    license="MIT",
    packages=["treeio"],
    install_requires=["cython"],
)
