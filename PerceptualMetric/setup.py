# -*- coding: utf-8 -*-

"""
ICTree: Automatic Perceptual Metrics for Tree Models
The Python package setup script.
"""

import distutils.errors as de
import glob
import multiprocessing as mp
import os
import pathlib
import platform
import setuptools as st
import setuptools.command.build_ext as ste
import sys


with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


class CMakeExtension(st.Extension):
    """ Simple extensions representing a CMake task, which just skips the original processing steps. """
    def __init__(self, name):
        super().__init__(name, sources=[ ])


class BuildExtensions(ste.build_ext):
    """
    Extensions builder, which takes care of custom build steps - the Cython TreeIO binding.
    Inspired by https://stackoverflow.com/a/48015772 .
    """

    def run(self):
        for ext in self.extensions:
            try: 
                self.build_cmake(ext)
            except de.DistutilsExecError as e: 
                print(f"Failed to build extension \"{ext.name}\" with \"{e}\". It will not be available!")
        super().run()

    def build_cmake(self, ext):
        # Backup original working directory to return at the end.
        cwd_bck = pathlib.Path(os.getcwd()).absolute()
        # Get path to the CMake root
        cmake_path, cmake_target = ext.name.split(":")
        cmake_root = (pathlib.Path(__file__).parent / cmake_path).absolute()

        # Make sure the requisite directories exist.
        tmp_dir = pathlib.Path(self.build_temp)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ext_dir = pathlib.Path(self.get_ext_fullpath(ext.name))
        ext_dir.mkdir(parents=True, exist_ok=True)

        # Prepare build configuration.
        python_path = pathlib.Path(sys.executable)
        os_platform = "win" if "windows" in platform.architecture()[1].lower() else "linux"
        os_arch = "x86" if platform.architecture()[0] == "32bit" else "x64"
        use_egl = True
        core_count = mp.cpu_count()
        build_type = "Debug" if self.debug else "Release"

        # Build the CMake command line.
        cmake_args = [
            # Use the current architecture - both "x86" and "x64" should work.
            f"-DBUILD_ARCHITECTURE={os_arch}",
            # Enable building of the Cython module.
            "-DPM_CYTHON_ENABLED=True",
            # Compile with headless rendering support using EGL, if requested.
            f"-DIO_USE_EGL={use_egl}",
            # Build using the requested build type - Release or Debug.
            f"-DCMAKE_BUILD_TYPE={build_type}",
            # Make sure we output the library to the correct directory.
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir.parent.absolute()}",
            # Provide CMake with a hint as to the location of the target (current) interpreter.
            f"-DPython_ROOT={python_path.absolute()}",
            f"-DPython_ROOT_DIR={python_path.parent.absolute()}",
            # Location of the CMake root.
            str(cmake_root),
        ]

        # Build the make command line.
        make_args = [
            # Configure for selected build type.
            "--config", build_type,
            # Parallel building for increased speed.
            f"-j{core_count}",
            # Build only the PyTreeIO bindings.
            "--target", f"{cmake_target}",
        ]

        # Prepare building scripts and build the library.
        os.chdir(str(tmp_dir))
        # Run the cmake twice in case of failure to make sure we have a valid configuration ready.
        try:
            self.spawn([ "cmake" ] + cmake_args)
        except de.DistutilsExecError as e:
            try:
                self.spawn([ "cmake" ] + cmake_args)
            except de.DistutilsExecError as e:
                print("Failed to configure CMake even with second attempt, quitting!")
                raise e
        # In case of actual build, run make as well.
        if not self.dry_run:
            self.spawn([ "cmake", "--build", "./" ] + make_args)

        # Return to the original working directory.
        os.chdir(str(cwd_bck))


class DataFileWrapper:
    def __init__(self, source_path: str, target_path: str):
        self._source_path = source_path
        self._target_path = target_path
        self._data_files = None

    def _generate_data_files(self):
        self._data_files = [
            ( f"{self._target_path}/{path.parent}", [ str(path_str) ])
            for path_str in glob.glob(f"{self._source_path}/**", recursive=True)
            for path in [ pathlib.Path(path_str) ]
            if path.is_file()
        ]

    def __getitem__(self, idx):
        self._generate_data_files()
        return self._data_files[idx]

    def __len__(self):
        self._generate_data_files()
        return len(self._data_files)


st.setup(
    name="ictree",
    version="0.0.1",
    author="Tomas Polasek, David Hrusa, Bedrich Benes, Martin Cadik",
    author_email="ipolasek@fit.vutbr.cz;hrusadav@gmail.com;bbenes@purdue.edu;cadik@fit.vutbr.cz",
    description="ICTree: Automatic Perceptual Metrics for Tree Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://cphoto.fit.vutbr.cz/ictree",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    packages=[
        "ictree",
        "ictree.growthwizard",
        "ictree.perceptree",
        "ictree.perceptree_bin",
        "ictree.treeindexer",
    ],
    package_dir={
        "ictree": "psrc/",
        "ictree.growthwizard": "psrc/growthwizard",
        "ictree.perceptree": "psrc/perceptree",
        "ictree.perceptree_bin": "bin",
        "ictree.treeindexer": "psrc/treeindexer",
    },
    data_files=DataFileWrapper("lib", "ictree"),
    ext_modules=[
        CMakeExtension("../:PyTreeIO"),
    ],
    cmdclass={
        "build_ext": BuildExtensions,
    },
    zip_safe=False,
    setup_requires=[ "setuptools_scm" ],
    include_package_data=True,
    python_requires=">=3.9.4",
)
