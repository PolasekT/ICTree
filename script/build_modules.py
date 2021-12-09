# -*- coding: utf-8 -*-

"""
Helper script used for building the PyTreeIO Python binding module.
"""

import argparse
import multiprocessing as mp
import os
import pathlib
import platform
import shutil
import subprocess
import sys


def build_module(cmake_path: str, cmake_target: str,
                 render_backend: str,
                 debug: bool, clean: bool, dry_run: bool):
    """
    Build a module with CMake.
    :param cmake_path: Relative path to the CMake root, starting from the project root.
    :param cmake_target: Name of the target module to build.
    :param render_backend: Rendering backend - "egl" or "glu".
    :param debug: Build the module in debug mode?
    :param clean: Cleanup the CMake configuration before building?
    :param dry_run: Skip building the module (True)?
    """

    # Backup original working directory to return at the end.
    cwd_bck = pathlib.Path(os.getcwd()).absolute()
    # Get path to the CMake root
    cmake_root = (pathlib.Path(__file__).parent / cmake_path).absolute()

    # Make sure the requisite directories exist.
    tmp_dir = pathlib.Path(cmake_root / "build")
    if clean and tmp_dir.exists() and tmp_dir.is_dir():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ext_dir = pathlib.Path(cmake_root / "PerceptualMetric/lib/treeio/lib/")
    ext_dir.mkdir(parents=True, exist_ok=True)

    # Prepare build configuration.
    python_path = pathlib.Path(sys.executable)
    os_platform = "win" if "windows" in platform.architecture()[1].lower() else "linux"
    os_arch = "x86" if platform.architecture()[0] == "32bit" else "x64"
    use_egl = render_backend.lower() == "egl"
    core_count = mp.cpu_count()
    build_type = "Debug" if debug else "Release"

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
        subprocess.check_call([ "cmake" ] + cmake_args)
    except subprocess.CalledProcessError as e:
        try:
            subprocess.check_call([ "cmake" ] + cmake_args)
        except subprocess.CalledProcessError as e:
            print("Failed to configure CMake even with second attempt, quitting!")
            raise e
    # In case of actual build, run make as well.
    if not dry_run:
        subprocess.check_call([ "cmake", "--build", "./" ] + make_args)

    # Return to the original working directory.
    os.chdir(str(cwd_bck))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--render-backend",
                        action="store",
                        default="egl", type=str,
                        metavar=("egl|glu"),
                        dest="render_backend",
                        help="Which render backend should the PyTreeIO use? "
                             "EGL for headless rendering, glu for windowed.")

    args = parser.parse_args(sys.argv[1:])

    build_module(
        cmake_path="../",
        cmake_target="PyTreeIO",
        render_backend=args.render_backend,
        debug=False, clean=False,
        dry_run=False,
    )
