##
# @author Tomas Polasek
# @date 26.4.2021
# @version 1.0
# @brief Python wrapper for TreeIO.
##

from cython.operator cimport dereference
from libcpp cimport bool

import pathlib

from enum import IntEnum
from typing import Optional, Union


cdef extern from "string" namespace "std":
    cdef cppclass string:
        string()
        string(char*)
        char* c_str()


cdef extern from "TreeIO/TreeIO.h" namespace "treeutil":
    cdef enum _CLoggerLevel "treeutil::Logger::Level":
        Debug = 0,
        Info = 1,
        Warning = 2,
        Error = 3

    cdef cppclass _CLogger "treeutil::Logger":
        @staticmethod
        void setLoggingLevel(_CLoggerLevel level);


cdef extern from "TreeIO/TreeIO.h" namespace "treeio":
    cdef struct _CTreeNodeData "treeio::TreeNodeData":
        pass
    cdef struct _CTreeRuntimeMetaData "treeio::TreeRuntimeMetaData":
        pass
    cdef struct _CTreeDynamicMetaData "treeio::TreeDynamicMetaData":
        pass
    cdef struct _CTreeMetaData "treeio::TreeMetaData":
        pass

    cdef cppclass _CArrayTreeT "treeio::ArrayTreeT"[NT, MT]:
        _CArrayTreeT() except +

        @staticmethod
        _CArrayTreeT[NT, MT] fromString(string) except +

        @staticmethod
        _CArrayTreeT[NT, MT] fromPath(string) except +

        string serialize()
        void printNodeInfo()


cdef extern from "TreeIO/Tree.h" namespace "trestat":
    cdef cppclass _CTreeStats "treestat::TreeStats":
        _CTreeStats() except +
        _CTreeStats(const _CArrayTreeT[_CTreeNodeData, _CTreeMetaData]&) except +

        void calculateStatistics(const _CArrayTreeT[_CTreeNodeData, _CTreeMetaData]&) except +
        void saveStatisticsToDynamic(_CArrayTreeT[_CTreeNodeData, _CTreeMetaData]&) except +


cdef extern from "TreeIO/Tree.h" namespace "treeio":
    cdef cppclass _CRuntimeMetaData "treeio::RuntimeMetaData":
        pass


cdef extern from "TreeIO/Render.h" namespace "treerndr":
    cdef cppclass _CRenderConfig "treerndr::RenderConfig":
        # Path to save the images to.
        string outputPath
        # Base name for the saved files. For each view several
        #   modalities will be produced. Their names use the following naming
        #   scheme: <BASE>_<VIEW_IDX>_<MODALITY>.png
        # Additionally, meta-data file will be created, sharing the same name
        #   but using ".json" extension instead.
        string baseName
        # Width of the rendered image.
        size_t width
        # Height of the rendered image.
        size_t height
        # Number of samples to render per pixel.
        size_t samples
        # Number of views to render.
        size_t viewCount
        # Distance of the camera from the origin.
        float cameraDistance
        # Height of the camera from the ground plane.
        float cameraHeight
        # Normalize the tree size to treeScale?
        bool treeNormalize
        # Scale to normalize the tree to, if treeNormalize is enabled.
        float treeScale

    cdef cppclass _CDitherConfig "treerndr::DitherConfig":
        # Path to save the images to.
        string outputPath
        # Seed used for repeatable dithered view generation.
        int seed
        # Number of dithered variants to generate.
        size_t ditherCount

        # Variance of the camera distance.
        float camDistanceVar

        # Strength of yaw dithering.
        float camYawDither
        # Minimal yaw dithering.
        float camYawDitherLow
        # Maximal yaw dithering.
        float camYawDitherHigh

        # Strength of pitch dithering.
        float camPitchDither
        # Minimal roll dithering.
        float camPitchDitherLow
        # Maximal pitch dithering.
        float camPitchDitherHigh

        # Strength of roll dithering.
        float camRollDither
        # Minimal roll dithering.
        float camRollDitherLow
        # Maximal roll dithering.
        float camRollDitherHigh

    cdef cppclass _CRenderHelper "treerndr::RenderHelper":
        _CRenderHelper() except +

        void renderTree(const _CArrayTreeT[_CTreeNodeData, _CTreeMetaData] &tree,
                        const _CRenderConfig &config) except +

        void renderDitheredTree(const _CArrayTreeT[_CTreeNodeData, _CTreeMetaData] &tree,
                                const _CRenderConfig &config, const _CDitherConfig &dither) except +


cdef extern from "TreeIO/Augment.h" namespace "treeaug":
    cdef cppclass _CTreeAugmenter "treeaug::TreeAugmenter":
        _CTreeAugmenter() except +


cdef extern from "treeio.h":
    cdef int testIntegration(string)
    cdef _CArrayTreeT[_CTreeNodeData, _CTreeMetaData] *treeConstruct()
    cdef void treeDestroy(_CArrayTreeT[_CTreeNodeData, _CTreeMetaData]*)
    cdef _CArrayTreeT[_CTreeNodeData, _CTreeMetaData] *treeFromString(string)
    cdef _CArrayTreeT[_CTreeNodeData, _CTreeMetaData] *treeFromPath(string)
    cdef bool treeSave(_CArrayTreeT[_CTreeNodeData, _CTreeMetaData]*, string)
    cdef void treeCopy(_CArrayTreeT[_CTreeNodeData, _CTreeMetaData]*,
                       _CArrayTreeT[_CTreeNodeData, _CTreeMetaData]*)
    cdef bool treeAugment(_CArrayTreeT[_CTreeNodeData, _CTreeMetaData]*, _CTreeAugmenter*,
                          int, bool, float, float, float, float, float, float, bool)


cdef class ArrayTree:
    cdef _CArrayTreeT[_CTreeNodeData, _CTreeMetaData] *thisPtr

    @classmethod
    def create_from_string(cls, serialized: str):
        tree = ArrayTree()
        tree.from_string(serialized.encode())
        return tree

    @classmethod
    def create_from_path(cls, path: Union[str, pathlib.Path]):
        tree = ArrayTree()
        tree.from_path(str(path).encode())
        return tree

    @classmethod
    def create_copy(cls, tree: ArrayTree):
        result = ArrayTree()
        result.copy_from(tree)
        return result

    def __cinit__(self, serialized: Optional[str] = None, path: Optional[Union[str, pathlib.Path]] = None):
        if serialized is not None:
            self.thisPtr = treeFromString(serialized.encode())
        elif path is not None:
            self.thisPtr = treeFromPath(str(path).encode())
        else:
            self.thisPtr = treeConstruct()
        if self.thisPtr == NULL:
            raise MemoryError("Failed to allocate ArrayTree!")

    def __init__(self, serialized: Optional[str] = None, path: Optional[Union[str, pathlib.Path]] = None):
        pass

    cdef set_ptr(self, _CArrayTreeT[_CTreeNodeData, _CTreeMetaData] *ptr):
        treeDestroy(self.thisPtr)
        del self.thisPtr

        self.thisPtr = ptr

    cdef _CArrayTreeT[_CTreeNodeData, _CTreeMetaData] *get_ptr(self):
        return self.thisPtr

    def from_string(self, serialized: str):
        self.set_ptr(treeFromString(serialized.encode()))

    def from_path(self, path: Union[str, pathlib.Path]):
        self.set_ptr(treeFromPath(str(path).encode()))

    def to_path(self, path: Union[str, pathlib.Path]) -> bool:
        return treeSave(self.thisPtr, str(path).encode())

    def __dealloc__(self):
        self.set_ptr(NULL)

    def print_node_info(self):
        self.thisPtr.printNodeInfo()

    def serialize(self) -> str:
        return self.thisPtr.serialize().decode("UTF-8")

    def copy_from(self, tree: ArrayTree):
        treeCopy(self.get_ptr(), tree.get_ptr())


cdef class TreeStats:
    cdef _CTreeStats *thisPtr

    def __cinit__(self, tree: Optional[ArrayTree] = None):
        self.thisPtr = new _CTreeStats()

        if self.thisPtr == NULL:
            raise MemoryError("Failed to allocate TreeStats!")

    def __init__(self, tree: Optional[ArrayTree] = None):
        if tree is not None:
            self.calculate_statistics(tree)

    def __dealloc__(self):
        del self.thisPtr
        self.thisPtr = NULL

    def calculate_statistics(self, tree: ArrayTree):
        self.thisPtr.calculateStatistics(dereference(tree.get_ptr()))

    def save_statistics(self, tree: ArrayTree):
        self.thisPtr.saveStatisticsToDynamic(dereference(tree.get_ptr()))


cdef class TreeRenderer:
    cdef _CRenderHelper *thisPtr

    def __cinit__(self):
        self.thisPtr = new _CRenderHelper()

        if self.thisPtr == NULL:
            raise MemoryError("Failed to allocate RenderHelper!")

    def __init__(self):
        pass

    def __dealloc__(self):
        del self.thisPtr
        self.thisPtr = NULL

    def render_dither_tree(self, tree: ArrayTree,
                           **kwargs,
                          ):

        cdef _CRenderConfig config
        if "output_path"      in kwargs: config.outputPath      = str(kwargs["output_path"]).encode()
        if "base_name"        in kwargs: config.baseName        = str(kwargs["base_name"]).encode()
        if "width"            in kwargs: config.width           = kwargs["width"]
        if "height"           in kwargs: config.height          = kwargs["height"]
        if "samples"          in kwargs: config.samples         = kwargs["samples"]
        if "view_count"       in kwargs: config.viewCount       = kwargs["view_count"]
        if "camera_distance"  in kwargs: config.cameraDistance  = kwargs["camera_distance"]
        if "camera_height"    in kwargs: config.cameraHeight    = kwargs["camera_height"]
        if "tree_normalize"   in kwargs: config.treeNormalize   = kwargs["tree_normalize"]
        if "tree_scale"       in kwargs: config.treeScale       = kwargs["tree_scale"]

        cdef _CDitherConfig dither
        if "dither_output_path"     in kwargs: dither.outputPath          = str(kwargs["dither_output_path"]).encode()
        if "dither_seed"            in kwargs: dither.seed                = kwargs["dither_seed"]
        if "dither_count"           in kwargs: dither.ditherCount         = kwargs["dither_count"]
        if "cam_distance_var"       in kwargs: dither.camDistanceVar      = kwargs["cam_distance_var"]
        if "cam_yaw_dither"         in kwargs: dither.camYawDither        = kwargs["cam_yaw_dither"]
        if "cam_yaw_dither_low"     in kwargs: dither.camYawDitherLow     = kwargs["cam_yaw_dither_low"]
        if "cam_yaw_dither_high"    in kwargs: dither.camYawDitherHigh    = kwargs["cam_yaw_dither_high"]
        if "cam_pitch_dither"       in kwargs: dither.camPitchDither      = kwargs["cam_pitch_dither"]
        if "cam_pitch_dither_low"   in kwargs: dither.camPitchDitherLow   = kwargs["cam_pitch_dither_low"]
        if "cam_pitch_dither_high"  in kwargs: dither.camPitchDitherHigh  = kwargs["cam_pitch_dither_high"]
        if "cam_roll_dither"        in kwargs: dither.camRollDither       = kwargs["cam_roll_dither"]
        if "cam_roll_dither_low"    in kwargs: dither.camRollDitherLow    = kwargs["cam_roll_dither_low"]
        if "cam_roll_dither_high"   in kwargs: dither.camRollDitherHigh   = kwargs["cam_roll_dither_high"]

        self.thisPtr.renderDitheredTree(
            dereference(tree.get_ptr()),
            config, dither,
        )


cdef class TreeAugmenter:
    cdef _CTreeAugmenter *thisPtr

    def __cinit__(self):
        self.thisPtr = new _CTreeAugmenter()

        if self.thisPtr == NULL:
            raise MemoryError("Failed to allocate TreeAugmenter!")

    def __init__(self):
        pass

    def __dealloc__(self):
        del self.thisPtr
        self.thisPtr = NULL

    def augment_tree(self, tree: ArrayTree,
                     seed: int = 0, normal: bool = False,
                     n_low: float = -1.0, n_high: float = 1.0, n_strength: float = 0.0,
                     b_low: float = -1.0, b_high: float = 1.0, b_strength: float = 0.0,
                     skip_leaves: bool = False,
                    ) -> bool:
        return treeAugment(
            tree.get_ptr(), self.thisPtr,
            seed, normal,
            n_low, n_high, n_strength,
            b_low, b_high, b_strength,
            skip_leaves,
        )


class LoggingLevel(IntEnum):
    # Debugging messages.
    Debug = 0,
    # Non-critical information messages.
    Info = 1,
    # Warning messages.
    Warning = 2,
    # Critical error messages.
    Error = 3


cdef class TreeLogger:
    @classmethod
    def set_logging_level(self, level: LoggingLevel):
        _CLogger.setLoggingLevel(int(level))


def test_statistics(serialized: bytes):
    tree = ArrayTree(serialized=serialized)
    print(tree.serialize())

    tree_stats = TreeStats(tree)
    tree_stats.save_statistics(tree)
    print(tree.serialize())


def test_integration(string s):
    return testIntegration(s)

