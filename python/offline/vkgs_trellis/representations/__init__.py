from .radiance_field import Strivec
from .octree import DfsOctree as Octree
from .gaussian import Gaussian

# Mesh imports are optional (requires kaolin) - only import when needed
# Uncomment the following line if you need mesh functionality:
# from .mesh import MeshExtractResult
MeshExtractResult = None


def __getattr__(name):
    global MeshExtractResult
    if name == "MeshExtractResult" and MeshExtractResult is None:
        from .mesh import MeshExtractResult

        globals()["MeshExtractResult"] = MeshExtractResult
    return globals().get(name)
