
import netgen
from netgen.occ import *
from netgen.meshing import MeshingStep

shape = Sphere(Pnt(0,0,0), 1)
#shape = Torus(p=(0, 0, 0), n=(0, 0, 1), R=1.0, r=0.3)
ngmesh = OCCGeometry(shape).GenerateMesh(maxh=0.1, perfstepsend=MeshingStep.MESHSURFACE)
mesh = Mesh(ngmesh, netgen_flags={"degree": 2})
