import numpy as np
import firedrake as fd
from firedrake import grad,div,inner,sym,dx,ds
from petsc4py import PETSc
from mpi4py import MPI

print_= print
print = PETSc.Sys.Print
comm = MPI.COMM_WORLD


from pathlib import Path
import sys
name="results/"+Path(sys.argv[0]).resolve().stem
print(f"{name=}")

n=2
mesh = fd.UnitSquareMesh(n, n)

# inf-sup stable element - Taylor - Hood
Ep = fd.FiniteElement("CG", mesh.ufl_cell(), 1)
Ev = fd.VectorElement("CG", mesh.ufl_cell(), 2)

# Alternative stable, pressure robust elements - Scott-Vogelius
Ep = fd.FiniteElement("DG", mesh.ufl_cell(), 1, variant="alfeld")
Ev = fd.VectorElement("CG", mesh.ufl_cell(), 2, variant="alfeld")

Evp = fd.MixedElement([Ev, Ep])
W = fd.FunctionSpace(mesh, Evp)

bc_wall = fd.DirichletBC(W.sub(0), fd.as_vector([0,0]), [1,2,3])
bc_top = fd.DirichletBC(W.sub(0), fd.as_vector([1,0]), [4])

# Collect boundary conditions
bcs = [bc_wall, bc_top]

# Define unknown and test function(s)
(v_, p_) = fd.TestFunctions(W)

# current unknown time step
w = fd.Function(W)
(v, p) = fd.split(w)

def a(v,v_) :
    return (inner(grad(v), grad(v_)))*dx 

def b(q,v) :
    return inner(div(v),q)*dx

# variational form
F_eq1 = a(v,v_) - b(p,v_)
F_eq2 = b(p_,v) 

F = F_eq1 + F_eq2

J = fd.derivative(F, w)

from scipy.sparse import csr_matrix
from scipy.linalg import null_space
from numpy.linalg import cond, det
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

A = fd.assemble(J, bcs=bcs)

A = A.petscmat
A = csr_matrix((A.getValuesCSR()[2],A.getValuesCSR()[1],A.getValuesCSR()[0])).todense()

print(f"{cond(A)=}  {det(A)=}")
print(f"{null_space(A)=}")

plt.spy(A)
plt.savefig(name+'_spyA.pdf', bbox_inches='tight')

# ---- pressure nullspace 
nullsp = fd.MixedVectorSpaceBasis(W, [W.sub(0), fd.VectorSpaceBasis(constant=True, comm=mesh.comm)])
#nullsp = None

problem=fd.NonlinearVariationalProblem(F,w,bcs,J)
solver=fd.NonlinearVariationalSolver(problem, nullspace=nullsp, options_prefix="")
solver.solve()


(v,p)=w.subfunctions
v.rename("velocity")
p.rename("pressure")
vfile = fd.VTKFile(name+"_sol.pvd")
vfile.write(v, p)

print(f' vnorm={fd.norm(v)}   pnorm={fd.norm(p)}   pressure mean value={fd.assemble(p*dx)}')

