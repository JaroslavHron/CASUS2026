import numpy as np
import firedrake as fd
from firedrake import grad,div,inner,sym,dx,ds,Constant
from petsc4py import PETSc
from mpi4py import MPI

print_= print
print = PETSc.Sys.Print
comm = MPI.COMM_WORLD


from pathlib import Path
import sys
name="results/"+Path(sys.argv[0]).resolve().stem
print(f"{name=}")


# Mesh
n=20
mesh = fd.UnitSquareMesh(n, n)

# inf-sup stable element - Taylor - Hood
Ep = fd.FiniteElement("CG", mesh.ufl_cell(), 1)
Ev = fd.VectorElement("CG", mesh.ufl_cell(), 2)
Ee = fd.FiniteElement("CG", mesh.ufl_cell(), 2)

# Alternative stable, pressure robust elements - Scott-Vogelius
#Ep = fd.FiniteElement("DG", mesh.ufl_cell(), 1, variant="alfeld")
#Ev = fd.VectorElement("CG", mesh.ufl_cell(), 2, variant="alfeld")

Evpe = fd.MixedElement([Ev, Ep, Ee])
W = fd.FunctionSpace(mesh, Evpe)

bcv_wall = fd.DirichletBC(W.sub(0), fd.as_vector([0,0]), [1,2,3,4])
bce_top = fd.DirichletBC(W.sub(2), 0, [4])
bce_down = fd.DirichletBC(W.sub(2), 1, [3])

# Collect boundary conditions
bcs = [bcv_wall, bce_top, bce_down]

dt = 0.5
t_end = 100
theta=Constant(1.0)   # Implicit Euler timestepping

g=Constant([0.0, -1.0]) # nondimensional gravity vector

# Benchmarks (doi: 10.1016/j.crme.2008.02.004 , doi: 10.1002/fld.395 )
Ra=1e4
Pr=0.71

# Define unknown and test function(s)
(v_, p_, e_) = fd.TestFunctions(W)

# current unknown time step
w = fd.Function(W)
(v, p, e) = fd.split(w)

# previous known time step
w0 = fd.Function(W)
(v0, p0, e0) = fd.split(w0)

D = sym(grad(v))
T = 2*Constant(Pr/Ra)*D
K = Constant(1.0/np.sqrt(Ra*Pr))

def a(v,v_) :
    return ( inner(grad(v)*v, v_)*dx + inner(T, grad(v_))*dx  + e*inner(g,v_)*dx ) 

def b(q,v) :
    return ( inner(div(v),q)*dx )

def c(v,e,e_) :
    return ( inner(v,grad(e))*e_*dx + inner(K*grad(e),grad(e_))*dx - inner(T,D)*e_*dx )

# variational form without time derivative in current time
Feq1 = a(v,v_) +  c(v,e,e_)
Feq2 = b(p_,v) - b(p,v_) 

# part of the equation without Lagrange multipliers
F = Feq1 

# variational forms without time derivative in previous time
F0=fd.replace(F, {w: w0}) 

#combine variational forms with time derivative
#
#  dw/dt + F(w,t) = 0 is approximated as
#  (w-w0)/dt + theta*F(w,t) + (1-theta)*F(w0,t0) = 0
#

vdot=Constant(1.0/dt)*inner((v-v0),v_)*dx
edot=Constant(1.0/dt)*inner((e-e0),e_)*dx
F = vdot + edot + theta*F + (1.0-theta)*F0 + Feq2

J = fd.derivative(F, w)

nullsp = fd.MixedVectorSpaceBasis(W, [W.sub(0), fd.VectorSpaceBasis(constant=True, comm=mesh.comm), W.sub(2)])


lu = {
    "snes_type": "newtonls",
    "snes_max_it": 40,
    "snes_rtol": 1e-10,
    "snes_atol": 1e-10,
    "snes_linesearch_type": "basic",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}

problem=fd.NonlinearVariationalProblem(F,w,bcs,J)
solver=fd.NonlinearVariationalSolver(problem, nullspace=nullsp, solver_parameters=lu, options_prefix="")

bce_top.apply(w)
bce_down.apply(w)

(v,p,e)=w.subfunctions
v.rename("v", "velocity")
p.rename("p", "pressure")
e.rename("e", "temperature")

# Time-stepping
t = 0.0

# Save to file
rfile = fd.VTKFile(name+"_vpe.pvd")
rfile.write(v,p,e,time=t)

while t < t_end:

    w0.assign(w)
    t += dt
    print(f"{t=}")
    solver.solve()

    rfile.write(v,p,e,time=t)

