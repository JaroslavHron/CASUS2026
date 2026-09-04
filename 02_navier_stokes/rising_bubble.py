import numpy as np
import firedrake as fd
from firedrake import grad,div,inner,sym,dx,ds,dS,outer,avg,jump,Constant
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
mesh = fd.RectangleMesh(n, 2*n, 1.0, 2.0)

# inf-sup stable element - Taylor - Hood
Ep = fd.FiniteElement("CG", mesh.ufl_cell(), 1)
Ev = fd.VectorElement("CG", mesh.ufl_cell(), 2)
El = fd.FiniteElement("CG", mesh.ufl_cell(), 1)

# Alternative stable, pressure robust elements - Scott-Vogelius
#Ep = fd.FiniteElement("DG", mesh.ufl_cell(), 1, variant="alfeld")
#Ev = fd.VectorElement("CG", mesh.ufl_cell(), 2, variant="alfeld")

Evpl = fd.MixedElement([Ev, Ep, El])
W = fd.FunctionSpace(mesh, Evpl)

# Define unknown and test function(s)
w = fd.Function(W)
w0 = fd.Function(W)

(v_, p_, l_) = fd.TestFunctions(W)

(v, p, l)=fd.split(w)
(v0, p0, l0)=fd.split(w0)

bcv= fd.DirichletBC(W.sub(0), fd.as_vector([0.0, 0.0]), [1,2,3,4])
bcs = [bcv]


# Time stepping parameters
dt = 0.1
t_end = 20.0
theta=Constant(0.5)   # theta schema
k=Constant(1.0/dt)
g=Constant([0.0,-1.0])

# surface tension
st=Constant(1.0)

rho1=1e2
rho2=1e3
mu1=1e1
mu2=1e0

eps=fd.CellDiameter(mesh)

x,y = fd.SpatialCoordinate(mesh)
# initial distance function

center = [0.5, 0.5]
radius = 0.25

buble1 = fd.sqrt((x-0.5)**2 + (y-1.5)**2) - 0.25
buble2 = fd.sqrt((x-.3)**2 + (y-1.3)**2) - 0.2
base = y - 0.5

def min(a,b):
    return(fd.conditional(a<b,a,b))

dist = min(base, min( buble1 , buble2))

def Sign(q):
    #return(fd.sign(q))
    #return fd.conditional(fd.lt(fd.abs(q),eps),q/eps,fd..sign(q))
    return q/fd.sqrt(q*q+eps*eps)

#def Delta(q):
#    return conditional(lt(abs(q),eps),(1.0/eps)*0.5*(1.0+cos(pi*q/eps)),Constant(0.0))

def rho(l):
    return(rho1 * 0.5* (1.0+ Sign(l)) + rho2 * 0.5*(1.0 - Sign(l)))

def nu(l):
   return(mu1 * 0.5* (1.0+ Sign(l)) + mu2 * 0.5*(1.0 - Sign(l)))


def EQ(v,p,l,v_,p_,l_):
    F_ls = inner(div(l*v),l_)*dx 

    T= -p*I + nu(l)*(grad(v)+grad(v).T)
    F_ns = inner(T,grad(v_))*dx + rho(l)*inner(grad(v)*v, v_)*dx - rho(l)*inner(g,v_)*dx

    rr=fd.sqrt(inner(grad(l),grad(l)))
    n=grad(l)/rr
    F_st = st*rr*inner(I-outer(n,n),grad(v_))*dx

    F = F_ls + F_ns + F_st
    return(F)

n = fd.FacetNormal(mesh)
I = fd.Identity(mesh.topological_dimension)
h = fd.CellDiameter(mesh)


F= ( k*(theta*rho(l)+(1.0-theta)*rho(l0))*inner(v-v0,v_)*dx
     + k*inner(l-l0,l_)*dx
     + theta*EQ(v,p,l,v_,p_,l_) + (1.0-theta)*EQ(v0,p,l0,v_,p_,l_)
     + div(v)*p_*dx
)

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
solver=fd.NonlinearVariationalSolver(problem, nullspace=nullsp, solver_parameters=lu, options_prefix="flow")

def reinit(l0):
    #implement here:
    #   given mesh and function l on the mesh
    # reinitialize function l such that |grad(l)|=1
    # and the zero levelset does not change (too much)

    l = fd.Function(l0.function_space())
    norm = lambda f: fd.sqrt(eps*eps+inner(f,f))
    R=((norm(grad(l))-1.0)**2)*dx(metadata={'quadrature_degree': 4}) + 1e3*((Sign(l)-Sign(l0))**2)*dx(metadata={'quadrature_degree': 4}) + 1e0*inner(jump(grad(l)),jump(grad(l)))*dS
    F=fd.derivative(R,l)
    J=fd.derivative(F,l)
    problem=fd.NonlinearVariationalProblem(F,l,[],J)
    solver=fd.NonlinearVariationalSolver(problem,  options_prefix="reinit")
    return(solver, l)


# Time-stepping
t = 0.0

(v,p,l) = w.subfunctions

v.rename("velocity")
p.rename("pressure")
l.rename("levelset")

l.interpolate(dist)

reinit_solver, ll = reinit(l)

# Save to file
rfile = fd.VTKFile(name+"_vpl.pvd")
rfile.write(v,p,l,time=t)

while t < t_end:

    w0.assign(w)
    t += dt
    print(f"{t=}")
    solver.solve()
    
    # reintialize the levelset
    ll.assign(l)
    reinit_solver.solve()
    l.assign(ll)

    rfile.write(v,p,l,time=t)

    V=fd.assemble(fd.conditional(fd.lt(l,0.0),1.0,0.0)*dx)
    print("  volume = {0:e}".format(V))
